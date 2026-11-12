"""Surface currents of the SWOT-paper forecasts against 15 m drifters, point by point along the drifter trajectories,
in the spirit of Uchida et al. (2026, GRL, doi:10.1029/2026GL121978, Fig. 4).

Truth: CMEMS In Situ TAC hourly drifter currents at 15 m (drogued), 2024 (/Odyssey/public/drifters/cmems/*_2024.nc).
Hourly speeds > 3 m/s are rejected, gaps <= 6 h are filled linearly, longer gaps split the series into segments,
segments shorter than 5 days are dropped. Each segment is low-passed with a 2nd-order Butterworth filter, 48 h cutoff,
applied forward and backward (Uchida et al. 2026), and the first and last 24 h of each segment are discarded. Every
hourly point between 20 and 60 deg N/S is kept, so one drifter gives many points. No Ekman correction (no 2024 wind
stress on Odyssey); it affects both inputs equally.

Forecasts: SSH = SLA + CNES-CLS22 MDT (zos of <key>.zarr in oceanbench_eval), 48 starts 2024-01-17..2024-12-11, lead day
k valid on start + k - 1 (lead_day_index k-1). Geostrophic and cyclogeostrophic currents with jaxparrow
(minimization_based, default settings, 2000 iterations; Bertrand et al.), computed on two bands (15-65 N, 65-15 S) with
16 cells of periodic padding in longitude. A daily forecast field is used for all hourly points of its valid UTC day and
interpolated bilinearly in space at the hourly drifter position.

Diagnostics: complex ratio r = w_drifter / w_forecast (w = u + i v), with |r| the speed mismatch and arg(r) the angle
from the forecast to the drifter velocity; polar histograms, share of points within 45 deg, median |r|, and the RMS
vector error, by system, input, subset (boundary-current boxes; points under the SWOT swath of day t0-1) and leadtime.
Rotary spectra of the drifter velocities (raw and low-passed; southern hemisphere conjugated, so that negative
frequencies are anticyclonic) justify the filter.

Usage (cwd = oceanbench_eval; python = venvs/cyclo, which has jaxparrow):
  python drifter_cyclo_eval.py drifters     # hourly points + rotary spectra  -> results/currents_cyclo/
  python drifter_cyclo_eval.py match KEY..  # forecast currents at the points (GPU) -> results/currents_cyclo/match_<key>.npz
  python drifter_cyclo_eval.py analyse      # tables + figures
"""
import glob
import sys
from pathlib import Path

import numpy
import pandas
import xarray
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, sosfiltfilt

EVAL = Path("/Odyssey/private/d21botvy/oceanbench_eval")
OUT = EVAL / "results" / "currents_cyclo"
DRIFTERS = "/Odyssey/public/drifters/cmems/drifter_*_2024.nc"
FIRST_DAYS = pandas.date_range("2024-01-17", "2024-12-11", freq="7D")
LEADS = 7
MAX_HOURLY_SPEED, MAX_GAP_H, MIN_SEGMENT_H, EDGE_H, CUTOFF_H = 3.0, 6, 120, 24, 48.0
LAT_MIN, LAT_MAX = 20.0, 60.0
BANDS = ((15.0, 65.0), (-65.0, -15.0))
PAD = 16
SYSTEMS = {  # paper name: (Nadir key, Nadir+SWOT key)
    "UNet-N": ("nadir_only", "nadir_swot"),
    "UNet-S": ("unet_swoterrscratch_nadir", "unet_swoterrscratch_nadir_swot"),
    "4DVarNet-N": ("fdv_leadfix_nadir", "fdv_leadfix_nadir_swot"),
    "4DVarNet-S": ("fdv29S_nadir", "fdv29S_nadir_swot"),
}
BOXES = {  # run_regions.py, as in the SLA tables of the paper: (lat_min, lat_max, lon_min, lon_max)
    "gulf_stream": (30.0, 45.0, -80.0, -45.0), "kuroshio": (25.0, 45.0, 125.0, 170.0), "agulhas": (-45.0, -25.0, 10.0, 45.0)}
MIN_SPEED = 0.25  # m/s, drifter (low-passed) speed, as the ORCAst set of drifter_currents_eval.py
SPECTRUM_WINDOW_H = 240


def in_boxes(lat, lon):
    return numpy.logical_or.reduce([(lat >= a) & (lat <= b) & (lon >= c) & (lon <= d) for a, b, c, d in BOXES.values()])


# ---------------------------------------------------------------- drifters

def rotary(w: numpy.ndarray) -> numpy.ndarray:
    """Rotary power spectrum of windows (n, SPECTRUM_WINDOW_H), hourly; fftshifted, frequencies in cpd."""
    w = (w - w.mean(1, keepdims=True)) * numpy.hanning(w.shape[1])
    return numpy.fft.fftshift(numpy.abs(numpy.fft.fft(w, axis=1)) ** 2, axes=1) / w.shape[1]


def build_drifters() -> None:
    sos = butter(2, 1.0 / CUTOFF_H, btype="low", fs=1.0, output="sos")
    tables, spectra = [], {g: {"raw": [], "lp": []} for g in ("all", "boundary")}
    for path in sorted(glob.glob(DRIFTERS)):
        ds = xarray.open_dataset(path)
        hourly = pandas.DataFrame({"lat": ds["lat"].values.astype(float),
                                   "lon": numpy.rad2deg(numpy.unwrap(numpy.deg2rad(ds["lon"].values.astype(float)))),
                                   "u": ds["ums"].values, "v": ds["vms"].values},
                                  index=pandas.to_datetime(ds["date"].values).round("h"))
        hourly = hourly[~hourly.index.duplicated()]
        hourly = hourly[numpy.hypot(hourly.u, hourly.v) <= MAX_HOURLY_SPEED].dropna()
        if len(hourly) < MIN_SEGMENT_H:
            continue
        full = hourly.reindex(pandas.date_range(hourly.index[0], hourly.index[-1], freq="h"))
        present = full.u.notna().values
        gap = ~present
        run = numpy.cumsum(numpy.r_[True, gap[1:] != gap[:-1]])
        long_gap = gap & (pandas.Series(run).map(pandas.Series(run).value_counts()).values > MAX_GAP_H)
        full = full.interpolate(limit_area="inside")
        full[long_gap] = numpy.nan  # only gaps <= MAX_GAP_H stay filled
        segment = numpy.cumsum(full.u.isna().values)  # a run of NaN (gap > MAX_GAP_H) separates segments
        for _, part in full[full.u.notna()].groupby(segment[full.u.notna().values]):
            if len(part) < MIN_SEGMENT_H:
                continue
            u_lp, v_lp = sosfiltfilt(sos, part.u.values), sosfiltfilt(sos, part.v.values)
            keep = slice(EDGE_H, len(part) - EDGE_H)
            frame = pandas.DataFrame({"time": part.index[keep], "lat": part.lat.values[keep],
                                      "lon": (part.lon.values[keep] + 180.0) % 360.0 - 180.0,
                                      "u": part.u.values[keep], "v": part.v.values[keep],
                                      "u_lp": u_lp[keep], "v_lp": v_lp[keep],
                                      "measured": present[full.index.get_indexer(part.index[keep])]})
            frame["drifter"] = int(Path(path).stem.split("_")[1])
            tables.append(frame)
            # rotary spectra on non-overlapping 10-day windows inside 20-60 deg
            n = len(frame) // SPECTRUM_WINDOW_H
            for i in range(n):
                window = frame.iloc[i * SPECTRUM_WINDOW_H:(i + 1) * SPECTRUM_WINDOW_H]
                lat = window.lat.mean()
                if not LAT_MIN <= abs(lat) <= LAT_MAX:
                    continue
                for kind, (u, v) in {"raw": (window.u, window.v), "lp": (window.u_lp, window.v_lp)}.items():
                    w = (u.values + 1j * v.values)[None]
                    w = numpy.conj(w) if lat < 0 else w
                    spectra["all"][kind].append(rotary(w)[0])
                    if in_boxes(numpy.array([lat]), numpy.array([window.lon.mean()]))[0]:
                        spectra["boundary"][kind].append(rotary(w)[0])
    table = pandas.concat(tables, ignore_index=True)
    table = table[(table.lat.abs() >= LAT_MIN) & (table.lat.abs() <= LAT_MAX)].reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "drifters_hourly_lp48_2024.csv.gz", index=False, float_format="%.5f")
    freq = numpy.fft.fftshift(numpy.fft.fftfreq(SPECTRUM_WINDOW_H, d=1.0 / 24.0))  # cpd
    numpy.savez(OUT / "rotary_spectra.npz", freq_cpd=freq,
                **{f"{g}_{k}": numpy.mean(v, axis=0) for g, d in spectra.items() for k, v in d.items()},
                **{f"{g}_n": len(d["raw"]) for g, d in spectra.items()})
    print(f"{table.drifter.nunique()} drifters, {len(table)} hourly points in 20-60 deg, "
          f"{len(spectra['all']['raw'])} spectral windows ({len(spectra['boundary']['raw'])} in boxes)", flush=True)


def load_points() -> pandas.DataFrame:
    points = pandas.read_csv(OUT / "drifters_hourly_lp48_2024.csv.gz", parse_dates=["time"])
    date = points.time.dt.floor("D")
    start = FIRST_DAYS.searchsorted(date, side="right") - 1
    lead = (date - FIRST_DAYS[numpy.maximum(start, 0)]).dt.days.values + 1
    ok = (start >= 0) & (lead >= 1) & (lead <= LEADS)
    points["start"], points["lead"] = numpy.where(ok, start, -1), numpy.where(ok, lead, -1)
    return points


# ---------------------------------------------------------------- forecasts

def band_currents(zos: numpy.ndarray, lat: numpy.ndarray, lon: numpy.ndarray):
    """Geostrophic and cyclogeostrophic currents (jaxparrow) on one latitude band, periodic padding in longitude."""
    import jax.numpy as jnp
    from jaxparrow.cyclogeostrophy import minimization_based
    ssh = numpy.concatenate([zos[:, -PAD:], zos, zos[:, :PAD]], axis=1)
    lon_pad = numpy.concatenate([lon[-PAD:] - 360.0, lon, lon[:PAD] + 360.0])
    lon2, lat2 = numpy.meshgrid(lon_pad, lat)
    r = minimization_based(jnp.asarray(lat2), jnp.asarray(lon2), ssh_t=jnp.asarray(ssh), is_grid_rectilinear=True,
                           rotate_to_geographic=False, return_geos=True)
    return lon_pad, {name: numpy.asarray(a) for name, a in (("ug", r.ug), ("vg", r.vg), ("uc", r.ucg), ("vc", r.vcg))}


def match(key: str, points: pandas.DataFrame) -> None:
    zos_all = xarray.open_zarr(EVAL / f"{key}.zarr").zos.sel(first_day_datetime=FIRST_DAYS.values)
    if "depth" in zos_all.dims:
        zos_all = zos_all.isel(depth=0)
    out = {name: numpy.full(len(points), numpy.nan, numpy.float32) for name in ("ug", "vg", "uc", "vc")}
    for s, first_day in enumerate(FIRST_DAYS):
        block = zos_all.isel(first_day_datetime=s).load()
        for lead in range(1, LEADS + 1):
            sel = numpy.flatnonzero((points.start.values == s) & (points.lead.values == lead))
            if len(sel) == 0:
                continue
            field = block.isel(lead_day_index=lead - 1)
            for lat_min, lat_max in BANDS:
                band = field.sel(latitude=slice(lat_min, lat_max))
                inside = sel[(points.lat.values[sel] >= lat_min) & (points.lat.values[sel] <= lat_max)]
                if len(inside) == 0:
                    continue
                lon_pad, currents = band_currents(band.values, band.latitude.values, band.longitude.values)
                xy = numpy.column_stack([points.lat.values[inside], points.lon.values[inside]])
                for name, values in currents.items():
                    interp = RegularGridInterpolator((band.latitude.values, lon_pad), values, bounds_error=False, fill_value=numpy.nan)
                    out[name][inside] = interp(xy)
        print(key, first_day.date(), flush=True)
    numpy.savez(OUT / f"match_{key}.npz", **out)


# ---------------------------------------------------------------- analysis

def ratio_stats(wd: numpy.ndarray, wf: numpy.ndarray) -> dict:
    r = wd / wf
    angle = numpy.rad2deg(numpy.angle(r))
    return {"n": len(r), "pct_angle45": 100 * numpy.mean(numpy.abs(angle) <= 45), "median_abs_angle": numpy.median(numpy.abs(angle)),
            "median_ratio": numpy.median(numpy.abs(r)), "speed_fc_cm_s": 100 * numpy.mean(numpy.abs(wf)),
            "speed_dr_cm_s": 100 * numpy.mean(numpy.abs(wd)), "rms_vec_err_cm_s": 100 * numpy.sqrt(numpy.mean(numpy.abs(wf - wd) ** 2))}


def analyse() -> None:
    sys.path.insert(0, str(EVAL))
    from swot_conditional import swot_masks
    points = load_points()
    fields = {key: numpy.load(OUT / f"match_{key}.npz") for pair in SYSTEMS.values() for key in pair}
    valid = (points.start.values >= 0) & (numpy.hypot(points.u_lp, points.v_lp).values > MIN_SPEED)
    for f in fields.values():
        valid &= numpy.isfinite(f["uc"]) & numpy.isfinite(f["ug"])
    p = points[valid].reset_index(drop=True)
    idx = numpy.flatnonzero(valid)
    grid_lat, grid_lon = numpy.arange(-80.0, 81.76, 0.25), numpy.arange(-180.0, 179.76, 0.25)
    masks = swot_masks(None, grid_lat, grid_lon, FIRST_DAYS.values)
    j = numpy.clip(numpy.round((p.lat.values - grid_lat[0]) / 0.25).astype(int), 0, len(grid_lat) - 1)
    i = numpy.round((p.lon.values - grid_lon[0]) / 0.25).astype(int) % len(grid_lon)
    subsets = {"all_20_60": numpy.ones(len(p), bool), "boundary": in_boxes(p.lat.values, p.lon.values),
               "swot_t-1": masks["swot_t-1"].values[p.start.values, j, i]}
    wd = (p.u_lp + 1j * p.v_lp).values
    rows, rng = [], numpy.random.default_rng(0)
    for system, keys in SYSTEMS.items():
        for kind in ("c", "g"):
            wf = {inp: (fields[k][f"u{kind}"][idx] + 1j * fields[k][f"v{kind}"][idx]) for inp, k in zip(("Nadir", "Nadir+SWOT"), keys)}
            for subset, member in subsets.items():
                for lead in range(1, LEADS + 1):
                    sel = member & (p.lead.values == lead)
                    stats = {inp: ratio_stats(wd[sel], w[sel]) for inp, w in wf.items()}
                    # paired bootstrap over starts of the change in pct_angle45 and in RMS vector error
                    starts = p.start.values[sel]
                    hit = {inp: numpy.abs(numpy.angle(wd[sel] / w[sel])) <= numpy.pi / 4 for inp, w in wf.items()}
                    err = {inp: numpy.abs(w[sel] - wd[sel]) ** 2 for inp, w in wf.items()}
                    agg = pandas.DataFrame({"s": starts, "hn": hit["Nadir"], "hs": hit["Nadir+SWOT"], "en": err["Nadir"], "es": err["Nadir+SWOT"]}
                                           ).groupby("s").agg(["sum", "size"])
                    boot_a, boot_e = [], []
                    for _ in range(1000):
                        b = agg.iloc[rng.integers(0, len(agg), len(agg))].sum()
                        boot_a.append(100 * (b[("hs", "sum")] - b[("hn", "sum")]) / b[("hn", "size")])
                        boot_e.append(100 * (numpy.sqrt(b[("es", "sum")] / b[("es", "size")]) / numpy.sqrt(b[("en", "sum")] / b[("en", "size")]) - 1))
                    for inp, st in stats.items():
                        rows.append({"system": system, "currents": "cyclogeostrophic" if kind == "c" else "geostrophic",
                                     "subset": subset, "lead": lead, "input": inp, "n_drifters": p.drifter.values[sel].size and len(numpy.unique(p.drifter.values[sel])),
                                     "n_starts": len(agg), **st,
                                     "d_pct_angle45_lo": numpy.percentile(boot_a, 2.5), "d_pct_angle45_hi": numpy.percentile(boot_a, 97.5),
                                     "d_rms_pct_lo": numpy.percentile(boot_e, 2.5), "d_rms_pct_hi": numpy.percentile(boot_e, 97.5)})
        print(system, "done", flush=True)
    table = pandas.DataFrame(rows)
    table.to_csv(OUT / "ratio_stats.csv", index=False, float_format="%.4f")
    numpy.savez(OUT / "lead1_ratios.npz", **{f"{system}|{inp}|{subset}": (wd / (fields[k]["uc"][idx] + 1j * fields[k]["vc"][idx]))[m & (p.lead.values == 1)]
                                             for system, keys in SYSTEMS.items() for inp, k in zip(("Nadir", "Nadir+SWOT"), keys)
                                             for subset, m in subsets.items()})
    lead1 = table[(table.lead == 1)].pivot_table(index=["currents", "subset", "system"], columns="input",
                                                   values=["pct_angle45", "median_ratio", "rms_vec_err_cm_s", "n"])
    print(lead1.round(2).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "drifters":
        build_drifters()
    elif sys.argv[1] == "match":
        pts = load_points()
        for k in sys.argv[2:]:
            match(k, pts)
    elif sys.argv[1] == "analyse":
        analyse()
