"""Example of the effect of SWOT on forecast geostrophic currents (after the figure of the 2024 SWOT OSE data challenge).

Day t0 = a weekly start of 2024 (default: the start whose previous day, t0-1, has the largest SWOT coverage of the Agulhas box).
Currents: geostrophic, u = -g/f dh/dy, v = g/f dh/dx on the 1/4 deg grid, h = SLA + CNES-CLS22 MDT, |lat| < 3 deg masked.
a) global speed of the leadtime 1 forecast with Nadir+SWOT input (UNet-N, valid on t0);
b) speed from the KaRIn swath of day t0-1 (SWOT cells of the CLS gridded input, where the nadir-only grid is empty);
c) Agulhas, leadtime 1 forecast with Nadir+SWOT input; d) same with Nadir input; e) c - d.
Usage (cwd oceanbench_eval, python venvs/cyclo): python make_swot_current_example.py <out png> [YYYY-MM-DD]
"""
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Odyssey/private/d21botvy/oceanbench_eval")
from build_challenger import load_mdt_on_grid  # noqa: E402

SWOT = "/Odyssey/public/swot_traces/cls/gridded_obs_sla_0.25_swot_2024_reformated.nc"
NADIR = "/Odyssey/public/swot_traces/cls/gridded_obs_sla_0.25_nadirs_2024_without_swon.nc"
BOX = (-45, -25, 10, 45)
G, OMEGA = 9.81, 7.2921e-5
STARTS = pd.date_range("2024-01-17", "2024-12-11", freq="7D")


def speed(h, lat, lon):
    f = 2 * OMEGA * np.sin(np.deg2rad(lat))[:, None]
    dy = 111195.0 * 0.25
    dx = dy * np.cos(np.deg2rad(lat))[:, None]
    dhdy = np.gradient(h, axis=0) / dy
    dhdx = np.gradient(h, axis=1) / dx
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = -G / f * dhdy, G / f * dhdx
    s = np.hypot(u, v)
    s[np.abs(lat) < 3, :] = np.nan
    return s


def main(out, date=None):
    sw, nd = xr.open_dataset(SWOT), xr.open_dataset(NADIR)
    lat, lon = sw.lat.values, sw.lon.values
    in_box = (lat[:, None] >= BOX[0]) & (lat[:, None] <= BOX[1]) & (lon[None] >= BOX[2]) & (lon[None] <= BOX[3])
    def swot_only(day):
        a = sw.sla_filtered.sel(time=day).values; b = nd.sla_filtered.sel(time=day).values
        return np.where(np.isfinite(a) & ~np.isfinite(b), a, np.nan)
    cov = {t: float(np.isfinite(swot_only(t - pd.Timedelta(days=1)))[in_box].mean()) for t in STARTS}
    t0 = pd.Timestamp(date) if date else max(cov, key=cov.get)
    print("start", t0.date(), "coverage of the box at t0-1:", round(cov.get(t0, np.nan), 2))
    zs = {k: xr.open_zarr(f"{k}.zarr").zos.sel(first_day_datetime=t0).isel(lead_day_index=0).squeeze().load() for k in ("nadir_swot", "nadir_only")}
    flat, flon = zs["nadir_swot"].latitude.values, zs["nadir_swot"].longitude.values
    mdt = load_mdt_on_grid(flat, flon).values
    fs = {k: speed(v.values, flat, flon) for k, v in zs.items()}
    dsw = swot_only(t0 - pd.Timedelta(days=1))
    # KaRIn field on the forecast grid (the CLS grid is -80..81.75 on a wider longitude range)
    ksw = sw.sla_filtered.sel(time=t0 - pd.Timedelta(days=1)).copy(data=dsw).sel(lat=slice(flat[0], flat[-1])).interp(lon=flon, lat=flat).values
    kmdt = mdt
    ks = speed(ksw + kmdt, flat, flon)
    la, lo = (flat >= BOX[0] - 3) & (flat <= BOX[1] + 3), (flon >= BOX[2] - 3) & (flon <= BOX[3] + 3)
    ext = [flon[lo][0], flon[lo][-1], flat[la][0], flat[la][-1]]
    sub = lambda a: a[np.ix_(la, lo)]
    near = np.isfinite(sub(ksw)); 
    from scipy.ndimage import binary_dilation
    near = binary_dilation(near, iterations=2)
    dd = np.abs(sub(fs["nadir_swot"]) - sub(fs["nadir_only"]))
    print("mean |c-d| (m/s) within 0.5 deg of the swath: %.3f, elsewhere in the box: %.3f; max %.2f" % (np.nanmean(dd[near]), np.nanmean(dd[~near]), np.nanmax(dd)))
    fig = plt.figure(figsize=(7.2, 7.8)); gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1, 1], width_ratios=[1, 1, 0.045], hspace=0.30, wspace=0.22)
    vmax = 0.8
    ax = fig.add_subplot(gs[0, :2]); im = ax.imshow(fs["nadir_swot"], origin="lower", extent=[flon[0], flon[-1], flat[0], flat[-1]], vmin=0, vmax=vmax, cmap="viridis", aspect="auto")
    ax.set_title("a) Forecast, Nadir+SWOT input, leadtime 1 (valid %s)" % t0.date(), fontsize=8, loc="left")
    ax.plot([BOX[2], BOX[3], BOX[3], BOX[2], BOX[2]], [BOX[0], BOX[0], BOX[1], BOX[1], BOX[0]], "w-", lw=0.8)
    panels = [(1, 0, "b) KaRIn swath of day t0-1", sub(ks)), (1, 1, "c) Forecast, Nadir+SWOT", sub(fs["nadir_swot"])), (2, 0, "d) Forecast, Nadir", sub(fs["nadir_only"]))]
    for r, c, t, a in panels:
        ax = fig.add_subplot(gs[r, c]); im2 = ax.imshow(a, origin="lower", extent=ext, vmin=0, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_title(t, fontsize=8, loc="left")
    ax = fig.add_subplot(gs[2, 1]); d = sub(fs["nadir_swot"]) - sub(fs["nadir_only"])
    im3 = ax.imshow(d, origin="lower", extent=ext, vmin=-0.2, vmax=0.2, cmap="RdBu_r", aspect="auto"); ax.set_title("e) c - d", fontsize=8, loc="left")
    cax = fig.add_subplot(gs[0:2, 2]); fig.colorbar(im, cax=cax, label="Geostrophic speed (m s$^{-1}$)")
    cax2 = fig.add_subplot(gs[2, 2]); fig.colorbar(im3, cax=cax2, label="Change (m s$^{-1}$)")
    for a in fig.axes: a.tick_params(labelsize=6)
    for a in fig.axes[:5]:
        a.set_xlabel("Longitude (\u00b0)", fontsize=6); a.set_ylabel("Latitude (\u00b0)", fontsize=6)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
