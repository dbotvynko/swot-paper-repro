"""Separation between drifters and virtual particles advected by the forecast currents, against leadtime, for the four
SWOT-paper systems with Nadir and Nadir+SWOT input (fig-separation).

Trajectories: oceanbench_eval/lagrangian_drifters.py outputs (6-day trajectories seeded at 15 m drifter positions,
20-60 deg N/S, at each forecast start; geostrophic currents of the forecast). Same trajectories for all eight runs
(common set). (a) mean separation per day relative to UNet-N with Nadir input (%; absolute values in the csv); (b) change with SWOT input, sep(N+S)/sep(N) - 1, with a 95 % bootstrap
interval over the starts.

Usage (cwd = oceanbench_eval): python make_separation_figure.py <out dir>
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import pandas

sys.path.insert(0, "/Odyssey/private/d21botvy/oceanbench_eval")
import lagrangian_drifters as ld  # noqa: E402

TRAJ = Path("/Odyssey/private/d21botvy/oceanbench_eval/results/lagrangian/trajectories")
VIEWS = Path("/Odyssey/private/d21botvy/oceanbench_eval")
RUNS = {
    "UNet-N": (TRAJ / "nadir_only.csv.gz", TRAJ / "nadir_swot.csv.gz"),
    "UNet-S": (TRAJ / "unet_swoterrscratch_nadir.csv.gz", TRAJ / "unet_swoterrscratch_nadir_swot.csv.gz"),
    "4DVarNet-N": (VIEWS / "fdv_view/results/lagrangian/trajectories/nadir_only.csv.gz", VIEWS / "fdv_view/results/lagrangian/trajectories/nadir_swot.csv.gz"),
    "4DVarNet-S": (VIEWS / "fdv29S_view/results/lagrangian/trajectories/nadir_only.csv.gz", VIEWS / "fdv29S_view/results/lagrangian/trajectories/nadir_swot.csv.gz"),
}
COLORS = dict(zip(RUNS, ["#0072B2", "#56B4E9", "#D55E00", "#E69F00"]))
REPETITIONS = 2000
plt.rcParams.update({"font.size": 8})


def main(out: Path) -> None:
    wide = {(s, i): ld.load_wide(str(p)) for s, pair in RUNS.items() for i, p in zip(("Nadir", "Nadir+SWOT"), pair)}
    index = None
    for w in wide.values():
        k = pandas.MultiIndex.from_arrays(w["key"].T)
        index = k if index is None else index.intersection(k)
    sep = {}
    for name, w in wide.items():
        take = pandas.MultiIndex.from_arrays(w["key"].T).get_indexer(index)
        picked = {k: v[take] for k, v in w.items() if k != "key"}
        sep[name] = ld.per_trajectory(picked)["sep"]
    starts = pandas.factorize(index.get_level_values(0))[0]
    valid = numpy.logical_and.reduce([numpy.isfinite(v).all(axis=1) for v in sep.values()])
    days = numpy.arange(1, sep[("UNet-N", "Nadir")].shape[1] + 1)
    rng = numpy.random.default_rng(0)
    rows = []
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ref = sep[("UNet-N", "Nadir")][valid].mean(0)
    for system in RUNS:
        a, b = sep[(system, "Nadir")][valid], sep[(system, "Nadir+SWOT")][valid]
        s = starts[valid]
        per_start_a = numpy.array([numpy.bincount(s, weights=a[:, d]) for d in range(len(days))])
        per_start_b = numpy.array([numpy.bincount(s, weights=b[:, d]) for d in range(len(days))])
        change = 100 * (b.mean(0) / a.mean(0) - 1)
        boot = []
        for _ in range(REPETITIONS):
            pick = rng.integers(0, per_start_a.shape[1], per_start_a.shape[1])
            boot.append(100 * (per_start_b[:, pick].sum(1) / per_start_a[:, pick].sum(1) - 1))
        lo, hi = numpy.percentile(boot, [2.5, 97.5], axis=0)
        ax_a.plot(days, 100 * (a.mean(0) / ref - 1), color=COLORS[system], lw=1.4, ls="--", marker="o", ms=2.5)
        ax_a.plot(days, 100 * (b.mean(0) / ref - 1), color=COLORS[system], lw=1.4, marker="o", ms=2.5, label=system)
        ax_b.fill_between(days, lo, hi, color=COLORS[system], alpha=0.15, lw=0)
        ax_b.plot(days, change, color=COLORS[system], lw=1.4, marker="o", ms=2.5, label=system)
        for d in range(len(days)):
            rows.append({"system": system, "day": days[d], "n_traj": int(valid.sum()), "sep_nadir_km": a[:, d].mean(),
                         "sep_nadir_swot_km": b[:, d].mean(), "change_pct": change[d], "ci_low": lo[d], "ci_high": hi[d]})
    ax_a.plot([], [], color="0.4", ls="--", label="Nadir input")
    ax_a.plot([], [], color="0.4", label="Nadir+SWOT input")
    ax_a.set_xlabel("Leadtime (days)"); ax_a.axhline(0, color="0.4", lw=0.6)
    ax_a.set_ylabel("Separation relative to UNet-N, Nadir (%)"); ax_a.set_title("(a) Separation by system and input", loc="left", fontsize=8)
    ax_b.axhline(0, color="0.4", lw=0.6)
    ax_b.set_xlabel("Leadtime (days)"); ax_b.set_ylabel("Change with SWOT input (%)"); ax_b.set_title("(b) Change in separation with SWOT", loc="left", fontsize=8)
    ax_a.legend(frameon=False, fontsize=6.5, ncol=1)
    for ax in (ax_a, ax_b):
        ax.grid(lw=0.3, color="0.85"); ax.set_xticks(days)
    fig.tight_layout()
    fig.savefig(out / "fig-separation.pdf")
    table = pandas.DataFrame(rows)
    table.to_csv(out / "fig-separation.csv", index=False, float_format="%.3f")
    print(f"{valid.sum()} trajectories, {len(numpy.unique(starts[valid]))} starts")
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
