"""Figures of the drifter currents evaluation (outputs of drifter_cyclo_eval.py) for the SWOT paper.

  fig-currents-polar.pdf : polar histograms of r = w_drifter / w_forecast (cyclogeostrophic), leadtime 1, rows = systems,
                           columns = boundary currents (Nadir, Nadir+SWOT), under the t0-1 swath (Nadir, Nadir+SWOT).
  fig-currents-lead.pdf  : change with SWOT input against leadtime: share of points within 45 deg and RMS vector error,
                           with 95 % paired bootstrap intervals over starts.
  fig-rotary-spectra.pdf : rotary spectra of the drifter velocities, raw and 48 h low-passed (Supporting Information).

Usage: python make_current_figures.py <paper figures dir>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import pandas

RES = Path("/Odyssey/private/d21botvy/oceanbench_eval/results/currents_cyclo")
SYSTEMS = ["UNet-N", "UNet-S", "4DVarNet-N", "4DVarNet-S"]
COLORS = dict(zip(SYSTEMS, ["#0072B2", "#56B4E9", "#D55E00", "#E69F00"]))  # Okabe-Ito, fixed order
SUBSETS = {"boundary": "Boundary currents", "swot_t-1": r"Under the $t_0-1$ swath"}
R_MAX = 3.0
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})


def polar(out: Path) -> None:
    ratios = numpy.load(RES / "lead1_ratios.npz")
    stats = pandas.read_csv(RES / "ratio_stats.csv")
    stats = stats[(stats.lead == 1) & (stats.currents == "cyclogeostrophic")]
    theta_edges = numpy.deg2rad(numpy.linspace(-180, 180, 37))
    r_edges = numpy.linspace(0, R_MAX, 31)
    columns = [(s, i) for s in SUBSETS for i in ("Nadir", "Nadir+SWOT")]
    fig, axes = plt.subplots(len(SYSTEMS), 4, figsize=(7.2, 7.6), subplot_kw={"projection": "polar"})
    for row, system in enumerate(SYSTEMS):
        for col, (subset, inp) in enumerate(columns):
            ax = axes[row, col]
            r = ratios[f"{system}|{inp}|{subset}"]
            h, _, _ = numpy.histogram2d(numpy.angle(r), numpy.clip(numpy.abs(r), 0, R_MAX - 1e-6), [theta_edges, r_edges])
            h = h / h.sum() / (numpy.diff(r_edges)[None] * 0.5 * (r_edges[1:] + r_edges[:-1])[None])  # density per area
            ax.pcolormesh(theta_edges, r_edges, h.T, cmap="Blues", vmin=0, vmax=numpy.percentile(h, 99.5), shading="flat", rasterized=True)
            ax.plot(numpy.linspace(-numpy.pi, numpy.pi, 200), numpy.ones(200), color="0.35", lw=0.6, ls="--")
            ax.plot([0], [1], marker="+", color="#00BFC4", ms=10, mew=1.6)
            ax.set_ylim(0, R_MAX); ax.set_yticks([1, 2]); ax.set_yticklabels(["1", "2"], fontsize=6)
            ax.set_xticks(numpy.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(["0°", "90°", "±180°", "-90°"], fontsize=6)
            ax.grid(lw=0.3, color="0.8")
            st = stats[(stats.system == system) & (stats.subset == subset) & (stats.input == inp)].iloc[0]
            ax.text(0.5, -0.16, f"{st.pct_angle45:.0f} % within 45°, median |r| {st.median_ratio:.2f}", transform=ax.transAxes,
                    ha="center", fontsize=6.5)
            if row == 0:
                ax.set_title(f"{SUBSETS[subset]}\n{inp}", fontsize=8, pad=14)
            if col == 0:
                ax.text(-0.42, 0.5, system, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.04, wspace=0.45, hspace=0.55)
    fig.savefig(out / "fig-currents-polar.pdf", dpi=250)
    plt.close(fig)


ALL_SUBSETS = {"all_20_60": "All points 20-60°", **SUBSETS}


def polar_by_subset(out: Path, r_max: float = 5.0) -> None:
    """Review figures: one per subset, rows = systems, columns = Nadir and Nadir+SWOT input, same colour scale in a figure."""
    ratios = numpy.load(RES / "lead1_ratios.npz")
    stats = pandas.read_csv(RES / "ratio_stats.csv")
    stats = stats[(stats.lead == 1) & (stats.currents == "cyclogeostrophic")]
    theta_edges = numpy.deg2rad(numpy.linspace(-180, 180, 37))
    r_edges = numpy.linspace(0, r_max, 26)
    area = numpy.diff(r_edges)[None] * 0.5 * (r_edges[1:] + r_edges[:-1])[None]
    for subset, title in ALL_SUBSETS.items():
        hist, outside = {}, {}
        for system in SYSTEMS:
            for inp in ("Nadir", "Nadir+SWOT"):
                r = ratios[f"{system}|{inp}|{subset}"]
                h, _, _ = numpy.histogram2d(numpy.angle(r), numpy.abs(r), [theta_edges, r_edges])  # |r| > r_max left out (share printed)
                hist[(system, inp)] = h / len(r) / area
                outside[(system, inp)] = 100 * numpy.mean(numpy.abs(r) > r_max)
        vmax = numpy.percentile(numpy.concatenate([h.ravel() for h in hist.values()]), 99.5)
        fig, axes = plt.subplots(len(SYSTEMS), 2, figsize=(5.6, 11.5), subplot_kw={"projection": "polar"})
        for row, system in enumerate(SYSTEMS):
            for col, inp in enumerate(("Nadir", "Nadir+SWOT")):
                ax = axes[row, col]
                ax.pcolormesh(theta_edges, r_edges, hist[(system, inp)].T, cmap="Blues", vmin=0, vmax=vmax, shading="flat", rasterized=True)
                ax.plot(numpy.linspace(-numpy.pi, numpy.pi, 200), numpy.ones(200), color="0.35", lw=0.6, ls="--")
                ax.plot([0], [1], marker="+", color="#D55E00", ms=6, mew=1.0)
                ax.set_ylim(0, r_max); ax.set_yticks([1, 2, 3, 4]); ax.set_yticklabels(["1", "2", "3", "4"], fontsize=6)
                ax.set_xticks(numpy.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(["0°", "90°", "±180°", "-90°"], fontsize=6)
                ax.grid(lw=0.3, color="0.8")
                st = stats[(stats.system == system) & (stats.subset == subset) & (stats.input == inp)].iloc[0]
                ax.text(0.5, -0.30, f"{st.pct_angle45:.1f} % within 45°, median |r| {st.median_ratio:.2f}\n"
                        f"RMS vector error {st.rms_vec_err_cm_s:.1f} cm/s; |r| > {r_max:.0f}: {outside[(system, inp)]:.1f} %", transform=ax.transAxes, ha="center", fontsize=6.5)
                if row == 0:
                    ax.set_title(f"{inp} input", fontsize=9, pad=16)
                if col == 0:
                    ax.text(-0.35, 0.5, system, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=9)
        fig.suptitle(f"{title}, leadtime 1: r = drifter / forecast velocity (cyclogeostrophic)\n"
                     f"angle = direction error, radius = speed ratio; + marks a perfect forecast; n = {int(st.n)} points", fontsize=8, y=0.995)
        fig.subplots_adjust(left=0.1, right=0.97, top=0.90, bottom=0.07, hspace=0.75, wspace=0.35)
        fig.savefig(out / f"review-polar-{subset}.png", dpi=170)
        plt.close(fig)


def lead(out: Path) -> None:
    stats = pandas.read_csv(RES / "ratio_stats.csv")
    stats = stats[stats.currents == "cyclogeostrophic"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    for col, subset in enumerate(SUBSETS):
        for system in SYSTEMS:
            t = stats[(stats.system == system) & (stats.subset == subset)].pivot(index="lead", columns="input")
            d_angle = t[("pct_angle45", "Nadir+SWOT")] - t[("pct_angle45", "Nadir")]
            d_rms = 100 * (t[("rms_vec_err_cm_s", "Nadir+SWOT")] / t[("rms_vec_err_cm_s", "Nadir")] - 1)
            for ax, d, lo, hi in ((axes[0, col], d_angle, "d_pct_angle45_lo", "d_pct_angle45_hi"), (axes[1, col], d_rms, "d_rms_pct_lo", "d_rms_pct_hi")):
                ax.fill_between(t.index, t[(lo, "Nadir")], t[(hi, "Nadir")], color=COLORS[system], alpha=0.15, lw=0)
                ax.plot(t.index, d, color=COLORS[system], lw=1.6, marker="o", ms=3, label=system)
        axes[0, col].set_title(SUBSETS[subset])
        for ax in axes[:, col]:
            ax.axhline(0, color="0.4", lw=0.6)
            ax.grid(lw=0.3, color="0.85")
        axes[1, col].set_xlabel("Leadtime (days)")
    axes[0, 0].set_ylabel("Change in share within 45°\n(percentage points)")
    axes[1, 0].set_ylabel("Change in RMS vector error (%)")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig-currents-lead.pdf")
    plt.close(fig)


def rotary(out: Path) -> None:
    s = numpy.load(RES / "rotary_spectra.npz")
    f = s["freq_cpd"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, (side, mask, label) in zip(axes, (("anticyclonic", f < 0, "Anticyclonic"), ("cyclonic", f > 0, "Cyclonic"))):
        for kind, ls, name in (("raw", "-", "hourly"), ("lp", "--", "48 h low-pass")):
            ax.loglog(numpy.abs(f[mask]), s[f"all_{kind}"][mask], color="#0072B2", ls=ls, lw=1.2, label=f"20-60°, {name}")
            ax.loglog(numpy.abs(f[mask]), s[f"boundary_{kind}"][mask], color="#D55E00", ls=ls, lw=1.2, label=f"boundary currents, {name}")
        ax.axvspan(2 * 7.2921e-5 * numpy.sin(numpy.deg2rad(20)) * 86400 / (2 * numpy.pi),
                   2 * 7.2921e-5 * numpy.sin(numpy.deg2rad(60)) * 86400 / (2 * numpy.pi), color="0.9", lw=0, zorder=0)
        for x in (0.5, 1.0, 2.0):
            ax.axvline(x, color="0.4", lw=0.6, ls=":")
        ax.set_title(label); ax.set_xlabel("Frequency (cpd)"); ax.grid(lw=0.3, color="0.85")
    axes[0].set_ylabel(r"Power (m$^2$ s$^{-2}$ cpd$^{-1}$, arbitrary scale)")
    axes[1].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "fig-rotary-spectra.pdf")
    plt.close(fig)


if __name__ == "__main__":
    target = Path(sys.argv[1])
    for job in (polar, lead, rotary):
        try:
            job(target)
            print(job.__name__, "ok")
        except FileNotFoundError as error:
            print(job.__name__, "skipped:", error)
