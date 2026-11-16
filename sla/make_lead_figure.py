"""Change in SLA RMSE with SWOT input against leadtime, UNet-N, 48 starts of 2024 (fig-lead).

change = RMSE(Nadir+SWOT) / RMSE(Nadir) - 1 (negative = SWOT helps), against DT along-track altimetry, globally (with its 95 %
bootstrap interval over the starts), by latitude band and in the boundary-current boxes. Same data and layout as the left
panel of the former fig1_change_by_lead_day.png (oceanbench_eval/paper_change_figures.py); the currents panel is dropped
(replaced by fig-currents-lead). Input: oceanbench_eval/results/swot_gain/sla_curves.csv (gain = -change), nothing recomputed.

Usage: python make_lead_figure.py <out dir>   -> fig-lead.pdf
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas  # noqa: E402

CURVES = Path("/Odyssey/private/d21botvy/oceanbench_eval/results/swot_gain/sla_curves.csv")
STYLES = {"global": ("Global", "#14181d", "-", 2.2), "n40_n82": ("40-82° N", "#4a3aa7", "-", 1.2),
          "n20_n40": ("20-40° N", "#2a78d6", "-", 1.2), "s20_n20": ("20° S-20° N", "#c0392b", "-", 1.6),
          "s40_s20": ("40-20° S", "#1baf7a", "-", 1.2), "s80_s40": ("80-40° S", "#0f5d63", "-", 1.2),
          "gulf_stream": ("Gulf Stream", "#eb6834", "--", 1.6), "kuroshio": ("Kuroshio", "#e87ba4", "--", 1.6),
          "agulhas": ("Agulhas", "#8a5a00", "--", 1.6)}
plt.rcParams.update({"font.size": 8})


def main(out: Path) -> None:
    table = pandas.read_csv(CURVES)
    table["change_pct"], table["lo"], table["hi"] = -table.gain_pct, -table.ci_high, -table.ci_low
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for area, (label, color, style, width) in STYLES.items():
        s = table[table.area == area].sort_values("lead_day")
        if s.empty:
            continue
        ax.plot(s.lead_day, s.change_pct, color=color, ls=style, lw=width, marker="o", ms=2.5, label=label)
        if area == "global":
            ax.fill_between(s.lead_day, s.lo, s.hi, color=color, alpha=0.15, lw=0)
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_xlabel("Leadtime (days)"); ax.set_xticks(range(1, 8))
    ax.set_ylabel("Change in SLA RMSE with SWOT input (%)")
    ax.grid(lw=0.3, color="0.85"); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig-lead.pdf")
    print("written", out / "fig-lead.pdf")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
