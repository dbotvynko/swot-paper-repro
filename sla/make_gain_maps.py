"""Map of the change in SLA RMSE with SWOT input, RMSE(Nadir+SWOT)/RMSE(Nadir) - 1, per 2 deg box (>= 200 along-track
points), one panel per system, over the 2024 starts (DT along-track altimetry, CNES-CLS22 MDT).

Input: <view>/results/swot_gain/maps.npz written by oceanbench_eval/swot_gain_maps.py (key sla_<lead>, gain = -change).
Usage: python make_gain_maps.py <out dir> [lead]   -> fig-gain-maps.pdf (default lead 1)
"""
import sys
from pathlib import Path

import cartopy

cartopy.config["data_dir"] = "/Odyssey/private/d21botvy/cartopy_data"
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import BoundaryNorm

EVAL = Path("/Odyssey/private/d21botvy/oceanbench_eval")
SYSTEMS = {"UNet-N": EVAL, "UNet-S": EVAL / "unetS_view", "4DVarNet-N": EVAL / "fdv_view", "4DVarNet-S": EVAL / "fdv29S_view"}
BOXES = {"gulf_stream": (30, 45, -80, -45), "kuroshio": (25, 45, 125, 170), "agulhas": (-45, -25, 10, 45)}
plt.rcParams.update({"font.size": 8})


def main(out: Path, lead: int) -> None:
    levels = numpy.arange(-30, 31, 5)
    norm = BoundaryNorm(levels, 256, extend="both")
    lat_edges, lon_edges = numpy.arange(-90, 90.1, 2), numpy.arange(-180, 180.1, 2)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), subplot_kw={"projection": ccrs.Robinson(central_longitude=0)})
    for ax, (system, view) in zip(axes.flat, SYSTEMS.items()):
        change = -numpy.load(view / "results" / "swot_gain" / "maps.npz")[f"sla_{lead}"]
        mesh = ax.pcolormesh(lon_edges, lat_edges, change, cmap="RdBu_r", norm=norm, transform=ccrs.PlateCarree(), rasterized=True)
        ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="none", zorder=2)
        ax.coastlines(lw=0.3, zorder=3)
        for a0, a1, b0, b1 in BOXES.values():
            ax.plot([b0, b1, b1, b0, b0], [a0, a0, a1, a1, a0], color="k", lw=0.7, transform=ccrs.PlateCarree(), zorder=4)
        for y in (-20, 20):
            ax.plot([-180, 180], [y, y], color="0.3", lw=0.5, ls="--", transform=ccrs.PlateCarree(), zorder=4)
        ax.set_global()
        ax.set_title(system, fontsize=9)
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.05, pad=0.04, aspect=40, ticks=levels)
    cbar.set_label(f"Change in SLA RMSE with SWOT input, leadtime {lead} (%); negative: SWOT helps")
    fig.savefig(out / "fig-gain-maps.pdf", dpi=250, bbox_inches="tight")
    print("written", out / "fig-gain-maps.pdf")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 1)
