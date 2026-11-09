"""Render the paired Okutama camera comparison as a local vector figure."""

from collections import defaultdict
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_okutama_past_camera as study


ARMS = ("stable_cv", "camera_cv", "stable_ols4", "camera_ols4")
LABELS = ("CV · align past", "CV · align + reproject", "OLS4 · align past", "OLS4 · align + reproject")


def panel(ax, readout, index, metric, unit, marker):
    colors = ["#286768", "#286768", "#945332", "#945332"]
    for j, (arm, color) in enumerate(zip(ARMS, colors, strict=True)):
        grouping = defaultdict(list)
        for row in readout["by_clip"]:
            value = row["arms"][arm][index]
            if value is not None:
                grouping[row["scenario"]].append(value)
        values = sorted(np.mean(v) for v in grouping.values())
        y = np.linspace(-.10, .10, len(values)) if len(values) > 1 else np.zeros(len(values))
        ax.scatter(values, j+.19+y, s=12, color=color, alpha=.50, linewidths=0, zorder=2)
        s = readout["all_windows"][arm][metric]
        if s is not None:
            lo, hi = s["ci95"]
            mean = s["site_mean"]
            ax.errorbar(mean, j-.12, xerr=[[mean-lo], [hi-mean]], fmt="o", color=color,
                        markersize=5, capsize=3, elinewidth=1.3, zorder=3)
    ax.axvline(0., color="#424442", linewidth=.8, linestyle="--")
    ax.set_ylim(len(ARMS)-.45, -.45)
    ax.set_xlabel(unit)
    ax.set_title(marker, loc="left", fontsize=10, fontweight="semibold", pad=9)
    ax.tick_params(axis="both", length=0, labelsize=8, pad=5)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e1e3e0", linewidth=.6)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    p = study.OUT/"readout.json"
    result = study.study.read(p)
    if (result["forecast_freeze"] != study.study.artifact(study.OUT/"forecast_freeze.json")
            or result["future_images_accessed"] or result["reserved_outcomes_opened"]):
        raise ValueError("Require bound past-image evidence")
    fig, ax = plt.subplots(1, 2, figsize=(9.1, 2.65), sharey=True, layout="constrained")
    panel(ax[0], result, 2, "lower_gain_vs_matched_raw_pp", "Conservative individual gain (pp)", "a   Individual forecast")
    panel(ax[1], result, 5, "pair_gain_lower_px", "Conservative pair gain (px)", "b   Relative motion")
    ax[0].set_yticks(np.arange(len(LABELS)), LABELS)
    fig.suptitle("Okutama-Action | Paired differences from the matching raw forecast", x=.012, ha="left", fontsize=11)
    out = study.LOCAL/"figure"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/"okutama_camera_gain.pdf", metadata={"CreationDate": None})
    fig.savefig(out/"okutama_camera_gain.png", dpi=150)
    plt.close(fig)
    print(out/"okutama_camera_gain.pdf")


if __name__ == "__main__":
    main()
