"""Render the fixed-versus-selected harm coverage exhibit from frozen results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"outputs/publication_readiness_2026_09/av2_harm_quantile_v1/postselection.json"
DEFAULT = ROOT/"data/stage_cvpr2027_experiments/submission_admin/av2_selection_coverage"


def draw(source, output):
    r = json.loads(source.read_bytes())
    if r["reserved_outcomes_opened"] or r["two_slot_scenarios"] != 340:
        raise ValueError("Expected the frozen development contrast")
    labels = ["Overall positive harm", "Easy positive harm", "Signed easy harm"]
    coverage = 100*np.asarray(r["pooled_coverage"])
    delta = 100*np.asarray(r["paired_delta"]["site_mean"])
    ci = 100*np.asarray(r["paired_delta"]["ci95"])
    y = np.arange(len(labels))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
                         "axes.titlesize": 10, "svg.hashsalt": "m3w-av2-coverage"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.65), sharey=True)
    fig.subplots_adjust(left=.24, right=.98, bottom=.23, top=.80, wspace=.28)
    dark, blue = "#333333", "#24627a"
    for ax in (a, b):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", linewidth=.6, alpha=.25)
        ax.set_axisbelow(True)
    a.scatter(coverage[0], y-.12, color=dark, marker="o", s=28, label="Frozen choice", zorder=3)
    a.scatter(coverage[1], y+.12, color=blue, marker="s", s=26, label="Choice optimized by estimate", zorder=3)
    a.axvline(80., color="#777777", linewidth=.7, linestyle="--")
    a.set(xlim=(65, 86), xticks=[65, 70, 75, 80, 85], yticks=y, yticklabels=labels,
          xlabel="Pooled upper coverage (%)", title="(a) Same estimator, different choice")
    a.invert_yaxis()
    b.errorbar(delta, y, xerr=np.vstack((delta-ci[:, 0], ci[:, 1]-delta)),
               color=blue, fmt="s", markersize=4.5, capsize=3, linewidth=1.25)
    b.axvline(0., color="#777777", linewidth=.8)
    b.set(xlim=(-16, 1), xticks=[-15, -10, -5, 0], xlabel="Paired coverage change (pp)",
          title="(b) City-balanced change, 95% CI")
    b.tick_params(labelleft=False)
    handles, names = a.get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncols=2, frameon=False)
    fig.text(.5, .03, "340 AV2 development scenarios; 3 seeds; exact matched two-edit menu.",
             ha="center", fontsize=8, color=dark)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(output.with_suffix(".png"), dpi=230)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    draw(SOURCE, p.parse_args().output)
