"""Render a paper figure from replay-verified aggregate results."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "outputs/publication_readiness_2026_09"
DEFAULT = ROOT / "data/stage_cvpr2027_experiments/native_pixel_gain_v1/figures"


def readout(name):
    directory = HOME / name
    path = directory / "analysis.json"
    raw = path.read_bytes()
    dig = hashlib.sha256(raw).hexdigest()
    for phase in ("score", "replay"):
        a = json.loads((directory / f"{phase}_execution.json").read_text())["artifact"]
        if ROOT / a["path"] != path or a["sha256"] != dig or a["bytes"] != len(raw):
            raise ValueError(f"Unverified readout: {name}")
    d = json.loads(raw)
    if d["independent_confirmation"]:
        raise ValueError("The figure is for held fitting sources only")
    return d


def render(out):
    obs = readout("native_observability_v1")
    cond = readout("native_conditional_gain_v1")["modes"]["support_matched"]["contrasts"]
    pixel = readout("native_pixel_gain_v1")["direct"]
    brier = [(label, obs["brier"][key]) for label, key in (
        ("Donor prevalence", "prevalence"), ("Motion only", "factor_motion"),
        ("Motion + box quality", "factor_quality"))]
    gain = [(label, value["full_grid_lower"]) for label, value in (
        ("Box support vs. OLS4", obs["direct"]["factor_quality_vs_ols4"]),
        ("Conditional vs. support only", cond["conditional_point_vs_support_only"]),
        ("Shared-step rank vs. conditional", cond["common_step_rank_vs_conditional_point"]),
        ("Background vs. no image", pixel["background_vs_no_image"]),
        ("Foreground vs. background", pixel["foreground_vs_background"]))]
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "svg.hashsalt": "m3w-support-utility-v1",
                         "axes.spines.right": False, "axes.spines.top": False})
    fig, ax = plt.subplots(2, 1, figsize=(7.4, 5.2), gridspec_kw={"height_ratios": (3, 5)})
    fig.subplots_adjust(left=.37, right=.96, top=.89, bottom=.14, hspace=.72)
    for i, (a, rows, title, label) in enumerate((
        (ax[0], brier, "A  |  Future-label availability", "Missing-step Brier score (lower is better)"),
        (ax[1], gain, "B  |  Forecast replacement", "Direct full-grid lower gain (points on observed CV error)"),
    )):
        point = np.array([r["point"] for _, r in rows])
        lo, hi = np.array([r["interval95"] for _, r in rows]).T
        y = np.arange(len(rows))
        for j, v in enumerate(point):
            color = ("#147d72" if i == 0 and j == 2 else
                     "#b24734" if i == 1 and hi[j] < 0 else "#35475b")
            a.errorbar(v, j, xerr=[[v-lo[j]], [hi[j]-v]], color=color,
                       marker="o", capsize=3, markersize=5, linewidth=1.3)
        a.set_yticks(y, [name for name, _ in rows])
        a.invert_yaxis()
        a.set_xlabel(label, labelpad=8)
        a.set_title(title, loc="left", fontweight="bold", pad=11)
        a.grid(axis="x", alpha=.2)
        a.tick_params(axis="y", length=0, pad=8)
    ax[1].axvline(0., color="#111827", linewidth=1, linestyle="--")
    fig.text(.37, .035, "Nine held fitting sources, three seeds; 95% source intervals.",
             fontsize=8, color="#475569")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "support_utility.svg", metadata={"Date": None})
    fig.savefig(out / "support_utility.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT)
    render(p.parse_args().output_dir)
