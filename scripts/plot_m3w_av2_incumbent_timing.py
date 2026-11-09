"""Render a small anonymous supplement figure from the two sealed AV2 readouts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/"outputs/publication_readiness_2026_09/av2_incumbent_timing_v1"
DEFAULT = ROOT/"data/stage_cvpr2027_experiments/submission_admin/av2_incumbent_timing"
MODES = ("replay", "geometry", "history")


def numbers():
    s, a = [json.loads((REPORT/k).read_bytes()) for k in ("summary.json", "capacity_attribution.json")]
    if any(x["reserved_outcomes_opened"] or x["independent_confirmation"] or not x["original_2pct"] for x in (s, a)):
        raise ValueError("Figure requires the original development-only protocol")
    c, d = s["counts"], a["counts"]
    if (c["two_slots"] != d["two_slots"] or c["local_available"] != d["local_timing"]
            or c["anchor_eligible"] != d["anchor_eligible"]):
        raise ValueError("Capacity and decision counts disagree")
    cap = [d[k] for k in ("local_timing", "same_actor_and_peak", "same_actors", "all_recipients")]
    top, release, passed = [[c[f"{k}_{x}"] for k in MODES] for x in ("top", "release", "pass")]
    if cap != sorted(cap) or cap[-1] > d["anchor_eligible"] or any(n > cap[0] for n in top):
        raise ValueError("Invalid nested opportunity counts")
    if any(p > n for p, n in zip(passed, release, strict=True)):
        raise ValueError("Passed a larger number than released")
    return c["two_slots"], cap, top, release, passed


def panel(ax, label, names, values, limit, xlabel):
    ax.set_title(label, loc="left", fontsize=9, fontweight="semibold", pad=8)
    bars = ax.barh(names, values, height=.53, color=["#788995"]*(len(values)-1)+["#27586a"])
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, limit)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#a5a5a5")
    ax.tick_params(axis="both", length=0, labelsize=8, pad=4)
    ax.grid(axis="x", color="#dddddd", linewidth=.5)
    ax.set_axisbelow(True)


def plot(destination):
    total, cap, top, release, passed = numbers()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.12, 1]})
    panel(axes[0], "A  Hindsight capacity",
          ["Timing only", "Actors & peaks fixed", "Actors fixed", "Any recipients"],
          cap, 68, "Views with a qualifying alternative")
    names = ["Replay only", "Geometry head", "History head"]
    panel(axes[1], "B  Top-ranked choices",
          [f"{k}  ({a}/{b})" if b else f"{k}  (none)" for k, a, b in zip(names, passed, release, strict=True)],
          top, 35, "Qualifying top-ranked choices")
    fig.subplots_adjust(left=.19, right=.97, top=.83, bottom=.28, wspace=.88)
    fig.text(.19, .085, f"{total:,} two-edit seed views. Parentheses in B: passes / releases; none means no release.",
             fontsize=7.5, color="#343434")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".pdf"), metadata={
        "Title": "AV2 fixed-budget timing alternatives", "Author": "", "CreationDate": None, "ModDate": None})
    fig.savefig(destination.with_suffix(".png"), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT, help="Path stem for the PDF and PNG")
    plot(p.parse_args().output)
