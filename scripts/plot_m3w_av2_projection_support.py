"""Plot the frozen AV2 support and qualification decomposition."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT/"outputs/publication_readiness_2026_09/av2_projection_support_v1/summary.json"
DEFAULT = ROOT/"data/stage_cvpr2027_experiments/submission_admin/av2_projection_support"


def draw(path):
    s = json.loads(SUMMARY.read_bytes())
    if s["independent_confirmation"] or s["reserved_outcomes_opened"]:
        raise ValueError("Expected development-only evidence")
    total = s["same_count"]
    orders = ["transferred_cost", "without_support", "actor_support", "known_support_oracle"]
    values = []
    for k in orders:
        r = s["matched"][k]
        values.append([r["qualified"], total-r["qualified"]-r["risk_fail"],
                       r["risk_fail"]-r["unsupported"], r["unsupported"]])
    v = np.array(values)
    assert (v >= 0).all() and (v.sum(1) == total).all()
    tax = s["risk_taxonomy"]
    supported = tax["supported"]
    # The fixed forecast qualifies before confidence in the verified parent.
    parent = json.loads((SUMMARY.parent.parent/"av2_position_release_v1/summary.json").read_bytes())
    qualified = parent["held_forced_eligible"]
    x = [qualified, supported-qualified, tax.get("observed_positive_harm", 0),
         tax.get("unknown_completion_bound_only", 0), tax.get("unsupported_reference", 0)]
    assert sum(x)+tax.get("easy_net_only", 0) == s["views"] and not tax.get("easy_net_only", 0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "svg.hashsalt": "m3w-av2-support"})
    c = ["#315c70", "#99b5c0", "#a5a5a5", "#e2e2e2"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.0, 3.05), gridspec_kw={"width_ratios": [1, 1.3]})
    fig.subplots_adjust(left=.21, right=.98, bottom=.26, top=.83, wspace=.55)
    for ax in (a, b):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", linewidth=.55, alpha=.25)
        ax.set_axisbelow(True)
    y = np.arange(len(x))
    a.barh(y, x, color=[c[0], c[1], c[2], c[2], c[3]], height=.54, edgecolor="#626262", linewidth=.4)
    for yy, val in zip(y, x, strict=True):
        a.text(val+12, yy, str(val), va="center", fontsize=7.5)
    a.set(yticks=y, yticklabels=["Qualified", "Risk pass, floor fails", "Observed harm", "Completion bound only",
        "No easy reference"], xlim=(0, max(x)*1.19), xlabel="Views", title=f"(a) All {s['views']:,} held views")
    a.invert_yaxis()
    y = np.arange(len(orders))
    left = np.zeros(len(orders))
    for i, label in enumerate(("Qualified", "Only floor fails", "Supported risk fails", "No easy reference")):
        b.barh(y, v[:, i], left=left, color=c[i], height=.54, edgecolor="#666666", linewidth=.4, label=label)
        for yy, start, val in zip(y, left, v[:, i], strict=True):
            if val >= 6:
                b.text(start+val/2, yy, f"{val:g}", ha="center", va="center", fontsize=7.5,
                       color="white" if i == 0 else "#202020")
        left += v[:, i]
    b.set(yticks=y, yticklabels=["Old cost", "No support term", "Learned support", "Known support*"],
          xlim=(0, total), xticks=np.linspace(0, total, 4), xlabel="Views at a fixed count",
          title=f"(b) Equal-count ranking (n = {total})")
    b.invert_yaxis()
    h, l = b.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(.56, .065), ncols=4, frameon=False, fontsize=7.5)
    fig.text(.58, .035, "* Hindsight diagnostic. Six AV2 development cities; three seeds.",
             ha="center", fontsize=7.2, color="#333333")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(path.with_suffix(".png"), dpi=230)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    draw(parser.parse_args().output)
