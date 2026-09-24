"""Descriptive tradeoff for all six registered controlled feature/policy arms."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "outputs/publication_readiness_2026_09/dimensionless_risk_v1"


def main():
    a = json.loads((PUBLIC / "analysis.json").read_text())
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "m3w-dimensionless-risk-v1"})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.9), sharey=True)
    policies = {"point": "#146e87", "population": "#8c397b", "selected": "#b66017"}
    actions = (("damped_velocity_005", "Damping .05"),
               ("transformer", "Transformer"), ("eqmotion", "EqMotion"))
    for ax, (action, title) in zip(axes, actions):
        def point(policy):
            r = a["summary"][action + "__" + policy]
            easy = max(-v["gain_percent"] for s in r["seeds"].values()
                       for v in s["subsets"]["positive_easy"]["by_scene"].values())
            return r["ADE"]["equal_scene_gain_percent"], easy
        ax.axhline(2, color="#aa2727", linestyle="--", linewidth=1)
        ax.axhline(0, color="#bbbbbb", linewidth=.7)
        for policy, color in policies.items():
            start, end = point("native_" + policy), point("dimensionless_" + policy)
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, alpha=.55)
            ax.scatter(*start, marker="o", facecolors="white", edgecolors=color,
                       linewidths=1.5, s=68, zorder=3)
            ax.scatter(*end, marker="^", c=color, s=64, zorder=4)
        ax.scatter(*point("old_strict"), marker="x", c="#222222", s=64, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("Equal-site ADE gain over CV (%)")
        ax.set_xlim(0, 6.5)
        ax.set_ylim(-1.35, 9.75)
        ax.grid(axis="y", color="#e8e8e8", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Worst positive-easy degradation (%)\nacross physical site / seed")
    legend = [Line2D([0], [0], color=c, label=p.title()) for p, c in policies.items()]
    legend += [Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white",
                      linestyle="none", label="Native features"),
               Line2D([0], [0], marker="^", color="#555555", linestyle="none", label="Dimensionless"),
               Line2D([0], [0], marker="x", color="#222222", linestyle="none", label="Old strict"),
               Line2D([0], [0], color="#aa2727", linestyle="--", label="2% ceiling")]
    fig.suptitle("Removing native-unit features can improve utility while worsening easy cases", y=.985)
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .065),
               ncol=4, frameon=False, fontsize=9)
    fig.text(.5, .012, "Four design-exposed SDD sites; three seeds. Descriptive readout, not safety certification. "
             "Zero-CV harms and all uncontrolled/count-matched results remain in the table.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=.078, right=.985, bottom=.27, top=.86, wspace=.16)
    svg = PUBLIC / "risk_tradeoff.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    preview = ROOT / "data/stage_cvpr2027_experiments/dimensionless_risk_v1/risk_tradeoff.png"
    fig.savefig(preview, dpi=145)
    plt.close(fig)
    print(preview)


if __name__ == "__main__":
    main()
