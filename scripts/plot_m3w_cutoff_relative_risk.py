"""All matched representation/rule tradeoffs, without a selected winner."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "outputs/publication_readiness_2026_09/cutoff_relative_risk_v1"


def main():
    a = json.loads((PUBLIC / "analysis.json").read_text())
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "m3w-cutoff-relative-risk-v1"})
    policies = {"point": "#146e87", "population": "#8c397b", "selected": "#b66017"}
    arms = (("native", "o"), ("dimensionless", "^"), ("cutoff", "s"))
    actions = (("damped_velocity_005", "Damping .05"),
               ("transformer", "Transformer"), ("eqmotion", "EqMotion"))

    def point(action, policy):
        r = a["summary"][action + "__" + policy]
        easy = max(-v["gain_percent"] for s in r["seeds"].values()
                   for v in s["subsets"]["positive_easy"]["by_scene"].values())
        return r["ADE"]["equal_scene_gain_percent"], easy

    values = [point(action, arm+"_"+policy) for action, _ in actions
              for arm, _ in arms for policy in policies]
    values += [point(action, "old_strict") for action, _ in actions]
    xlo, xhi = min(0., min(v[0] for v in values))-.3, max(v[0] for v in values)+.5
    ylo, yhi = min(0., min(v[1] for v in values))-.7, max(2., max(v[1] for v in values))+1.
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 5.1), sharey=True)
    for ax, (action, title) in zip(axes, actions):
        ax.axhline(2, color="#aa2727", linestyle="--", linewidth=1)
        ax.axhline(0, color="#bbbbbb", linewidth=.7)
        for policy, color in policies.items():
            coordinates = [point(action, arm+"_"+policy) for arm, _ in arms]
            ax.plot([v[0] for v in coordinates], [v[1] for v in coordinates], color=color, alpha=.35)
            for (arm, marker), p in zip(arms, coordinates):
                ax.scatter(*p, marker=marker, facecolors="white" if arm == "native" else color,
                           edgecolors=color, linewidths=1.2, s=60, zorder=4)
        ax.scatter(*point(action, "old_strict"), marker="x", c="#222222", s=64, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("Equal-site ADE gain over CV (%)")
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.grid(axis="y", color="#e8e8e8", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Worst positive-easy degradation (%)\nacross physical site / seed")
    legend = [Line2D([0], [0], color=c, label=p.title()) for p, c in policies.items()]
    for (arm, marker), label in zip(arms, ("Native", "Dimensionless", "Cutoff-relative")):
        legend.append(Line2D([0], [0], marker=marker, color="#555555",
            markerfacecolor="white" if arm == "native" else "#555555", linestyle="none", label=label))
    legend += [Line2D([0], [0], marker="x", color="#222222", linestyle="none", label="Old strict"),
               Line2D([0], [0], color="#aa2727", linestyle="--", label="2% ceiling")]
    fig.suptitle("Risk features: restored cutoff context versus normalized shape alone", y=.985)
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .067),
               ncol=4, frameon=False, fontsize=9)
    fig.text(.5, .014, "Four design-exposed SDD sites; three seeds. Descriptive, not independent safety evidence. "
             "Zero-CV harms and all controls remain in the results table.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.078, right=.985, bottom=.28, top=.86, wspace=.16)
    svg = PUBLIC / "risk_tradeoff.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    preview = ROOT / "data/stage_cvpr2027_experiments/cutoff_relative_risk_v1/risk_tradeoff.png"
    fig.savefig(preview, dpi=145); plt.close(fig)
    print(preview)


if __name__ == "__main__":
    main()
