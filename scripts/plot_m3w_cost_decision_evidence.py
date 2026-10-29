"""Plot verified aggregate evidence; never rank/select a favorable model or seed."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_m3w_cost_decision_evidence import build, PUBLIC, STUDIES


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    evidence = build()
    names = ['Continuous costs', 'Record decision', 'Published loss', 'Support-unit repair']
    arms = ('physical_only', 'physical_neural', 'neural_only')
    colors = ('#0072B2', '#D55E00', '#009E73')
    labels = ('Physical only', 'Physical + frozen neural', 'Frozen neural only')
    rows = {(r['study'], r['policy']): r for r in evidence['policies'] if r['scope'] == 'inner_held'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout='constrained')
    y = np.arange(len(STUDIES))
    for i, (arm, color, label) in enumerate(zip(arms, colors, labels)):
        selected = [rows[study, arm] for study in STUDIES]
        positions = y+(i-1)*.21
        points = np.array([r['known_ADE_gain_floor_percent'] for r in selected])
        bounds = np.array([r['conditional_CI95_percent'] for r in selected])
        axes[0].errorbar(points, positions, xerr=np.stack((points-bounds[:, 0], bounds[:, 1]-points)),
                         fmt='o', color=color, capsize=3, markersize=5, label=label)
        for row, position in zip(selected, positions):
            supported = row['supported_record_head_views']
            required = row['required_record_head_views']
            axes[1].scatter([100*supported/required], [position], color=color, s=27)
            axes[1].annotate(f'{supported}/{required}', (100*supported/required, position),
                             xytext=(6, 0), textcoords='offset points', va='center', fontsize=8)
    for ax in axes:
        ax.set_yticks(y, names); ax.invert_yaxis(); ax.grid(axis='x', color='#dddddd', linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].axvline(0, color='#777777', linewidth=.8)
    axes[0].set_xlabel('Known-label ADE gain over floor (%)\nConditional 95% locality-bootstrap CI')
    axes[1].set_xlabel('Required record/head views supported (%)\nRepeated views are not independent recordings')
    axes[1].set_xlim(-3, 112); axes[1].axvline(100, color='#777777', linestyle='--', linewidth=.8)
    fig.suptitle('Mean forecast gain is not complete selected-risk support', fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=3, frameon=False)
    folder = PUBLIC/'cost_decision_figures_20261029'; folder.mkdir(exist_ok=True)
    path = folder/'utility_and_required_risk.png'; fig.savefig(path, dpi=180); plt.close(fig)
    (folder/'caption.md').write_text(
        '# Figure Caption\n\nExposed SOURCE development, not independent confirmation. '
        'All54fits and all three head seeds per method are retained; the trajectory producer is frozen. '
        'Left: known-label utility only, not missing-outcome completion bounds. '
        'Right: full-policy all/easy2%risk support on the complete required recording roster. '
        'Undefined empty risks are not passes. The published method is a native-loss adapter. '
        'Its preregistered cost-gated control and completion-bound contrasts are reported in the evidence table, '
        'not selected from this plot. Methods have different objectives/capacities, so this is not a single-factor '
        'comparison except for the old/new support-unit pair. No deployment or world-model efficacy follows.\n')
    print(json.dumps(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                         real_aggregate_evidence=True, scientific_recalculation=False)))


if __name__ == '__main__':
    main()
