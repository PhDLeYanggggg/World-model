"""Plot frozen six-study aggregate evidence, never a new model selection."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.report_m3w_feature_cost_evidence import DEST, STUDIES


def main():
    value = json.loads(DEST.with_suffix('.json').read_text())
    names = ['Continuous cost', 'Recording policy', 'Published deferral',
             'Support-unit repair', 'Shared reference', 'Static scene context']
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), gridspec_kw={'width_ratios':[1.25, 1.]})
    y = np.arange(6)
    for i, study in enumerate(STUDIES):
        row = value['inner_held_mixed_minus_physical'][study]['full_known_pp']
        point, ci = row['point'], row['CI95']
        if point is None:
            raise ValueError('Full fixed comparison required')
        axes[0].plot(ci, [i, i], color='#3f637d', lw=2)
        axes[0].scatter([point], [i], color='#3f637d', s=35, zorder=3)
    axes[0].axvline(0, color='#555555', ls='--', lw=1)
    axes[0].set_yticks(y, names); axes[0].invert_yaxis()
    axes[0].set_xlabel('Mixed minus physical-only known ADE gain (pp)')
    axes[0].set_title('Mixed bank does not establish neural gain', fontsize=12)
    for offset, arm, label, color in ((-.16, 'physical_only', 'Physical-only', '#46728d'),
                                     (.16, 'physical_neural', 'Physical + neural', '#a34b4b')):
        rows = [next(r for r in value['policies'] if r['study']==study and
                     r['scope']=='inner_held' and r['policy']==arm) for study in STUDIES]
        counts = [r['supported_record_head_views'] for r in rows]
        if any(r['required_record_head_views'] != 186 for r in rows):
            raise ValueError('Same required roster required')
        axes[1].barh(y+offset, counts, height=.29, color=color, label=label)
        for i, count in enumerate(counts):
            axes[1].text(count+2, i+offset, str(count), va='center', fontsize=9)
    axes[1].set_yticks(y, []); axes[1].invert_yaxis()
    axes[1].set_xlim(0, 201); axes[1].set_xticks([0, 62, 124, 186])
    axes[1].axvline(186, color='#555555', ls='--', lw=1)
    axes[1].set_xlabel('Required recording/head views supported / 186')
    axes[1].set_title('None supports every required recording', fontsize=12)
    axes[1].legend(loc='upper right', bbox_to_anchor=(.9, .95), frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.17); ax.set_axisbelow(True)
    fig.suptitle('Exposed SOURCE development: fixed forecast bank, no deployment promotion', fontsize=13)
    fig.text(.5, .015, '3,000 locality bootstrap; repeated views are not independent recordings. Empty risk is not estimable.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    folder = DEST.parent/'feature_cost_figures_20261029'; folder.mkdir(exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(folder/('neural_increment_and_record_risk.'+suffix), dpi=180)
    plt.close(fig)
    print(json.dumps(dict(source=str(DEST.with_suffix('.json').relative_to(ROOT)),
        plot=str(folder.relative_to(ROOT)), scientific_recalculation=False)))


if __name__ == '__main__':
    main()
