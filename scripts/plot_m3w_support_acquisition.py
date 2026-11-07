"""Plot verified support-acquisition capacity, never raw trajectories."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/support_acquisition'
OUT = ROOT/'data/stage_cvpr2027_experiments/evidence_manuscript_v10/utility_tradeoff_build'
ROWS = (
    ('insert_only', 'Insert only'),
    ('early_joint', 'Joint edits, insertion before first switch'),
    ('ordered_joint', 'Joint edits, no deletion before insertion'),
    ('free_joint', 'Joint edits, unrestricted order'),
    ('shortlist_joint', 'Ordered joint edits, old eight-slot list'),
)


def main():
    raw = (STUDY/'analysis.json').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    for phase in ('score', 'replay'):
        if json.loads((STUDY/f'{phase}_execution.json').read_text())['artifact']['sha256'] != digest:
            raise ValueError('Require exact aggregate replay')
    d = json.loads(raw)
    if not d['oracle_only'] or d['inference_policy_evaluated'] or not d['original_rules_unchanged']:
        raise ValueError('A figure for hindsight capacity only')
    g = d['groups']
    n = np.array([[g[k]['modes'][r]['checked_feasible'] for k in ('zero_direct', 'active_no_easy')]
                  for r, _ in ROWS])
    if g['all_missing']['views'] != 35 or (n > [18, 17]).any():
        raise ValueError('Unexpected cohort')
    fig, ax = plt.subplots(figsize=(8.6, 3.6), layout='constrained')
    y = np.arange(len(n))
    for j, (color, label) in enumerate((('#24596a', 'No original switch (18)'),
                                        ('#9b542d', 'Switched without easy reference (17)'))):
        left = n[:, :j].sum(1)
        ax.barh(y, n[:, j], left=left, height=.6, color=color, label=label)
        for i, v in enumerate(n[:, j]):
            if v:
                ax.text(left[i]+v/2, i, str(v), color='white', ha='center', va='center', weight='bold')
    ax.axvline(35, color='#727980', linewidth=1, linestyle='--')
    for i, v in enumerate(n.sum(1)):
        ax.text(v+.45, i, str(v), color='#17242c', va='center', weight='bold')
    ax.set(yticks=y, yticklabels=[v for _, v in ROWS], xlim=(0, 37),
           xticks=[0, 5, 10, 15, 20, 25, 30, 35],
           xlabel='Checked feasible views (of 35 with undefined easy support)',
           title='Hindsight support acquisition under the original risk and utility rules')
    ax.invert_yaxis()
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=2, frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT/f'support_acquisition.{ext}', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
