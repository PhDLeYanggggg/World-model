"""Render the exposed-development capacity counts without reading outcomes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/native_joint_slate_capacity_v1'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path,
        default=ROOT/'data/stage_cvpr2027_experiments/native_joint_slate_capacity_v1/capacity.pdf')
    a = p.parse_args()
    doc = json.loads((PUBLIC/'development.json').read_text())
    if doc['role'] != 'development' or doc['deployable_policy']:
        raise ValueError('Only the label-aware diagnostic may be plotted')
    data = doc['summary']
    counts = []
    for name in ('joint_only', 'four_option'):
        r = data['options'][f'{name}_vs_ols4']
        n = (r['proved_infeasible'], r['nonpositive_upper_bound'], r['positive_witnesses'])
        if sum(n) != data['views'] or r['unresolved_no_witness']:
            raise ValueError('Stack requires exhaustive, verified categories')
        counts.append(n)
    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    fig.subplots_adjust(left=.19, right=.97, top=.76, bottom=.31)
    colors = ('#CB6A43', '#D9DFE6', '#147D75')
    legend = ('Risk infeasible', 'Feasible, no positive OLS4 gain', 'Feasible, positive OLS4 gain')
    for j, row in enumerate(counts):
        left = 0
        for i, (n, c) in enumerate(zip(row, colors)):
            ax.barh(j, n, left=left, height=.54, color=c, label=legend[i] if j == 0 else None)
            ax.text(left+n/2, j, str(n), ha='center', va='center', weight='bold',
                    color='white' if i == 2 else '#18272F', fontsize=11)
            left += n
    ax.set_yticks((0, 1), ('Joint only', 'Four options'))
    ax.set_xlim(0, data['views'])
    ax.set_xticks((0, 25, 50, 75, 102))
    ax.set_xlabel('Recording-seed views (34 recordings, three seeds)')
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.02, .94, 'Capacity of the frozen forecast proposals', size=13, weight='bold', va='top')
    fig.text(.02, .82, 'Hindsight diagnostic under the unchanged 2% risk rule', size=9, color='#43505A', va='top')
    fig.legend(loc='lower center', bbox_to_anchor=(.5, .025), frameon=False, ncols=3,
               fontsize=7.8, columnspacing=1.)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
