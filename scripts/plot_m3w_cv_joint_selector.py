"""Plot the causal joint-choice contrasts on the registered development roster."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('readout', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    r = json.loads(a.readout.read_text())
    if r['independent_confirmation'] or not r['matched_total_peak_per_query']:
        raise ValueError('Require the matched development readout')
    keys = ('joint_vs_independent', 'joint_vs_additive', 'joint_vs_additive_matched')
    labels = ('Independent', 'Additive pair terms', 'Additive, equal total peak')
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.75), constrained_layout=True, sharey=True)
    data = [[r['agent']['effects'][k]['completion_lower_pp'] for k in keys],
            [r['pair'][k]['lower_pixel'] for k in keys]]
    for ax, units, rows in zip(axes, ('CV-normalized pp', 'Native pixels'), data, strict=True):
        ax.axvline(0., linewidth=.7, color='#777777')
        for y, (x, c) in enumerate(zip(rows, ('#235781', '#235781', '#9a4c2e'), strict=True)):
            if x is None or x['point'] is None:
                ax.text(.02, y, 'not estimable', transform=ax.get_yaxis_transform(), va='center')
                continue
            val, (lo, hi) = x['point'], x['interval95']
            ax.plot([lo, hi], [y, y], color=c, linewidth=1.4)
            ax.scatter(val, y, c=c, s=25, zorder=3)
        ax.set(xlabel=f'Paired full-grid lower gain ({units})', yticks=range(len(keys)), yticklabels=labels)
        ax.spines[['right', 'top']].set_visible(False)
        ax.grid(axis='x', alpha=.16)
    axes[0].invert_yaxis()
    axes[0].set_title('Agent displacement')
    axes[1].set_title('Pair displacement')
    fig.suptitle('Coupled choice versus controls, at identical query edit counts')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)


if __name__ == '__main__':
    main()
