"""Plot checked hindsight feasibility separately from past-only recording passes."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_m3w_decision_bottlenecks import ROOT, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/stage_cvpr2027_experiments/recipient_feasibility.pdf')
    args = parser.parse_args()
    fixed = load('native_fixed_recipient_risk_bound_v1', 'analysis.json', 'replay_execution.json')
    exact = load('native_exact_count_joint_capacity_v1', 'analysis.json', 'replay_execution.json')
    policy = load('native_recipient_shortfall_v1', 'analysis.json', 'replay_execution.json')
    n = exact['views']
    if fixed['views'] != n or policy['grouped']['views'] != n or n != 102:
        raise ValueError('Require the same recording-seed roster')
    values = [(0, f['not_ruled_out'], f['counts']['provably_infeasible'])
              for k in ('fixed_neural_count', 'free_neural_count') for f in [fixed['bounds'][k]]]
    values += [(a['verified_feasible'], a['unresolved'], a['proved_infeasible'])
               for k in ('fixed_nomination', 'full_forecast_bank') for a in [exact['risk_capacity'][k]]]
    if any(sum(row) != n for row in values):
        raise ValueError('Incomplete feasibility categories')

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.1, 3.45), layout='constrained',
                                     gridspec_kw={'width_ratios': [1.25, 1]})
    names = ('Fixed people, fixed neural count', 'Fixed people, free neural count',
             'All people, one path each', 'All people, all paths')
    colors = ('#137471', '#DEDFE5', '#C46042')
    legends = ('Verified feasible', 'Unresolved / not ruled out', 'Proved infeasible')
    for i, row in enumerate(values):
        offset = 0
        for j, (v, c, label) in enumerate(zip(row, colors, legends)):
            left.barh(i, v, left=offset, height=.51, color=c, label=label if i == 0 else None)
            if v:
                left.text(offset+v/2, i, str(v), ha='center', va='center', fontsize=8,
                          color='white' if j == 0 else '#102E35')
            offset += v
    left.set_yticks(range(4), names, fontsize=8)
    left.set_title('A  Hindsight feasibility', loc='left', fontsize=11, weight='bold')
    left.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0, -.21), handlelength=1.4)

    arms = ('gain', 'support_first', 'shortfall')
    data = [policy['grouped']['risk'][key] for key in arms]
    if len({d['switches'] for d in data}) != 1:
        raise ValueError('Only matched-count actual policies belong in this panel')
    for i, cell in enumerate(data):
        v = cell['supported_views']
        right.barh(i, n, height=.51, color='#EDF0F1')
        right.barh(i, v, height=.51, color=colors[0])
        right.text(v+2, i, f'{v}/{n}', va='center', fontsize=8, color='#102E35')
    right.set_yticks(range(3), ('Gain', 'Support-first', 'Risk shortfall'), fontsize=8)
    right.set_title('B  Past-only policy passes', loc='left', fontsize=11, weight='bold')
    for ax in (left, right):
        ax.invert_yaxis()
        ax.set_xlim(0, n)
        ax.set_xticks((0, 25, 50, 75, n))
        ax.set_xlabel('Recording-seed views', fontsize=8)
        ax.tick_params(axis='both', length=0, labelsize=8)
        ax.grid(axis='x', color='#E8ECEF', linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'left', 'right']].set_visible(False)
        ax.spines['bottom'].set_color('#BAC4C9')
    fig.suptitle('Feasibility and realized recording passes are different questions',
                 weight='bold', fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
