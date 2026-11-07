"""Plot uncertainty removal at a matched missing-position budget."""
import argparse
from pathlib import Path

from plot_m3w_decision_bottlenecks import ROOT, load, plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'data/stage_cvpr2027_experiments/pair_annotation_budget.pdf')
    args = p.parse_args()
    data = load('native_pair_annotation_budget_v1', 'analysis.json', 'replay_execution.json')
    rows = data['curves']
    x = [v['positions'] for v in rows]
    if x[-1] != data['influential_positions'] or data['labels_collected']:
        raise ValueError('Expected the complete conditional budget curve')
    fig, ax = plt.subplots(figsize=(6.4, 3.8), layout='constrained')
    for key, label, color, style in (
        ('dependency_greedy', 'Dependency-aware', '#167A73', '-'),
        ('incident_mass', 'Incident mass', '#4C70A5', '--'),
        ('chronological', 'Chronological', '#B75C41', ':'),
    ):
        ax.plot(x, [100*v[key] for v in rows], linestyle=style, color=color, label=label, linewidth=2)
    chosen = data['budgets']['0.95']['positions']['dependency_greedy']
    ax.axhline(95, color='#717981', linewidth=.8, linestyle='--', zorder=0)
    ax.plot([chosen], [95], 'o', color='#167A73', markersize=5, zorder=4)
    ax.annotate(f'{chosen:,} positions at 95%', xy=(chosen, 95), xytext=(1240, 68),
                fontsize=8.5, color='#174742', ha='left',
                arrowprops=dict(arrowstyle='-', color='#547975', linewidth=.75))
    ax.set(title='Targeted labels resolve more of the fixed pair comparison',
           xlabel='Requested missing positions (if annotatable)',
           ylabel='Original disagreement debit removed (%)', xlim=(0, x[-1]), ylim=(0, 103))
    ax.set_xticks((0, 1000, 2000, 3000), ('0', '1,000', '2,000', '3,000'))
    ax.set_yticks((0, 25, 50, 75, 95))
    ax.grid(axis='y', alpha=.17)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
