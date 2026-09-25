"""Plot all three seeds of the frozen event-risk utility/safety tradeoff."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_conditional_risk_v1'


def main():
    raw = (PUBLIC/'analysis.json').read_bytes()
    result = json.loads(raw)
    audit = json.loads((PUBLIC/'accounting_audit.json').read_text())
    if audit['analysis_sha256'] != hashlib.sha256(raw).hexdigest() or not audit['all_passed']:
        raise ValueError('Verified accounting required')
    groups = ['Previous pointwise', 'All-event ratio', 'Easy-event ratio', 'Easy + support guard']
    colors = ['#63676b', '#3678a0', '#117d69', '#ab5b28']
    metrics = []
    for seed in (17, 29, 43):
        metrics.append([result['references'][str(seed)]['pointwise']['full'],
            result['policies'][f'{seed}_all_neural_underharm4_no_guard']['full'],
            result['policies'][f'{seed}_easy_neural_underharm4_no_guard']['full'],
            result['policies'][f'{seed}_easy_neural_underharm4_source_zero_guard']['full']])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout='constrained')
    for ax, field, title in [(axes[0], 'ADE_vs_CV', 'Prediction gain decreases'),
                            (axes[1], 'positive_easy_ADE_vs_CV', 'Worst-locality easy harm decreases')]:
        for g, (label, color) in enumerate(zip(groups, colors)):
            for s, marker in enumerate(('o', 's', '^')):
                metric = metrics[s][g][field]
                y = metric['equal_scene_gain_percent'] if ax is axes[0] else -metric['worst_scene_gain_percent']
                ax.scatter(g+(s-1)*.14, y, color=color, marker=marker, s=55, zorder=3)
        ax.axhline(0, color='#999999', linewidth=.8)
        ax.set_xticks(range(4), ['Previous\npointwise', 'All-event\nratio', 'Easy-event\nratio', 'Easy +\nsupport guard'])
        ax.set_title(title, fontsize=12)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Equal-locality ADE improvement over CV (%)')
    axes[1].set_ylabel('Worst positive-easy degradation over CV (%)')
    axes[1].axhline(2, color='#a3342a', linestyle='--', linewidth=1, label='2% observed limit')
    axes[1].legend(frameon=False, fontsize=9)
    handles = [plt.Line2D([], [], color='#444444', marker=m, linestyle='None', label=f'Seed {s}')
               for s, m in zip((17, 29, 43), ('o', 's', '^'))]
    axes[0].legend(handles=handles, frameon=False, fontsize=9)
    fig.suptitle('Frozen forecasts and utility; new risk moments\nSource-development points, not independent safety certification', fontsize=13)
    fig.savefig(PUBLIC/'risk_utility_tradeoff.png', dpi=160)
    fig.savefig(PUBLIC/'risk_utility_tradeoff.svg')
    p = PUBLIC/'risk_utility_tradeoff.svg'
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
