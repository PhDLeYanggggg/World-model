"""Plot actual one-step probe effects without implying generalization lift."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_aux_gradient as run
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def main():
    doc = json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    plt.rcParams.update({'font.size': 9, 'svg.hashsalt': 'european_aux_gradient_v1'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    pairs = [('projection_minus_true', 'cost4'), ('projection_minus_true', 'easy_harm_positive'),
             ('projected_true_minus_shuffled', 'cost4'), ('projected_true_minus_shuffled', 'easy_harm_positive')]
    for ax, (contrast, metric) in zip(axes.flat, pairs):
        rows = [r for r in doc['comparisons'] if r['pair'] == 'full' and r['arm'] == 'cap_aux'
                and r['contrast'] == contrast and r['metric'] == metric]
        labels = []
        for i, row in enumerate(rows):
            labels.append(row['assignment'])
            if row['status'] == 'not_estimable':
                ax.text(0, i, 'not estimable'); continue
            p, ci = row['point'], row['CI']
            color = '#138070' if ci[0] > 0 else '#b54548' if ci[1] < 0 else '#52616b'
            ax.hlines(i, ci[0], ci[1], color=color, linewidth=1.8)
            ax.scatter([p], [i], c=color, s=24, zorder=3)
        ax.axvline(0, color='#888888', linewidth=.8)
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_title(contrast.replace('_', ' ')+'\n'+metric.replace('_', ' '))
        ax.set_xlabel('Actual fitting-probe MSE reduction (%)')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useMathText=True)
        ax.grid(axis='x', alpha=.2); ax.invert_yaxis()
    fig.suptitle('Same-state AdamW interventions: full inputs, frozen cap-auxiliary models\n'
                 'Three seeds; 3000 descriptive locality resamples; exposed fitting data, not held-out lift', fontsize=12)
    fig.savefig(run.PUBLIC/'adamw_probe_effects.svg', metadata={'Date': None})
    fig.savefig(run.PRIVATE/'adamw_probe_effects.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__': main()
