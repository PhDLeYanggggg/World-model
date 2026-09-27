"""Show every registered source assignment, retaining negative and missing cells."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_regime_transport as run
from scripts.report_m3w_european_regime_transport import PRIMARY
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def figure(doc, contrasts, name, subtitle):
    fig, axes = plt.subplots(2, len(contrasts), figsize=(19, 8), constrained_layout=True)
    for row, family in enumerate(('full', 'motion_only')):
        for col, (contrast, title) in enumerate(contrasts):
            ax = axes[row, col]
            values = doc['outer_cut3']['contrasts'][contrast][family]
            for i, metrics in enumerate(values.values()):
                m = metrics[PRIMARY]
                if 'CI' not in m:
                    ax.text(0, i, 'not estimable')
                    continue
                p, (lo, hi) = m['point'], m['CI']
                ax.errorbar(p, i, xerr=np.array([[p-lo], [hi-p]]), fmt='o',
                    color='#00796b' if lo > 0 else '#b23b3b' if hi < 0 else '#555555', capsize=4)
            ax.axvline(0, color='black', linewidth=.8)
            ax.set_yticks(range(len(values)),
                [k.replace('producer', 'P').replace('_controller', ' / C') for k in values])
            ax.set_title(f'{family}\n{title}')
            ax.set_xlabel('Expected easy-harm MSE gain (%)')
            ax.grid(axis='x', alpha=.2)
    fig.suptitle(subtitle+'\nUnchanged outer cut; 3 seeds; 3,000 four-locality resamples; exposed source development', fontsize=13)
    fig.savefig(run.PUBLIC/(name+'.svg'), metadata={'Date': None})
    fig.savefig(run.PRIVATE/(name+'.png'), dpi=120)
    plt.close(fig)


def main():
    doc = json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    plt.rcParams.update({'svg.hashsalt': 'm3w-regime-transport-v1', 'font.size': 10})
    for mode in ('raw', 'scaled'):
        figure(doc, [(key+'_'+mode, title) for key, title in (
            ('cut_at_two', 'Cut3 vs cut2 at two fitting sites'),
            ('cut_at_three', 'Cut3 vs cut2 at three fitting sites'),
            ('regime_at_cut2', 'Three vs two fitting sites at cut2'),
            ('regime_at_cut3', 'Three vs two fitting sites at cut3'))],
            'crossed_'+mode, f'Crossed fitting-regime contrasts: {mode} cost heads, not trajectory gains')
    figure(doc, [('scale_'+cell, cell+': scaled vs raw') for cell in run.method.CELLS],
        'magnitude_transport', 'Frozen magnitude readout: all four fitting cells, not a policy comparison')


if __name__ == '__main__':
    main()
