"""Show every fitting view; the empirical floor is not a deployable oracle."""
import csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import run_m3w_european_fixed_cap_diagnostic as run


def main():
    with (run.PUBLIC/'view_metrics.csv').open() as source:
        rows = list(csv.DictReader(source))
    plt.rcParams.update({'svg.hashsalt': 'm3w-fixed-cap-v1', 'svg.fonttype': 'none', 'font.size': 10})
    banks = ['original_in_sample', 'oof', 'in_sample_next', 'in_sample_prev']
    labels = ['Original\nin-sample', 'Inner\nOOF', 'Cyclic\nnext', 'Cyclic\nprevious']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    for i, pair in enumerate(['full', 'motion_only']):
        for j, weights in enumerate(['uniform_positive_envelope', 'registered_risk_weights']):
            ax = axes[i, j]
            for k, bank in enumerate(banks):
                rr = sorted((r for r in rows if (r['pair'], r['bank'], r['weighting']) == (pair, bank, weights)), key=lambda r: r['tag'])
                assert len(rr) == 72
                values = np.array([float(r['floor_share_percent']) for r in rr if r['floor_share_percent']])
                offsets = np.linspace(-.2, .2, len(values))
                ax.scatter(k+offsets, values, s=15, alpha=.6, color='#257b91' if i == 0 else '#b34e66')
                if len(values):
                    ax.hlines(np.median(values), k-.25, k+.25, color='#202020', lw=2)
            ax.set_xticks(range(4), labels)
            ax.set_ylim(-2, 102); ax.grid(axis='y', alpha=.2)
            ax.set_title(pair.replace('_', ' ')+' / '+('uniform' if j == 0 else 'registered risk weights'))
            ax.set_ylabel('Empirical floor / fitting easy-harm MSE (%)')
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Frozen harm cap: all fitting-view empirical projection floors')
    fig.text(.5, .02, 'Dots: dependent fitting views. Black line: median. No confidence interval or conditional-bias claim.\n'
             'Projection uses realized labels offline; it is neither an achievable prediction nor a deployment result.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .96))
    fig.savefig(run.PUBLIC/'fixed_cap_diagnostic.svg', metadata={'Date': None})
    fig.savefig(run.PRIVATE/'fixed_cap_diagnostic_preview.png', dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
