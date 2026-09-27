"""All registered assignment intervals, including negative contrasts."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_oof_magnitude as run
from scripts.report_m3w_european_oof_magnitude import PRIMARY
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    doc = json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    plt.rcParams.update({'svg.hashsalt': 'm3w-oof-magnitude-v1', 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, family in enumerate(('full', 'motion_only')):
        for col, contrast in enumerate(('scaled_true_vs_scaled_cost', 'scaled_true_vs_raw_true')):
            ax = axes[row, col]; values = doc['contrasts'][contrast][family]
            for i, metrics in enumerate(values.values()):
                m = metrics[PRIMARY]
                if 'CI' not in m:
                    ax.text(0, i, 'not estimable'); continue
                p, (lo, hi) = m['point'], m['CI']
                ax.errorbar(p, i, xerr=np.array([[p-lo], [hi-p]]), fmt='o',
                    color='#00796b' if lo > 0 else '#b23b3b' if hi < 0 else '#555555', capsize=4)
            ax.axvline(0, color='black', linewidth=.8)
            ax.set_yticks(range(len(values)), [k.replace('producer', 'P').replace('_controller', ' / C') for k in values])
            ax.set_title(f'{family}: '+contrast.replace('_', ' '))
            ax.set_xlabel('Expected easy-harm cost MSE gain (%)'); ax.grid(axis='x', alpha=.2)
    fig.suptitle('Honest OOF magnitude: exposed source-held costs\n3 seeds;3,000 four-locality resamples;not trajectory/policy lift', fontsize=13)
    fig.savefig(run.PUBLIC/'source_held_costs.svg', metadata={'Date': None})
    fig.savefig(run.PRIVATE/'source_held_costs.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__': main()
