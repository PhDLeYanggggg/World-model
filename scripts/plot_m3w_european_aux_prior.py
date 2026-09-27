"""Plot every registered assignment,including intervals that favor the controls."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_aux_prior as run
from scripts.report_m3w_european_aux_prior import PRIMARY
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    doc = json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    plt.rcParams.update({'svg.hashsalt': 'm3w-aux-prior-v1', 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, family in enumerate(('full', 'motion_only')):
        for column, contrast in enumerate(('repair_vs_cost_only', 'repair_vs_old_true')):
            ax = axes[row, column]; values = doc['contrasts'][contrast][family]
            for i, (assignment, metrics) in enumerate(values.items()):
                m = metrics[PRIMARY]
                if 'CI' not in m:
                    ax.text(0, i, 'not estimable'); continue
                p, (lo, hi) = m['point'], m['CI']
                ax.errorbar(p, i, xerr=np.array([[p-lo], [hi-p]]), fmt='o',
                    color='#00796b' if lo > 0 else '#b23b3b' if hi < 0 else '#555555', capsize=4)
            ax.axvline(0, color='black', linewidth=.8)
            ax.set_yticks(range(len(values)), [k.replace('producer','P').replace('_controller',' / C') for k in values])
            ax.set_title(f'{family}: '+contrast.replace('_', ' '))
            ax.set_xlabel('Easy-harm cost MSE gain (%)'); ax.grid(axis='x', alpha=.2)
    fig.suptitle('Auxiliary prior repair: exposed source-held costs\n3 seeds; 3,000 four-locality resamples; not trajectory/policy lift', fontsize=13)
    fig.savefig(run.PUBLIC/'source_held_costs.svg', metadata={'Date': None})
    fig.savefig(run.PRIVATE/'source_held_costs.png', dpi=130)
    plt.close(fig)
    print(json.dumps({'svg':run.artifact(run.PUBLIC/'source_held_costs.svg')}))


if __name__ == '__main__': main()
