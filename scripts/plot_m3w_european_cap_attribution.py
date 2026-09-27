"""Plot every assignment in the fixed primary attribution contrasts."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'm3w-cap-attribution-v1'
import matplotlib.pyplot as plt
from scripts import run_m3w_european_cap_attribution as run


def main():
    doc = json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    comparisons = [
        ('history_neighbors_envelope_coupled_vs_history_neighbors_frozen_cap', 'Relax cap, same history + neighbors'),
        ('history_neighbors_envelope_coupled_vs_score_only_envelope_coupled', 'Relaxed: history + neighbors / scores'),
        ('history_neighbors_envelope_coupled_vs_old_summary_envelope_coupled', 'Relaxed: history + neighbors / summaries'),
        ('history_neighbors_envelope_coupled_vs_ordered_history_envelope_coupled', 'Relaxed: add neighbor history')]
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), layout='constrained')
    for i, (name, title) in enumerate(comparisons):
        for j, pair in enumerate(('full', 'motion_only')):
            ax = axes[i, j]; groups = doc['contrasts'][pair][name]
            for k, metrics in enumerate(groups.values()):
                v = metrics['primary']
                if v['status'] != 'measured':
                    ax.text(0, k, 'not estimable', fontsize=8); continue
                low, high = v['CI']; color = '#167561' if low > 0 else '#b64242' if high < 0 else '#646b72'
                ax.hlines(k, low, high, color=color, linewidth=2)
                ax.plot(v['point'], k, 'o', color=color, markersize=4)
            ax.set_yticks(range(len(groups)), [g.replace('producer', 'P').replace('_controller', ' / C') for g in groups])
            ax.invert_yaxis(); ax.axvline(0, color='black', linewidth=.7)
            ax.set_xscale('symlog', linthresh=1); ax.grid(axis='x', alpha=.2)
            ax.set_title(title+'\n'+pair.replace('_', ' '), fontsize=11)
            ax.set_xlabel('Easy-harm MSE gain (%) [symmetric-log axis]', fontsize=9)
    fig.suptitle('Frozen-cap attribution on fitting-development localities\nThree cached seeds; four-locality bootstrap; no new training or independent test', fontsize=12)
    fig.savefig(run.PUBLIC/'cap_attribution.svg', metadata={'Date': None})
    fig.savefig(run.PRIVATE/'cap_attribution_preview.png', dpi=110)
    plt.close(fig)


if __name__ == '__main__': main()
