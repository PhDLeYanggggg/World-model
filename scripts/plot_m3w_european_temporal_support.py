"""Aggregate exploratory intervals; do not plot windows as independent trials."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'm3w-temporal-support-v1'
import matplotlib.pyplot as plt
from scripts import run_m3w_european_temporal_support as run


def main():
    doc = json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    comparisons = [('history_neighbors_vs_score_only', 'History + neighbors / score only'),
        ('history_neighbors_vs_old_summary', 'History + neighbors / old summaries'),
        ('ordered_history_vs_score_only', 'Ordered history / score only'),
        ('history_neighbors_vs_ordered_history', 'Add neighbor history')]
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), layout='constrained')
    for i, (name, label) in enumerate(comparisons):
        for j, pair in enumerate(('full', 'motion_only')):
            ax = axes[i, j]; groups = doc['contrasts'][pair][name]
            for k, (assignment, metrics) in enumerate(groups.items()):
                v = metrics['primary']
                if v['status'] != 'measured':
                    ax.text(0, k, 'not estimable', fontsize=8); continue
                low, high = v['CI']; point = v['point']
                color = '#167561' if low > 0 else '#b64242' if high < 0 else '#646b72'
                ax.hlines(k, low, high, color=color, linewidth=2)
                ax.plot(point, k, 'o', color=color, markersize=4)
            ax.set_yticks(range(len(groups)), [g.replace('producer', 'P').replace('_controller', ' / C') for g in groups])
            ax.invert_yaxis(); ax.axvline(0, color='black', linewidth=.7)
            ax.set_xscale('symlog', linthresh=1)
            ax.set_title(label+'\n'+pair.replace('_', ' '), fontsize=11)
            ax.set_xlabel('Easy-harm MSE gain (%) [symmetric-log axis]', fontsize=9)
            ax.grid(axis='x', alpha=.2)
    fig.suptitle('Inner fitting-locality information screen\nThree seeds; four-locality paired bootstrap; no independent test', fontsize=13)
    fig.savefig(run.PUBLIC/'context_intervals.svg', metadata={'Date': None})
    fig.savefig(run.PRIVATE/'context_intervals_preview.png', dpi=110); plt.close(fig)
    rows = json.loads((run.PRIVATE/'readout.json').read_text())['rows']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for pair, color in [('full', '#167561'), ('motion_only', '#b64242')]:
        support = [r['scoring_support'][0] for r in rows if r['input']['pair'] == pair]
        points = {(s['easy_harm_rows'], s['event_tracks'], s['event_track_mass_effective_count']) for s in support}
        x, y, z = zip(*sorted(points))
        axes[0].scatter(x, y, s=12, alpha=.45, color=color, label=pair.replace('_', ' '))
        axes[1].scatter(y, z, s=12, alpha=.45, color=color, label=pair.replace('_', ' '))
    for ax in axes:
        ax.set_xscale('symlog', linthresh=1); ax.set_yscale('symlog', linthresh=1)
        ax.grid(alpha=.15); ax.legend(fontsize=9)
    axes[0].set_xlabel('Easy-harm rows'); axes[0].set_ylabel('Distinct event-bearing tracks')
    axes[1].set_xlabel('Distinct event-bearing tracks'); axes[1].set_ylabel('Harm-mass effective track count')
    fig.suptitle('Repeated windows and concentrated event mass\nDependent views; identical plotted triples deduplicated; not independent sample size', fontsize=12)
    fig.savefig(run.PUBLIC/'event_support.svg', metadata={'Date': None})
    fig.savefig(run.PRIVATE/'event_support_preview.png', dpi=110); plt.close(fig)


if __name__ == '__main__': main()
