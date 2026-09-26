"""Plot every registered contrast, retaining adverse and unsupported intervals."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_strong_cap_auxiliary as run
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    data = json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())['contrasts']
    plt.rcParams.update({'svg.hashsalt': 'm3w-strong-cap-aux-v1', 'font.family': 'DejaVu Sans',
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    contrasts = [('aux_vs_control', 'True auxiliary / matched original control', '#1764ab'),
                 ('aux_vs_original', 'True auxiliary / original frozen', '#b86410'),
                 ('aux_vs_shuffled', 'True auxiliary / shuffled', '#158060'),
                 ('control_vs_original', 'Reconstructed / original', '#666666'),
                 ('shuffled_vs_control', 'Shuffled / matched control', '#9c427b')]
    metrics = [('envelope_positive__harm_MSE_gain_percent', 'Easy-harm MSE gain (%)'),
               ('all__H_all_MSE_gain_percent', 'All-harm MSE gain (%)'),
               ('envelope_positive__top10_gain_pp', 'Top-10% harm capture gain (pp)'),
               ('envelope_positive__coverage_log_error_reduction', 'Absolute log coverage-error reduction')]
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    for col, family in enumerate(['full', 'motion_only']):
        assignments = list(data['aux_vs_control'][family])
        for row, (metric, title) in enumerate(metrics):
            ax = axes[row, col]; missing = 0
            for j, (key, label, color) in enumerate(contrasts):
                for i, assignment in enumerate(assignments):
                    value = data[key][family][assignment][metric]; pos = i+(j-2)*.13
                    if 'CI' not in value:
                        missing += 1; continue
                    ax.plot(value['CI'], [pos, pos], color=color, lw=1)
                    ax.plot(value['point'], pos, 'o', color=color, ms=3,
                            label=label if row == col == i == 0 else None)
            ax.axvline(0, color='#222222', lw=.8, linestyle='--')
            ax.set_yticks(range(6), [a.replace('producer', 'P').replace('_controller', ' / C') for a in assignments])
            ax.set_ylim(-.85, 5.6); ax.grid(axis='x', alpha=.18)
            ax.set_title(family.replace('_', ' ')+' | '+title, loc='left', fontsize=10)
            ax.set_xlabel('Positive favors the first model')
            if missing:
                ax.text(.01, .02, f'{missing} not estimable; retained in tables', transform=ax.transAxes, fontsize=8)
    fig.suptitle('Cap-event supervision with the strong original estimator retained', fontsize=15, y=.988)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .965), ncol=2, frameon=False)
    fig.text(.5, .013, 'Three seeds averaged within four localities; 3,000 paired resamples; assignments overlap.\n'
             'Source development only. No new policy evaluation, independent confirmation or deployment.', ha='center')
    fig.tight_layout(rect=[0, .05, 1, .9])
    fig.savefig(run.PUBLIC/'strong_cap_auxiliary_contrasts.svg', metadata={'Date': None})
    fig.savefig(run.PRIVATE/'strong_cap_auxiliary_contrasts.png', dpi=115)
    plt.close(fig)


if __name__ == '__main__': main()
