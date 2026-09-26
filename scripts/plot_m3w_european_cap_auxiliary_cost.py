"""All paired assignment intervals, including adverse and unsupported results."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_cap_auxiliary_cost as run
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    data = json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())['contrasts']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.hashsalt': 'm3w-cap-aux-cost-v1',
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    contrasts = [('aux_vs_control', 'True auxiliary vs cost only', '#1764ab'),
                 ('aux_vs_original', 'True auxiliary vs original', '#b86410'),
                 ('aux_vs_shuffled', 'True auxiliary vs shuffled', '#158060'),
                 ('control_vs_original', 'Cost only vs original', '#6a6a6a'),
                 ('shuffled_vs_control', 'Shuffled vs cost only', '#9c427b')]
    metrics = [('envelope_positive__harm_MSE_gain_percent', 'Easy-harm MSE gain (%)'),
               ('all__H_all_MSE_gain_percent', 'All-harm MSE gain (%)'),
               ('envelope_positive__top10_gain_pp', 'Top-10% harm capture gain (pp)'),
               ('envelope_positive__coverage_log_error_reduction', 'Absolute log coverage-error reduction')]
    fig, axs = plt.subplots(4, 2, figsize=(14, 14))
    for col, pair in enumerate(['full', 'motion_only']):
        assignments = list(data['aux_vs_control'][pair])
        for row, (metric, title) in enumerate(metrics):
            ax = axs[row, col]; missing = []
            for j, (key, _, color) in enumerate(contrasts):
                for i, assignment in enumerate(assignments):
                    value = data[key][pair][assignment][metric]; pos = i+(j-2)*.13
                    if 'CI' not in value:
                        missing.append(key+'/'+assignment); continue
                    low, high = value['CI']; point = value['point']
                    ax.plot([low, high], [pos, pos], color=color, lw=1.2)
                    ax.plot(point, pos, 'o', color=color, ms=3)
            ax.axvline(0, color='#222222', lw=.8, linestyle='--')
            ax.set_yticks(range(6), [a.replace('producer', 'P').replace('_controller', ' / C') for a in assignments])
            ax.set_ylim(-.85, 5.6); ax.grid(axis='x', alpha=.18)
            ax.set_title(pair.replace('_', ' ')+' | '+title, loc='left', fontsize=10)
            ax.set_xlabel('Positive favors the first model')
            if missing:
                ax.text(.01, .02, f'{len(missing)} contrasts not estimable; retained in tables',
                        transform=ax.transAxes, fontsize=8, color='#9a2222')
    handles = [Line2D([0], [0], color=c, marker='o', label=label, lw=1.2, ms=3)
               for _, label, c in contrasts]
    fig.suptitle('Does cap-event supervision improve expected forecasting costs?', fontsize=15, y=.988)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .965), ncol=3, frameon=False)
    fig.text(.5, .013, '3 seeds averaged within each of 4 localities; 3,000 paired resamples. Assignments overlap.\n'
             'Source development only; no independent confirmation, new policy evaluation or deployment.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .05, 1, .91])
    fig.savefig(run.PUBLIC/'cap_auxiliary_cost_contrasts.svg', metadata={'Date': None})
    fig.savefig(run.PRIVATE/'cap_auxiliary_cost_contrasts.png', dpi=115)
    plt.close(fig)


if __name__ == '__main__': main()
