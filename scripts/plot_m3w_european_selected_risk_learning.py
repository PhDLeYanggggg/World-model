"""Plot all matched fitting traces and the predeclared source-only contrasts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_selected_risk_learning_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_selected_risk_learning_v1'
COLORS = {'mean': '#1d718f', 'selected': '#b63e55'}


def main():
    fits = json.loads((PUBLIC/'training_metrics.json').read_text())
    seeds = json.loads((PUBLIC/'seed_averaged_metrics.json').read_text())
    assert len(fits) == 72
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for row, pair in enumerate(('full', 'motion_only')):
            for col, field in enumerate(('moment_mse', 'selected_group_mse')):
                ax = axes[row, col]
                for arm, color in COLORS.items():
                    take = [v for v in fits if v['arm'] == arm and '_'+pair+'_'+arm+'/' in v['path']]
                    assert len(take) == 18
                    steps = sorted(set.intersection(*[{p['step'] for p in v['fit']['trace']} for v in take]))
                    values = np.array([[{p['step']:p[field] for p in v['fit']['trace']}[s] for s in steps] for v in take])
                    for value in values: ax.plot(steps, value, color=color, alpha=.12, linewidth=.7)
                    ax.plot(steps, np.median(values, axis=0), color=color, linewidth=2, label=arm+' (median)')
                ax.set_title(pair+' / '+field.replace('_', ' '))
                ax.set_ylabel('Fixed-batch normalized error')
                ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)
                if row == 1: ax.set_xlabel('Optimizer updates')
        axes[0, 0].legend(frameon=False)
        fig.suptitle('Training diagnostics, not held-scene performance')
        fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(PUBLIC/'training_loss.svg')
        fig.savefig(PRIVATE/'training_loss_preview.png', dpi=120); plt.close(fig)

        comparisons = [('selected_joint_vs_mean_joint__all', 'Selected loss vs mean loss'),
                       ('selected_joint_vs_selected_dual__all', 'Query allocation vs individual'),
                       ('selected_joint_vs_selected_hash_matched__all', 'Query allocation vs matched count')]
        fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
        labels = []
        for pair in ('full', 'motion_only'):
            labels.extend(pair+' / '+name for name in sorted(seeds[pair]))
        for ax, (key, title) in zip(axes, comparisons):
            i = 0
            for pair in ('full', 'motion_only'):
                color = '#1d718f' if pair == 'full' else '#b63e55'
                for name in sorted(seeds[pair]):
                    value = seeds[pair][name][key]
                    if value.get('status') != 'not_estimable':
                        lo, hi = value['CI']; point = value['gain_percent']
                        ax.plot([lo, hi], [i, i], color=color, linewidth=1.2)
                        ax.scatter(point, i, color=color, s=22, zorder=3)
                    i += 1
            ax.axvline(0, color='#777777', linewidth=.8)
            ax.set_title(title, fontsize=10); ax.set_xlabel('All-ADE gain (%) / 95% CI')
            ax.grid(axis='x', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
        axes[0].set_yticks(range(len(labels)), labels); axes[0].invert_yaxis()
        fig.suptitle('Three-seed means, 3,000 locality resamples; overlapping source roles')
        fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(PUBLIC/'source_contrasts.svg')
        fig.savefig(PRIVATE/'source_contrasts_preview.png', dpi=120); plt.close(fig)


if __name__ == '__main__': main()
