"""Descriptive TRAIN calibration plot, not a held-out performance claim."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.report_m3w_selected_cost_audit import PUBLIC


def main():
    value = json.loads((PUBLIC/'summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)
    pools = ('all_legal', 'gate_passing', 'selected')
    for ax, key, title in zip(axes, ('harm_prediction_ratio', 'easy_harm_prediction_ratio'),
                              ('All harm H', 'Easy harm EH')):
        for family, label, color, style in (('no_scene', 'Motion only', '#69747b', '--'),
                                            ('scene', 'Static scene context', '#286b79', '-')):
            data = value['families'][family]['physical_neural']['pools']
            ratios = [data[p]['median'][key] for p in pools]
            ax.plot(range(3), ratios, marker='o', lw=2, ls=style, color=color, label=label)
            for i, ratio in enumerate(ratios):
                if family == 'scene':
                    ax.annotate(f'{ratio:.3f}', (i, ratio), xytext=(0, 10),
                                textcoords='offset points', ha='center', fontsize=10)
        ax.axhline(1, color='#9c5353', lw=1, ls=':')
        ax.set_yscale('log'); ax.set_ylim(.01, 4)
        ax.set_xticks(range(3), ['All legal', 'Gate passing', 'Selected'])
        ax.set_title(title, fontsize=12); ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Median predicted / observed cost mass (log scale)')
    axes[0].legend(frameon=False, fontsize=10, loc='lower left')
    fig.suptitle('Cost accuracy changes inside the policy region', fontsize=13)
    fig.text(.5, .02, 'Exposed TRAIN only; defined record/head ratios, not independent samples or a safety guarantee.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .92))
    folder = PUBLIC/'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder/'selection_conditioned_cost_ratios.png', dpi=180); plt.close(fig)
    print(json.dumps(dict(plot=str(folder.relative_to(ROOT)), fresh_training=False)))


if __name__ == '__main__':
    main()
