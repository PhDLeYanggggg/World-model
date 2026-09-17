"""Aggregate scientific figure, without source imagery or row trajectories."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/spatial_motion'
os.environ.setdefault('MPLCONFIGDIR', str(PRIVATE/'matplotlib_cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(PRIVATE/'font_cache'))
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'm3w_spatial_motion'
import matplotlib.pyplot as plt
import numpy as np


def main():
    directory = ROOT/'outputs/publication_readiness_2026_09/spatial_motion'
    report = json.loads((directory/'report.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    colors = {'quality_control': '#54666f', 'lowpass_pool': '#278978',
              'lowpass_grid': '#aa7830', 'native_grid': '#ba475f'}
    for j, (variant, color) in enumerate(colors.items()):
        for fold in range(3):
            trials = [t for t in report['trials'] if t['variant'] == variant and t['fold'] == fold]
            y = [t['vs_CV']['improvement_percent'] for t in trials]
            x = fold+(j-1.5)*.16
            axes[0].scatter(x+np.linspace(-.02, .02, 3), y, color=color, s=20, alpha=.8)
            axes[0].plot([x-.05, x+.05], [np.mean(y)]*2, color=color, linewidth=2,
                         label=variant.replace('_', ' ') if fold == 0 else None)
        summary = report['summary'][variant+'_row_log']['vs_CV']
        values = summary['per_seed_gain_percent']
        axes[1].scatter(j+np.linspace(-.06, .06, 3), values, color=color, s=25)
        axes[1].plot([j-.12, j+.12], [summary['gain_percent']]*2, color=color, linewidth=2)
    axes[0].set_xticks(range(3), ['ETH', 'Hotel', 'Grouped Zara'])
    axes[1].set_xticks(range(4), ['Quality', 'Lowpass\npool', 'Lowpass\ngrid', 'Native\ngrid'])
    for axis in axes:
        axis.axhline(0, color='black', linewidth=.8)
        axis.set_ylabel('Past-normalized ADE gain vs CV (%)')
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_title('Held fit scene / training seed')
    axes[1].set_title('Equal-scene aggregate / training seed')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Native detail and spatial motion: no stable forecast improvement', fontsize=12)
    fig.text(.5, .015, 'Three historically used scenes; fixed 4,000-update models. Not independent confirmation.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(directory/'scene_seed_results.svg', metadata={'Date': None})
    fig.savefig(PRIVATE/'scene_seed_results.png', dpi=140)
    plt.close(fig)
    print(PRIVATE/'scene_seed_results.png')


if __name__ == '__main__':
    main()
