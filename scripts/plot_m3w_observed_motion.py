"""Aggregate-only scientific SVG, no source imagery or person trajectories."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'data/stage_cvpr2027_experiments/observed_motion_v2/matplotlib_cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'data/stage_cvpr2027_experiments/observed_motion_v2/font_cache'))
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'm3w_observed_motion_v2'
import matplotlib.pyplot as plt
import numpy as np


def main():
    directory = ROOT/'outputs/publication_readiness_2026_09/observed_motion_v2'
    report = json.loads((directory/'report.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharex=True)
    colors = {'quality_control': '#54666f', 'magnitude': '#278978', 'directed': '#ba475f'}
    for axis, objective in zip(axes, ('row_log', 'scene_ade_harm')):
        for j, (variant, color) in enumerate(colors.items()):
            for fold in range(3):
                ts = [t for t in report['trials'] if t['variant'] == variant and t['objective'] == objective and t['fold'] == fold]
                y = [t['vs_CV']['improvement_percent'] for t in ts]
                x = fold + (j-1)*.20
                axis.scatter(x+np.linspace(-.025, .025, 3), y, color=color, s=20, alpha=.8)
                axis.plot([x-.05, x+.05], [np.mean(y)]*2, color=color, linewidth=2,
                          label=variant.replace('_', ' ') if fold == 0 else None)
        axis.axhline(0, color='black', linewidth=.8)
        axis.set_xticks(range(3), ['ETH', 'Hotel', 'Grouped Zara'])
        axis.set_title(objective.replace('_', ' '))
        axis.set_ylabel('Past-normalized ADE gain vs CV (%)')
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Past image motion: fixed forecasts on held fit scenes', fontsize=12)
    fig.text(.5, .015, 'Three historically used scenes; each point is one seed. Not independent confirmation.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(directory/'scene_seed_results.svg', metadata={'Date': None})
    private = ROOT/'data/stage_cvpr2027_experiments/observed_motion_v2/scene_seed_results.png'
    fig.savefig(private, dpi=140)
    plt.close(fig)
    print(private)


if __name__ == '__main__':
    main()
