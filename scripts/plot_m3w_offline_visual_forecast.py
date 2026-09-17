"""Plot aggregate evidence only; never source video frames or individual tracks."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/offline_visual_forecast'
os.environ['MPLCONFIGDIR'] = str(PRIVATE / 'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'm3w-offline-visual-forecast'
import matplotlib.pyplot as plt
import numpy as np


def main():
    directory = ROOT / 'outputs/publication_readiness_2026_09/offline_visual_forecast'
    report = json.loads((directory / 'report.json').read_text())
    arms = ['geometry', 'mask_only', 'current_rgb', 'past_rgb']
    labels = ['Geometry', 'Mask only', 'Current RGB', 'Past RGB']
    colors = ['#525252', '#9a9a9a', '#167d8d', '#c25065']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout='constrained')
    for fold, (ax, title) in enumerate(zip(axes, ['ETH', 'Hotel', 'Zara (grouped)'])):
        for index, (arm, color) in enumerate(zip(arms, colors)):
            values = [t['vs_CV']['improvement_percent'] for t in report['trials'] if t['fold'] == fold and t['arm'] == arm]
            ax.scatter(index + np.array([-.12, 0, .12]), values, color=color, s=27)
            ax.plot([index - .22, index + .22], [np.mean(values)] * 2, color=color, linewidth=2)
        ax.axhline(0, color='black', linewidth=.8, linestyle='--')
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(4), labels, rotation=25, ha='right')
        ax.set_ylabel('Normalized ADE gain vs CV (%)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Full fit cohort: no neural arm beats CV on a held physical scene', fontsize=13)
    fig.supxlabel('Points: three seeds. Short lines: seed means. Panel scales differ; higher is better.', fontsize=10)
    svg = io.StringIO()
    fig.savefig(svg, format='svg', metadata={'Date': None})
    (directory / 'scene_seed_results.svg').write_text(
        '\n'.join(line.rstrip() for line in svg.getvalue().splitlines()) + '\n')
    fig.savefig(PRIVATE / 'scene_seed_results.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
