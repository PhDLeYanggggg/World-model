"""Aggregate training-versus-transfer evidence; no source media or row identities."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/objective_alignment'
os.environ['MPLCONFIGDIR'] = str(PRIVATE / 'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'm3w-objective-alignment'
import matplotlib.pyplot as plt


def main():
    directory = ROOT / 'outputs/publication_readiness_2026_09/objective_alignment'
    report = json.loads((directory / 'report.json').read_text())
    arms = ['row_log', 'row_ade', 'scene_log', 'scene_ade', 'scene_ade_harm']
    labels = ['Row / log', 'Row / ADE', 'Scene / log', 'Scene / ADE', 'Scene / ADE\n+ harm']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for fold, (name, color, marker) in enumerate(zip(
            ['ETH', 'Hotel', 'Zara (grouped)'], ['#555555', '#b13e50', '#167d8d'], ['o', 's', '^'])):
        for index, arm in enumerate(arms):
            trials = [t for t in report['trials'] if t['arm'] == arm and t['fold'] == fold]
            xs = [index + (fold-1)*.2 + (i-1)*.05 for i in range(3)]
            axes[0].scatter(xs, [t['training_equal_scene_primary_gain_percent'] for t in trials],
                            c=color, marker=marker, s=23, label=name if index == 0 else None)
            axes[1].scatter(xs, [t['vs_CV']['primary_ADE']/t['vs_CV']['reference_ADE'] for t in trials],
                            c=color, marker=marker, s=23)
    axes[0].axhline(0, color='black', linestyle='--', linewidth=.8)
    axes[0].set_title('Training gain increases with aligned loss')
    axes[0].set_ylabel('Training primary ADE gain vs CV (%)')
    axes[0].legend(title='Held site of each training fold', fontsize=9)
    axes[1].axhline(1, color='black', linestyle='--', linewidth=.8)
    axes[1].set_yscale('log')
    axes[1].set_ylim(.93, 10)
    axes[1].set_title('Every held-scene forecast remains worse than CV')
    axes[1].set_ylabel('Held-scene ADE / CV ADE (log axis; below 1 is better)')
    for ax in axes:
        ax.set_xticks(range(5), labels, rotation=20, ha='right')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.supxlabel('45 matched fits; points are seed/fold results, not independent new sites.', fontsize=10)
    buffer = io.StringIO()
    fig.savefig(buffer, format='svg', metadata={'Date': None})
    (directory / 'training_transfer.svg').write_text('\n'.join(s.rstrip() for s in buffer.getvalue().splitlines())+'\n')
    fig.savefig(PRIVATE / 'training_transfer.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
