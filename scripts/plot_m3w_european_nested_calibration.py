"""Show every nested neural calibration view, not a selected winner."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.report_m3w_european_cv_reference import sha

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_nested_calibration_v1'


def main():
    a = json.loads((PUBLIC/'analysis.json').read_text())
    audit = json.loads((PUBLIC/'accounting_audit.json').read_text())
    if not audit['all_passed'] or audit['analysis_sha256'] != sha(PUBLIC/'analysis.json'):
        raise ValueError('Verified complete results required')
    rows = [(d, e, r) for d in (1, 2) for e in ('all', 'easy')
        for r in ('none', 'population_rescale', 'selected_risk_grid')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True, layout='constrained')
    colors = ['#126a7c', '#b34f32', '#7a4386']
    for i, seed in enumerate((17, 29, 43)):
        for j, (d, event, rule) in enumerate(rows):
            key = f'{seed}_direction{d}_neural_{event}_{rule}'
            p = a['policies'][key]
            y = j+(i-1)*.2
            for ax, m in zip(axes[:2], (p['ADE_vs_CV'], a['neural_vs_damping'][key]['all'])):
                val, interval = m['equal_scene_gain_percent'], m['scene_bootstrap_ci95']
                if val is not None:
                    ax.plot(val, y, 'o', color=colors[i], markersize=4)
                if interval is not None:
                    ax.plot(interval, [y, y], color=colors[i], linewidth=1)
            easy = p['positive_easy_ADE_vs_CV']['worst_scene_gain_percent']
            if easy is not None:
                axes[2].plot(-easy, y, 'x' if p['zero_CV']['harmed_rows'] else 'o',
                    color=colors[i], markersize=5, label=f'Seed {seed}' if j == 0 else None)
    labels = {'none': 'uncalibrated', 'population_rescale': 'population rescale', 'selected_risk_grid': 'selected-risk fit'}
    axes[0].set_yticks(np.arange(len(rows)), [f'Rotation {d} / {e} / {labels[r]}' for d,e,r in rows])
    axes[0].invert_yaxis()
    for ax, title in zip(axes, ('Neural ADE gain vs CV', 'Neural gain vs matched damping', 'Worst-locality easy degradation')):
        ax.set_title(title, fontsize=11)
        ax.axvline(0, color='#777777', linewidth=.8)
        ax.grid(axis='x', color='#dddddd', linewidth=.5)
        ax.set_xlabel('Percent')
        ax.spines[['top', 'right']].set_visible(False)
    axes[2].axvline(2, color='#b34f32', linestyle='--', label='2% limit')
    axes[2].set_xscale('symlog', linthresh=2)
    fig.legend(*axes[2].get_legend_handles_labels(), loc='outside lower center', ncols=4)
    fig.suptitle('Nested source calibration: all 36 neural views\n'
        '3,000 conditional locality bootstrap resamples; x means a zero-CV case was harmed', fontsize=13)
    svg = PUBLIC/'calibration_comparison.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(PUBLIC/'calibration_comparison.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
