"""Source-development repair contrasts, including failed safety and strong controls."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_cv_reference_v1'


def main():
    from report_m3w_european_cv_reference import sha
    r = json.loads((PUBLIC/'analysis.json').read_text())
    audit = json.loads((PUBLIC/'accounting_audit.json').read_text())
    if audit['analysis_sha256'] != sha(PUBLIC/'analysis.json') or not audit['all_passed']:
        raise ValueError('Verified result accounting required')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'svg.hashsalt': 'm3w-cv-reference-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    seeds = [17, 29, 43]
    rows = [r['seeds'][str(seed)+'_neural_underharm4']['full'] for seed in seeds]
    y = np.arange(3)
    for offset, name, color, label in [(-.18, 'previous_pointwise', '#777777', 'Old selected-baseline reference'),
                                     (.18, 'pointwise', '#117864', 'CV-reference repair')]:
        values = [-row[name]['positive_easy_ADE_vs_CV']['equal_scene_gain_percent'] for row in rows]
        bars = axes[0].barh(y+offset, values, height=.32, label=label, color=color)
        axes[0].bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    axes[0].axvline(2, color='#a04000', linestyle='--', linewidth=1, label='2% mean limit')
    axes[0].set_yticks(y, [f'Seed {v}' for v in seeds]); axes[0].invert_yaxis()
    axes[0].set_xlim(0, 16)
    axes[0].set_xlabel('Equal-locality positive-easy degradation (%)')
    axes[0].set_title('Partial repair, not a worst-locality safety pass')
    axes[0].legend(loc='lower right', frameon=False, fontsize=8)
    for i, row in enumerate(rows):
        m = row['pointwise']['ADE_vs_fixed_damping097']
        lo, hi = m['scene_bootstrap_ci95']
        axes[1].plot([lo, hi], [i, i], color='#21618c', linewidth=2)
        axes[1].scatter(m['equal_scene_gain_percent'], i, color='#21618c', s=40)
    axes[1].axvline(0, color='black', linewidth=.8)
    axes[1].set_yticks(y, [f'Seed {v}' for v in seeds]); axes[1].invert_yaxis()
    axes[1].set_xlabel('ADE gain over fixed damping 0.97 (%)')
    axes[1].set_title('3,000-locality-bootstrap intervals include zero')
    fig.suptitle('Frozen forecasts, new cost heads; source development only', fontsize=13)
    for ext in ('png', 'svg'):
        path = PUBLIC/('repair_contrasts.'+ext)
        fig.savefig(path, dpi=160, metadata={'Date': None} if ext == 'svg' else None)
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)
    print(str(PUBLIC/'repair_contrasts.png'))


if __name__ == '__main__':
    main()
