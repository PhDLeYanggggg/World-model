"""Plot every registered risk policy without selecting an outcome winner."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/net_easy_moment_guarded_v1'
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'm3w-net-easy-moment-v1'
import matplotlib.pyplot as plt
import numpy as np


def main():
    rows = json.loads((PUBLIC/'compact_results.json').read_text())['rows']
    fig, axes = plt.subplots(3, 2, figsize=(12, 13), gridspec_kw={'width_ratios':[1.4, 1.]})
    colors = ['#555555', '#333333', '#999999', '#777777', '#087F8C', '#AC3973', '#267B49', '#B54B2A', '#62529B']
    for i, action in enumerate(('damped_velocity_005', 'transformer', 'eqmotion')):
        group = [r for r in rows if r['action']==action]
        y = np.arange(len(group))
        for j, r in enumerate(group):
            ax = axes[i, 0]; ci = r['CI']; val = r['ADE_gain']
            ax.errorbar(val, j, xerr=[[max(0., val-ci[0])], [max(0., ci[1]-val)]],
                fmt='o', color=colors[j], capsize=3, markersize=5)
            axes[i, 1].barh(j, r['worst_site_seed_easy_degradation'], color=colors[j], height=.65)
        axes[i, 0].set_yticks(y, [r['policy'].replace('_', ' ') for r in group], fontsize=9)
        axes[i, 1].set_yticks(y, [])
        for ax in axes[i]:
            ax.invert_yaxis(); ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
            ax.spines[['right', 'top']].set_visible(False)
        axes[i, 0].axvline(0, color='#444444', lw=.8)
        axes[i, 1].axvline(2, color='#B12525', linestyle='--', lw=1.1, label='2% ceiling')
        axes[i, 1].legend(fontsize=8, loc='lower right')
        axes[i, 0].set_title(action.replace('_', ' '), loc='left', fontsize=12)
        axes[i, 0].set_xlabel('Equal-site ADE gain over CV (%) with site-bootstrap CI95')
        axes[i, 1].set_xlabel('Worst site/seed easy degradation (%)')
    fig.suptitle('Net versus positive easy risk: all fixed controls', fontsize=16, y=.995)
    fig.text(.5, .012, 'SDD development only | obs8/pred12 annotation pixels | 4 sites, 3 seeds | not a safety certificate', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .03, 1, .98))
    svg = PUBLIC/'risk_tradeoff.svg'; fig.savefig(svg, metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    png = ROOT/'data/stage_cvpr2027_experiments/net_easy_moment_guarded_v1/risk_tradeoff_preview.png'
    fig.savefig(png, dpi=140); plt.close(fig)
    print(json.dumps(dict(svg=str(svg), preview=str(png))))


if __name__ == '__main__':
    main()
