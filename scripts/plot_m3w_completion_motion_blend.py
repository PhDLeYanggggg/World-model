"""Render the matched observed-versus-full-grid pair comparison."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v13 as evidence


def build(path):
    refs = {}
    pair = evidence.verified_analysis('native_relative_path_training_v1', refs)
    blend = evidence.verified_analysis('native_completion_motion_blend_v1', refs)
    if pair['shared_substitutions'] != blend['common_substitutions'] or blend['common_substitutions'] != 9675:
        raise ValueError('Expected identical registered recipients')
    values = [pair['relative_geometry']['with_relative_vs_individual'], *(
        blend['relative_geometry'][f'{k}_vs_reference'] for k in ('fixed_half', 'observed_fit', 'completion_fit'))]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    colors = ('#21455b', '#586875', '#b4573b', '#17746e')
    labels = ('Full pair correction', 'Fixed half correction', 'Learned: observed loss', 'Learned: with missing debit')
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.08), sharey=True, gridspec_kw={'wspace': .31})
    for ax, title, key, limit in zip(axes, ('A  Co-observed steps', 'B  Full-grid lower contrast'),
                                    ('observed_ade_reduction', 'grid_ade_lower'),
                                    ((-.008, .077), (-.67, .040)), strict=True):
        ax.axvline(0, color='#7d8487', linestyle='--', linewidth=.8, zorder=1)
        for i, (v, color) in enumerate(zip(values, colors, strict=True)):
            m = v[key]
            lo, hi = m['interval95']
            ax.errorbar(m['point'], i, xerr=[[m['point']-lo], [hi-m['point']]], fmt='o',
                        color=color, linewidth=1.4, markersize=5, capsize=3, zorder=3)
        ax.set(xlim=limit, title=title, xlabel='Relative-path ADE reduction (pixels)')
        ax.grid(axis='x', linewidth=.65, alpha=.3)
        ax.spines[['left', 'top', 'right']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_ylim(len(labels)-.55, -.45)
    fig.suptitle('Pair corrections versus the same individual-motion reference',
                 fontweight='bold', y=.975)
    fig.text(.02, .028, 'Positive favors correction. Separate horizontal scales. '
             '95% paired source-bootstrap intervals, with seeds averaged within source.', fontsize=7.6)
    fig.subplots_adjust(left=.24, right=.98, top=.79, bottom=.23)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    return refs


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=evidence.prior.DATA/'completion_motion_blend_figure.pdf')
    args = p.parse_args()
    print(json.dumps(build(args.output), sort_keys=True))
