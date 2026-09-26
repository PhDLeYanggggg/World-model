"""Fixed primary contrasts and all predefined context-sign summaries."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import run_m3w_european_context_residual as run


def main():
    a = json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    plt.rcParams.update({'svg.hashsalt':'m3w-context-residual-v1', 'font.size':10, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, key, title in zip(axes[:2], ('original_context_vs_self', 'original_context_vs_global'),
            ('Context correction vs original', 'Context correction vs global bias')):
        for pair, color, shift in (('full', '#267b91', -.12), ('motion_only', '#ba4657', .12)):
            roles = sorted(a['contrasts'][key][pair]); points, lo, hi = [], [], []
            for role in roles:
                v = a['contrasts'][key][pair][role]['envelope_positive__harm_MSE_gain_percent']
                assert 'CI' in v; points.append(v['point']); lo.append(v['CI'][0]); hi.append(v['CI'][1])
            yy = np.arange(len(roles))+shift
            ax.hlines(yy, lo, hi, color=color); ax.scatter(points, yy, color=color, s=24, label=pair, zorder=3)
        ax.set_yticks(np.arange(len(roles)), [r.replace('producer', 'P').replace('_controller', ' / C') for r in roles])
        ax.axvline(0, color='#666666', ls='--', lw=1); ax.set_title(title)
        ax.set_xlabel('Easy-harm MSE improvement (%)\nThree-seed mean; 95% locality bootstrap CI')
        ax.grid(axis='x', alpha=.18); ax.legend(loc='best', frameon=False)
    ax = axes[2]; p = a['patterns']['full']['original']; names = run.method.NAMES
    same = np.array([p[n]['held_same_sign'] for n in names]); opposite = np.array([p[n]['held_opposite_sign'] for n in names])
    ax.barh(np.arange(len(names)), same, color='#267b91', label='Held same sign')
    ax.barh(np.arange(len(names)), opposite, left=same, color='#ba4657', label='Held opposite sign')
    ax.set_yticks(np.arange(len(names)), [n.replace('_', ' ') for n in names])
    ax.set_xlabel('Dependent supported context cells\nConsistent sign in all three fitting localities')
    ax.set_title('Does fitting bias direction repeat?'); ax.legend(frameon=False); ax.grid(axis='x', alpha=.18)
    for ax in axes: ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Fixed cost probes, not new trajectory gains or independent calibration')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(run.PUBLIC/'context_probe.svg', metadata={'Date':None})
    fig.savefig(run.PRIVATE/'context_probe_preview.png', dpi=130); plt.close(fig)


if __name__ == '__main__': main()
