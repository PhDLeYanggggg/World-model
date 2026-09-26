"""Registered nested-residual contrasts and all inner-head training losses."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import run_m3w_european_nested_residual as run


def main():
    a = json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    plt.rcParams.update({'svg.hashsalt':'m3w-nested-residual-v1', 'font.size':9, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    contrasts = [('original', 'Original outer estimator'), ('outer_context', 'Prior in-sample context repair'),
                 ('oof_global', 'OOF global-bias repair'), ('next', 'Matched cyclic control: next'),
                 ('prev', 'Matched cyclic control: previous')]
    for ax, (key, title) in zip(axes.flat, contrasts):
        for pair, color, shift in (('full', '#267b91', -.12), ('motion_only', '#ba4657', .12)):
            values = a['contrasts']['oof_context_vs_'+key][pair]; roles = sorted(values)
            points, lo, hi = [], [], []
            for role in roles:
                v = values[role]['envelope_positive__harm_MSE_gain_percent']
                assert 'CI' in v
                points.append(v['point']); lo.append(v['CI'][0]); hi.append(v['CI'][1])
            yy = np.arange(len(roles))+shift
            ax.hlines(yy, lo, hi, color=color)
            ax.scatter(points, yy, color=color, s=24, label=pair, zorder=3)
        ax.set_yticks(np.arange(len(roles)), [r.replace('producer', 'P').replace('_controller', ' / C') for r in roles])
        ax.axvline(0, color='#666666', ls='--', lw=1)
        ax.set_title('OOF context repair vs\n'+title)
        ax.set_xlabel('Easy-harm MSE improvement (%)\n95% locality bootstrap CI; three-seed means')
        ax.grid(axis='x', alpha=.18); ax.legend(frameon=False, fontsize=8)
    heads = run.completed_heads()
    ax = axes.flat[5]
    steps = list(range(200, 2001, 200))
    for pair, color in (('full', '#267b91'), ('motion_only', '#ba4657')):
        rows = [h for h in heads if ('_'+pair+'_') in h['input']['tag']]
        losses = np.array([[next(t['cost_loss'] for t in h['fit']['trace'] if t['step']==s) for s in steps] for h in rows])
        lo, median, hi = np.quantile(losses, [.25, .5, .75], axis=0)
        ax.plot(steps, median, color=color, label=pair)
        ax.fill_between(steps, lo, hi, color=color, alpha=.16)
    ax.set_title('All 432 inner risk heads\nFixed training-batch loss, not held performance')
    ax.set_xlabel('Optimizer updates per head'); ax.set_ylabel('Normalized cost loss: median and IQR')
    ax.legend(frameon=False); ax.grid(alpha=.18)
    for ax in axes.flat: ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Nested residual supervision: source-development evidence, not trajectory gain or independent calibration')
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(run.PUBLIC/'nested_residual.svg', metadata={'Date':None})
    fig.savefig(run.PRIVATE/'nested_residual_preview.png', dpi=130); plt.close(fig)


if __name__ == '__main__': main()
