"""Display every registered contrast, not a selected favorable assignment."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import run_m3w_european_risk_conditioned_residual as run


def main():
    a = json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    plt.rcParams.update({'svg.hashsalt': 'm3w-risk-conditioned-residual-v1', 'font.size': 9, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(4, 2, figsize=(14, 17))
    labels = [('common_context', 'Previous common-event context'), ('original', 'Original outer estimator'),
              ('inner_event', 'Previous inner-event OOF'), ('outer_context', 'Prior three-locality context'),
              ('risk_only', 'New score-only OOF'), ('risk_next', 'New score/context cyclic next'),
              ('risk_prev', 'New score/context cyclic previous')]
    for ax, (key, title) in zip(axes.flat, labels):
        for pair, color, offset in [('full', '#267b91', -.12), ('motion_only', '#ba4657', .12)]:
            values = a['contrasts']['risk_oof_vs_'+key][pair]; roles = sorted(values)
            q = [values[r]['envelope_positive__harm_MSE_gain_percent'] for r in roles]
            yy = np.arange(len(roles))+offset
            ax.hlines(yy, [x['CI'][0] for x in q], [x['CI'][1] for x in q], color=color)
            ax.scatter([x['point'] for x in q], yy, color=color, label=pair, s=24, zorder=3)
        ax.set_yticks(np.arange(len(roles)), [r.replace('producer', 'P').replace('_controller', ' / C') for r in roles])
        ax.axvline(0, color='#666666', ls='--', lw=1); ax.grid(axis='x', alpha=.18)
        ax.set_title('Risk-conditioned OOF vs\n'+title)
        ax.set_xlabel('Easy-harm MSE improvement (%)\n95% locality bootstrap CI; three-seed means')
        ax.legend(frameon=False, fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0, .95,
        'Evidence boundaries\n\n864 fixed ridge probes; no new neural training.\n'
        'Six overlapping source assignments, four localities each.\n'
        'Three-seed means; 3,000 paired locality resamples.\n'
        'No multiplicity adjustment or safety guarantee.\n\n'
        'All trajectories and other three costs are unchanged.\n'
        'Independent selection/calibration/confirmation unopened.\n'
        'Cost estimation is not trajectory utility or deployment.', va='top', linespacing=1.6)
    fig.suptitle('Risk-conditioned residual transfer: source-development comparisons, all retained')
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(run.PUBLIC/'risk_conditioned_residual.svg', metadata={'Date': None})
    fig.savefig(run.PRIVATE/'risk_conditioned_residual_preview.png', dpi=100)
    plt.close(fig)


if __name__ == '__main__': main()
