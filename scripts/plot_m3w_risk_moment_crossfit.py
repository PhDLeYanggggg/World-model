"""Aggregate locality intervals, not independent-row uncertainty."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import run_m3w_european_risk_moment_crossfit as run


def main():
    d = json.loads((run.PUBLIC/'summary.json').read_text())['by_candidate']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), layout='constrained')
    order = [('dimensionless','fit'), ('dimensionless','held'), ('damped','fit'), ('damped','held')]
    names = ['Neural / fitting', 'Neural / held', 'Damping / fitting', 'Damping / held']
    colors = ['#167d8d', '#167d8d', '#a85326', '#a85326']
    def point(ax, x, y, color, factor=1, offset=0, marker='o'):
        if x['point'] is None or x['ci95'] is None: return
        v = x['point']*factor; lo, hi = [c*factor for c in x['ci95']]
        ax.errorbar(v, y+offset, xerr=[[v-lo],[hi-v]], fmt=marker, color=color, capsize=3, markersize=5)
    for ax, task in zip(axes.flat[:2], ['reference','harm']):
        for i, (arm, scope) in enumerate(order):
            point(ax, d[arm][f'{scope}_{task}_MSE_skill_percent'], 3-i, colors[i])
        ax.set_yticks(range(4), names[::-1]); ax.axvline(0, color='#666666', linestyle='--', linewidth=.8)
        ax.set_xlabel('MSE improvement over fitting-only constant (%)')
        ax.set_title('Reference-cost prediction' if task == 'reference' else 'Positive-harm prediction')
    ax = axes[1, 0]
    for i, (arm, scope) in enumerate(order):
        point(ax, d[arm][f'{scope}_screen_actual_harm_ratio'], 3-i, colors[i], factor=100, offset=.09)
        point(ax, d[arm][f'{scope}_screen_predicted_harm_ratio'], 3-i, '#636363', factor=100, offset=-.09, marker='s')
    ax.set_yticks(range(4), names[::-1]); ax.axvline(2, color='#b22222', linestyle='--', linewidth=1)
    ax.set_xlabel('Positive harm / CV error on screened rows (%)')
    ax.set_title('Screened risk: actual (circle), predicted (square)', fontsize=10)
    ax = axes[1, 1]
    for i, (arm, moment) in enumerate([('dimensionless','reference'),('dimensionless','harm'),('damped','reference'),('damped','harm')]):
        point(ax, d[arm][f'held_screen_{moment}_actual_over_predicted'], 3-i, colors[i])
    ax.set_yticks(range(4), ['Damping / harm','Damping / reference','Neural / harm','Neural / reference'])
    ax.axvline(1, color='#666666', linestyle='--', linewidth=.8)
    ax.set_xlabel('Actual / predicted moment on held screened rows')
    ax.set_title('Held screened-moment calibration', fontsize=10)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='x', alpha=.15); ax.tick_params(labelsize=8)
    fig.suptitle('Lower average prediction error does not calibrate the intervention subset', fontsize=13)
    fig.supxlabel('12 opened source localities; 3 seeds; 3,000 locality bootstrap draws. Diagnostic screen, not deployment.', fontsize=9)
    fig.savefig(run.PUBLIC/'moment_evidence.png', dpi=150, metadata={'Software':'M3W fixed aggregate plot'})
    plt.close(fig)


if __name__ == '__main__': main()
