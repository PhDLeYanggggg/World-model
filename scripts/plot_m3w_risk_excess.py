"""Fixed aggregate views of the loss contrast and its uncalibrated screen."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import run_m3w_european_risk_excess as run


def main():
    d = json.loads((run.PUBLIC/'summary.json').read_text())['by_candidate']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), layout='constrained')
    colors = dict(dimensionless='#167d8d', damped='#a85326')
    def point(ax, value, y, arm, factor=1):
        if value['point'] is None or value['ci95'] is None: return
        v = value['point']*factor; lo, hi = [x*factor for x in value['ci95']]
        ax.errorbar(v, y, xerr=[[v-lo],[hi-v]], fmt='o', color=colors[arm], capsize=3, markersize=5)
    order = [('dimensionless','fit'),('dimensionless','held'),('damped','fit'),('damped','held')]
    ax = axes[0, 0]
    for i, (arm, role) in enumerate(order): point(ax, d[arm][role+'_MSE_gain_vs_control_percent'], 3-i, arm)
    ax.set_yticks(range(4), ['Damping / held','Damping / fitting','Neural / held','Neural / fitting'])
    ax.axvline(0, color='#666666', linestyle='--', linewidth=.8)
    ax.set_title('Direct score objective vs two-moment objective', fontsize=10)
    ax.set_xlabel('Signed-score MSE improvement (%)')
    policy_order = [('dimensionless','control'),('dimensionless','new'),('damped','control'),('damped','new')]
    names = ['Damping / direct','Damping / moments','Neural / direct','Neural / moments']
    specs = [(axes[0,1], 'screen_positive_harm_ratio', 100, 2, 'Held screened positive harm', 'Positive harm / CV error (%)'),
             (axes[1,0], 'screen_rate', 100, None, 'Held diagnostic coverage', 'Rows accepted by score <= 0 (%)'),
             (axes[1,1], 'easy_ADE_gain_vs_CV_percent', 1, -2, 'Held easy net ADE gain', 'Gain vs CV (%); dashed line = -2%')]
    for ax, key, factor, reference, title, label in specs:
        for i, (arm, policy) in enumerate(policy_order): point(ax, d[arm]['held_'+policy+'_'+key], 3-i, arm, factor)
        ax.set_yticks(range(4), names); ax.set_title(title, fontsize=10); ax.set_xlabel(label)
        if reference is not None: ax.axvline(reference, color='#b22222', linestyle='--', linewidth=1)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='x', alpha=.15); ax.tick_params(labelsize=8)
    fig.suptitle('A matched risk-objective change: prediction error and screening behavior', fontsize=13)
    fig.supxlabel('12 opened source localities; 3 seeds; 3,000 locality bootstrap draws. All-risk-only screen, not deployment.', fontsize=9)
    fig.savefig(run.PUBLIC/'risk_excess_evidence.png', dpi=150, metadata={'Software':'M3W fixed aggregate plot'})
    plt.close(fig)


if __name__ == '__main__': main()
