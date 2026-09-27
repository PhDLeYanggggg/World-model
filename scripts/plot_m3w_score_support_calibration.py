"""Fixed aggregate figure with explicit worst-view safety failures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_score_support_calibration_v1'


def main():
    d = json.loads((PUBLIC/'summary.json').read_text())
    policies = ['guarded','supported','calibrated','calibrated_supported']
    fig, axes = plt.subplots(1,3,figsize=(13,5.4),layout='constrained')
    for arm, offset, color in [('dimensionless',-.1,'#167d8d'),('damped',.1,'#a85326')]:
        for i,p in enumerate(policies):
            m = d['summary'][arm]['excess_'+p]
            for ax,key,mult in zip(axes,['all_gain_percent','switch_rate','easy_gain_percent'],[1,100,1]):
                r = m[key]
                if r['point'] is None: continue
                pt = r['point']*mult; lo,hi = [v*mult for v in r['ci95']]
                ax.errorbar(pt,i+offset,xerr=[[pt-lo],[hi-pt]],fmt='o',color=color,label=arm if i==0 else None,capsize=3)
    for ax,title in zip(axes,['Net ADE gain vs CV (%)','Intervention rate (%)','Net easy ADE gain vs CV (%)']):
        ax.set_yticks(range(4),policies); ax.invert_yaxis(); ax.set_xlabel(title)
        ax.axvline(0,color='#777777',linestyle='--'); ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(loc='best',fontsize=8); axes[2].axvline(-2,color='red',linestyle=':')
    worst = d['worst_views']['dimensionless']['excess_calibrated_supported']['worst_easy_gain_percent']
    fig.suptitle('Fixed signed-score policies: independent calibration sources within development')
    fig.supxlabel('12 development localities; 3 seeds; 3,000 locality-bootstrap draws. No risk certificate or deployment.\n'
        f'Worst calibrated-supported neural view easy gain: {worst:.2f}%. Both objective families reported in the table.',fontsize=9)
    fig.savefig(PUBLIC/'calibration_evidence.png',dpi=150,metadata={'Software':'M3W fixed aggregate plot'})
    plt.close(fig)


if __name__ == '__main__': main()
