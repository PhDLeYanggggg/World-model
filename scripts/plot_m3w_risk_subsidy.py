"""All fixed controls and adverse cases; no visual outcome selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt']='m3w-risk-subsidy-v1'
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/risk_subsidy_v1'


def main():
    rows=json.loads((PUBLIC/'compact_results.json').read_text())['rows']
    fig,axes=plt.subplots(3,2,figsize=(15,14),gridspec_kw={'width_ratios':[1.5,1.]})
    palette=['#333333','#777777','#999999','#555555','#B34B2D','#167A9C','#827E24','#257248','#7F508B','#328D9B','#7F9B42']
    for i,action in enumerate(('damped_velocity_005','transformer','eqmotion')):
        group=[r for r in rows if r['action']==action]
        for j,r in enumerate(group):
            value=r['ADE_gain'];lo,hi=r['CI']
            axes[i,0].errorbar(value,j,xerr=[[max(0,value-lo)],[max(0,hi-value)]],fmt='o',capsize=3,color=palette[j])
            axes[i,1].barh(j,r['worst_easy_degradation'],height=.65,color=palette[j])
            axes[i,1].text(.985,j,f"{r['zero_CV_harmed']} harmed",ha='right',va='center',fontsize=8,
                transform=axes[i,1].get_yaxis_transform(),bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
        axes[i,0].set_yticks(range(len(group)),[r['policy'].replace('_',' ') for r in group],fontsize=9)
        axes[i,1].set_yticks(range(len(group)),[])
        for ax in axes[i]:
            ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.spines[['right','top']].set_visible(False)
        axes[i,0].axvline(0,color='#444444',lw=.8)
        axes[i,1].axvline(2,color='#B12525',ls='--',lw=1.1)
        axes[i,1].set_xlim(0,max(2.5,max(r['worst_easy_degradation'] for r in group)*1.2))
        axes[i,0].set_title(action.replace('_',' '),loc='left')
        axes[i,0].set_xlabel('Equal-site ADE gain over CV (%) and site-bootstrap CI95')
        axes[i,1].set_xlabel('Worst site/seed easy degradation (%); zero-CV harm counts at right')
    fig.suptitle('Query risk credit and denominator: all fixed controls',fontsize=16,y=.995)
    fig.text(.5,.012,'Development-exposed SDD | obs8/pred12 | 4 sites, 3 seeds | 2% line is not a safety certificate',ha='center')
    fig.tight_layout(rect=(0,.03,1,.98))
    svg=PUBLIC/'risk_tradeoff.svg';fig.savefig(svg,metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    preview=ROOT/'data/stage_cvpr2027_experiments/risk_subsidy_v1/risk_tradeoff_preview.png'
    fig.savefig(preview,dpi=140);plt.close(fig)
    print(json.dumps(dict(svg=str(svg),preview=str(preview))))


if __name__=='__main__':main()
