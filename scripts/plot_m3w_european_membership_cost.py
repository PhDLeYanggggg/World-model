"""Registered objective traces and all expected-cost contrasts, including failures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_membership_cost_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_membership_cost_v1'


def main():
    fits=json.loads((PUBLIC/'training_metrics.json').read_text()); agg=json.loads((PUBLIC/'aggregate_metrics.json').read_text())
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','svg.hashsalt':'m3w-membership-cost-v1'}):
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,arm in zip(axes,('direct','conditional')):
            for pair,color in (('full','#1d718f'),('motion_only','#b63e55')):
                ts=[r['fit']['trace'] for r in fits if (r['pair'],r['arm'])==(pair,arm)]
                steps=sorted(set.intersection(*[{v['step'] for v in t} for t in ts]))
                vs=np.array([[{v['step']:v['objective'] for v in t}[s] for s in steps] for t in ts])
                ax.plot(steps,np.median(vs,axis=0),color=color,label=pair)
                ax.fill_between(steps,*np.quantile(vs,[.25,.75],axis=0),color=color,alpha=.12)
            ax.set_title(arm); ax.set_xlabel('Updates'); ax.set_ylabel('Arm-specific fixed-batch objective')
            ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
        axes[0].legend(frameon=False)
        fig.suptitle('Median and IQR / objectives and scales differ across arms')
        fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(PUBLIC/'training_loss.svg',metadata={'Date':None})
        fig.savefig(PRIVATE/'training_preview.png',dpi=120); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,6),sharey=True)
        for ax,comp in zip(axes,('conditional_vs_direct','conditional_vs_original','conditional_vs_constant')):
            cs=agg['contrasts'][comp]; labels=[]; i=0
            for pair in ('full','motion_only'):
                for role,ms in sorted(cs[pair].items()):
                    labels.append(pair+' / '+role); v=ms['envelope_positive__harm_MSE_gain_percent']
                    color='#1d718f' if pair=='full' else '#b63e55'
                    if 'CI' in v:
                        ax.plot(v['CI'],[i,i],color=color); ax.scatter(v['point'],i,color=color,s=20)
                    else: ax.text(0,i,'not estimable')
                    i+=1
            ax.set_title(comp.replace('conditional_vs_','vs ')); ax.axvline(0,color='#888',linewidth=.8)
            ax.set_xlabel('Easy-harm MSE gain (%)'); ax.grid(axis='x',alpha=.2); ax.spines[['top','right']].set_visible(False)
        axes[0].set_yticks(range(len(labels)),labels); axes[0].invert_yaxis()
        fig.suptitle('Conditional factorization / positive disagreement / 3,000 locality resamples')
        fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(PUBLIC/'paired_contrasts.svg',metadata={'Date':None})
        fig.savefig(PRIVATE/'contrasts_preview.png',dpi=120); plt.close(fig)


if __name__=='__main__': main()
