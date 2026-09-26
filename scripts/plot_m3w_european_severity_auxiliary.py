"""Fixed loss and paired-contrast figures, retaining negative results."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_severity_auxiliary as run
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    run.registration()
    new=json.loads((run.PUBLIC/'training_metrics.json').read_text())
    old=json.loads((run.parent.PUBLIC/'training_metrics.json').read_text())
    fits=old+[dict(r,arm='severity_aux') for r in new]
    agg=json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.hashsalt':'severity-auxiliary-v1'})
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    def curve(ax,trace,key,color,label):
        steps=sorted(set.intersection(*[{v['step'] for v in t} for t in trace]))
        vs=np.array([[{v['step']:v[key] for v in t}[s] for s in steps] for t in trace])
        ax.plot(steps,np.median(vs,axis=0),color=color,label=label)
        ax.fill_between(steps,*np.quantile(vs,[.25,.75],axis=0),color=color,alpha=.1)
    for ax,pair in zip(axes[:2],('full','motion_only')):
        for arm,color in (('cost_only','#444444'),('membership_aux','#23718e'),('severity_aux','#b54856')):
            ts=[r['fit']['trace'] for r in fits if (r['pair'],r['arm'])==(pair,arm)]
            curve(ax,ts,'cost_loss',color,arm)
        ax.set_title(pair+' / normalized cost MSE'); ax.legend(frameon=False)
    for pair,color in (('full','#23718e'),('motion_only','#b54856')):
        curve(axes[2],[r['fit']['trace'] for r in new if r['pair']==pair],'severity_BCE',color,pair)
    axes[2].set_title('New weighted BCE only'); axes[2].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel('Updates'); ax.set_ylabel('Fixed fitting-batch loss'); ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Median and IQR across dependent fits; fitting loss is not held-out improvement')
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(run.PUBLIC/'training_loss.svg',metadata={'Date':None})
    fig.savefig(run.PRIVATE/'training_preview.png',dpi=120); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,6),sharey=True)
    for ax,comp,title in zip(axes,('severity_vs_original','severity_vs_control','severity_vs_ordinary'),
        ('Weighted vs original','Weighted vs cost-only','Weighted vs ordinary auxiliary')):
        labels=[]; i=0
        for pair in ('full','motion_only'):
            for role,ms in sorted(agg['contrasts'][comp][pair].items()):
                labels.append(pair+' / '+role); v=ms['envelope_positive__harm_MSE_gain_percent']
                color='#23718e' if pair=='full' else '#b54856'
                if 'CI' in v: ax.plot(v['CI'],[i,i],color=color); ax.scatter(v['point'],i,color=color,s=20)
                else: ax.text(0,i,'not estimable')
                i+=1
        ax.set_title(title); ax.axvline(0,color='#888888',lw=.8); ax.set_xlabel('Easy-harm MSE gain (%)')
        ax.grid(axis='x',alpha=.2); ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(range(len(labels)),labels); axes[0].invert_yaxis()
    fig.suptitle('Positive disagreement / 3,000 locality resamples / exploratory source development')
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(run.PUBLIC/'paired_contrasts.svg',metadata={'Date':None})
    fig.savefig(run.PRIVATE/'contrasts_preview.png',dpi=120); plt.close(fig)


if __name__=='__main__': main()
