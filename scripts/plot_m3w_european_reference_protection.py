"""Matched loss and all-source contrast figures; no seed selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_reference_protection_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_reference_protection_v1'


def main():
    fits=json.loads((PUBLIC/'training_metrics.json').read_text())
    seeds=json.loads((PUBLIC/'seed_averaged_metrics.json').read_text()); assert len(fits)==36
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
        for i,pair in enumerate(('full','motion_only')):
            take=[r for r in fits if r['pair']==pair]
            for j,(component,title) in enumerate(((0,'All-reference cost'),(3,'Easy-positive harm'))):
                ax=axes[i,j]
                for arm,color in (('continued','#1d718f'),('protected','#b63e55')):
                    traces=[r['arms'][arm]['fit']['trace'] for r in take]
                    steps=sorted(set.intersection(*[{v['step'] for v in t} for t in traces]))
                    values=np.array([[{v['step']:v['component_mse'][component] for v in t}[s] for s in steps] for t in traces])
                    for v in values: ax.plot(steps,v,color=color,alpha=.15,linewidth=.7)
                    ax.plot(steps,np.median(values,axis=0),color=color,label=arm,linewidth=2)
                ax.set_title(pair+' / '+title); ax.set_ylabel('Fixed-batch normalized MSE'); ax.grid(alpha=.2)
                ax.spines[['top','right']].set_visible(False)
                if i==1: ax.set_xlabel('Additional optimizer updates')
        axes[0,0].legend(frameon=False)
        fig.suptitle('Continuation fitting diagnostics, not held-source performance')
        fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(PUBLIC/'training_loss.svg')
        fig.savefig(PRIVATE/'training_loss_preview.png',dpi=120); plt.close(fig)
        comparisons=[('protected_joint_vs_continued_joint__all','Protected vs continued'),
            ('continued_joint_vs_mean_joint__all','Extra training vs original'),
            ('protected_joint_vs_raw_neural__all','Protected vs old neural')]
        fig,axes=plt.subplots(1,3,figsize=(13,6),sharey=True)
        labels=[p+' / '+g for p in ('full','motion_only') for g in sorted(seeds[p])]
        for ax,(key,title) in zip(axes,comparisons):
            i=0
            for pair in ('full','motion_only'):
                color='#1d718f' if pair=='full' else '#b63e55'
                for group in sorted(seeds[pair]):
                    v=seeds[pair][group][key]
                    if v.get('status')!='not_estimable':
                        ax.plot(v['CI'],[i,i],color=color); ax.scatter(v['gain_percent'],i,color=color,s=22,zorder=3)
                    i+=1
            ax.axvline(0,color='#777777',linewidth=.8); ax.set_title(title,fontsize=10)
            ax.set_xlabel('All-ADE gain (%) / 95% CI'); ax.grid(axis='x',alpha=.2); ax.spines[['top','right']].set_visible(False)
        axes[0].set_yticks(range(len(labels)),labels); axes[0].invert_yaxis()
        fig.suptitle('Three-seed locality means / 3,000 bootstrap resamples / source development')
        fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(PUBLIC/'source_contrasts.svg')
        fig.savefig(PRIVATE/'source_contrasts_preview.png',dpi=120); plt.close(fig)


if __name__=='__main__': main()
