"""All-head fitting traces and paired locality-held harm retrieval."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_harm_tail_crossfit_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_harm_tail_crossfit_v1'


def main():
    fits=json.loads((PUBLIC/'training_metrics.json').read_text()); assert len(fits)==144
    contrasts=json.loads((PUBLIC/'aggregate_metrics.json').read_text())['contrasts']
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True)
        for i,pair in enumerate(('full','motion_only')):
            traces=[r['fit']['trace'] for r in fits if r['pair']==pair]
            steps=sorted(set.intersection(*[{v['step'] for v in t} for t in traces]))
            ax=axes[i]; color='#1d718f' if i==0 else '#b63e55'
            values=np.array([[{v['step']:v['moment_mse'] for v in t}[s] for s in steps] for t in traces])
            for v in values: ax.plot(steps,v,color=color,alpha=.12,linewidth=.6)
            ax.plot(steps,np.median(values,axis=0),color=color,linewidth=2,label='Median / 72 heads')
            ax.set_title(pair); ax.set_ylabel('Fixed-batch normalized four-moment MSE')
            ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
            ax.set_xlabel('Optimizer updates')
        axes[0].legend(frameon=False)
        fig.suptitle('Locality-excluded fitting, not a policy-improvement result')
        fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(PUBLIC/'training_loss.svg')
        fig.savefig(PRIVATE/'training_loss_preview.png',dpi=120); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,6),sharey=True)
        labels=[p+' / '+g for p in ('full','motion_only') for g in sorted(contrasts[p])]
        for ax,(subset,title) in zip(axes,(('all','All evaluable rows'),('envelope_positive','Positive forecast disagreement'))):
            index=0
            for pair in ('full','motion_only'):
                color='#1d718f' if pair=='full' else '#b63e55'
                for group in sorted(contrasts[pair]):
                    v=contrasts[pair][group][subset+'__moment_vs_envelope']
                    if v.get('status')!='not_estimable':
                        ax.plot(v['CI_pp'],[index,index],color=color)
                        ax.scatter(v['point_pp'],index,color=color,s=24,zorder=3)
                    else: ax.text(0,index,'not estimable',fontsize=8)
                    index+=1
            ax.axvline(0,color='#777777',linewidth=.8); ax.set_title(title)
            ax.set_xlabel('Top 10% harm capture: moment minus envelope (pp)')
            ax.grid(axis='x',alpha=.2); ax.spines[['top','right']].set_visible(False)
        axes[0].set_yticks(range(len(labels)),labels); axes[0].invert_yaxis()
        fig.suptitle('Three-seed locality means / 3,000 resamples of four held localities')
        fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(PUBLIC/'harm_ranking.svg')
        fig.savefig(PRIVATE/'harm_ranking_preview.png',dpi=120); plt.close(fig)


if __name__=='__main__': main()
