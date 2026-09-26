"""Matched readout fitting and complete locality-level contrasts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_frozen_harm_readout_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_frozen_harm_readout_v1'


def main():
    fits=json.loads((PUBLIC/'training_metrics.json').read_text()); assert len(fits)==288
    cs=json.loads((PUBLIC/'aggregate_metrics.json').read_text())['contrasts']
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True)
        for ax,pair in zip(axes,('full','motion_only')):
            for arm,color in (('mean_features','#1d718f'),('fractional_features','#b63e55')):
                traces=[r['fit']['trace'] for r in fits if r['pair']==pair and r['arm']==arm]
                steps=sorted(set.intersection(*[{v['step'] for v in t} for t in traces]))
                values=np.array([[{v['step']:v['harm_mse'] for v in t}[s] for s in steps] for t in traces])
                ax.plot(steps,np.median(values,axis=0),color=color,label=arm,linewidth=2)
                ax.fill_between(steps,*np.quantile(values,[.25,.75],axis=0),color=color,alpha=.12)
            ax.set_title(pair); ax.set_xlabel('Readout updates'); ax.set_ylabel('Fixed-batch two-harm MSE')
            ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.2)
        axes[0].legend(frameon=False)
        fig.suptitle('Frozen representation readouts: median and IQR / 72 heads per arm and pair')
        fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(PUBLIC/'training_loss.svg')
        fig.savefig(PRIVATE/'training_loss_preview.png',dpi=120); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,6),sharey=True)
        labels=[p+' / '+g for p in ('full','motion_only') for g in sorted(cs['fractional_vs_matched'][p])]
        for ax,comparison in zip(axes,('fractional_vs_matched','fractional_vs_original')):
            i=0
            for pair in ('full','motion_only'):
                for group in sorted(cs[comparison][pair]):
                    v=cs[comparison][pair][group]['envelope_positive__harm_MSE_gain_percent']
                    color='#1d718f' if pair=='full' else '#b63e55'
                    if 'CI' in v:
                        ax.plot(v['CI'],[i,i],color=color); ax.scatter(v['point'],i,color=color,s=20)
                    else: ax.text(0,i,'not estimable',fontsize=8)
                    i+=1
            ax.axvline(0,color='#777777',linewidth=.8); ax.set_title(comparison)
            ax.set_xlabel('Conditional easy-harm MSE gain (%)'); ax.grid(axis='x',alpha=.2)
            ax.spines[['top','right']].set_visible(False)
        axes[0].set_yticks(range(len(labels)),labels); axes[0].invert_yaxis()
        fig.suptitle('Positive favors fractional-feature readout / 3,000 locality resamples')
        fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(PUBLIC/'paired_contrasts.svg')
        fig.savefig(PRIVATE/'paired_contrasts_preview.png',dpi=120); plt.close(fig)


if __name__=='__main__': main()
