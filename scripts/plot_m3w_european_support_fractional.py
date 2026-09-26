"""Fixed-batch loss and all-role paired held-locality effects."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_support_fractional_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_support_fractional_v1'


def main():
    fits=json.loads((PUBLIC/'training_metrics.json').read_text()); assert len(fits)==144
    controls=json.loads((ROOT/'outputs/publication_readiness_2026_09/european_harm_tail_crossfit_v1/training_metrics.json').read_text())
    cs=json.loads((PUBLIC/'aggregate_metrics.json').read_text())['contrasts']
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True)
        for i,pair in enumerate(('full','motion_only')):
            ax=axes[i]
            for name,rows,color in (('mean',controls,'#1d718f'),('fractional',fits,'#b63e55')):
                traces=[r['fit']['trace'] for r in rows if r['pair']==pair]
                steps=sorted(set.intersection(*[{v['step'] for v in t} for t in traces]))
                vals=np.array([[{v['step']:v['moment_mse'] for v in t}[s] for s in steps] for t in traces])
                ax.plot(steps,np.median(vals,axis=0),color=color,label=name,linewidth=2)
                ax.fill_between(steps,*np.quantile(vals,[.25,.75],axis=0),color=color,alpha=.12)
            ax.set_title(pair); ax.set_ylabel('Fixed-batch moment MSE'); ax.set_xlabel('Updates')
            ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
        axes[0].legend(frameon=False)
        fig.suptitle('Fitting diagnostics: median and interquartile range / 72 heads per arm')
        fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(PUBLIC/'training_loss.svg')
        fig.savefig(PRIVATE/'training_loss_preview.png',dpi=120); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,6),sharey=True)
        labels=[p+' / '+g for p in ('full','motion_only') for g in sorted(cs[p])]
        keys=[('harm_MSE_gain_percent','Conditional easy-harm MSE gain (%)'),
            ('top10_gain_pp','Conditional top10 harm capture (pp)'),
            ('coverage_log_error_reduction','Absolute log-coverage error reduction')]
        for ax,(key,title) in zip(axes,keys):
            i=0
            for pair in ('full','motion_only'):
                for group in sorted(cs[pair]):
                    v=cs[pair][group]['envelope_positive__'+key]; color='#1d718f' if pair=='full' else '#b63e55'
                    if 'CI' in v:
                        ax.plot(v['CI'],[i,i],color=color); ax.scatter(v['point'],i,color=color,s=20,zorder=3)
                    else: ax.text(0,i,'not estimable',fontsize=8)
                    i+=1
            ax.axvline(0,color='#777777',linewidth=.8); ax.set_xlabel(title,fontsize=9)
            ax.grid(axis='x',alpha=.2); ax.spines[['top','right']].set_visible(False)
        axes[0].set_yticks(range(len(labels)),labels); axes[0].invert_yaxis()
        fig.suptitle('Fractional minus matched mean: positive favors new objective / 3,000 locality resamples')
        fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(PUBLIC/'paired_contrasts.svg')
        fig.savefig(PRIVATE/'paired_contrasts_preview.png',dpi=120); plt.close(fig)


if __name__=='__main__': main()
