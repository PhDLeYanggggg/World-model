"""Display the complete membership training and source-role contrasts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_easy_membership_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_easy_membership_v1'


def main():
    fits=json.loads((PUBLIC/'training_metrics.json').read_text()); agg=json.loads((PUBLIC/'aggregate_metrics.json').read_text())
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','svg.hashsalt':'m3w-easy-membership-v1'}):
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,pair in zip(axes,('full','motion_only')):
            for arm,color in (('linear','#1d718f'),('mlp','#b63e55')):
                ts=[r['fit']['trace'] for r in fits if (r['pair'],r['arm'])==(pair,arm)]
                steps=sorted(set.intersection(*[{v['step'] for v in t} for t in ts]))
                vs=np.array([[{v['step']:v['BCE'] for v in t}[s] for s in steps] for t in ts])
                ax.plot(steps,np.median(vs,axis=0),color=color,label=arm)
                ax.fill_between(steps,*np.quantile(vs,[.25,.75],axis=0),color=color,alpha=.12)
            ax.set_title(pair); ax.set_xlabel('Updates'); ax.set_ylabel('Fixed-batch membership BCE'); ax.grid(alpha=.2)
            ax.spines[['top','right']].set_visible(False)
        axes[0].legend(frameon=False); fig.suptitle('Median and IQR / 72 classifiers per arm and pair')
        fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(PUBLIC/'training_loss.svg',metadata={'Date':None}); fig.savefig(PRIVATE/'training_preview.png',dpi=120); plt.close(fig)
        for section,sep,stem,label in (
            ('contrasts','__envelope_positive__','paired_contrasts','training prevalence'),
            ('conditional_constant_sensitivity','__','conditional_sensitivity','training conditional prevalence'),
        ):
            fig,axes=plt.subplots(1,2,figsize=(12,6),sharey=True); cs=agg[section]
            labels=[p+' / '+g for p in ('full','motion_only') for g in sorted(cs[p])]
            for ax,arm in zip(axes,('linear','mlp')):
                i=0
                for pair in ('full','motion_only'):
                    for group in sorted(cs[pair]):
                        v=cs[pair][group][arm+sep+'Brier_skill_percent']; color='#1d718f' if pair=='full' else '#b63e55'
                        if 'CI' in v:
                            ax.plot(v['CI'],[i,i],color=color); ax.scatter(v['point'],i,color=color,s=20)
                        else: ax.text(0,i,'not estimable')
                        i+=1
                ax.set_title(arm); ax.axvline(0,color='#888',linewidth=.8); ax.set_xlabel('Brier skill vs '+label+' (%)')
                ax.grid(axis='x',alpha=.2); ax.spines[['top','right']].set_visible(False)
            axes[0].set_yticks(range(len(labels)),labels); axes[0].invert_yaxis()
            fig.suptitle('Conditional membership probability error / 3,000 locality resamples')
            fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(PUBLIC/(stem+'.svg'),metadata={'Date':None})
            preview='contrasts' if section=='contrasts' else 'sensitivity'
            fig.savefig(PRIVATE/(preview+'_preview.png'),dpi=120); plt.close(fig)


if __name__=='__main__': main()
