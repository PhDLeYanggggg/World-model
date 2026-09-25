"""Scientific figures for fixed-batch loss and every predeclared source contrast."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_easy_harm_sampling_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_easy_harm_sampling_v1'


def main():
    fits = json.loads((PUBLIC/'training_metrics.json').read_text())
    seeds = json.loads((PUBLIC/'seed_averaged_metrics.json').read_text()); assert len(fits)==36
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'}):
        fig,axes = plt.subplots(1,2,figsize=(10,4.7),sharex=True)
        for ax,pair in zip(axes,('full','motion_only')):
            take = [r for r in fits if r['pair']==pair]; assert len(take)==18
            for field,label,color in (('control_fit','Uniform mean control','#1d718f'),
                                      ('fit','Importance-corrected exposure','#b63e55')):
                steps = sorted(set.intersection(*[{t['step'] for t in r[field]['trace']} for r in take]))
                values = np.array([[{t['step']:t['moment_mse'] for t in r[field]['trace']}[s]
                    for s in steps] for r in take])
                for v in values: ax.plot(steps,v,color=color,alpha=.13,linewidth=.8)
                ax.plot(steps,np.median(values,axis=0),color=color,label=label,linewidth=2)
            ax.set_title(pair); ax.set_xlabel('Optimizer updates'); ax.set_ylabel('Fixed-batch normalized MSE')
            ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
        axes[0].legend(frameon=False,fontsize=8)
        fig.suptitle('Training diagnostics only; unchanged expected objective')
        fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(PUBLIC/'training_loss.svg')
        fig.savefig(PRIVATE/'training_loss_preview.png',dpi=120); plt.close(fig)
        comparisons = [('corrected_joint_vs_mean_joint__all','Sampling vs uniform control'),
            ('corrected_joint_vs_raw_neural__all','Against old neural rule'),
            ('corrected_joint_vs_corrected_dual__all','Joint vs individual allocation')]
        fig,axes = plt.subplots(1,3,figsize=(13,6),sharey=True)
        labels = [p+' / '+g for p in ('full','motion_only') for g in sorted(seeds[p])]
        for ax,(key,title) in zip(axes,comparisons):
            i=0
            for pair in ('full','motion_only'):
                color = '#1d718f' if pair=='full' else '#b63e55'
                for group in sorted(seeds[pair]):
                    v=seeds[pair][group][key]
                    if v.get('status')!='not_estimable':
                        ax.plot(v['CI'],[i,i],color=color,linewidth=1.2)
                        ax.scatter(v['gain_percent'],i,color=color,s=22,zorder=3)
                    i+=1
            ax.axvline(0,color='#777777',linewidth=.8); ax.set_title(title,fontsize=10)
            ax.set_xlabel('All-ADE gain (%) / 95% CI'); ax.grid(axis='x',alpha=.2)
            ax.spines[['top','right']].set_visible(False)
        axes[0].set_yticks(range(len(labels)),labels); axes[0].invert_yaxis()
        fig.suptitle('Three-seed means / 3,000 locality resamples / source development')
        fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(PUBLIC/'source_contrasts.svg')
        fig.savefig(PRIVATE/'source_contrasts_preview.png',dpi=120); plt.close(fig)


if __name__ == '__main__': main()
