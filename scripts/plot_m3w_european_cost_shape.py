"""Fixed shape contrasts and guards, preserving all assignment intervals."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_cost_shape as run
from scripts.report_m3w_european_cost_shape import PRIMARY,GUARDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def panels(doc,comparisons,metric,title,name):
    fig,axes=plt.subplots(2,len(comparisons),figsize=(6*len(comparisons),9),squeeze=False)
    fig.subplots_adjust(left=.065,right=.985,bottom=.105,top=.875,hspace=.50,wspace=.3)
    for row,family in enumerate(('full','motion_only')):
        for col,(contrast,label) in enumerate(comparisons):
            ax=axes[row,col]; values=doc['contrasts'][contrast][family]; limits=[0.]
            for i,m in enumerate(values.values()):
                v=m[metric]
                if 'CI' not in v: ax.text(0,i,'not estimable'); continue
                point,(lo,hi)=v['point'],v['CI']; limits.extend((lo,point,hi))
                ax.errorbar(point,i,xerr=np.array([[point-lo],[hi-point]]),fmt='o',
                    color='#00796b' if lo>0 else '#b23b3b' if hi<0 else '#555555',capsize=4)
            ax.axvline(0,color='black',linewidth=.8)
            ax.set_yticks(range(len(values)),[k.replace('producer','P').replace('_controller',' / C') for k in values])
            ax.set_title(family+'\n'+label)
            if metric==PRIMARY:
                ax.set_xscale('symlog',linthresh=1.); lo,hi=min(limits),max(limits)
                ax.set_xlim(lo-.1*max(abs(lo),1),hi+.1*max(abs(hi),1))
                xlabel='Expected easy-harm MSE gain (%)\nsymlog;linear between -1 and +1'
            else:
                xlabel={GUARDS[0]:'Top10 harm capture gain (percentage points)',
                    GUARDS[1]:'Absolute log-coverage error reduction',GUARDS[2]:'All-envelope H_all MSE gain (%)'}[metric]
            ax.set_xlabel(xlabel); ax.grid(axis='x',alpha=.2)
    fig.suptitle(title+'\n3 seeds;3,000 four-locality resamples;exposed source development',fontsize=13)
    fig.savefig(run.PUBLIC/(name+'.svg'),metadata={'Date':None})
    fig.savefig(run.PRIVATE/(name+'.png'),dpi=100); plt.close(fig)


def main():
    doc=json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    plt.rcParams.update({'svg.hashsalt':'m3w-cost-shape-v1','font.size':10})
    panels(doc,[(f'shape_mass_cost_only_vs_{m}_cost_only','Cost-only shape mass vs '+m)
        for m in ('raw','scaled','mass','shape_L2')],PRIMARY,'Cost-shape primary comparisons;not trajectory gain','shape_controls')
    panels(doc,[('shape_true_vs_shape_cost','True auxiliary vs cost-only'),
        ('shape_true_vs_shape_shuffled','True auxiliary vs shuffled')],PRIMARY,'Matched auxiliary comparisons','shape_auxiliary')
    panels(doc,[(f'shape_L2_{a}_vs_scaled_{a}',a+':shape L2 vs origin L2') for a in ('cost_only','cap_aux','shuffled_aux')],
        PRIMARY,'Shape-only controls without mean constraints','shape_L2_controls')
    for i,metric in enumerate(GUARDS):
        panels(doc,[(f'shape_mass_cost_only_vs_{m}_cost_only','Cost-only shape mass vs '+m) for m in ('raw','scaled')],
            metric,['Tail-capture guard','Coverage-log guard','All-harm MSE guard'][i]+';positive favors shape',f'shape_guard_{i}')


if __name__=='__main__': main()
