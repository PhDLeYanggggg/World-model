"""Visualize frozen fitting-batch geometry separately from actual AdamW changes."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_task_gradients as run
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    _,identity=run.registration(); done=json.loads((run.PUBLIC/'completion_checks.json').read_text())
    assert done['identity']==identity; rows=[]
    for ref in done['groups']:
        assert run.artifact(ROOT/ref['path'])==ref
        g=json.loads((ROOT/ref['path']).read_text())
        rows.extend(dict(pair=g['pair'],**r['diagnostic']) for r in g['records'] if r['arm']=='membership_aux')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'m3w-task-gradients-v1'})
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,pair,color in zip(axs,['full','motion_only'],['#2166ac','#24856a']):
        use=[r for r in rows if r['pair']==pair]
        x=np.array([r['shared_BCE_relation']['cost']['cosine'] for r in use])
        y=np.array([100*r['auxiliary_minus_cost_step']['cost']/r['before']['cost'] for r in use])
        assert np.isfinite(x).all() and np.isfinite(y).all()
        ax.scatter(x,y,s=28,c=color,alpha=.8,edgecolors='white',linewidths=.5)
        ax.axhline(0,color='#666666',lw=.8); ax.axvline(0,color='#999999',lw=.8,ls='--')
        ax.set_title(pair.replace('_',' ').title()+' | 72 dependent fitting views')
        ax.set_xlabel('Shared cost / membership gradient cosine')
        ax.grid(alpha=.15); ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Virtual auxiliary minus cost-only loss\n(% of frozen pre-step cost; positive = worse)')
    fig.suptitle('Frozen fitting-batch diagnosis, not held-out model improvement',fontsize=13)
    fig.text(.5,.025,'Saved AdamW momentum; one disposable step per arm. No policy or checkpoint update.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95))
    fig.savefig(run.PUBLIC/'gradient_optimizer.svg',metadata={'Date':None})
    fig.savefig(run.PRIVATE/'gradient_optimizer.png',dpi=150); plt.close(fig)


if __name__=='__main__': main()
