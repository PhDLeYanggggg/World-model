"""Descriptive recording concentration and radial association; no selected views."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_severity_transport as run
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    run.registration(); refs=json.loads((run.PUBLIC/'completion_checks.json').read_text())['groups']
    rows=[json.loads((ROOT/r['path']).read_text()) for r in refs]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.hashsalt':'severity-transport-v1'})
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for pair,color,marker in (('full','#23718e','o'),('motion_only','#b54856','x')):
        rs=[f for r in rows if r['pair']==pair for f in r['records']]
        xs=[]; ys=[]; u=[]; v=[]
        for r in rs:
            e=r['error_accounting']['original']['positive_disagreement']
            if e['status']!='computed': continue
            x=e['groups']['recording']['positive_row_excess']['top1_share']
            y=r['gradients']['weighted_BCE']['recording_gradient_norm_mass']['top1_share']
            p=e['partitions']['radial_outside']
            if x is not None and y is not None: xs.append(x); ys.append(y)
            if p['positive_excess_share'] is not None: u.append(p['row_share']); v.append(p['positive_excess_share'])
        axes[0].scatter(xs,ys,color=color,marker=marker,s=22,alpha=.7,label=pair)
        axes[1].scatter(u,v,color=color,marker=marker,s=22,alpha=.7,label=pair)
    axes[0].set(xlabel='Largest recording / positive row-excess mass',ylabel='Largest recording / fitting gradient norm mass',
        title='Separate held-error and fitting concentration',xlim=(-.03,1.03),ylim=(-.03,1.03))
    axes[1].plot([0,1],[0,1],color='#666666',lw=1,linestyle='--')
    axes[1].set(xlabel='Held rows outside fitting radial cut',ylabel='Outside share of positive row-excess mass',
        title='Radial association, not conditional support',xlim=(-.03,1.03),ylim=(-.03,1.03))
    xs=np.arange(3); keys=('ordinary_BCE','weighted_BCE','easy_harm_cost')
    for pair,color,shift in (('full','#23718e',-.12),('motion_only','#b54856',.12)):
        rs=[f for r in rows if r['pair']==pair for f in r['records']]; centers=[]; lower=[]; upper=[]
        for k in keys:
            a=[r['gradients'][k]['recording_gradient_norm_mass']['top1_share'] for r in rs]
            a=[v for v in a if v is not None]; q=np.quantile(a,[.25,.5,.75])
            centers.append(q[1]); lower.append(q[1]-q[0]); upper.append(q[2]-q[1])
        axes[2].errorbar(xs+shift,centers,yerr=[lower,upper],fmt='o',color=color,label=pair,capsize=3)
    axes[2].set(xticks=xs,xticklabels=['Ordinary BCE','Weighted BCE','Easy-harm cost'],ylim=(0,1),
        title='Fixed-batch gradient norm concentration',ylabel='Largest recording share / median and IQR')
    for ax in axes:
        ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False)
    fig.suptitle('Frozen source development: 144 dependent views, zero optimizer updates')
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(run.PUBLIC/'transport_diagnostic.svg',metadata={'Date':None})
    fig.savefig(run.PRIVATE/'transport_preview.png',dpi=120); plt.close(fig)


if __name__=='__main__': main()
