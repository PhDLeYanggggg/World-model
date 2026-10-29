"""All fixed retrieval controls, aggregate SVG only; preview remains private."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts import run_m3w_coupled_local_cost as run


def main():
    p=run.PUBLIC;data=json.loads((p/'summary.json').read_text());check=json.loads((p/'verification.json').read_text())
    if not check['exact_replay'] or check['summary_sha256']!=run.sha(p/'summary.json'):
        raise ValueError('Complete verified fitting summary required')
    modes=list(run.api.MODES);banks=list(run.parent.api.BANKS)
    plt.rcParams.update({'font.size':10,'svg.hashsalt':run.NAME,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
    for i,(bank,color,label) in enumerate(zip(banks,['#2864a0','#aa4249'],['Physical bank','Physical + neural bank'])):
        values=[data['modes'][m]['accuracy'][bank] for m in modes]
        if any(v['point'] is None or v['CI95'] is None for v in values):
            raise ValueError('Do not silently plot an undefined primary accuracy as zero')
        y=np.array([v['point'] for v in values]);ci=np.array([v['CI95'] for v in values]);x=np.arange(4)+(i-.5)*.2
        axes[0].errorbar(x,y,yerr=np.stack((y-ci[:,0],ci[:,1]-y)),fmt='o',color=color,capsize=4,label=label)
        support=[data['modes'][m]['safety'][bank]['supported_record_views'] for m in modes]
        axes[1].bar(x,support,width=.18,color=color,label=label)
    labels=['Local\nknown','Local\nbounds','Global\nbounds','Shuffled\nbounds']
    for ax in axes:
        ax.set_xticks(np.arange(4),labels);ax.grid(axis='y',color='#dddddd',linewidth=.6);ax.set_axisbelow(True)
    axes[0].axhline(0,color='#777777',linewidth=.8);axes[0].set_ylabel('Known ADE gain over floor (%)')
    axes[1].axhline(444,color='#777777',linestyle=':',linewidth=.9);axes[1].set_ylim(0,465)
    axes[1].set_ylabel('Required risk views supported (of 444)')
    handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.9),ncol=2,frameon=False)
    fig.suptitle('Coupled local-cost control: complete fitting results',fontsize=13,y=.97)
    fig.text(.5,.03,'Intervals: 3,000 locality resamples, fitting only. Repeated context/control-seed views are dependent.',ha='center',fontsize=8.6)
    fig.subplots_adjust(left=.09,right=.98,bottom=.19,top=.76,wspace=.30)
    folder=p/'figures';folder.mkdir(exist_ok=True);out=folder/'control_comparison.svg'
    fig.savefig(out,metadata={'Date':None});out.write_text('\n'.join(s.rstrip() for s in out.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'control_comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(public_aggregate_SVG=1,raw_data_media_exported=False)))


if __name__=='__main__':
    main()
