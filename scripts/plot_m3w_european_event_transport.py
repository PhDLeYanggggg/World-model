"""All registered primary contrasts, including negative and inconclusive ones."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import run_m3w_european_event_transport as run


def main():
    a=json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    plt.rcParams.update({'svg.hashsalt':'m3w-event-transport-v1','font.size':9,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(16,10))
    labels=[('old_oof','Previous inner-event OOF'),('original','Original outer estimator'),
            ('outer_context','Prior three-locality context'),('common_global','Common-event global bias'),
            ('common_next','Common-event cyclic next'),('common_prev','Common-event cyclic previous')]
    for ax,(key,title) in zip(axes.flat,labels):
        for pair,color,offset in [('full','#267b91',-.12),('motion_only','#ba4657',.12)]:
            values=a['contrasts']['common_oof_vs_'+key][pair]; roles=sorted(values)
            q=[values[r]['envelope_positive__harm_MSE_gain_percent'] for r in roles]
            yy=np.arange(len(roles))+offset
            ax.hlines(yy,[x['CI'][0] for x in q],[x['CI'][1] for x in q],color=color)
            ax.scatter([x['point'] for x in q],yy,color=color,label=pair,s=24,zorder=3)
        ax.set_yticks(np.arange(len(roles)),[r.replace('producer','P').replace('_controller',' / C') for r in roles])
        ax.axvline(0,color='#666666',ls='--',lw=1); ax.grid(axis='x',alpha=.18)
        ax.set_title('Common-event OOF context vs\n'+title)
        ax.set_xlabel('Easy-harm MSE improvement (%)\n95% locality bootstrap CI; three-seed means')
        ax.legend(frameon=False,fontsize=8); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Common-event residual transport: source-development only; no trajectory or deployment claim')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(run.PUBLIC/'event_transport.svg',metadata={'Date':None})
    fig.savefig(run.PRIVATE/'event_transport_preview.png',dpi=120); plt.close(fig)


if __name__=='__main__': main()
