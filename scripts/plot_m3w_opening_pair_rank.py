"""Render the bound comparison of first-two-query shortlist recall."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT/'outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/opening_pair_rank'
OUT = ROOT/'data/stage_cvpr2027_experiments/evidence_manuscript_v10/utility_tradeoff_build'


def main():
    raw = (PUB/'analysis.json').read_bytes()
    a = json.loads(raw)
    d = json.loads((PUB/'diverse_shortlist.json').read_text())
    h = hashlib.sha256(raw).hexdigest()
    if d['analysis_sha256']!=h or json.loads((PUB/'replay_execution.json').read_text())['artifact']['sha256']!=h:
        raise ValueError('Require verified matching evidence')
    old,new = a['old_recall']['first_two'],a['recall']['first_two']
    values = ([old['joint_top1'],new['pointwise_top1'],new['pair_top1']],
              [old['joint_top8'],new['pointwise_top8'],new['pair_top8']])
    diverse = [[d['recall']['first_two'][m][k] for m in ('old','pointwise','pair')]
               for k in ('flat','row','family')]
    fig,axes = plt.subplots(1,2,figsize=(9,3.3),sharey=True,layout='constrained')
    labels = ('Previous joint','Pointwise','Pairwise')
    for ax,data,names,title in zip(axes,(values,diverse),
        (('Top 1','Top 8'),('Flat','Row diverse','Family diverse')),
        ('Training objective','Same eight-slot budget')):
        width = .66/len(data)
        for j,(v,n,c) in enumerate(zip(data,names,('#286376','#9b563d','#66704a'))):
            b = ax.bar(np.arange(3)+(j-(len(data)-1)/2)*width,v,width*.94,label=n,color=c)
            ax.bar_label(b,padding=2,fontsize=8)
        ax.axhline(new['bank'],color='#707070',linestyle='--',linewidth=.8)
        ax.set(xticks=np.arange(3),xticklabels=labels,ylim=(0,18),yticks=[0,5,10,15],title=title)
        ax.legend(loc='upper left',bbox_to_anchor=(.02,.79),fontsize=7,frameon=False)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Views with a useful action / 18')
    fig.suptitle('First two queries, exposed zero-direct views (dashed: full bank)')
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'opening_rank_recall.{ext}',dpi=180)


if __name__=='__main__':
    main()
