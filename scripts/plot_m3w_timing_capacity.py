"""Render the verified equal-total timing result for the working supplement."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/timing_capacity'
OUT = ROOT/'data/stage_cvpr2027_experiments/evidence_manuscript_v10/utility_tradeoff_build'


def main():
    raw = (STUDY/'analysis.json').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    for phase in ('score','replay'):
        if json.loads((STUDY/f'{phase}_execution.json').read_text())['artifact']['sha256'] != digest:
            raise ValueError('Require byte-identical score and replay evidence')
    data = json.loads(raw)
    if len(data['records'])!=102 or sum(r['total'] for r in data['records'])!=6952:
        raise ValueError('Plot the registered population and count')
    fig,ax = plt.subplots(figsize=(8.2,3.5),layout='constrained')
    names = ('Observed risk','Add final predicted budgets','Add every predicted prefix')
    labels = ['Fixed by query','Same recording total']
    for i,(old,new) in enumerate(zip(('fixed','fixed_terminal','fixed_prefix'),
                                    ('total','total_terminal','total_prefix'))):
        for j,(key,scope) in enumerate(((old,'prior'),(new,'modes'))):
            states = [r[scope][key]['status'] for r in data['records']]
            count = [sum(s in ('optimal','feasible_unproven') for s in states),
                     states.count('infeasible'), states.count('unresolved')]
            start = 0
            for h,(n,c,name) in enumerate(zip(count,('#286376','#b56c44','#e0e3e6'),
                    ('Checked feasible','Proven infeasible','Unresolved'))):
                y = 3*i+j
                ax.barh(y,n,left=start,height=.82,color=c,label=name if i==j==0 else None)
                if n:
                    ax.text(start+n/2,y,str(n),ha='center',va='center',fontsize=8,
                            color='white' if h<2 else '#17272e',weight='bold')
                start += n
    ticks = [3*i+j for i in range(3) for j in range(2)]
    ax.set(yticks=ticks,yticklabels=[names[i]+'\n'+labels[j] for i in range(3) for j in range(2)],
           xlim=(0,102),xticks=[0,25,50,75,102],xlabel='Recording views (34 recordings, 3 seeds)',
           title='Retiming at the same recording total: hindsight only')
    ax.tick_params(axis='y',length=0,labelsize=8)
    ax.spines[['top','right','left']].set_visible(False)
    ax.invert_yaxis()
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=3,frameon=False,fontsize=8)
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'timing_capacity.{ext}',dpi=200)
    light = {k:v for k,v in data.items() if k!='records'}
    light['analysis_sha256'] = digest
    (STUDY/'summary.json').write_text(json.dumps(light,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':
    main()
