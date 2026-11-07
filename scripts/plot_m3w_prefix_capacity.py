"""Create the capacity figure from the verified aggregate, not forecast arrays."""
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/prefix_capacity'
OUT = ROOT/'data/stage_cvpr2027_experiments/evidence_manuscript_v10/utility_tradeoff_build'
LABELS = (
    ('free_capped', 'Free counts, query cap'),
    ('fixed', 'Fixed counts'),
    ('fixed_terminal', 'Fixed + final predicted budgets'),
    ('fixed_prefix', 'Fixed + every predicted prefix'),
)


def main():
    raw = (STUDY/'analysis.json').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if any(json.loads((STUDY/f'{phase}_execution.json').read_text())['artifact']['sha256'] != digest
           for phase in ('score','replay')):
        raise ValueError('Require matching scoring and aggregate replay receipts')
    data = json.loads(raw)
    status = [data['summary'][key]['status_counts'] for key,_ in LABELS]
    count = np.array([[r.get('optimal',0)+r.get('feasible_unproven',0),
                       r.get('infeasible',0), r.get('unresolved',0)] for r in status])
    if not (count.sum(1)==102).all() or not data['label_aware_only']:
        raise ValueError('Plot the complete hindsight population only')
    light = {k:data[k] for k in ('summary','transitions','solver_registration_counts',
                                  'label_aware_only','original_rules_unchanged',
                                  'independent_confirmation','deployment_changed')}
    light['analysis_sha256'] = digest
    (STUDY/'summary.json').write_text(json.dumps(light,indent=2,sort_keys=True)+'\n')
    fig, ax = plt.subplots(figsize=(8.6,3.25),layout='constrained')
    edges = np.column_stack((np.zeros(len(count)),count.cumsum(axis=1)))
    y = np.arange(len(count))
    for j,(color,label) in enumerate(zip(('#286376','#b56c44','#e0e3e6'),
                            ('Checked feasible', 'Proven infeasible', 'Unresolved'))):
        ax.barh(y,count[:,j],left=edges[:,j],height=.58,color=color,label=label)
        for i,k in enumerate(count[:,j]):
            if k:
                ax.text(edges[i,j]+k/2,i,str(k),ha='center',va='center',
                        color='white' if j<2 else '#17272e',fontsize=9,weight='bold')
    ax.set(yticks=y,yticklabels=[v for _,v in LABELS],xlim=(0,102),xticks=[0,25,50,75,102],
           xlabel='Recording views (34 recordings, 3 forecaster seeds)',
           title='Hindsight capacity under unchanged observed risk rules')
    ax.invert_yaxis()
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=3,frameon=False,fontsize=8)
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'prefix_capacity.{ext}',dpi=200)


if __name__ == '__main__':
    main()
