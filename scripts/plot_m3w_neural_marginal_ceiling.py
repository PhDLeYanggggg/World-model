"""Plot only verified aggregate oracle statistics,not learned performance."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_neural_marginal_ceiling_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_neural_marginal_ceiling_v1'


def main():
    path=PUBLIC/'summary.json';data=json.loads(path.read_text())
    check=json.loads((PUBLIC/'verification.json').read_text())
    if not check['exact_replay'] or check['summary_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Verified marginal-ceiling aggregates required')
    roles=('fit','inner_held');labels=('Fitting','Exposed inner-held');x=np.arange(2)
    fig,axes=plt.subplots(1,2,figsize=(12,5.3))
    fig.subplots_adjust(left=.07,right=.985,top=.79,bottom=.20,wspace=.25)
    for offset,key,label,color in ((-.18,'physical_oracle_gain_floor_percent','Physical oracle','#567b91'),
            (.18,'mixed_oracle_gain_floor_percent','Physical + neural oracle','#429580')):
        rows=[data[s]['all'][key] for s in roles];means=np.array([r['point'] for r in rows])
        err=np.array([[r['point']-r['CI95'][0],r['CI95'][1]-r['point']] for r in rows]).T
        axes[0].bar(x+offset,means,width=.32,color=color,label=label,yerr=err,capsize=4)
    rows=[data[s]['all']['neural_marginal_gain_floor_pp'] for s in roles]
    means=np.array([r['point'] for r in rows]);err=np.array([[r['point']-r['CI95'][0],r['CI95'][1]-r['point']] for r in rows]).T
    axes[1].bar(x,means,width=.45,color=['#bc8957','#6779a8'],yerr=err,capsize=5)
    for i,value in enumerate(means):axes[1].text(i+.09,value+.10,f'{value:.2f} pp',ha='left',fontsize=10)
    axes[0].set(title='Known-label oracle utility',ylabel='ADE gain over floor (%)',ylim=(0,49))
    axes[1].set(title='Paired extra saved-path opportunity',ylabel='Neural marginal gain (percentage points)',ylim=(0,4.5))
    axes[0].legend(frameon=False,fontsize=9,loc='upper left')
    for ax in axes:
        ax.set_xticks(x,labels);ax.spines[['top','right']].set_visible(False)
        ax.yaxis.grid(True,color='#dddddd',linewidth=.6);ax.set_axisbelow(True)
    fig.suptitle('Future-informed candidate opportunity, not deployed performance',fontsize=14,y=.96)
    fig.text(.02,.04,'SOURCE development; conditional 95% locality intervals. Oracle gaps do not bound differences between causal learned policies.',fontsize=8.5)
    (PUBLIC/'figures').mkdir(exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    fig.savefig(PUBLIC/'figures/oracle_opportunity.svg');fig.savefig(PRIVATE/'oracle_opportunity.png',dpi=150)
    plt.close(fig);print(json.dumps(dict(verified_aggregate_plot=True,raw_data_or_weights=False)))


if __name__=='__main__':main()
