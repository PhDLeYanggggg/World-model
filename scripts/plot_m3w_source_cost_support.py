"""SOURCE cost-support figure from public scalar receipts only."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/publication_readiness_2026_09'
PUBLIC = BASE/'european_actual_floor_projection_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_actual_floor_projection_v1'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    support_path = BASE/'european_selection_bank_support_v1/verification.json'
    source_path = PUBLIC/'complete.json'
    support,source = (json.loads(p.read_text()) for p in (support_path,source_path))
    if not support['all_record_support_counts_exact'] or len(source['rows']) != 9:
        raise ValueError('Complete verified SOURCE support and all nine projections required')
    for row in source['rows']:
        old = json.loads((BASE/'european_calibration_source_fit_v1/cases'/(row['trial']+'_mean_absolute.json')).read_text())
        if row['prediction_sha256'] != old['full_prediction_sha256']:
            raise ValueError('Original SOURCE cost inference identity differs')
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','svg.hashsalt':'m3w_source_cost_support',
        'axes.spines.right':False,'axes.spines.top':False})
    fig,axes = plt.subplots(1,3,figsize=(12.3,4.9))
    for i,(scope,label) in enumerate((('all_legal','All legal'),('neural_legal','Neural'),('old_selected','Old selected'))):
        cells = [c['scopes'][scope] for c in support['cases']]
        n = sum(c['actions'] for c in cells)
        value = 100*sum(c['cost_sum_overrun_beyond_5ppm_actions'] for c in cells)/n
        axes[0].bar(i,value,color=('#657485','#ba675e','#3f8d82')[i],width=.62)
        axes[0].text(i,value+.25,f'{value:.2f}%',ha='center',fontsize=9)
        axes[0].text(i,-2.25,f'n={n:,}',ha='center',fontsize=7.5)
    axes[0].set_xticks(range(3),['All legal','Neural','Old selected'])
    axes[0].set_ylim(-2.8,17); axes[0].set_yticks([0,5,10,15])
    axes[0].set_ylabel('Predicted B+H above actual radius (%)')
    axes[0].set_title('Cost-support mismatch on SOURCE')
    originals = [r['scores']['native_normalized']['original']['total'] for r in source['rows']]
    projected = [r['scores']['native_normalized']['projected']['total'] for r in source['rows']]
    reduction = 100*(np.array(originals)-np.array(projected))/originals
    axes[1].bar(np.arange(9),reduction,color='#3f8d82',width=.72)
    axes[1].set_xticks(np.arange(9),[r['trial'].replace('single','F').replace('_seed','\nS') for r in source['rows']],fontsize=7)
    axes[1].set_ylim(0,1.1); axes[1].set_ylabel('Full fitting-objective reduction (%)')
    axes[1].set_title('Zero-update effect is small')
    keys = ('B','H','R','ER','EH'); offset = .18
    for sign,arm,color in ((-1,'original','#657485'),(1,'projected','#3f8d82')):
        values = []
        for key in keys:
            value = sum(sum(c['query_visits']*c['components'][key] for c in r['scores']['native_normalized'][arm]['by_site'].values())
                /r['scores']['native_normalized'][arm]['query_visits'] for r in source['rows'])/9
            values.append(value)
        axes[2].barh(np.arange(5)+sign*offset,values,height=.32,color=color,label=arm.capitalize())
    axes[2].set_yticks(range(5),keys); axes[2].invert_yaxis()
    axes[2].set_xlabel('Query-weighted RMS-scaled cost MSE')
    axes[2].set_title('Reference errors remain unchanged')
    axes[2].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.grid(color='#dddddd',linewidth=.5); ax.set_axisbelow(True)
    fig.suptitle('Causal cost support: a mechanism check, not a deployed-policy result',fontsize=12)
    fig.text(.5,.08,'SOURCE fitting only. Nine frozen producers; repeated action views are not independent. '
        '5ppm is descriptive, not a new risk threshold.',ha='center',fontsize=8)
    fig.text(.5,.035,'No new training or policy evaluation. All 444 SOURCE record views have outcome-informed '
        'safe support; causal learnability is unproved.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.07,right=.98,top=.80,bottom=.29,wspace=.5)
    out = PUBLIC/'figures'; out.mkdir(exist_ok=True)
    preview = PRIVATE/'figures'; preview.mkdir(exist_ok=True)
    svg,png = out/'SOURCE_cost_support.svg',preview/'SOURCE_cost_support.png'
    fig.savefig(svg,metadata={'Date':None}); fig.savefig(png,dpi=170); plt.close(fig)
    receipt = dict(result_source='fresh_plot_complete_public_scalar_SOURCE_evidence',
        support_verification_sha256=sha(support_path),projection_complete_sha256=sha(source_path),
        script_sha256=sha(__file__),svg_sha256=sha(svg),private_png_sha256=sha(png),
        old9_SOURCE_predictions_exact=True,private_arrays_read=False,
        policy_or_training_changed=False,independent_confirmation=False)
    (out/'figure_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(svg=str(svg),preview=str(png))))


if __name__ == '__main__': main()
