"""Public aggregate-only interpretation of label-aware frozen-cost support."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_frozen_cost_support as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import verify


def build():
    p=run.PUBLIC;reg=json.loads(run.REG.read_text())
    for rel,digest in reg['bindings'].items():
        if run.sha(ROOT/rel)!=digest:raise ValueError('Frozen diagnostic definition changed')
    bundle=verify(ROOT,p);complete=json.loads((p/'complete.json').read_text());check=json.loads((p/'verification.json').read_text())
    if (reg['parent_summary_sha256']!=run.sha(run.parent.PUBLIC/'summary.json')
            or reg['parent_complete_sha256']!=run.sha(run.parent.PUBLIC/'complete.json')
            or reg['parent_registration_sha256']!=run.sha(run.parent.REG)
            or reg['parent_verification_sha256']!=run.sha(run.parent.PUBLIC/'verification.json')
            or complete['registration_sha256']!=run.sha(run.REG)
            or complete['summary_sha256']!=run.sha(p/'summary.json')
            or check['summary_sha256']!=run.sha(p/'summary.json') or check['complete_sha256']!=run.sha(p/'complete.json')
            or not check['exact72_group_numeric_recalculation'] or not check['all_original_control_scores_exact']):
        raise ValueError('Verified complete original diagnostic and parent required')
    own=aggregate_groups(run.NAME);parent=aggregate_groups(run.parent.NAME)
    original_refs={Path(r['path']).name:r['sha256'] for r in json.loads((run.parent.PUBLIC/'complete.json').read_text())['groups']}
    rows=[own[Path(r['path']).name] for r in complete['groups']];records=set()
    for row in rows:
        key=f'{row["context"]}_{row["source"]}_seed{row["seed"]}_fit.json';old=parent[key]
        if (row['registration_sha256']!=run.sha(run.REG) or row['parent_group_sha256']!=original_refs[key]
                or row['cost_prediction_sha256']!=old['cost_prediction_sha256'] or row['ids_sha256']!=old['ids_sha256']):
            raise ValueError('Original cohort and moment hashes required')
        for a in run.api.CONTROLS:
            if row['result']['controls'][a]!=old['result']['policies'][a]['policy']:raise ValueError('Original full scores changed')
            for rec,r in row['result']['recordings'].items():
                if r['controls'][a]['actual']!=old['result']['recordings'][rec][a]:raise ValueError('Original recording scores changed')
        records.update((row['source'],rec) for rec in row['result']['recordings'])
    summary=json.loads((p/'summary.json').read_text())
    rebuilt=run.api.summarize(rows,reg['contexts'],json.loads(run.parent.CONFIG.read_text()))
    rebuilt.update(unique_target_rows=149510,unique_recordings=len(records))
    if summary!=rebuilt or len(records)!=74:raise ValueError('Complete canonical diagnostic differs')
    reasons={a:[dict(context=r['context'],source=r['source'],seed=r['seed'],recording=rec,reason=v[a]['oracle']['infeasibility_reason'])
        for r in rows for rec,v in r['result']['recordings'].items() if not v[a]['oracle']['budget_feasible']] for a in run.api.SUPPORTS}
    return dict(summary=summary,complete=complete,verification=check,bundle=bundle,unsupported=reasons,
        parent_summary=json.loads((run.parent.PUBLIC/'summary.json').read_text()),pilot=json.loads((p/'pilot.json').read_text()))


def interval(v):
    return 'not_estimable' if v['CI95'] is None else f'{v["point"]:.6f} [{v["CI95"][0]:.6f},{v["CI95"][1]:.6f}]'


def render(v):
    s=v['summary'];p=run.PUBLIC
    lines=['# Frozen Cost Gates: Where Safe Benefit Is Lost','','## Material Passport','',
        'fresh_run:complete72-group label-aware support calculation and separate exact numerical replay.',
        'cached_verified:forecasts,cost heads,frozen actor actions,all original rows and complete control scores.',
        'not_run:new training,independent roles or deployment. Parent outcomes were already read before this diagnostic was registered.',
        'These are future-label-aware diagnostic oracles,not model performance and never inference features.','',
        '## Nested Candidate Support','',
        '|Candidate restriction|Known positive-benefit upper pp [conditional95%CI]|Safe free-count witness gain pp [conditional95%CI]|Supported record views|Positive record views|',
        '|---|---:|---:|---:|---:|']
    for a,r in s['supports'].items():
        lines.append(f'|{a}|{interval(r["positive_upper_pp"])}|{interval(r["safe_witness_gain_pp"])}|{r["supported_record_views"]}/{r["required_record_views"]}|{r["positive_record_views"]}|')
    lines+=['','The upper column ignores safety and is not an attainable model score. The witness column uses true labels to choose actions and easy anchors.',
        'Pointwise-safe witnesses satisfy every original predicted query budget,but use free counts rather than the trained actor counts.',
        'The pointwise restriction is not the full joint feasible set. Its failure would not prove joint infeasibility.','',
        '|Restriction|Lost known positive-benefit upper pp [conditional95%CI]|','|---|---:|']
    for name,r in s['upper_benefit_losses'].items():lines.append(f'|{name}|{interval(r)}|')
    lines+=['','## Frozen Actions Versus Label-Aware Keep/Drop','',
        '|Policy|Actual supported record views|Deletion-oracle supported views|Deletion-oracle positive views|Known selected row views|Predicted pointwise-safe|Predicted safe with observed harm-budget excess|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in s['controls'].items():
        lines.append('|'+name+'|'+'|'.join(str(r[k]) for k in ('actual_supported_record_views','label_aware_deletion_supported_record_views',
            'label_aware_deletion_positive_record_views','known_selected_row_views','predicted_pointwise_safe_row_views',
            'predicted_pointwise_safe_but_observed_row_risk_over2_percent'))+'|')
    lines+=['','A deletion oracle uses future labels and need not retain predicted query feasibility after removing balancing actions.',
        'The last column counts H-.02R>0 or H_easy-.02R_easy>0. A zero reference does not define a risk ratio;the excess inequality still has its original meaning.',
        'Row-level predicted/observed mismatch is a calibration diagnostic,not a replacement for whole-record safety.','',
        '## Unsupported Records','']
    for name,r in s['supports'].items():lines.append(f'- {name}: {r["infeasible_reasons"]}')
    lines+=['','All recordings remain,including those without estimable easy-reference support. Empty risk is not safe0.',
        'Full details are in `unsupported_record_views.json`.','',
        '## What This Can And Cannot Resolve','',
        'A positive pointwise-safe witness demonstrates that this conservative subset contains useful label-known actions under frozen predicted budgets.',
        'A gap from the actual policy locates missed decision opportunities;it does not prove causal features can identify those opportunities.',
        'Predicted-positive headroom loss quantifies useful actions rejected by that frozen support gate.',
        'Actual versus keep/drop support measures whether oracle rejection of current mistakes could fix recording risk;the trained model has no access to that oracle.',
        'This is not an independent-generalization test,not proof of optimal query-count allocation,and not a reason to relax2%or discard difficult records.','',
        '## Execution','',
        f'Complete diagnostic PID{v["complete"]["pid"]},timed phase{v["complete"]["seconds"]:.6f}s,peakRSS{v["complete"]["peak_RSS_bytes"]}B.',
        'Native arm64 CPU4/interop1/workers0. All72original numerical results and control scores replay exactly.',
        'No optimizer updates,threshold search,new forecast training or independent-role opening.','',
        '![Frozen cost support diagnosis](figures/frozen_support.svg)','',
        '## Scope','',
        '149,510unique targets,74recordings,12exposed localities;72views repeat contexts/critic seeds. Shared forecast producer43.',
        '3,000conditional paired locality bootstrap draws average repeated views before resampling;not independent confirmation or familywise significance.',
        'EuropeanSquares oblique detector-silver/local-image,obs8/pred12/rawstride12. Not SDDt50,meters,seconds,human gold,true3D or foundation.',
        'Wholly-unknown envelopes do not resolve missing interior labels. Stage5C/SMC off. No deployment or submission-ready claim.','']
    (p/'readout.md').write_text('\n'.join(lines));run.write(p/'unsupported_record_views.json',v['unsupported'])
    run.write(p/'evidence.json',dict(summary_sha256=run.sha(p/'summary.json'),complete_sha256=run.sha(p/'complete.json'),
        verification_sha256=run.sha(p/'verification.json'),bundle_sha256=v['bundle']['archive_sha256'],
        exact_original_control_scores=True,new_training=False,label_aware_diagnostic_only=True,
        independent_confirmation=False,deployment_changed=False,submission_ready=False))


def figure(v):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    s=v['summary'];fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');x=np.arange(3)
    for offset,key,color,label in ((-.18,'positive_upper_pp','#326a95','Positive-benefit upper'),(.18,'safe_witness_gain_pp','#b85042','Label-aware safe witness')):
        z=[s['supports'][a][key] for a in run.api.SUPPORTS];y=np.array([r['point'] for r in z])
        axes[0].bar(x+offset,y,width=.35,color=color,label=label)
        axes[0].errorbar(x+offset,y,yerr=np.array([[r['point']-r['CI95'][0],r['CI95'][1]-r['point']] for r in z]).T,
            fmt='none',color='#222222',capsize=3)
    axes[0].set_xticks(x,['Causal bank','Positive gate','Pointwise cost']);axes[0].set_ylabel('Known reference-normalized benefit (pp)')
    axes[0].legend(fontsize=9);axes[0].set_title('Future-label-aware diagnostics,not model scores',fontsize=11)
    counts=[s['supports'][a]['supported_record_views'] for a in run.api.SUPPORTS]
    axes[1].bar(x,counts,color=['#326a95','#9a6a16','#b85042'],width=.65)
    axes[1].axhline(s['supports'][run.api.SUPPORTS[0]]['required_record_views'],ls='--',color='#555555',label='All required views')
    axes[1].set_xticks(x,['Causal bank','Positive gate','Pointwise cost']);axes[1].set_ylabel('Oracle-supported recording views');axes[1].legend(fontsize=9)
    axes[1].set_title('Whole cohort retained,no deployment certificate',fontsize=11)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('SOURCE only: frozen actions and costs,partial labels',fontsize=12)
    folder=run.PUBLIC/'figures';folder.mkdir(exist_ok=True);svg=folder/'frozen_support.svg';fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'frozen_support.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    v=build();render(v);figure(v);print(json.dumps(dict(groups=v['summary']['groups'],supports=v['summary']['supports'],new_training=False)))
