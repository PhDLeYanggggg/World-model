"""Independently check stored choices/costs and publish every fixed comparison."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_development_evaluation import content_digest
from src.evaluation.m3w_frozen_interaction import CONTROLS


def display_roundoff_bound(arm,agent_count):
    # Legacy score means retain float32; solver coefficients are accumulated in float64.
    # Selected gains are nonnegative under both frozen policies. Bound sum/divide
    # roundoff separately, without relaxing the optimizer's primal-dual certificate.
    u=np.finfo(np.float32).eps/2
    if arm['mean_predicted_gain']<0:
        raise ValueError('Nonnegative selected-gain precondition required')
    steps=2*agent_count+8
    if steps*u>=1:raise ValueError('Unsupported roundoff bound size')
    gamma=steps*u/(1-steps*u)
    magnitude=1+abs(arm['full_objective'])+abs(arm['unary_objective'])+abs(arm['product_objective'])
    return gamma*abs(arm['mean_predicted_gain'])+64*np.finfo(np.float64).eps*magnitude


def independent_check(rows,queries,summary,policy):
    by_query = {}
    row_ids = set()
    largest_display_gap=0.
    for row in rows:
        key = (row['recording_id'],row['frame_id'],row['horizon_raw'])
        identity = (*key,row['agent_id'])
        if identity in row_ids:raise ValueError('Duplicate agent query')
        row_ids.add(identity)
        by_query.setdefault(key,[]).append(row)
        for name in CONTROLS:
            selected = row['arms'][name]
            fixed = row['arms']['uncontrolled' if selected['switch'] else 'floor']
            for metric in ('ade','fde'):
                if selected[metric] != fixed[metric]:
                    raise ValueError('Selected forecast error is not the fixed floor/candidate error')
    query_ids = [(q['recording_id'],q['frame_id'],q['horizon_raw']) for q in queries]
    if len(set(query_ids))!=len(query_ids) or set(query_ids)!=set(by_query):
        raise ValueError('Scene query membership duplicated or incomplete')
    if len(rows)!=summary['full']['agent_query_count'] or len(queries)!=summary['full']['unique_scene_queries']:
        raise ValueError('Summary query population changed')
    for query in queries:
        key = (query['recording_id'],query['frame_id'],query['horizon_raw'])
        block = by_query[key]
        if len(block) != query['agent_count']:raise ValueError('Agent membership changed')
        for control,name in zip(('independent','unary_geometry','joint'),CONTROLS):
            arm = query['arms'][control]
            count = sum(r['arms'][name]['switch'] for r in block)
            if query['matched'] and count != query['reference_count']:raise ValueError('Claimed count match false')
            np.testing.assert_allclose(arm['full_objective'],
                -arm['mean_predicted_gain']+policy['pair_weight']*arm['mean_pair_proxy'],rtol=0,atol=1e-12)
            display_gap=abs(arm['full_objective']-arm['unary_objective']-arm['product_objective'])
            largest_display_gap=max(largest_display_gap,display_gap)
            if display_gap>display_roundoff_bound(arm,query['agent_count']):
                raise ValueError('Objective display discrepancy exceeds float32 reduction bound')
            if arm['solver_optimal']:
                cert = arm['numerical'];tol=cert['objective_tolerance']*(1+abs(cert['recomputed_primal']))
                if not cert['numerical_certificate_pass'] or not -tol<=cert['absolute_gap']<=tol:
                    raise ValueError('Unchecked numerical success')
                if control in ('unary_geometry','joint'):
                    direct=arm['unary_objective']+(arm['product_objective'] if control=='joint' else 0.)
                    np.testing.assert_allclose(direct,cert['recomputed_primal'],rtol=0,atol=1e-12)
        if query['matched']:
            np.testing.assert_allclose(query['predicted_full_objective_advantage'],
                query['arms']['unary_geometry']['full_objective']-query['arms']['joint']['full_objective'],rtol=0,atol=1e-12)
    for metric in ('ade','fde'):
        eligible = [r for r in rows if r['baseline_'+metric] is not None]
        if len(eligible)!=summary['full']['label_coverage'][metric]:
            raise ValueError('Summary label coverage changed')
        for left,right in ((CONTROLS[2],CONTROLS[1]),(CONTROLS[2],CONTROLS[0]),(CONTROLS[1],CONTROLS[0])):
            paired = summary['contrasts']['all'][left+'_minus_'+right][metric]
            if eligible:
                expected=float(np.mean([np.mean([r['arms'][left][metric]-r['arms'][right][metric]
                    for r in eligible if r['physical_scene']==s]) for s in sorted({r['physical_scene'] for r in eligible})]))
                np.testing.assert_allclose(expected,paired['left_minus_right_error'],rtol=0,atol=1e-12)
    for name in CONTROLS:
        for subset in ('all','easy','hard'):
            threshold = summary['full']['slice_definition']
            eligible = [r for r in rows if r['baseline_ade'] is not None]
            if subset=='easy':eligible=[r for r in eligible if r['baseline_ade']<=threshold['easy_baseline_error_at_most']]
            if subset=='hard':eligible=[r for r in eligible if r['baseline_ade']>=threshold['hard_baseline_error_at_least']]
            m=summary['full']['arms'][name][subset]
            if len(eligible)!=m['count']:raise ValueError('Summary label cohort changed')
            sites=sorted({r['physical_scene'] for r in eligible})
            if eligible:
                expected=float(np.mean([np.mean([r['arms'][name]['ade'] for r in eligible if r['physical_scene']==s]) for s in sites]))
                np.testing.assert_allclose(expected,m['selected_error'],rtol=0,atol=1e-12)
            if len(sites)==1 and m.get('bootstrap',{}).get('status')!='not_run_insufficient_physical_scenes':
                raise ValueError('Single-site uncertainty overclaim')
    native_checks=0
    for recording,reported in summary['full']['dataset_local_metrics']['recordings'].items():
        for metric in ('ade','fde'):
            eligible=[r for r in rows if r['recording_id']==recording and r['baseline_'+metric] is not None]
            if not eligible:continue
            for name in CONTROLS:
                expected=float(np.mean([r['arms'][name][metric]*r['scale'] for r in eligible]))
                np.testing.assert_allclose(expected,reported['arms'][name][metric],rtol=0,atol=1e-12)
                native_checks+=1
    return dict(rows_checked=len(rows),scene_queries_checked=len(queries),
        selected_forecast_error_checks=len(rows)*len(CONTROLS)*2,
        aggregate_ade_checks=len(CONTROLS)*3,paired_ade_fde_checks=6,
        native_recording_checks=native_checks,unique_row_membership_checked=True,
        max_float32_display_decomposition_gap=largest_display_gap,
        solver_certificate_tolerance_changed=False,no_confirmatory_uncertainty=True)


def mechanism_activity(rows,queries):
    joint,unary=CONTROLS[2],CONTROLS[1]
    changed=[r for r in rows if r['arms'][joint]['switch']!=r['arms'][unary]['switch']]
    labeled=[r for r in changed if r['baseline_ade'] is not None]
    delta=np.asarray([r['arms'][joint]['ade']-r['arms'][unary]['ade'] for r in labeled])
    coefficient_gains=[]
    for q in queries:
        a,b=q['arms']['unary_geometry'],q['arms']['joint']
        coefficient_gains.append((a['unary_objective']+a['product_objective'])-
                                 (b['unary_objective']+b['product_objective']))
    return dict(status='post_hoc_descriptive_not_selection',
        identity_changed_scene_queries=sum(q['joint_minus_unary_switch_identities']>0 for q in queries),
        identity_changed_agent_queries=len(changed),labeled_changed_agents=len(labeled),
        changed_agents_better=int(np.sum(delta<0)),changed_agents_worse=int(np.sum(delta>0)),
        changed_agents_equal=int(np.sum(delta==0)),
        max_objective_product_bound=max((q['absolute_product_sum_bound'] for q in queries),default=0.),
        mean_objective_product_bound=float(np.mean([q['absolute_product_sum_bound'] for q in queries])),
        max_coefficient_objective_advantage=max(coefficient_gains,default=0.),
        coefficient_advantage_not_realized_gain=True,
        physical_safety_evidence=False)


def main():
    cfg=json.loads((ROOT/'configs/m3w_frozen_interaction_v1.json').read_text())
    if file_digest(ROOT/cfg['protocol'])!=cfg['protocol_file_sha256']:
        raise ValueError('Frozen parent protocol changed')
    protocol=json.loads((ROOT/cfg['protocol']).read_text())
    out=ROOT/cfg['report'];out.mkdir(parents=True,exist_ok=True)
    records,verifications,bindings,activity=[],[],{},[]
    detailed={}
    for family in cfg['families']:
        directory=ROOT/cfg['output']/family
        completion=json.loads((directory/'completion.json').read_text())
        identity=json.loads((directory/'identity.json').read_text())
        report=json.loads((directory/'report.json').read_text())
        if completion['run_sha256']!=content_digest(identity) or completion['report_sha256']!=file_digest(directory/'report.json'):
            raise ValueError('Incomplete or changed frozen family')
        if report['identity_sha256']!=completion['run_sha256']:
            raise ValueError('Report identity changed')
        bindings[str((directory/'report.json').relative_to(ROOT))]=completion['report_sha256']
        for p,h in identity['code_sha256'].items():
            if file_digest(ROOT/p)!=h:raise ValueError('Changed execution code')
        for p,h in identity['source_exports'].items():
            if file_digest(ROOT/p)!=h:raise ValueError('Changed original export')
        parts={}
        receipt_paths=set()
        for info in completion['receipts']:
            receipt=ROOT/info['path']
            if info['path'] in receipt_paths:raise ValueError('Duplicate batch receipt')
            receipt_paths.add(info['path'])
            if file_digest(receipt)!=info['sha256']:raise ValueError('Changed batch receipt')
            saved=json.loads(receipt.read_text());cache=receipt.with_name(receipt.name.replace('.receipt.json','.json'))
            if saved['run_sha256']!=completion['run_sha256'] or saved['cache_sha256']!=file_digest(cache):
                raise ValueError('Changed private choices')
            key=saved['key'];part=json.loads(cache.read_text())
            p=parts.setdefault(key[1],dict(rows=[],queries=[]))
            p['rows'].extend(part['rows']);p['queries'].extend(part['queries'])
        detailed[family]=report['results']
        if set(parts)!=set(report['results']):raise ValueError('Reported candidate set differs from receipts')
        for candidate,result in report['results'].items():
            part=parts[candidate]
            policy_name=candidate.rsplit('_',1)[-1]
            checks=independent_check(part['rows'],part['queries'],result,protocol['development_evaluation']['policies'][policy_name])
            verifications.append(dict(family=family,candidate=candidate,**checks))
            activity.append(dict(family=family,candidate=candidate,**mechanism_activity(part['rows'],part['queries'])))
            summary=result['full'];arms=summary['arms']
            for name in CONTROLS:
                m=arms[name]
                records.append(dict(family=family,candidate=candidate,control=name,
                    ade=m['all']['selected_error'],fde=m['secondary_fde']['selected_error'],
                    gain_percent=m['all']['improvement_pct'],easy_degradation_percent=100*m['easy']['degradation_fraction'],
                    hard_gain_percent=m['hard']['improvement_pct'],positive_harm=m['all']['mean_positive_harm'],
                    p95_error=m['all']['agent_window_p95_error_descriptive'],switch_rate=m['switch_rate_all_past_supported'],
                    proximity=m['mean_query_excess_proximity_proxy']))
        del parts
    csv_path=out/'all_controls.csv'
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    populations=[{k:r['full'][k] for k in ('physical_scene_count','recording_count','agent_query_count','unique_scene_queries','label_coverage')}
                 for results in detailed.values() for r in results.values()]
    if any(p!=populations[0] for p in populations):raise ValueError('Fixed comparison populations differ')
    aggregate=dict(result_source='fresh_run_independent_choice_and_aggregate_reduction',source_bindings=bindings,
        per_candidate_population=populations[0],
        verification=verifications,all_fixed_results=detailed,
        mechanism_activity=activity,
        private_rows_independent_observations=False,scene_ci='not_run_only_one_physical_development_site',
        primary_metric_changed=False,new_training=False,selection_performed=False,independent_confirmation=False,
        stage5c_executed=False,smc_enabled=False)
    (out/'analysis.json').write_text(json.dumps(aggregate,indent=2,allow_nan=False)+'\n')
    lines=['# Frozen Neural Forecasts: Coupling Comparison','',
        'Post-hoc development evidence on two already explored UCY recordings of one physical site.',
        'No new training, threshold selection, primary-metric change or deployment. All 24 fixed combinations retained.',
        f"Each combination: {populations[0]['unique_scene_queries']:,} scene queries, {populations[0]['agent_query_count']:,} agent queries; "
        f"{populations[0]['label_coverage']['ade']:,} ADE labels and {populations[0]['label_coverage']['fde']:,} endpoint labels.",
        'Agent queries can overlap in time. Repetition across 24 fits does not multiply the independent population.','',
        '## Same-Predictor Contrasts','',
        'Negative ADE/FDE difference favors full joint over unary geometry. Errors retain the original past normalization.','',
        '| Family | Candidate | Joint minus unary ADE | Joint minus unary FDE | Nonzero matched queries | Switch disagreements |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    delta_key=CONTROLS[2]+'_minus_'+CONTROLS[1]
    for family,results in detailed.items():
        for candidate,result in results.items():
            m=result['contrasts']['all'][delta_key];q=result['query_summary']
            lines.append(f"| {family} | {candidate} | {m['ade']['left_minus_right_error']:.12g} | {m['fde']['left_minus_right_error']:.12g} | {q['nonzero_matches']} | {q['joint_unary_switch_disagreements']} |")
    lines.extend(['','## Coupling Opportunity And Replay','',
        'Each row has 970 scene queries. Possible products need not change an optimum; changed identities need not improve forecast accuracy.',
        'These counts repeat the same observed population across fixed fits and are not independent samples.','',
        '| Family | Candidate | Possible product queries | Proxy-advantage queries | Unmatched queries | New/legacy risk-only disagreements | Legacy row changes |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |'])
    for family,results in detailed.items():
        for candidate,result in results.items():
            q=result['query_summary']
            lines.append(f"| {family} | {candidate} | {q['possible_product_queries']} | {q['proxy_advantage_queries']} | {q['unmatched_queries']} | {q['new_reference_legacy_switch_disagreements']} | {result['original_replay']['legacy_control_rows_different']} |")
    lines.extend(['','## Changed Decisions, Not New Independent Cases','',
        'Post-hoc descriptive attribution only; favorable rows are not used for selection or a new success criterion.','',
        '| Family | Candidate | Changed scene queries | Changed agent queries | ADE-labeled changed agents | Better | Worse | Equal |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for a in activity:
        lines.append(f"| {a['family']} | {a['candidate']} | {a['identity_changed_scene_queries']} | {a['identity_changed_agent_queries']} | {a['labeled_changed_agents']} | {a['changed_agents_better']} | {a['changed_agents_worse']} | {a['changed_agents_equal']} |")
    lines.extend(['','## All New Controls','',
        'I = risk-only independent; U = geometry-aware independent at exact count; J = full joint at exact count.',
        'The CSV and analysis JSON retain FDE, positive harm, tail errors, coverage, raw-recording errors and all legacy controls.','',
        '| Family | Candidate | Arm | ADE gain vs CV (%) | Easy degradation (%) | Hard gain (%) | Switch rate |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |'])
    for row in records:
        lines.append(f"| {row['family']} | {row['candidate']} | {dict(zip(CONTROLS,'IUJ'))[row['control']]} | {row['gain_percent']:.7g} | {row['easy_degradation_percent']:.7g} | {row['hard_gain_percent']:.7g} | {row['switch_rate']:.7g} |")
    lines.extend(['','## Evidence Limits','',
        'Exactly one physical site: scene-bootstrap confidence intervals are unavailable, not replaced by window bootstraps.',
        'Three seed values describe fit variability only. Repeated head/policy exports are not independent agents or scenes.',
        'Matched past-agent counts do not imply equal scored-label coverage, equal realized risk or physical safety.',
        'Original future validity and fixed forecast errors are verified before admitting each query. No future label influences decisions.',
        'The historical model bytes are executed from a verified private mirror; the newer checkout remains unchanged.',
        'No Stage5C, SMC, seconds, metric, foundation or submission-readiness claim.',''])
    (out/'complete_results.md').write_text('\n'.join(lines))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/cfg['output']/'matplotlib_cache'))
    os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/cfg['output']/'render_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(left,right)=plt.subplots(1,2,figsize=(12,10),sharey=True,gridspec_kw={'width_ratios':[1.25,1]})
    labels=[];y=0
    colors=dict(zip(CONTROLS,('#2171b5','#2b8c4f','#b6373e')))
    for family,results in detailed.items():
        for candidate,result in results.items():
            labels.append(family+' '+candidate.replace('seed','s').replace('_',' '))
            for offset,name in zip((-.18,0,.18),CONTROLS):
                gain=result['full']['arms'][name]['all']['improvement_pct']
                left.scatter(-gain,y+offset,color=colors[name],s=22)
            delta=result['contrasts']['all'][delta_key]['ade']['left_minus_right_error']
            right.scatter(delta,y,color='#69499a',s=28)
            y+=1
    left.set_yticks(range(y),labels,fontsize=8);left.invert_yaxis()
    for ax in (left,right):
        ax.axvline(0,color='#444444',linewidth=.7);ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    left.set_xlabel('ADE increase over CV (%)');left.set_title('All fixed controls; smaller is better',fontsize=11)
    right.set_xlabel('Joint minus unary ADE (past-normalized)')
    nonzero=sum(r['contrasts']['all'][delta_key]['ade']['left_minus_right_error']!=0
                for results in detailed.values() for r in results.values())
    right.set_title(f'Joint versus unary: {nonzero}/{len(verifications)} nonzero contrasts',fontsize=11)
    right.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
    from matplotlib.lines import Line2D
    left.legend([Line2D([0],[0],marker='o',linestyle='',color=colors[x]) for x in CONTROLS],
                ['Risk-only','Unary geometry','Full joint'],loc='best',fontsize=8)
    fig.suptitle('Frozen neural forecast controls: one explored UCY site',fontsize=13)
    fig.text(.5,.012,'Three training seeds; no independent-scene CI, model selection or new deployment.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.97));fig.savefig(out/'comparison.svg')
    fig.savefig(ROOT/cfg['output']/'comparison_preview.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(status='independent_reduction_complete',fixed_combinations=len(verifications),
        decision_rows=sum(x['rows_checked'] for x in verifications),
        selected_error_checks=sum(x['selected_forecast_error_checks'] for x in verifications),
        physical_development_sites=1),indent=2))


if __name__=='__main__':main()
