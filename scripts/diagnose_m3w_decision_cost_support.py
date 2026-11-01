"""Post-hoc SOURCE-only soft support bound; no optimizer or model-selection step."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_decision_cost_critic as run
from src.evaluation import m3w_soft_policy_support as api

REG=run.PUBLIC/'soft_support_registration.json'
FILES=(Path(__file__),ROOT/'src/evaluation/m3w_soft_policy_support.py',ROOT/'tests/test_m3w_soft_policy_support.py',run.PUBLIC/'soft_support_protocol.md')


def register():
    run.once(REG,dict(bindings={str(p.relative_to(ROOT)):run.sha(p) for p in FILES},
        main_registration_sha256=run.sha(run.REG),training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
        role='posthoc_SOURCE_training_support_diagnostic_not_primary_or_selection',
        own_policy_outcomes_read=False,main_definition_changed=False,new_training=False,independent_data_opened=False))


def diagnose():
    frozen=json.loads(REG.read_text())
    for rel,digest in frozen['bindings'].items():
        if run.sha(ROOT/rel)!=digest:raise ValueError('Diagnostic definition changed')
        run.source.frozen.BASE.engine.committed(ROOT/rel)
    if frozen['training_freeze_sha256']!=run.sha(run.PUBLIC/'training_freeze.json'):raise ValueError('Main fitted models changed')
    cfg,values,store,banks,models,_=run.admission();start,beat,_=run.runtime(cfg)
    old=run.aggregate_groups(run.source.frozen.source.parent.NAME);main=run.aggregate_groups(run.source.frozen.NAME)
    rows=[]
    for context,items in sorted(values[2]['contexts'].items()):
        packets,changed=run.previous.context_packets(context,items,values,store,banks,beat)
        for seed in cfg['seeds']:
            data,_,_=run.source.prepare(context,seed,packets,changed,store,models,old,main,beat);c=data['causal'];lay=run.api.training_layout(data)
            capacity=run.np.maximum(c['envelope'],c['prediction'][:,:,:2].sum(2))/c['scales'][:,None]
            best=api.best_case_risk(lay['risk'],capacity,c['allowed'],lay['temperature'])
            minimum=api.minimum_switch_probability(capacity,c['allowed'],lay['temperature'])
            total=run.np.zeros((len(lay['record_keys']),3));run.np.add.at(total,lay['record'],best)
            total[:,2]-=lay['easy_constant'];total/=lay['record_scales'][:,None]
            positive=total>1e-10
            rows.append(dict(context=context,seed=seed,rows=len(c['ids']),records=len(lay['record_keys']),
                temperature=lay['temperature'],minimum_switch_probability_quantiles=run.np.quantile(minimum,[0,.25,.5,.75,1]).tolist(),
                best_case_positive_constraint_counts=positive.sum(0).tolist(),
                records_with_any_positive_lower_bound=int(positive.any(1).sum()),
                maximum_best_case_record_risk=total.max(0).tolist(),
                by_record=[dict(source=str(k[0]),recording=str(k[1]),normalized_best_case_constraint=v.tolist(),
                    positive_lower_bound=p.tolist()) for k,v,p in zip(lay['record_keys'],total,positive)],
                original_training_input_hashes=data['input_hashes'],labels_used_only_for_offline_training_diagnosis=True))
            beat(state='soft_support_context_complete',contexts=len(rows),total=18,context=context,seed=seed)
            del data,c,lay,best,capacity,minimum
        del packets,changed
    summary=dict(context_seed_views=18,record_views=sum(r['records'] for r in rows),
        records_with_any_strictly_positive_relaxed_lower_bound=sum(r['records_with_any_positive_lower_bound'] for r in rows),
        positive_constraint_counts=run.np.sum([r['best_case_positive_constraint_counts'] for r in rows],0).tolist(),
        numerical_positive_reporting_floor=1e-10,original_risk_budget=.02,
        probability_box_is_relaxation_of_actual_cost_head=True,no_deployment_safety_guarantee=True,
        diagnostic_registration_sha256=run.sha(REG),main_outcomes_or_training_definition_changed=False)
    run.once(run.PUBLIC/'soft_support_diagnostic.json',dict(summary=summary,contexts=rows))
    run.write(run.PRIVATE/'soft_support_runtime.json',dict(seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    count=summary['records_with_any_strictly_positive_relaxed_lower_bound']
    lines=['# Soft Decision Support: Training Diagnostic','',
        'Post-hoc SOURCE-only analysis declared during fitting,before own policy readout. No new training or gate/selection change.','',
        f'All18 context/critic-seed views and{summary["record_views"]} original record views retained.',
        f'{count} record views have at least one strictly positive relaxed best-case expected-risk bound.',
        f'Constraint counts(all positive harm,easy positive harm,easy net degradation):{summary["positive_constraint_counts"]}.',
        'The bound relaxes the network to arbitrary per-action logits in its causal capacity box. It ignores shared parameters and moment coupling.',
        'A positive bound means even this larger soft-policy family cannot meet that training constraint at its frozen temperature.',
        'A nonpositive bound does not prove the actual network can attain it. This is not a hard-decoder impossibility result.',
        'The positive reporting floor1e-10 is numerical,not a relaxation of the original2%risk requirement.','',
        '|Context/seed|Record views|Any positive relaxed lower bound|Median minimum soft intervention probability|','|---|---:|---:|---:|']
    for r in rows:lines.append(f'|{r["context"]}/{r["seed"]}|{r["records"]}|{r["records_with_any_positive_lower_bound"]}|{r["minimum_switch_probability_quantiles"][2]:.6f}|')
    lines+=['','128-corner exhaustive synthetic checks agree with fractional-root bisection.',
        'This can diagnose a smooth-support/actual-fallback mismatch,not independent efficacy or a need to relax safety.',
        'All labels stay in SOURCE supervision/diagnosis. No future input,test goal or held confirmation labels.','']
    (run.PUBLIC/'soft_support_diagnostic.md').write_text('\n'.join(lines));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('register','run'));args=p.parse_args()
    register() if args.phase=='register' else diagnose()
