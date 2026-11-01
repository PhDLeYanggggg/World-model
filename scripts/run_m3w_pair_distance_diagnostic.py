"""Post-frozen-action pair-distance accuracy;not a new primary or deployment gate."""
import fcntl
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import verify_m3w_pair_frozen_actions as replay
from scripts import run_m3w_pair_action_allocation as run
from src.evaluation import m3w_pair_distance_diagnostic as api

PUBLIC=run.PUBLIC/'pair_distance_diagnostic'; REG=PUBLIC/'registration.json'
FILES=(Path(__file__),ROOT/'scripts/verify_m3w_pair_frozen_actions.py',ROOT/'src/evaluation/m3w_pair_distance_diagnostic.py',
    ROOT/'tests/test_m3w_pair_distance_diagnostic.py',PUBLIC/'protocol.md')


def register():
    run.once(REG,dict(bindings={str(p.relative_to(ROOT)):run.sha(p) for p in FILES},
        allocation_registration_sha256=run.sha(run.REG),type='secondary_posthoc_frozen_action_label_diagnostic',
        own_full_policy_results_inspected=False,registered_after_main_experiment_started=True,
        no_primary_metric_or_gate_change=True,no_inference_change=True,independent_confirmation=False))


def evaluate():
    for rel,digest in json.loads(REG.read_text())['bindings'].items():
        if run.sha(ROOT/rel)!=digest: raise ValueError('Diagnostic definition changed')
    rows=[]
    def collect(key,expected,packet,origin,choices,target,valid):
        result=api.evaluate(packet,origin,choices,target_labels=target,valid_labels=valid.astype(bool))
        row=dict(group=key,context=expected['context'],source=expected['source'],seed=expected['seed'],result=result,
            registration_sha256=run.sha(REG),frozen_action_sha256=expected['action_sha256'])
        run.once(PUBLIC/'groups'/key,row); rows.append(row)
    replay.verify(diagnostic=collect)
    if len(rows)!=72: raise ValueError('Full original72-group roster required')
    total=api.empty(); per_source={}
    for row in rows:
        value=row['result']['total']; source=per_source.setdefault(row['source'],api.empty())
        for dest in (total,source):
            for k in ('requested_pair_steps','known_pair_steps','unknown_pair_steps','unknown_disagreement_mass'): dest[k]+=value[k]
            for a in dest['known_error_mass']: dest['known_error_mass'][a]+=value['known_error_mass'][a]
    result=dict(groups=72,total=api.finish(total),by_source={k:api.finish(v) for k,v in sorted(per_source.items())},
        repeated_views_not_independent_pairs=True,CI_not_computed_for_pooled_point=True,
        source_fitting_secondary_diagnostic=True,original_gates_changed=False,
        independent_confirmation=False,deployment_changed=False,submission_ready=False)
    run.once(PUBLIC/'summary.json',result)
    lines=['# Actual Pair-Distance Accuracy Diagnostic','',
        'Secondary post-hoc label diagnostic on frozen original actions;not a new primary endpoint or safety metric.',
        'All72 original groups are retained. Future positions/masks enter only this evaluation after actions are frozen.',
        'Complete current-query target pairs share the same twelve requested future raw-frame steps.','',
        'For each pair/step,error is absolute predicted-minus-observed pair distance divided by the geometric mean of frozen past scales.',
        'For missing pair labels,the difference between policies is bounded by their predicted-distance disagreement(reverse triangle inequality).',
        'Known contribution and completion bounds divide by all requested pair-steps,not only known labels.',
        'Zero-pair groups remain not_estimable. Pooled points weight pair-step views and are not independent-sample confidence intervals.','',
        '| Source | Requested pairs/steps | Known | Unknown | Known contribution | Completion lower | Completion upper |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for source,value in result['by_source'].items():
        gain=value['full_requested_mean_gain']; cell=lambda x:'not_estimable' if x is None else f'{x:+.9f}'
        lines.append(f'|{source}|{value["requested_pair_steps"]}|{value["known_pair_steps"]}|{value["unknown_pair_steps"]}|{cell(gain["known_contribution"])}|{cell(gain["completion_lower"])}|{cell(gain["completion_upper"])}|')
    lines+=['',json.dumps(result['total'],indent=2),'',
        'Positive differences favor pair-aware assignments. This is local image-distance accuracy,not body collision,metric/seconds or physical safety.',
        'The original all/easy ADE risk and main completion-lower utility gates remain unchanged. No threshold/model was selected using this diagnostic.','']
    (PUBLIC/'readout.md').write_text('\n'.join(lines)); print(json.dumps(dict(groups=72,diagnostic_complete=True,inference_changed=False)))


if __name__=='__main__':
    if sys.argv[1:]==['register']: register()
    elif not sys.argv[1:]:
        with (run.PRIVATE/'process.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); evaluate()
    else: raise ValueError('Use register or no arguments')
