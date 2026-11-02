"""Label-free action and solver accounting from the complete frozen grid."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_independent_risk_calibration_v1'


def checked(ref):
    path = ROOT/ref['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('Frozen allocation artifact changed')
    return path


def main():
    freeze = json.loads((PUBLIC/'policy.json').read_text())
    refs = freeze['model_record_receipts']; reg = json.loads((PUBLIC/'registration.json').read_text())
    if len(refs) != 918:
        raise ValueError('Complete nine-pipeline/102-record grid required')
    role = json.loads(checked(reg['roles']).read_text())
    members = {r['source_member'] for r in role['recordings'] if r['role'] == 'risk_calibration_reserved'}
    trials = {r['trial'] for r in reg['frozen_models']}; seen = set(); summary = {}
    for reference in refs:
        policy = json.loads(checked(reference).read_text())
        inference = json.loads(checked(policy['inference']).read_text())
        inputs = json.loads(checked(inference['input_receipt']).read_text())
        identity = (inputs['source_member'], inference['trial'])
        if identity in seen or identity[0] not in members or identity[1] not in trials:
            raise ValueError('Unique admitted recording/pipeline required')
        seen.add(identity)
        choices = np.load(checked(policy['choices']), mmap_mode='r', allow_pickle=False)
        if (choices.shape != (13, 4, inference['rows']) or choices.dtype != np.int8
                or (choices < -1).any() or (choices >= 7).any()):
            raise ValueError('Unchanged full-grid choice schema required')
        for j, multiplier in enumerate(policy['multipliers']):
            meta = policy['allocation'][j]
            bucket = summary.setdefault(multiplier, dict(model_record_views=0, queries_per_policy=0,
                maximum_target_agents=0, actions={a:np.zeros(8, np.int64) for a in policy['policy_names']},
                solver_reasons={a:Counter() for a in ('joint_matched', 'joint_free')},
                nonoptimal_queries=Counter(), changed_queries=Counter()))
            bucket['model_record_views'] += 1; bucket['queries_per_policy'] += meta['queries']
            bucket['maximum_target_agents'] = max(bucket['maximum_target_agents'], meta['maximum_agents'])
            for ai, arm in enumerate(policy['policy_names']):
                bucket['actions'][arm] += np.bincount(choices[j, ai]+1, minlength=8)
            for arm in ('joint_matched', 'joint_free'):
                reasons = meta['reason_counts'][arm]
                if sum(reasons.values()) != meta['queries']:
                    raise ValueError('Complete query accounting required')
                bucket['solver_reasons'][arm].update(reasons)
            bucket['nonoptimal_queries'].update(meta['nonoptimal_queries'])
            bucket['changed_queries'].update(meta['changed_queries'])
    if seen != {(m,t) for m in members for t in trials}:
        raise ValueError('No missing required recording or pipeline')
    rows = []
    names = ('floor_fallback', 'constant_position', 'causal_cv', 'damped090', 'damped097', 'ols4', 'ols8', 'frozen_neural')
    for multiplier, bucket in sorted(summary.items()):
        actions = {}
        for arm, histogram in bucket['actions'].items():
            total = int(histogram.sum()); selected = total-int(histogram[0])
            actions[arm] = dict(counts=dict(zip(names, map(int, histogram))), repeated_target_views=total,
                               selected=selected, switch_rate=selected/total if total else None)
        if actions['independent']['selected'] != actions['joint_matched']['selected']:
            raise ValueError('Even pooled matched intervention counts must remain equal')
        rows.append(dict(multiplier=multiplier, model_record_views=bucket['model_record_views'],
            queries_per_policy=bucket['queries_per_policy'], maximum_target_agents=bucket['maximum_target_agents'],
            actions=actions, solver_reasons={k:dict(v) for k,v in bucket['solver_reasons'].items()},
            nonoptimal_queries=dict(bucket['nonoptimal_queries']), changed_queries=dict(bucket['changed_queries'])))
    output = dict(result_source='fresh_run_label_free_accounting_of_cached_verified_frozen_choices_and_solver_metadata',
        rows=rows, full_model_record_grids=918, per_query_count_proof='existing_allocator_not_reexecuted_by_this_report',
        execution={k:freeze[k] for k in ('pid', 'seconds', 'peak_RSS_bytes')},
        future_labels_read_by_report=False, accuracy_or_risk_efficacy_claim=False,
        milp_reoptimized=False, physical_pair_interactions_optimized=False,
        all_visible_agents_control_claim=False, deployment_changed=False)
    (PUBLIC/'causal_allocation_report.json').write_text(json.dumps(output, indent=2, sort_keys=True)+'\n')
    lines = ['# Frozen Allocation And Solver Accounting', '',
        'Only frozen choice matrices and allocator metadata are read. Counts concern',
        'past-supported target agents, not all visible agents. No target labels, accuracy',
        'or observed safety enter this report. Repeated row views are not independent.',
        'This is not a new per-query proof or solver rerun. Nonoptimal or numerical',
        'postcheck cases retain the feasible independent parent; do not call every query',
        'a globally optimal joint solution.', '',
        '|Lambda|Policy|Selected / repeated target views|Switch %|Nonoptimal queries|',
        '|---:|---|---:|---:|---:|']
    for row in rows:
        for arm, counts in row['actions'].items():
            lines.append(f'|{row["multiplier"]}|{arm}|{counts["selected"]}/{counts["repeated_target_views"]}|'
                         f'{100*counts["switch_rate"]:.6f}|{row["nonoptimal_queries"].get(arm,"N/A")}|')
    lines += ['', 'N/A means no MILP for that arm, not a measured zero solver error.',
        'Exact fast paths, successful MILP solves and distinct parent-fallback reasons',
        'are retained separately in the JSON. Joint allocation pools expected error',
        'budgets; it does not optimize physical pairwise conflicts. Actual risk still',
        'requires the separately frozen calibration readout.', '']
    (PUBLIC/'causal_allocation_report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(full_grids=918, multipliers=len(rows), labels_read=False)))


if __name__ == '__main__':
    main()
