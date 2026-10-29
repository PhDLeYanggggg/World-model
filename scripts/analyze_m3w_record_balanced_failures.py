"""Post-fit record-weight and constraint diagnosis, without new data scoring."""
import argparse
from collections import Counter
import json
import math
import statistics

from scripts import report_m3w_record_balanced_policy as report
from scripts.analyze_m3w_cost_sensitive_failures import classify, dual_snapshot

run = report.run


def record_weights(state):
    keys, weights, factors = (state[k] for k in
        ('record_keys', 'utility_weights', 'constraint_factor'))
    if not len(keys) or len(keys) != len(weights) or len(keys) != len(factors):
        raise ValueError('Aligned nonempty recording weights required')
    if not math.isclose(float(weights.sum()), 1., abs_tol=1e-6):
        raise ValueError('Unchanged normalized utility weights required')
    result = {}
    for key, weight, factor in zip(keys, weights, factors):
        pair = json.loads(key)
        if (not isinstance(pair, list) or len(pair) != 2
                or any(not isinstance(s, str) for s in pair)):
            raise ValueError('Literal source and recording string IDs required')
        key = tuple(pair)
        w, f = float(weight), float(factor)
        if (key in result or not math.isfinite(w*f) or w <= 0 or f <= 0
                or not math.isclose(w*f, 1./len(keys), rel_tol=1e-6)):
            raise ValueError('Distinct recordings with uniform total constraint weight required')
        result[key] = dict(utility_weight=w, constraint_factor=f,
            direction='increased' if f > 1.+1e-6 else 'decreased' if f < 1.-1e-6 else 'unchanged')
    return result


def collect():
    _, finish, summary, _ = report.collect()
    learned, parents = run.models(), run.previous.models()
    weights, snapshots = {}, {'cost_sensitive': [], 'record_balanced': []}
    for key, ref in sorted(learned.items()):
        state = run.base.core.read_checkpoint(run.ROOT/ref['checkpoint']['path'])
        weights[key] = record_weights(state)
        snapshots['record_balanced'].append(dual_snapshot(state))
        parent = run.base.core.read_checkpoint(run.ROOT/parents[key]['checkpoint']['path'])
        snapshots['cost_sensitive'].append(dual_snapshot(parent))
    counts, sources, directions, seen = {}, {}, {}, set()
    for ref in finish['groups']:
        row = json.loads((run.ROOT/ref['path']).read_text())
        key = row['context'], row['seed']
        for rec, arms in row['result']['recordings'].items():
            group = row['source'], rec
            identity = (*key, *group)
            if identity in seen:
                raise ValueError('Repeated record/context/head accounting')
            seen.add(identity)
            direction = weights[key][group]['direction']
            for arm, policy in arms.items():
                c = classify(policy)
                counts.setdefault(arm, Counter()).update(c)
                sources.setdefault(row['source'], {}).setdefault(arm, Counter()).update(c)
                directions.setdefault(direction, {}).setdefault(arm, Counter()).update(c)
    expected = {(*key, *group) for key, groups in weights.items() for group in groups}
    if seen != expected:
        raise ValueError('Every trained recording must appear in the readout')
    for arm, c in counts.items():
        if (c['total'] != summary['safety'][arm]['required_record_views'] or
                c['supported'] != summary['safety'][arm]['budget_record_views']):
            raise ValueError('Complete recording totals must match frozen scores')
    duals = {}
    for arm, values in snapshots.items():
        duals[arm] = {}
        for channel in ('all', 'easy'):
            rows = [v[channel] for v in values]
            positive = [x for r in rows for x in r['positive_scaled_risk']]
            d = {k: sum(r[k] for r in rows) for k in
                 ('record_views', 'positive_risk_before_last_update', 'dual_at_cap',
                  'positive_risk_with_dual_at_cap')}
            d.update(max_dual=max(r['max_dual'] for r in rows),
                median_positive_scaled_risk=statistics.median(positive) if positive else None)
            duals[arm][channel] = d
    return dict(result_source='fresh_analysis_of_cached_verified_fitting_scores_and_final_checkpoints',
        complete_sha256=run.sha(run.PUBLIC/'complete.json'), analyzer_sha256=run.sha(__file__),
        full_record_counts=counts, by_source=sources, by_constraint_weight_direction=directions,
        dual_trace_snapshot=duals, record_accounting_views=len(seen),
        trace_timing='risk_before_final_optimizer_step_dual_after_step_not_final_action_risk',
        inference='Weight-direction slices are post-fit descriptions, not randomized subgroup effects.',
        inner_held_scored=False, independent_roles_read=False, deployment_changed=False)


def markdown(data):
    lines = ['# Record-Constraint Diagnosis', '',
        'Frozen TRAIN outcomes and checkpoint metadata only. No new model or evaluation cohort.',
        'Counts repeat 74 source-qualified recordings across contexts and three policy-head seeds.', '',
        '|Policy|Supported / required|Known all over|Known easy over|All upper over|Easy upper over|Undefined easy|Net easy over|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, c in data['full_record_counts'].items():
        lines.append(f"|{arm}|{c['supported']}/{c['total']}|"+'|'.join(str(c.get(k, 0)) for k in
            ('known_positive_risk_over', 'known_easy_positive_risk_over', 'all_selected_risk_upper_over',
             'easy_selected_risk_upper_over', 'easy_selected_risk_upper_undefined', 'easy_degradation_upper_over'))+'|')
    lines += ['', '## Constraint Weight Direction', '',
        'Only constraint penalties change. Utility weights and the total constraint weight remain fixed.',
        'These slices describe final outcomes; they do not isolate a causal effect within each recording.', '',
        '|Direction|Cost-sensitive supported|Record-balanced supported|Cost-sensitive known all over|Record-balanced known all over|',
        '|---|---:|---:|---:|---:|']
    for direction, arms in sorted(data['by_constraint_weight_direction'].items()):
        a, b = arms['cost_sensitive'], arms['record_balanced']
        lines.append(f"|{direction}|{a['supported']}/{a['total']}|{b['supported']}/{b['total']}|"
            f"{a.get('known_positive_risk_over', 0)}|{b.get('known_positive_risk_over', 0)}|")
    lines += ['', '## Dual Snapshot', '',
        'Excess is just before the final optimizer update, duals just after it. Final safety uses the full readout, not this trace.', '',
        '|Policy/channel|Positive excess|Dual at cap|Both|Maximum dual|Median positive scaled excess|',
        '|---|---:|---:|---:|---:|---:|']
    for arm, channels in data['dual_trace_snapshot'].items():
        for name, d in channels.items():
            lines.append(f"|{arm}/{name}|{d['positive_risk_before_last_update']}|{d['dual_at_cap']}|"
                f"{d['positive_risk_with_dual_at_cap']}|{d['max_dual']:.6g}|{d['median_positive_scaled_risk']}|")
    lines += ['', 'All original recordings and the 2% budget remain. Empty or unsupported ratios are not safety passes.',
        'This diagnosis supports the next training-only repair; it does not select a deployment threshold or prove generalization.', '']
    return '\n'.join(lines)


def main(check=False):
    data = collect()
    values = {'constraint_diagnosis.json': json.dumps(data, indent=2, allow_nan=False)+'\n',
              'constraint_diagnosis.md': markdown(data)}
    for name, text in values.items():
        path = run.PUBLIC/name
        if check:
            if path.read_text() != text:
                raise ValueError('Post-fit diagnosis replay differs')
        elif path.exists():
            raise ValueError('Use --check for immutable completed diagnosis')
        else:
            path.write_text(text)
    print(json.dumps(dict(check=check,scope='fitting_only',record_views=data['record_accounting_views'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    main(parser.parse_args().check)
