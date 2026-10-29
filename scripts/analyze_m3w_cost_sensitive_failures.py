"""Post-fit diagnosis from frozen scores and checkpoints; no new data readout."""
import argparse
from collections import Counter
import json
import math
import statistics

from scripts import report_m3w_cost_sensitive_policy as report

run = report.run
RATIOS = ('known_positive_risk', 'known_easy_positive_risk',
          'all_selected_risk_upper', 'easy_selected_risk_upper',
          'easy_degradation_upper')


def classify(policy):
    report.check_policy(policy)
    out = dict(total=1, supported=int(run.readout.budget_status(policy)['supported']),
               empty=int(policy['selected_count'] == 0))
    for key in RATIOS:
        value = policy[key]
        if value is not None and not math.isfinite(value):
            raise ValueError('Finite ratio or explicit unknown required')
        state = 'undefined' if value is None else 'over' if value > .02+1e-12 else 'within'
        out[key+'_'+state] = 1
    return out


def dual_snapshot(state):
    trace = state['trace'][-1]
    if trace['step'] != state['step'] or state['step'] != 2000:
        raise ValueError('Final registered update required')
    dual = state['dual'].double()
    risk = run.api.torch.tensor(trace['exact_record_risk'], dtype=run.api.torch.float64)
    scales = state['record_scales'].double()
    if dual.shape != risk.shape or risk.shape != scales.shape or dual.shape[1] != 2:
        raise ValueError('Aligned all/easy record statistics required')
    cap = state['settings']['dual_maximum']
    if not (run.api.torch.isfinite(dual).all() and run.api.torch.isfinite(risk).all()
            and run.api.torch.isfinite(scales).all() and (scales > 0).all()
            and (dual >= 0).all() and (dual <= cap).all()):
        raise ValueError('Invalid dual/risk/scaling snapshot')
    out = {}
    for column, name in enumerate(('all', 'easy')):
        positive = risk[:, column] > 1e-12
        saturated = dual[:, column] >= cap-1e-6
        out[name] = dict(record_views=len(dual), positive_risk_before_last_update=int(positive.sum()),
            dual_at_cap=int(saturated.sum()), positive_risk_with_dual_at_cap=int((positive & saturated).sum()),
            max_dual=float(dual[:, column].max()),
            positive_scaled_risk=(risk[positive, column]/scales[positive, column]).tolist())
    return out


def collect():
    _, finish, summary, _ = report.collect()
    learned = run.models()
    parents = run.diag.whole.models()
    counts, sources, transitions = {}, {}, Counter()
    for ref in finish['groups']:
        row = json.loads((run.ROOT/ref['path']).read_text())
        for arms in row['result']['recordings'].values():
            for arm, policy in arms.items():
                counts.setdefault(arm, Counter()).update(classify(policy))
                sources.setdefault(row['source'], {}).setdefault(arm, Counter()).update(classify(policy))
            before = run.readout.budget_status(arms['exact_gradient'])['supported']
            after = run.readout.budget_status(arms['cost_sensitive'])['supported']
            transitions[f'{int(before)}_to_{int(after)}'] += 1
    for arm, count in counts.items():
        if (count['total'] != summary['safety'][arm]['required_record_views'] or
                count['supported'] != summary['safety'][arm]['budget_record_views']):
            raise ValueError('Record accounting mismatch')
    snapshots = {}
    for arm in ('exact_gradient', 'cost_sensitive'):
        values = []
        for (context, seed), ref in sorted(learned.items()):
            if arm == 'exact_gradient':
                ref = parents[context, seed, arm]
            values.append(dual_snapshot(run.base.core.read_checkpoint(run.ROOT/ref['checkpoint']['path'])))
        snapshots[arm] = {}
        for channel in ('all', 'easy'):
            rows = [v[channel] for v in values]
            positive = [x for r in rows for x in r['positive_scaled_risk']]
            snapshots[arm][channel] = {
                key:sum(r[key] for r in rows) for key in
                ('record_views','positive_risk_before_last_update','dual_at_cap','positive_risk_with_dual_at_cap')}
            snapshots[arm][channel].update(max_dual=max(r['max_dual'] for r in rows),
                median_positive_scaled_risk=statistics.median(positive) if positive else None)
    return dict(result_source='fresh_analysis_of_cached_verified_fitting_scores_and_final_checkpoints',
        complete_sha256=run.sha(run.PUBLIC/'complete.json'), analyzer_sha256=run.sha(__file__),
        full_record_counts=counts, by_source=sources, budget_transitions=transitions,
        dual_trace_snapshot=snapshots,
        trace_timing='risk_before_last_optimizer_update_dual_after_last_update_not_final_action_risk',
        independent_roles_read=False, inner_held_scored=False, deployment_changed=False,
        distinction='Aggregate source budgets do not replace the required per-record budgets.')


def markdown(data):
    out = ['# Post-Fit Constraint Diagnosis', '',
        'This diagnosis uses frozen fitting scores and checkpoint metadata only. No new evaluation cohort is opened.',
        'Counts repeat recording/forecast contexts across three head seeds, not independent recordings.', '',
        '|Policy|Supported / required|Known all over|Known easy over|All upper over|Easy upper over|Undefined easy|Easy degradation over|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, c in data['full_record_counts'].items():
        out.append(f"|{arm}|{c['supported']}/{c['total']}|"+'|'.join(str(c.get(k,0)) for k in
            ('known_positive_risk_over','known_easy_positive_risk_over','all_selected_risk_upper_over',
             'easy_selected_risk_upper_over','easy_selected_risk_upper_undefined','easy_degradation_upper_over'))+'|')
    out += ['', '## Source Concentration', '',
        '|Source|Exact-gradient known all over / records|Cost-sensitive known all over / records|Cost-sensitive supported|',
        '|---|---:|---:|---:|']
    for source, arms in sorted(data['by_source'].items()):
        a,b=arms['exact_gradient'],arms['cost_sensitive']
        out.append(f"|{source}|{a.get('known_positive_risk_over',0)}/{a['total']}|{b.get('known_positive_risk_over',0)}/{b['total']}|{b['supported']}|")
    out += ['', '## Dual Snapshot', '',
        'Risk is measured just before the final optimizer update; duals are from just after it.',
        'This is a training-mechanism diagnostic, not the final deployed-action safety score above.', '',
        '|Policy/channel|Positive excess|Dual at cap|Both|Maximum dual|Median positive scaled excess|',
        '|---|---:|---:|---:|---:|---:|']
    for arm, channels in data['dual_trace_snapshot'].items():
        for name,d in channels.items():
            out.append(f"|{arm}/{name}|{d['positive_risk_before_last_update']}|{d['dual_at_cap']}|{d['positive_risk_with_dual_at_cap']}|{d['max_dual']:.6g}|{d['median_positive_scaled_risk']}|")
    out += ['', 'The budget remains 2%. Empty ratios remain not_estimable. Utility gains cannot waive the full-roster safety requirement.',
        'This post-fit analysis motivates a next experiment; it does not choose a new model, threshold, or deployment policy.', '']
    return '\n'.join(out)


def main(check=False):
    data=collect()
    values={'constraint_diagnosis.json':json.dumps(data,indent=2,allow_nan=False)+'\n',
            'constraint_diagnosis.md':markdown(data)}
    for name,text in values.items():
        path=run.PUBLIC/name
        if check:
            if path.read_text()!=text:raise ValueError('Post-fit diagnosis replay differs')
        elif path.exists():raise ValueError('Use --check for immutable diagnosis')
        else:path.write_text(text)
    print(json.dumps(dict(check=check,scope='fitting_only',record_counts=data['full_record_counts'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    main(parser.parse_args().check)
