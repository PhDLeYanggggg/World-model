"""Arithmetic verification and descriptive readout of frozen direct policies."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from scripts import run_m3w_direct_policy as run
from scripts.report_m3w_shared_train_risk import check_policy, equal_number
from src.evaluation.m3w_easy_constrained_policy import budget_status


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mean(values):
    return statistics.fmean(values) if values and None not in values else None


def validate_result(data):
    require(data['rows'] == data['known']+data['unknown'] and not data['inference_label_filter'],
            'Full inference cohort required')
    require(set(data['policies']) == set(run.readout.POLICIES), 'Policy roster changed')
    checks = 0
    masses = ('known_rows', 'unknown_rows', 'selected_count', 'selected_unknown',
        'selected_unknown_envelope_mass', 'selected_known_benefit_mass',
        'selected_known_harm_mass', 'selected_known_reference_mass',
        'selected_known_easy_reference_mass', 'selected_known_easy_harm_mass',
        'selected_known_easy_benefit_mass', 'full_known_easy_reference_mass')
    for arm, policy in data['policies'].items():
        check_policy(policy); checks += 1
        for recording in data['recordings'].values():
            require(set(recording) == set(run.readout.POLICIES), 'Recording policy omitted')
            check_policy(recording[arm]); checks += 1
        for key in masses:
            equal_number(policy[key], sum(r[arm][key] for r in data['recordings'].values()))
    require(set(data['comparisons']) == {b+'_minus_'+a for a, b in run.readout.PAIRS},
            'Comparison roster changed')
    for pair in data['comparisons'].values():
        left, right = pair['matched_left'], pair['matched_right']
        check_policy(left); check_policy(right); checks += 2
        require(left['selected_count'] == right['selected_count'] == pair['matched_count'],
                'Count-matched comparison changed')
        require(pair['matched_risk_role'] == 'not_a_safety_gate', 'Matched control is utility only')
        require(pair['matched_empty'] == (pair['matched_count'] == 0), 'Empty comparison concealed')
        if pair['matched_empty']:
            require(pair['matched_risk_empty_status'] == 'not_estimable'
                    and left['all_selected_risk_upper'] is None
                    and right['easy_selected_risk_upper'] is None, 'Empty risk cannot pass')
        for mode in ('full', 'matched'):
            p = pair[mode]; den = data['full_known_reference_mass']
            equal_number(p['lower_mass'], p['known_difference_mass']-p['unknown_exchanged_envelope_mass'])
            equal_number(p['upper_mass'], p['known_difference_mass']+p['unknown_exchanged_envelope_mass'])
            for bound in ('lower', 'upper'):
                equal_number(p[bound+'_pp'], 100*p[bound+'_mass']/den if den > 0 else None)
    return checks


def aggregate(rows):
    out = dict(views=len(rows), row_occurrences=sum(r['result']['rows'] for r in rows),
        unknown_occurrences=sum(r['result']['unknown'] for r in rows),
        safety={}, comparisons={}, independent_observations=False)
    for arm in run.readout.POLICIES:
        ps = [r['result']['policies'][arm] for r in rows]
        recordings = [p[arm] for r in rows for p in r['result']['recordings'].values()]
        item = dict(selected=sum(p['selected_count'] for p in ps),
            unknown_selected=sum(p['selected_unknown'] for p in ps),
            budget_views=sum(budget_status(p)['supported'] for p in ps),
            budget_record_views=sum(budget_status(p)['supported'] for p in recordings),
            record_views=len(recordings), empty_views=sum(p['selected_count'] == 0 for p in ps),
            undefined_easy_views=sum(p['easy_selected_risk_upper'] is None for p in ps),
            reasons=dict(Counter(reason for p in ps for reason in p['reasons'])))
        for channel in ('', 'easy_'):
            key = 'known_'+channel+'positive_risk'; upper = channel+'selected_risk_upper' if channel else 'all_selected_risk_upper'
            item[channel+'known_violations'] = sum(p[key] is not None and p[key] > .02+1e-12 for p in ps)
            item[channel+'unknown_only_upper_failures'] = sum(p[key] is not None and p[key] <= .02+1e-12
                and p[upper] is not None and p[upper] > .02+1e-12 for p in ps)
            item[channel+'worst_defined_upper'] = max((p[upper] for p in ps if p[upper] is not None), default=None)
        item['easy_degradation_violations'] = sum(p['easy_degradation_upper'] is not None
            and p['easy_degradation_upper'] > .02+1e-12 for p in ps)
        out['safety'][arm] = item
    for left, right in run.readout.PAIRS:
        name = right+'_minus_'+left
        ps = [r['result']['comparisons'][name] for r in rows]
        item = dict(empty_matched_views=sum(p['matched_empty'] for p in ps))
        for mode in ('full', 'matched'):
            for bound in ('lower_pp', 'upper_pp'):
                item[mode+'_'+bound] = mean([p[mode][bound] for p in ps])
        out['comparisons'][name] = item
    return out


def validate():
    reg, cfg, _ = run.admission(); models = run.models()
    freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    done = json.loads((run.PUBLIC/'complete.json').read_text())
    require(freeze['optimizer_steps'] == 108000 and not freeze['independent_roles_read'], 'Training scope changed')
    require(done['registration_sha256'] == run.sha(run.REG)
        and done['training_freeze_sha256'] == run.sha(run.PUBLIC/'training_freeze.json')
        and done['summary_sha256'] == run.sha(run.PUBLIC/'summary.json')
        and not done['independent_roles_read'], 'Readout binding changed')
    rows, checks = [], 0
    for ref in done['groups']:
        path = run.ROOT/ref['path']
        require(not path.is_symlink() and path.resolve().is_relative_to((run.PUBLIC/'groups').resolve())
                and run.sha(path) == ref['sha256'], 'Readout receipt changed')
        row = json.loads(path.read_text())
        require(row['registration_sha256'] == run.sha(run.REG)
            and row['training_freeze_sha256'] == run.sha(run.PUBLIC/'training_freeze.json')
            and not any(row[k] for k in ('independent_roles_read', 'deployment_changed', 'whole_pipeline_OOF')),
            'Provenance boundary changed')
        item = next(i for i in reg['contexts'][row['context']] if i['packet']['identity']['source'] == row['source'])
        part = 'fit_recordings' if row['scope'] == 'fit' else 'held_recordings'
        require(set(row['result']['recordings']) == set(item['partition'][part]), 'Whole-recording split changed')
        checks += validate_result(row['result']); rows.append(row)
    summary = json.loads((run.PUBLIC/'summary.json').read_text())
    require(run.readout.summarize(rows) == summary, 'Summary no longer matches groups')
    training = {}
    for arm in run.api.ARMS:
        states = [run.api.core.read_checkpoint(run.ROOT/m['checkpoint']['path'])
                  for m in models.values() if m['arm'] == arm]
        require(len(states) == 18 and all(s['step'] == 2000 and not s['new_forecaster']
            and not s['held_used_for_training'] for s in states), 'Training count or role changed')
        training[arm] = dict(fits=18, parameters_each=sorted({s['parameters'] for s in states}),
            mean_first_loss=mean([s['trace'][0]['loss'] for s in states]),
            mean_final_loss=mean([s['trace'][-1]['loss'] for s in states]),
            mean_final_gain_lower=mean([s['trace'][-1]['gain_lower'] for s in states]),
            mean_final_action_fraction=mean([s['trace'][-1]['action_mean'] for s in states]),
            mean_soft_hard_gap=mean([s['trace'][-1]['soft_hard_gap'] for s in states]),
            final_duals_at_cap=sum(int((s['dual'] >= cfg['training']['dual_maximum']).sum()) for s in states),
            final_dual_count=sum(s['dual'].numel() for s in states),
            mean_max_final_dual=mean([float(s['dual'].max()) for s in states]),
            total_row_draws=sum(s['row_draws'] for s in states))
    detail = dict(scopes={}, training=training, arithmetic_policy_checks=checks,
        complete_sha256=run.sha(run.PUBLIC/'complete.json'), reporter_sha256=run.sha(Path(__file__)),
        result_source='cached_verified', new_training=False, new_inference=False,
        independent_confirmation=False, no_sampling_CI_claim=True)
    for scope in ('fit', 'inner_held'):
        part = [r for r in rows if r['scope'] == scope]
        detail['scopes'][scope] = dict(overall=aggregate(part), slices={})
        for key in ('source', 'context', 'seed'):
            detail['scopes'][scope]['slices'][key] = {str(v): aggregate([r for r in part if r[key] == v])
                                                    for v in sorted({r[key] for r in part})}
    return freeze, done, summary, detail


def markdown(freeze, done, summary, detail):
    fmt = lambda x: 'not estimable' if x is None else f'{x:+.6f}'
    lines = ['# Direct Neural Intervention: Fixed-Final Readout', '', '## What Ran', '',
        f"54 new neural policy fits,108,000 optimizer updates; {freeze['seconds']:.3f}s training,",
        f"{done['seconds']:.3f}s readout; peak training RSS {freeze['peak_RSS_bytes']:,} bytes.",
        'Fresh_run: policy fitting and fixed-final inference. Cached_verified: original',
        'forecaster/risk artifacts, source data and separate receipt arithmetic.',
        'No new trajectory forecaster, JEPA, residual or independent evaluation.',
        'All models frozen before inner-held scoring. Three head seeds are not three',
        'independent forecasting seeds. Prior exposure makes inner-held exploratory.', '',
        '## Full Policy Safety', '',
        'Original 2% all/easy positive-harm/reference and easy-preservation requirements.',
        'Risk is error-related, not physical safety. Undefined risks do not pass.', '',
        '| Cohort | Policy | Rows | Selected | Unknown selected | Budget views | Budget recording views | Known all/easy violations | Undefined easy |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope, block in detail['scopes'].items():
        agg = block['overall']
        for arm, p in agg['safety'].items():
            lines.append(f"| {scope} | {arm} | {agg['row_occurrences']} | {p['selected']} | {p['unknown_selected']} | {p['budget_views']}/{agg['views']} | {p['budget_record_views']}/{p['record_views']} | {p['known_violations']}/{p['easy_known_violations']} | {p['undefined_easy_views']} |")
    lines += ['', 'Row and recording occurrences repeat across contexts/seeds; not independent sample counts.',
        'The floor makes zero interventions. Its undefined selected risk is not zero or a safety certificate.',
        '', '## Paired Utility', '',
        '| Cohort | Contrast | Full lower/upper (pp) | Equal-count lower/upper (pp) | Empty matched views |',
        '|---|---|---:|---:|---:|']
    for scope, block in detail['scopes'].items():
        for name, p in block['overall']['comparisons'].items():
            lines.append(f"| {scope} | {name} | {fmt(p['full_lower_pp'])} / {fmt(p['full_upper_pp'])} | {fmt(p['matched_lower_pp'])} / {fmt(p['matched_upper_pp'])} | {p['empty_matched_views']} |")
    lines += ['', 'Intervals are deterministic partial-label completion bounds, not confidence intervals.',
        'Each view uses the full known floor-reference error denominator, then equal-view averaging.',
        'They are not relative FDE improvements. Matched risks are descriptive, never safety gates.',
        'Unknown shared actions cancel in paired utility; exchanged unknowns retain envelope uncertainty.', '',
        '## Training Diagnostics', '', '```json', json.dumps(detail['training'], indent=2), '```', '',
        'Dual-weighted objectives vary during training, so first/final loss is not a fixed validation loss.',
        'Final minibatch constraints do not prove whole-recording or held-recording safety.', '',
        '## Complete Source and Head-Seed Slices', '',
        '| Cohort | Slice | Primary full utility vs original (pp) | Primary matched utility vs original (pp) | Primary budget views |',
        '|---|---|---:|---:|---:|']
    for scope, block in detail['scopes'].items():
        for key, slices in block['slices'].items():
            for value, agg in slices.items():
                p = agg['comparisons']['hard_record_minus_origin']; s = agg['safety']['hard_record']
                lines.append(f"| {scope} | {key}:{value} | {fmt(p['full_lower_pp'])} / {fmt(p['full_upper_pp'])} | {fmt(p['matched_lower_pp'])} / {fmt(p['matched_upper_pp'])} | {s['budget_views']}/{agg['views']} |")
    lines += ['', '## Evidence Boundary', '',
        f"Primary full-roster budget supported: {summary['primary_full_roster_budget_supported']}.",
        'No secondary winner substitutes for hard_record. No new MSE gate or threshold search.',
        'obs8/pred12 rawstride12, pixel/local detector-silver; not metric, seconds, human gold,',
        'true3D, foundation, independent confirmation or submission-readiness evidence.',
        'Independent selection/calibration/confirmation untouched; deployment unchanged.',
        'Stage5C and SMC remain off.']
    return '\n'.join(lines)+'\n'


def execute(check=False):
    freeze, done, summary, detail = validate(); md = markdown(freeze, done, summary, detail)
    path, table = run.PUBLIC/'readout.json', run.PUBLIC/'readout.md'
    if check:
        require(detail == json.loads(path.read_text()) and md == table.read_text(), 'Report replay differs')
    else:
        run.once(path, detail)
        if table.exists():
            require(table.read_text() == md, 'Existing report changed')
        else:
            table.write_text(md)
    print(json.dumps(dict(views=144, fits=54, policy_checks=detail['arithmetic_policy_checks'],
        primary_full_roster_budget_supported=summary['primary_full_roster_budget_supported'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    execute(parser.parse_args().check)
