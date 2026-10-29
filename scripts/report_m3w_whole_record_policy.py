"""Separate arithmetic checks and complete-roster whole-record policy tables."""
import argparse
import json
from pathlib import Path
import statistics

from scripts import run_m3w_whole_record_policy as run
from scripts.report_m3w_shared_train_risk import check_policy, equal_number


def require(ok, message):
    if not ok: raise ValueError(message)


def validate_result(data):
    require(set(data['policies']) == set(run.readout.POLICIES), 'Policy roster changed')
    require(data['rows'] == data['known']+data['unknown'] and not data['inference_label_filter'], 'Inference cohort changed')
    checks = 0
    masses = ('known_rows','unknown_rows','selected_count','selected_unknown','selected_unknown_envelope_mass',
        'selected_known_benefit_mass','selected_known_harm_mass','selected_known_reference_mass',
        'selected_known_easy_reference_mass','selected_known_easy_harm_mass',
        'selected_known_easy_benefit_mass','full_known_easy_reference_mass')
    for arm, policy in data['policies'].items():
        check_policy(policy); checks += 1
        for rec in data['recordings'].values():
            check_policy(rec[arm]); checks += 1
        for key in masses:
            equal_number(policy[key], sum(r[arm][key] for r in data['recordings'].values()))
    require(set(data['comparisons']) == {b+'_minus_'+a for a,b in run.readout.PAIRS}, 'Comparison roster changed')
    for c in data['comparisons'].values():
        check_policy(c['matched_left']); check_policy(c['matched_right']); checks += 2
        require(c['matched_left']['selected_count'] == c['matched_right']['selected_count'] == c['matched_count'], 'Count mismatch')
        require(c['matched_empty'] == (c['matched_count'] == 0)
            and c['matched_risk_role'] == 'utility_only_not_safety_gate', 'Empty or matched safety contract changed')
        if c['matched_empty']:
            require(c['empty_risk_status'] == 'not_estimable' and c['matched_right']['easy_selected_risk_upper'] is None,
                    'Empty risk promoted')
        for mode in ('full','matched'):
            pair = c[mode]; den = data['full_known_reference_mass']
            equal_number(pair['lower_mass'],pair['known_difference_mass']-pair['unknown_exchanged_envelope_mass'])
            equal_number(pair['upper_mass'],pair['known_difference_mass']+pair['unknown_exchanged_envelope_mass'])
            for bound in ('lower','upper'):
                equal_number(pair[bound+'_pp'],100*pair[bound+'_mass']/den if den > 0 else None)
    return checks


def collect():
    reg, _, _ = run.admission(True); models = run.models()
    done = json.loads((run.PUBLIC/'complete.json').read_text())
    freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text()); rows = []; checks = 0
    require(done['registration_sha256'] == run.sha(run.REG)
        and done['training_freeze_sha256'] == run.sha(run.PUBLIC/'training_freeze.json')
        and done['summary_sha256'] == run.sha(run.PUBLIC/'summary.json')
        and not done['independent_roles_read'] and not freeze['independent_roles_read'], 'Readout lineage changed')
    for ref in done['groups']:
        path = run.ROOT/ref['path']
        require(not path.is_symlink() and path.resolve().is_relative_to((run.PUBLIC/'groups').resolve())
            and run.sha(path) == ref['sha256'], 'Group receipt changed')
        row = json.loads(path.read_text())
        require(row['registration_sha256'] == run.sha(run.REG)
            and row['training_freeze_sha256'] == run.sha(run.PUBLIC/'training_freeze.json')
            and not any(row[k] for k in ('independent_roles_read','deployment_changed','whole_pipeline_OOF')), 'Scope changed')
        oldpath = run.previous.PUBLIC/'groups'/path.name
        require(run.sha(oldpath) == row['parent_group_sha256'], 'Original readout changed')
        old = json.loads(oldpath.read_text())
        for new, former in (('origin','origin'),('stochastic','hard_record')):
            require(row['result']['policies'][new] == old['result']['policies'][former]
                and row['action_sha256'][new] == old['action_sha256'][former]
                and row['score_sha256'][new] == old['score_sha256'][former], 'Frozen control mismatch')
        item = next(i for i in reg['contexts'][row['context']] if i['packet']['identity']['source'] == row['source'])
        key = 'fit_recordings' if row['scope'] == 'fit' else 'held_recordings'
        require(set(row['result']['recordings']) == set(item['partition'][key]), 'Recording roster changed')
        checks += validate_result(row['result']); rows.append(row)
    summary = json.loads((run.PUBLIC/'summary.json').read_text())
    require(summary == run.readout.summarize(rows), 'Summary differs')
    detail = dict(policy_checks=checks, frozen_control_checks=288, scopes={}, training={},
        result_source='cached_verified', complete_sha256=run.sha(run.PUBLIC/'complete.json'),
        reporter_sha256=run.sha(Path(__file__)), independent_confirmation=False, deployment_changed=False)
    for arm in run.api.ARMS:
        fits = [v for v in models.values() if v['arm'] == arm]
        detail['training'][arm] = dict(fits=len(fits), optimizer_steps=sum(f['step'] for f in fits),
            seconds=sum(f['seconds'] for f in fits), sampled_primal_rows=sum(f['row_draws'] for f in fits),
            whole_record_forward_rows=sum(f['whole_record_forward_rows'] for f in fits),
            full_gradient_rows=sum(f['full_gradient_rows'] for f in fits))
    for scope in ('fit','inner_held'):
        group = [r for r in rows if r['scope'] == scope]
        detail['scopes'][scope] = dict(slices={}, unique_source_recordings=len({(r['source'], rec)
            for r in group for rec in r['result']['recordings']}), normalized_mass_pp={})
        for key in ('source','context','seed'):
            detail['scopes'][scope]['slices'][key] = {str(v):run.readout.aggregate([r for r in group if r[key] == v])
                                                   for v in sorted({r[key] for r in group})}
        for arm in run.readout.POLICIES:
            detail['scopes'][scope]['normalized_mass_pp'][arm] = {key:statistics.fmean(
                100*r['result']['policies'][arm][key]/r['result']['full_known_reference_mass'] for r in group)
                for key in ('selected_known_benefit_mass','selected_known_harm_mass','selected_unknown_envelope_mass')}
    return freeze, done, summary, detail


def markdown(freeze, done, summary, detail):
    fmt = lambda x: 'not estimable' if x is None else f'{x:+.6f}'
    lines = ['# Whole-Recording Policy Readout','','## Material Passport','',
        f"Fresh_run:36 neural policy fits,72,000 updates;training {freeze['seconds']:.3f}s;readout {done['seconds']:.3f}s.",
        f"Training peak RSS {freeze['peak_RSS_bytes']:,} bytes. All finals frozen before held scoring.",
        'Cached_verified: source data,forecast/risk models,original and stochastic policies.',
        'Not_run: independent test,new forecaster,deployment. Head seeds are not new forecasting seeds.',
        'Held recordings were exposed historically. This is exploratory,not whole-pipeline OOF.', '',
        '## Full Policy Safety','',
        '| Cohort | Policy | Selected | Unknown | Budget views | Budget recording views | Known all/easy violations | Easy degradation violations | Undefined easy |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope, block in summary['scopes'].items():
        for arm, p in block['safety'].items():
            lines.append(f"| {scope} | {arm} | {p['selected']} | {p['unknown_selected']} | {p['budget_views']}/72 | {p['budget_record_views']}/{p['required_record_views']} | {p['known_all_violations']}/{p['known_easy_violations']} | {p['easy_degradation_violations']} | {p['undefined_easy_views']} |")
    lines += ['', 'Original2% all/easy positive harm/reference and easy-preservation requirements unchanged.',
        'Undefined selected risk does not pass. Floor selects nobody; no fictitious safe zero ratio.', '',
        '## Paired Utility', '', '| Cohort | Contrast | Full lower/upper pp | Same-count lower/upper pp | Empty matched |',
        '|---|---|---:|---:|---:|']
    for scope, block in summary['scopes'].items():
        for name, p in block['comparisons'].items():
            lines.append(f"| {scope} | {name} | {fmt(p['full_lower_pp'])} / {fmt(p['full_upper_pp'])} | {fmt(p['matched_lower_pp'])} / {fmt(p['matched_upper_pp'])} | {p['empty_matched_views']} |")
    lines += ['', 'Deterministic partial-label completion bounds,not sampling confidence intervals or relative FDE gains.',
        'Full known reference denominator in each view,then equal-view averaging. Count matching is utility only.',
        'No posthoc policy selection; exact_gradient remains primary.', '', '## Compute and Exposure', '',
        '```json',json.dumps(detail['training'],indent=2),'```',
        'Exact_dual matches original sampled primal draws. Exact_gradient has greater deterministic gradient',
        'exposure; both see every eligible TRAIN row for dual updates. Update-matched,not compute-matched.', '',
        '## Complete Slices', '', '| Cohort | Slice | Primary vs original full pp | Primary vs stochastic full pp | Primary budget views |',
        '|---|---|---:|---:|---:|']
    for scope, block in detail['scopes'].items():
        for key, slices in block['slices'].items():
            for value, d in slices.items():
                a, b = [d['comparisons']['exact_gradient_minus_'+x] for x in ('origin','stochastic')]
                lines.append(f"| {scope} | {key}:{value} | {fmt(a['full_lower_pp'])} / {fmt(a['full_upper_pp'])} | {fmt(b['full_lower_pp'])} / {fmt(b['full_upper_pp'])} | {d['safety']['exact_gradient']['budget_views']}/{d['views']} |")
    lines += ['', 'Record/row views repeat across contexts/seeds; not independent samples.',
        f"Primary full budget supported: {summary['primary_full_budget_supported']}.",
        'No independent confirmation or deployment change. Empirical feasibility is not population safety.',
        'obs8/pred12 rawstride12,pixel/local detector-silver;no metric,seconds,human gold,true3D,foundation claim.',
        'Stage5C/SMC remain off.']
    return '\n'.join(lines)+'\n'


def execute(check=False):
    freeze, done, summary, detail = collect(); md = markdown(freeze,done,summary,detail)
    path, table = run.PUBLIC/'readout.json',run.PUBLIC/'readout.md'
    if check:
        require(detail == json.loads(path.read_text()) and md == table.read_text(),'Report replay differs')
    else:
        run.once(path,detail)
        if table.exists(): require(table.read_text() == md,'Frozen report changed')
        else: table.write_text(md)
    print(json.dumps(dict(policy_checks=detail['policy_checks'],frozen_control_checks=288,
        primary_full_budget_supported=summary['primary_full_budget_supported'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--check',action='store_true'); execute(p.parse_args().check)
