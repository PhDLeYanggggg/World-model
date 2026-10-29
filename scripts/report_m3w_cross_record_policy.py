"""Independent mass checks, complete-roster tables and source bootstrap."""
import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scripts import run_m3w_cross_record_policy as run
from scripts.report_m3w_shared_train_risk import check_policy, equal_number


def require(condition, message):
    if not condition: raise ValueError(message)


def validate_result(d):
    require(set(d['policies']) == set(run.readout.POLICIES), 'Fixed policy roster changed')
    require(d['known']+d['unknown'] == d['rows'] and not d['inference_label_filter'], 'Cohort changed')
    checks = 0
    masses = ('known_rows','unknown_rows','selected_count','selected_unknown','selected_unknown_envelope_mass',
        'selected_known_benefit_mass','selected_known_harm_mass','selected_known_reference_mass',
        'selected_known_easy_reference_mass','selected_known_easy_harm_mass',
        'selected_known_easy_benefit_mass','full_known_easy_reference_mass')
    for arm,p in d['policies'].items():
        check_policy(p); checks += 1
        for rec in d['recordings'].values(): check_policy(rec[arm]); checks += 1
        for key in masses: equal_number(p[key],sum(r[arm][key] for r in d['recordings'].values()))
    require(set(d['comparisons']) == {b+'_minus_'+a for a,b in run.readout.PAIRS}, 'Fixed contrasts changed')
    for c in d['comparisons'].values():
        check_policy(c['matched_left']); check_policy(c['matched_right']); checks += 2
        require(c['matched_left']['selected_count'] == c['matched_right']['selected_count'] == c['matched_count'], 'Counts differ')
        require(c['matched_empty'] == (c['matched_count'] == 0)
            and c['matched_risk_role'] == 'utility_only_not_safety_gate','Matched utility promoted to safety')
        if c['matched_empty']:
            require(c['empty_risk_status'] == 'not_estimable'
                and c['matched_left']['easy_selected_risk_upper'] is None
                and c['matched_right']['easy_selected_risk_upper'] is None,'Empty risk promoted')
        for mode in ('full','matched'):
            p = c[mode]; den = d['full_known_reference_mass']
            equal_number(p['lower_mass'],p['known_difference_mass']-p['unknown_exchanged_envelope_mass'])
            equal_number(p['upper_mass'],p['known_difference_mass']+p['unknown_exchanged_envelope_mass'])
            for bound in ('lower','upper'):
                equal_number(p[bound+'_pp'],100*p[bound+'_mass']/den if den > 0 else None)
    return checks


def source_bootstrap(rows):
    sources = sorted({r['source'] for r in rows})
    if len(sources) < 2: raise ValueError('At least two source clusters required')
    ix = np.random.default_rng(1342177).integers(len(sources),size=(3000,len(sources)))
    out = {}
    for left,right in run.readout.PAIRS:
        name = right+'_minus_'+left; out[name] = {}
        for mode in ('full','matched'):
            values = np.array([[statistics.fmean(r['result']['comparisons'][name][mode][bound]
                for r in rows if r['source'] == s) for bound in ('lower_pp','upper_pp')] for s in sources])
            if not np.isfinite(values).all(): raise ValueError('No unidentified source utility dropped')
            resampled = values[ix].mean(1)
            endpoints = np.quantile(resampled,[.025,.975],axis=0)
            out[name][mode] = dict(source_mean_completion_bounds_pp=values.mean(0).tolist(),
                lower_endpoint_percentiles_pp=endpoints[:,0].tolist(),
                upper_endpoint_percentiles_pp=endpoints[:,1].tolist(),
                outer_bootstrap_completion_envelope_pp=[float(endpoints[0,0]),float(endpoints[1,1])])
    return dict(seed=1342177,draws=3000,source_clusters=len(sources),sources=sources,
        resampling_unit='source_after_averaging_repeated_contexts_and_head_seeds',comparisons=out,
        independent_confirmation=False,row_bootstrap=False,conditional_on_frozen_forecasters=True,
        interpretation='exploratory_source_bootstrap_of_partial_label_completion_endpoints_not_FDE_CI')


def collect():
    reg,_,_ = run.admission(); learned = run.models()
    finish = json.loads((run.PUBLIC/'complete.json').read_text())
    freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    require(finish['registration_sha256'] == run.sha(run.REG)
        and finish['training_freeze_sha256'] == run.sha(run.PUBLIC/'training_freeze.json')
        and finish['summary_sha256'] == run.sha(run.PUBLIC/'summary.json'),'Lineage changed')
    rows, checks = [],0
    for ref in finish['groups']:
        path = run.ROOT/ref['path']
        require(path.resolve().is_relative_to((run.PUBLIC/'groups').resolve()) and not path.is_symlink()
            and run.sha(path) == ref['sha256'],'Changed readout')
        row = json.loads(path.read_text())
        require(row['registration_sha256'] == run.sha(run.REG)
            and row['training_freeze_sha256'] == run.sha(run.PUBLIC/'training_freeze.json')
            and not any(row[k] for k in ('independent_roles_read','deployment_changed','whole_pipeline_OOF')),'Scope changed')
        oldpath = run.previous.PUBLIC/'groups'/path.name
        require(run.sha(oldpath) == row['parent_group_sha256'],'Frozen reference changed')
        old = json.loads(oldpath.read_text())
        for arm in ('origin','exact_gradient'):
            for field in ('action_sha256','score_sha256'):
                require(row[field][arm] == old[field][arm],'Control prediction changed')
            require(row['result']['policies'][arm] == old['result']['policies'][arm],'Control score changed')
        item = next(i for i in reg['contexts'][row['context']] if i['packet']['identity']['source'] == row['source'])
        key = 'fit_recordings' if row['scope'] == 'fit' else 'held_recordings'
        require(set(row['result']['recordings']) == set(item['partition'][key]),'Recording roster changed')
        checks += validate_result(row['result']);rows.append(row)
    summary = json.loads((run.PUBLIC/'summary.json').read_text())
    require(summary == run.readout.summarize(rows),'Summary changed')
    detail = dict(result_source='cached_verified_fresh_readout_arithmetic',policy_checks=checks,
        frozen_control_checks=2*len(rows),complete_sha256=run.sha(run.PUBLIC/'complete.json'),
        reporter_sha256=run.sha(Path(__file__)),training={},scopes={})
    for arm in run.api.ARMS:
        fits = [f for f in learned.values() if f['arm'] == arm]
        detail['training'][arm] = dict(fits=len(fits),optimizer_steps=sum(f['step'] for f in fits),
            seconds=sum(f['seconds'] for f in fits),meta_forward_rows=sum(f['meta_forward_rows'] for f in fits),
            final_preupdate_positive_record_channels=sum(v>0 for f in fits for p in f['final_trace']['exact_record_risk'] for v in p),
            final_duals_at_cap=sum(v>=100 for f in fits for p in f['final_trace']['dual'] for v in p),
            final_dual_count=sum(len(p) for f in fits for p in f['final_trace']['dual']),
            final_inner_clipped_sides=sum(v<1 for f in fits for v in f['final_trace']['inner_clip_factors']))
    for context in reg['contexts']:
        for seed in reg['seeds']:
            a,b = (learned[context,seed,arm] for arm in run.api.ARMS)
            require(a['episode_hash'] == b['episode_hash'] and a['meta_forward_rows'] == b['meta_forward_rows'],
                    'Matched training episodes/exposure differ')
    for scope in ('fit','inner_held'):
        rs = [r for r in rows if r['scope'] == scope]
        item = dict(unique_source_recordings=len({(r['source'],rec) for r in rs for rec in r['result']['recordings']}),
                    slices={},normalized_mass_pp={},retention_vs_origin={})
        for key in ('source','context','seed'):
            item['slices'][key] = {str(v):run.readout.aggregate([r for r in rs if r[key] == v])
                                  for v in sorted({r[key] for r in rs})}
        for arm in run.readout.POLICIES:
            item['normalized_mass_pp'][arm] = {k:statistics.fmean(100*r['result']['policies'][arm][k]
                /r['result']['full_known_reference_mass'] for r in rs) for k in
                ('selected_known_benefit_mass','selected_known_harm_mass','selected_unknown_envelope_mass')}
        masses = item['normalized_mass_pp'];bk,hk = 'selected_known_benefit_mass','selected_known_harm_mass'
        for arm in ('exact_gradient',*run.api.ARMS):
            item['retention_vs_origin'][arm] = dict(
                benefit_fraction=masses[arm][bk]/masses['origin'][bk] if masses['origin'][bk]>0 else None,
                harm_fraction=masses[arm][hk]/masses['origin'][hk] if masses['origin'][hk]>0 else None,
                known_net_change_pp=masses[arm][bk]-masses[arm][hk]-masses['origin'][bk]+masses['origin'][hk])
        detail['scopes'][scope] = item
    bootstrap = source_bootstrap([r for r in rows if r['scope'] == 'inner_held'])
    return freeze,finish,summary,detail,bootstrap


def markdown(freeze,done,summary,detail,bootstrap):
    def f(x): return 'not_estimable' if x is None else f'{x:+.6f}'
    lines = ['# Cross-Recording Meta-Policy Readout','','## Evidence Scope','',
        f"Fresh_run: {freeze['count']} neural policy fits, {freeze['optimizer_steps']:,} updates; training {freeze['seconds']:.3f}s.",
        f"Peak RSS {freeze['peak_RSS_bytes']:,} bytes; readout {done['seconds']:.3f}s.",
        'Cached_verified: frozen source data, original forecasters/moments, exact_gradient and origin controls.',
        'Not_run: independent test, new trajectory forecaster, deployment. Head seeds are not forecasting seeds.',
        'Both episode sides are TRAIN. No test-time adaptation. All finals frozen before the exposed inner-held readout.',
        'All/easy positive-harm budgets and easy-degradation requirement remain 2%; matched utility is not safety.',
        'Empty matched risk is not_estimable. Completion bounds are not CIs or FDE improvements.','',
        '## Full-Policy Safety','','|Scope|Policy|Budget views|Record budget views|Known all/easy violations|Easy degradation violations|Undefined easy|Actions / unknown|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for scope,s in summary['scopes'].items():
        for arm,p in s['safety'].items():
            lines.append(f"|{scope}|{arm}|{p['budget_views']}/{p['required_views']}|{p['budget_record_views']}/{p['required_record_views']}|{p['known_all_violations']}/{p['known_easy_violations']}|{p['easy_degradation_violations']}|{p['undefined_easy_views']}|{p['selected']}/{p['unknown_selected']}|")
    lines += ['','## Utility Completion Bounds','','Equal-weight view means; not pooled row accuracy.','',
        '|Scope|Contrast|Full lower / upper pp|Count-matched lower / upper pp|Empty matched views|',
        '|---|---|---:|---:|---:|']
    for scope,s in summary['scopes'].items():
        for name,c in s['comparisons'].items():
            lines.append(f"|{scope}|{name}|{f(c['full_lower_pp'])} / {f(c['full_upper_pp'])}|{f(c['matched_lower_pp'])} / {f(c['matched_upper_pp'])}|{c['empty_matched_views']}|")
    lines += ['','## Paired Source Bootstrap','',
        '3,000 draws of 12 source clusters, seed 1342177. Average repeated contexts/head seeds within source first.',
        'Intervals below combine source-bootstrap endpoint uncertainty with outcome-completion bounds; they are exploratory, not independent confirmation.',
        'A positive upper endpoint is not a positive result. No row/bootstrap independence claim.','',
        '|Held contrast|Full outer envelope pp|Matched outer envelope pp|','|---|---:|---:|']
    for name,modes in bootstrap['comparisons'].items():
        a,b = (modes[k]['outer_bootstrap_completion_envelope_pp'] for k in ('full','matched'))
        lines.append(f'|{name}|{f(a[0])} / {f(a[1])}|{f(b[0])} / {f(b[1])}|')
    lines += ['','## Held Source and Seed Slices','',
        '|Slice|Primary budget views|Primary known all/easy violations|Cross minus same matched lower / upper pp|','|---|---:|---:|---:|']
    for kind in ('source','seed'):
        for key,s in detail['scopes']['inner_held']['slices'][kind].items():
            p=s['safety']['cross_record_meta'];c=s['comparisons']['cross_record_meta_minus_same_record_meta']
            lines.append(f"|{kind}:{key}|{p['budget_views']}/{p['required_views']}|{p['known_all_violations']}/{p['known_easy_violations']}|{f(c['matched_lower_pp'])} / {f(c['matched_upper_pp'])}|")
    lines += ['','## Known Benefit and Harm Retained','','Normalized masses, relative to frozen origin; unknown outcomes remain separately bounded.','',
        '|Scope|Policy|Benefit retained|Harm retained|Known net change pp|','|---|---|---:|---:|---:|']
    for scope,info in detail['scopes'].items():
        for arm,p in info['retention_vs_origin'].items():
            lines.append(f"|{scope}|{arm}|{100*p['benefit_fraction']:.3f}%|{100*p['harm_fraction']:.3f}%|{f(p['known_net_change_pp'])}|")
    lines += ['','## Compute and Checks','',
        '|Arm|Fits|Updates|Seconds|Support+query forward row occurrences|Final clipped inner sides|','|---|---:|---:|---:|---:|---:|']
    for arm,t in detail['training'].items():
        lines.append(f"|{arm}|{t['fits']}|{t['optimizer_steps']}|{t['seconds']:.3f}|{t['meta_forward_rows']}|{t['final_inner_clipped_sides']}|")
    lines += ['',f"Separate arithmetic checks: {detail['policy_checks']} policy summaries, {detail['frozen_control_checks']} frozen controls.",
        f"Primary full-budget support: {summary['primary_full_budget_supported']}. Deployment remains unchanged.",
        '74 fit / 31 held source-qualified recordings repeat across contexts and head seeds.',
        'No independent calibration, confirmation, metric/seconds, true-3D, foundation or submission-readiness claim.',
        'Stage5C not executed; SMC not enabled.','']
    return '\n'.join(lines)


def main(check=False):
    freeze,done,summary,detail,bootstrap = collect()
    outputs = {'readout.md':markdown(freeze,done,summary,detail,bootstrap),
        'readout.json':json.dumps(detail,indent=2,allow_nan=False)+'\n',
        'source_bootstrap.json':json.dumps(bootstrap,indent=2,allow_nan=False)+'\n'}
    for name,text in outputs.items():
        path=run.PUBLIC/name
        if check:
            require(path.read_text() == text,'Reporter replay differs: '+name)
        else:
            if path.exists(): require(path.read_text() == text,'Existing report differs: '+name)
            else: path.write_text(text)
    print(json.dumps(dict(checked=check,policy_checks=detail['policy_checks'],frozen_control_checks=detail['frozen_control_checks'])))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();main(args.check)
