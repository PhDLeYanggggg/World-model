"""Report the registered reference-input contrast without selecting a policy."""
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_expert_reference as run
from scripts.diagnose_m3w_role_separated_selection import moment_summary
from scripts.report_m3w_moving_complete_donor_policy import category


def utility_supported(a):
    ci=a['primary']['completion_lower_pp']['interval95']
    return bool(ci is not None and ci[0]>0)


def candidate_screen(a):
    s=a['policies']['expert_context']['joint_free']['safety']
    return bool(s['required_record_views']==72 and s['original_record_contract_pass'] and utility_supported(a))


def components(reg,a):
    errors=[];selected=[];common=[];caps=[];data=run.parent.full.load_data()
    for node in reg['jobs']:
        own,old=run.parent.own_and_floor(node);key=node['job']['key'];ix=old['local_ids']
        target=data['target_eval'][ix]-data['origin'][ix,None];valid=data['valid'][ix]
        packet=dict(paths=old['paths'],floor=old['floor'],cv=old['cv'],
                    causal=dict(envelope=run.np.linalg.norm(old['paths']-old['floor'][:,None],axis=-1).max(-1)))
        truth=run.parent.previous.method.amplitude.targets(packet,target,valid,node['job']['easy_cut'])['moments']
        for seed in (17,29,43):
            banks={arm:run.parent.load_cache(run.read(run.PUBLIC/'inference'/f'{key}_{arm}_seed{seed}.json')['cache']) for arm in run.method.ARMS}
            banks['frozen']=run.parent.load_cache(run.read(run.parent.PUBLIC/'inference'/f'{key}_role_separated_seed{seed}.json')['cache'])
            for site in node['part']['names']:
                take=own['sites']==site;known=take&own['observed'];w=own['weights'][known];w=w/w.sum()
                row,action=own['row'][known],own['action'][known]
                origin=run.method.base.read_checkpoint(run.checked(run.read(run.parent.PUBLIC/'fits'/f'{key}_{site}_seed{seed}_role_separated.json')['checkpoint']))
                rms=origin['preprocess']['rms'][[2,3]];y=own['moments'][known][:,[2,3]]
                for arm,b in banks.items():
                    pp=b['conditional'][row,action][:,[2,3]]/own['scales'][known,None]
                    delta=(pp-y)/rms
                    errors.append(dict(job=key,source=site,seed=seed,arm=arm,known_candidate_rows=len(row),
                        reference_MSE=float(w@(delta[:,0]**2)),easy_reference_MSE=float(w@(delta[:,1]**2)),
                        reference_bias=float(w@delta[:,0]),easy_reference_bias=float(w@delta[:,1]),
                        cohort='same_all_legal_known_candidates_and_frozen_known_RMS'))
                    for j in (1,4):run.np.testing.assert_array_equal(b['moments'][...,j],banks['frozen']['moments'][...,j])
                    run.np.testing.assert_array_equal(b['probability'],banks['frozen']['probability'])
                    caps.append(dict(job=key,source=site,seed=seed,arm=arm,
                        additional_benefit_caps=int((b['conditional'][own['row'][take],own['action'][take],0]
                            <banks['frozen']['conditional'][own['row'][take],own['action'][take],0]).sum())))
                    metric=next(r for r in a['rows'] if r['job']==key and r['source']==site and r['seed']==seed and r['arm']==arm)
                    for policy in ('independent','joint_free'):
                        mask=(old['sites']==site)&(b[policy]>=0)
                        for rec,v in metric['policies'][policy]['recordings'].items():
                            at=run.np.flatnonzero(mask&(old['recordings'].astype(str)==rec));choice=b[policy][at]
                            summary=moment_summary(b['moments'][at,choice].sum(0),truth[at,choice].sum(0))
                            for lhs,rhs in (('actual_completion_all_risk','completion_all_risk_upper'),
                                            ('actual_completion_easy_risk','completion_easy_risk_upper')):
                                if v[rhs] is None:assert summary[lhs] is None
                                else:run.np.testing.assert_allclose(summary[lhs],v[rhs],rtol=1e-10,atol=1e-12)
                            selected.append(dict(job=key,source=site,seed=seed,arm=arm,policy=policy,recording=rec,
                                reason=category(v),unknown_selected=v['unknown_selected'],**summary))
                    frozen_metric=next(r for r in a['rows'] if r['job']==key and r['source']==site and r['seed']==seed and r['arm']=='frozen')
                    for rec in frozen_metric['policies']['joint_free']['recordings']:
                        at=run.np.flatnonzero((old['sites']==site)&(old['recordings'].astype(str)==rec)&(banks['frozen']['joint_free']>=0))
                        choice=banks['frozen']['joint_free'][at]
                        common.append(dict(source=site,seed=seed,arm=arm,recording=rec,
                            cohort='frozen_joint_free_choices_identical_for_all_arms',
                            **moment_summary(b['moments'][at,choice].sum(0),truth[at,choice].sum(0))))
    return dict(reference_errors=errors,own_selected_calibration=selected,frozen_choice_calibration=common,
                benefit_caps=caps,all_candidate_harm_support_exact=True,posthoc_not_policy_selection=True)


def f(x):return 'NA' if x is None else f'{x:+.6f}'


def main():
    a=run.read(run.PUBLIC/'analysis.json');cfg,reg=run.register()
    if len(a['rows'])!=72 or not a['cached_frozen_control_exact']:raise ValueError('Full registered readout required')
    d=components(reg,a);run.once(run.PUBLIC/'reference_diagnostics.json',d)
    primary=a['primary']['completion_lower_pp'];ci=primary['interval95']
    lines=['# Shared Frozen-Expert Context For Reference Learning','',
        'This is a completed model-side input contrast on already-exposed own-B.',
        'All 48 reference fits and 48 complete exact refits precede policy readout.',
        'Harm and label-support predictions remain bitwise frozen for every candidate.',
        'Both new arms refit the same reference network; only shared expert context differs.','',
        '## Registered Primary','',
        f'Expert-context minus history-refit joint-free paired completion-lower gain: **{f(primary["point"])} pp**,',
        f'exploratory 95% source-component interval **[{f(ci[0])}, {f(ci[1])}]**.' if ci else 'Required primary is not estimable.',
        'Eight source components, with three risk seeds averaged within source before',
        '3,000 resamples. Teacher seed remains17; no independent C/H readout.','',
        '## Complete Policies','',
        '| Reference model | Decision | Known gain vs CV % | Separate lower % | Original risk supported | Known easy violations | Easy NA |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm in ('frozen',*run.method.ARMS):
        for name,p in a['policies'][arm].items():
            s=p['safety'];lines.append(f'| {arm} | {name} | {f(p["source_seed_macro_known_gain"])} | '
                f'{f(p["source_seed_macro_completion_lower_gain"])} | {s["supported_record_views"]}/72 | '
                f'{s["known_easy_violations"]} | {s["undefined_easy_views"]} |')
    lines+=['','72 views repeat24recordings over3risk seeds; they are not72independent scenes.',
        'Separate lower gains are not the paired lower contrast. Available-label',
        'image-coordinate obs8/pred12ADE is not official complete12ADE or SDD raw t50.','',
        '## Reference Prediction On The Same Known Candidate Cohort','',
        '| Model | R MSE | ER MSE | R bias | ER bias |','|---|---:|---:|---:|---:|']
    for arm in ('frozen',*run.method.ARMS):
        rr=[r for r in d['reference_errors'] if r['arm']==arm]
        lines.append('| '+arm+' | '+' | '.join(f(float(run.np.mean([r[k] for r in rr])))
            for k in ('reference_MSE','easy_reference_MSE','reference_bias','easy_reference_bias'))+' |')
    lines+=['','All entries use the same source-specific frozen known-only RMS and weights.',
        'This is conditional reference prediction, not a calibrated risk bound.','',
        '## Selected-Risk Diagnosis','',
        '| Model | Decision | Pass | Observed harm violation | Completion-only | Missing support | ER overpredicted |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm in ('frozen',*run.method.ARMS):
        for policy in ('independent','joint_free'):
            rr=[r for r in d['own_selected_calibration'] if r['arm']==arm and r['policy']==policy]
            cat=Counter(r['reason'] for r in rr)
            vals=[cat[k] for k in ('passes_original_record_contract','observed_harm_violation','unknown_completion_bound_violation','missing_reference_support')]
            lines.append(f'| {arm} | {policy} | '+' | '.join(map(str,vals))+f' | {sum(r["predicted_easy_reference"]>r["actual_observed_easy_reference"] for r in rr)} |')
    lines+=['','Categories are mutually exclusive in observed-violation,completion-only,',
        'support order. Undefined easy ratios remain NA, never zero harm. The linked',
        'diagnostics also use identical frozen joint choices to separate reference',
        'prediction changes from changed selection. Benefit caps can change utility;',
        'there is no claim of a pure denominator-only effect.','',
        '## Verdict','',
        'Registered utility evidence: '+('positive exploratory contrast' if utility_supported(a) else 'not supported')+'.',
        'Complete development candidate: '+('PASS' if candidate_screen(a) else '**FAIL / NA**')+'.',
        'Original per-recording2%positive all/easy harm,signed easy-net and reference',
        'support requirements are unchanged. Empty utility controls remain NA. No',
        'identity-fallback exception, new test threshold or automatic deployment.','',
        'fresh_run:shared-context derivation,48fits,48full refits,new inference and readout.',
        'cached_verified:parent features/labels,forecasts,role checkpoints and full controls.',
        'not_run:new independent C/H,three forecaster seeds,new dynamics or deployment.',
        'Original24-source-group main remains NA. Annotation-generation causality',
        'unverified; no metric/seconds,true3D,foundation,human-gold,Stage5C or SMC claim.','',
        '[Protocol](protocol.md) | [Reproduction](operation_zh.md) | [Failure analysis](failure_analysis.md)']
    (run.PUBLIC/'README.md').write_text('\n'.join(lines)+'\n')
    gates=['# Shared-Expert Reference Gates','', '| Gate | Status | Evidence |','|---|---|---|',
        '| Real source-excluded reference fitting | PASS | 48 models,4096updates each |',
        '| Complete exact refits | PASS | All48, matched initialization and draws |',
        '| Frozen harm/support | PASS | Every candidate exact |',
        '| Original frozen controls | PASS | Complete metrics exact |',
        '| Shared causal context | PASS by interface/alignment | Annotation-generation causality unverified |',
        '| Registered paired utility | '+('PASS, exploratory' if utility_supported(a) else 'FAIL / NA')+' | Primary fixed before execution |']
    for arm in run.method.ARMS:
        for name,p in a['policies'][arm].items():
            s=p['safety'];gates.append(f'| {arm}/{name} complete original risk | {"PASS" if s["original_record_contract_pass"] else "FAIL / NA"} | {s["supported_record_views"]}/72 |')
    gates+=['| Complete candidate | '+('PASS' if candidate_screen(a) else 'FAIL / NA')+' | Primary and all required risk views |',
        '| Independent confirmation/deployment | NOT RUN / FALSE | No new C/H opened |',
        '| Original24-source-group main | NA, not passed | Unchanged estimand |',
        '| Stage5C/SMC | FALSE | Not executed |']
    (run.PUBLIC/'world_model_gate_expert_reference.md').write_text('\n'.join(gates)+'\n')
    print('Full reference-input contrast and unchanged original-risk gates reported.')


if __name__=='__main__':main()
