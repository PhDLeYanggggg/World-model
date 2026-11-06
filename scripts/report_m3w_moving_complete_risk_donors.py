"""Report model-side bound supervision without equating gain to safety."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_complete_risk_donors as run
from scripts.report_m3w_moving_complete_donor_policy import category


def eligible(policy):
    safety=policy['safety']
    return bool(safety['required_record_views']==24 and safety['original_record_contract_pass']
                and policy['source_macro_completion_lower_gain']>0)


def component_metrics(reg):
    records=[]
    for node in reg['jobs']:
        key=node['job']['key']
        own=run.load_cache(next(r['cache'] for r in run.read(run.PUBLIC/'own.json')['banks'] if r['job']==key))
        before=run.load_cache(run.read(run.full.PUBLIC/'inference_v2'/f'{key}.json')['cache'])
        after=run.load_cache(run.read(run.PUBLIC/'inference'/f'{key}.json')['cache'])
        active=before['active']
        run.np.testing.assert_array_equal(before['ids'][active],own['ids'])
        for site in node['part']['names']:
            for si,seed in enumerate((17,29,43)):
                oldstate=run.neural.rel.base.read_checkpoint(run.checked(run.full.risk_checkpoint(key,site,seed)))
                scale=oldstate['preprocess']['target_scale']
                for role,mask in [('observed',own['observed_future']),('bound',~own['observed_future'])]:
                    at=(own['sites']==site)&mask
                    w=own['weights'][at]
                    if len(w):w=w/w.sum()
                    mse={name:float(run.np.sum(w[:,None]*((risk[si,active[at]]-own['target'][at])/scale)**2)/2)
                         if len(w) else None
                         for name,risk in [('known_only',before['risk']),('complete_risk',after['risk'])]}
                    records.append(dict(job=key,source=site,seed=seed,label_role=role,rows=int(at.sum()),
                                        normalized_MSE=mse,normalizer='frozen_known_only_fitting_target_RMS'))
    return dict(rows=records,unknown_MSE_is_bound_target_error_not_ground_truth=True,
                label_role_is_not_an_inference_feature=True)


def target_scale_diagnostics(reg):
    rows=[]
    for node in reg['jobs']:
        key=node['job']['key']
        for site in node['part']['names']:
            for seed in (17,29,43):
                old=run.neural.rel.base.read_checkpoint(run.checked(run.full.risk_checkpoint(key,site,seed)))
                new=run.neural.rel.base.read_checkpoint(run.checked(
                    run.read(run.PUBLIC/'fits'/f'{key}_{site}_seed{seed}.json')['checkpoint']))
                rows.append(dict(job=key,source=site,seed=seed,
                    RMS_ratio=(new['preprocess']['target_scale']/old['preprocess']['target_scale']).tolist()))
    ratios=run.np.asarray([r['RMS_ratio'] for r in rows])
    return dict(rows=rows,minimum=ratios.min(0).tolist(),median=run.np.median(ratios,axis=0).tolist(),
                maximum=ratios.max(0).tolist(),posthoc_fitting_scale_diagnostic_not_model_selection=True)


def main():
    p=run.PUBLIC;a=run.read(p/'analysis.json');old=run.read(run.full.PUBLIC/'analysis.json')
    if len(a['rows'])!=8 or a['fresh_models']!=24 or not a['original_rules_unchanged']:
        raise ValueError('Complete registered model comparison required')
    components=component_metrics(run.read(run.REG))
    run.once(p/'component_metrics.json',components)
    scales=target_scale_diagnostics(run.read(run.REG))
    run.once(p/'target_scale_diagnostics.json',scales)
    lines=['# Complete-Risk Donor Supervision','',
        'I added the conservative unknown-label contributions omitted by known-only',
        'donor fitting. All 24 new Torch heads and 24 full exact refits are complete.',
        'The same features, source exclusions, network, budget and zero threshold remain.',
        'This is exposed-development model repair, not a new independent test.','',
        '## Complete Policy Comparison','',
        '| Training / policy | Available-label gain % | Completion lower gain % | Selected | Unknown selected | Original supported views |',
        '|---|---:|---:|---:|---:|---:|']
    for training,metrics in [('known_only',old['policies']),('complete_risk',a['policies'])]:
        names=('forest64','seed17','mean','unanimity') if training=='known_only' else tuple(metrics)
        for name in names:
            m=metrics[name];s=m['safety']
            lines.append(f'| {training} / {name} | {m["source_macro_known_ADE_gain"]:+.6f} | '
                f'{m["source_macro_completion_lower_gain"]:+.6f} | {m["selected"]} | {m["unknown_selected"]} | '
                f'{s["supported_record_views"]}/{s["required_record_views"]} |')
    lines+=['','Equal-source macro gain versus the original training-defined floor.',
        'Available-label ADE is not complete-12 official ADE. Completion lower gain',
        'retains the original unknown-label path envelopes. Empty support stays NA.','',
        '## Original Risk And Support','',
        '| Policy | Known all >2% | Known easy >2% | Completion all >2% | Completion easy >2% | Easy reference NA |',
        '|---|---:|---:|---:|---:|---:|']
    categories={}
    for name,m in a['policies'].items():
        s=m['safety']
        lines.append(f'| {name} | {s["known_all_violations"]} | {s["known_easy_violations"]} | '
            f'{s["completion_all_violations"]} | {s["completion_easy_violations"]} | {s["undefined_easy_views"]} |')
        counts={}
        for row in a['rows']:
            for view in row['policies'][name]['recordings'].values():
                k=category(view);counts[k]=counts.get(k,0)+1
        categories[name]=counts
    run.once(p/'failure_taxonomy.json',categories)
    lines+=['','## Paired Change From Known-Only Consensus','',
        'This uses the envelope between the two policies, not subtraction of their',
        'separate pessimistic lower gains. A higher individual lower bound is not',
        'proof that the paired improvement is positive.','',
        '| Policy | Known pp | Completion lower pp | Exploratory source-bootstrap interval for lower pp |',
        '|---|---:|---:|---:|']
    for name,m in a['paired_vs_known_only'].items():
        lower=m['completion_lower_pp'];lo,hi=lower['interval95']
        lines.append(f'| {name} | {m["known_pp"]["point"]:+.6f} | {lower["point"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] |')
    lines+=['','3,000 paired resamples of 8 source components; not windows, independent',
        'confirmation or safety intervals. All policies and seed slices are retained.','',
        '## Observed Versus Bound Target Error','',
        '| Label role | Known-only MSE | Complete-risk MSE | Improved source/seed views |',
        '|---|---:|---:|---:|']
    for role in ('observed','bound'):
        rows=[r for r in components['rows'] if r['label_role']==role and r['rows']>0]
        old_mse=run.np.mean([r['normalized_MSE']['known_only'] for r in rows])
        new_mse=run.np.mean([r['normalized_MSE']['complete_risk'] for r in rows])
        improved=sum(r['normalized_MSE']['complete_risk']<r['normalized_MSE']['known_only'] for r in rows)
        lines.append(f'| {role} | {old_mse:.6f} | {new_mse:.6f} | {improved}/{len(rows)} |')
    lines+=['','MSE uses each frozen known-only fitting RMS for a common comparison scale.',
        'Unknown-role MSE measures prediction of a conservative bound contribution,',
        'not error against an observed future. The new fitting RMS is inflated by',
        f'median factors {scales["median"][0]:.3f} (all) and {scales["median"][1]:.3f} (easy).',
        'This is consistent with large bound targets disrupting observed-risk',
        'regression, not proof that normalization alone caused the failure.','',
        '## Decision','']
    passing=[name for name,m in a['policies'].items() if eligible(m)]
    lines+=['Original complete-risk screen: '+(', '.join(passing)+' pass.' if passing else '**No new policy passes.**'),
        'No deployment upgrade follows this exposed-development result. The original',
        '2% rules, support/NA semantics and old failures remain. No identity exception.','',
        '## What The Treatment Changes','',
        'Known features and targets match exactly before fitting. New unknown targets',
        'are conservative bounds, not ground-truth future error. Rebalancing complete',
        'source/query pools and fitting-only normalization are part of the treatment.',
        'The comparison does not isolate those mechanisms. The unknown-role mask is',
        'used for training/reporting only, never inference or a query rejection gate.',
        'The existing two-channel head still does not directly ensure selected easy',
        'reference support; stronger fitted scores alone may therefore be insufficient.','',
        'fresh_run: complete-risk supervision, 24 fits, 24 exact refits and full readout.',
        'cached_verified: known donors, forecasts, proposer and known-only controls.',
        'not_run: new outer C/H, independent validation, deployment or forecaster seeds.',
        'Original 24-group main remains NA. Native obs8/pred12 image coordinates and',
        'silver annotations; not verified metric/seconds, true 3D or foundation.',
        'Source annotation-generation causality remains unverified. Stage5C/SMC off.','',
        '[Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)']
    (p/'README.md').write_text('\n'.join(lines)+'\n')
    gates=['# Complete-Risk Model Gates','',
        '| Gate | Status | Evidence |','|---|---|---|',
        '| Model treatment actually trained | PASS | 24 complete Torch fits |',
        '| Complete exact refits | PASS | All 24, same fixed budgets |',
        '| Known feature and label parity | PASS | Checked before fitting |',
        '| Future-label inference separation | PASS by interface | Same causal120 input only |']
    for name,m in a['policies'].items():
        gates.append(f'| {name} original complete risk + positive lower gain | {"PASS" if eligible(m) else "FAIL / NA"} | '
                     f'{m["safety"]["supported_record_views"]}/24 supported views |')
    gates+=['| Independent confirmation | NOT RUN | Already-exposed own-B |',
        '| Deployment / submission | FALSE | No independent confirmation or promotion |',
        '| Stage5C / SMC | FALSE | Not executed |']
    (p/'world_model_gate_complete_risk_donors.md').write_text('\n'.join(gates)+'\n')
    print('Complete-risk model comparison reported; original rules unchanged.')


if __name__=='__main__':main()
