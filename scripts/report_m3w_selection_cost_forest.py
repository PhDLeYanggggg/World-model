"""Public-scalar matched forest report; cannot fit, retune or open confirmation."""
import json
import math
from pathlib import Path

from scripts import run_m3w_selection_cost_forest as run

ROOT = run.ROOT
KEYS = ('benefit','harm','reference','easy_reference','easy_harm')


def read_ref(ref):
    p = ROOT/ref['path']
    if not p.is_relative_to(ROOT/'outputs'):
        raise ValueError('Only public scalar receipts permitted')
    return json.loads(run.checked(ref).read_text())


def pool(cells):
    count = sum(c['known_action_views'] for c in cells)
    return dict(known_action_views=count,unknown_action_views=sum(c['unknown_action_views'] for c in cells),
        observed_known_mass={k:sum(c['observed_known_mass'][k] for c in cells) for k in KEYS},
        predicted_known_mass={k:sum(c['predicted_known_mass'][k] for c in cells) for k in KEYS},
        normalized_MSE={k:sum(c['known_action_views']*c['normalized_MSE'][k]
            for c in cells if c['known_action_views'])/count if count else None for k in KEYS})


def main():
    public = run.PUBLIC; read = lambda n:json.loads((public/n).read_text())
    cfg = json.loads(run.CONFIG.read_text()); complete = read('complete.json'); replay = read('verification.json')
    rows = [read_ref(r) for r in complete['groups']]; summary = read('summary.json')
    if (run.summarize(rows,cfg) != summary or replay['summary_sha256'] != run.sha(public/'summary.json')
            or not all(replay[k] for k in ('all18_SOURCE_cost_inferences_exact',
                'all18_selection_cost_inferences_exact','all54_metric_groups_exact','mean_control_exact'))):
        raise ValueError('Complete matched SOURCE/selection replay required before reporting')
    freeze = read('training_freeze.json'); fits = [read_ref(ref) for ref in freeze['fits']]
    if len(fits) != 18 or any(r['trees'] != 128 or r['minimum_query_visits'] < 4 or r['unknown_supervised_rows'] for r in fits):
        raise ValueError('All complete registered forests and SOURCE query support required')
    source = ROOT/'outputs/publication_readiness_2026_09/european_calibration_source_fit_v1/complete.json'
    controls = [read_ref(r) for r in json.loads(source.read_text())['cases'] if r['path'].endswith('_mean_absolute.json')]
    if len(controls) != 9 or {r['trial'] for r in controls} != {r['trial'] for r in fits}:
        raise ValueError('All nine matching original SOURCE controls required')
    pools = {'mean_absolute':{scope:pool([g['diagnosis'][scope] for c in controls for g in c['groups']])
        for scope in ('all_legal','selected')}}
    for arm in run.api.ARMS:
        pools[arm] = {scope:pool([r['SOURCE_fit_diagnosis'][scope] for r in fits if r['arm'] == arm])
            for scope in ('all_legal','selected')}
        a, b = pools[arm]['all_legal'], pools['mean_absolute']['all_legal']
        if a['known_action_views'] != b['known_action_views'] or a['unknown_action_views'] != b['unknown_action_views'] or any(
                not math.isclose(a['observed_known_mass'][k],b['observed_known_mass'][k],rel_tol=1e-10,abs_tol=1e-8) for k in KEYS):
            raise ValueError('Fixed-bank observed support and cost masses changed')
    (public/'SOURCE_cost_diagnosis.json').write_text(json.dumps(dict(
        result_source='fresh_public_scalar_aggregation_cached_verified_SOURCE_inferences',arms=pools,
        all_legal_support_and_target_mass_match=True,SOURCE_fitting_resubstitution_not_validation=True,
        selected_pools_differ_by_arm=True,control_source_freeze_sha256=run.sha(source),
        training_freeze_sha256=run.sha(public/'training_freeze.json')),indent=2,sort_keys=True)+'\n')
    number = lambda x:'N/A' if x is None else f'{x:.6g}'
    ratio = lambda a,b:None if b <= 0 else a/b
    interval = lambda r:'N/A' if r['CI95'] is None else f'{r["point"]:+.6f} [{r["CI95"][0]:+.6f}, {r["CI95"][1]:+.6f}]'
    lines = ['# Current-Bank Nonlinear Cost Control: Complete', '', '## Material Passport', '',
        'Fresh_run:18matched ExtraTrees SOURCE fits, one complete primary-arm refit,',
        '18development cost/action inferences,54metric groups and full separate replay.',
        'Cached_verified: original mean_absolute neural cost head, frozen forecasters,',
        'causal inputs, normalization, complete query-draw multiplicities and labels.',
        'No new neural/trajectory training, threshold search or independent-role opening.', '',
        '## Why This Comparison Is Matched', '',
        'All nine current producers use SOURCE folds and actual training seeds17/29/43.',
        'Each cost head trains on the original eight producer-excluded SOURCE localities.',
        'Both forests use444causal fields, action identity and causal envelope; the',
        'context arm adds fixed leave-self-out same-query means. Original query visits,',
        'legal-action/agent weighting and absolute-gain quadratic objective are preserved.',
        'The fixed forest context and post-fit physical projection differ from neural',
        'learned-hidden pooling and coupled decoding. This is not a one-factor capacity',
        'proof or matched-FLOP benchmark. The old forest experiments remain distinct.', '',
        '## Complete Development Policies', '',
        '|Estimator|Known ADE gain % [95% CI]|Mean switch %|Supported views|Known all/easy violations|Undefined easy|Worst signed easy net %|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('mean_absolute',*run.api.ARMS):
        s = summary['safety'][arm]; switch = 100*sum(r['policies'][arm]['switch_rate'] for r in rows)/len(rows)
        lines.append(f'|{arm}|{interval(summary["accuracy"][arm])}|{switch:.6f}|'
            f'{s["supported_record_views"]}/{s["required_record_views"]}|{s["known_all_risk_violations"]}/'
            f'{s["known_easy_risk_violations"]}|{s["undefined_easy_views"]}|{number(s["worst_known_easy_net_degradation_percent"])}|')
    lines += ['', 'All38,102targets/28recordings/six exposed development localities remain.',
        'The252pipeline/recording risk views repeat recordings, not independent samples.',
        'Original2%all/easy positive-harm, easy-net and finite-support rules are unchanged.',
        'Empty selected easy reference is not a safe zero. Partial/unknown labels remain.', '',
        '## Registered Contrast', '',
        '|Forest minus original mean_absolute|Known pp [95% CI]|Direct paired completion-lower pp [95% CI]|',
        '|---|---:|---:|']
    for arm in run.api.ARMS:
        lines.append(f'|{arm}|{interval(summary["effects"][arm]["known_pp"])}|'
                     f'{interval(summary["effects"][arm]["completion_lower_pp"])}|')
    lines += ['', 'Primary:context_forest FULL direct paired completion-lower contrast, not a',
        'selected best arm or the subtraction of two individual lower bounds.',
        f'Primary pass:**{summary["primary_pass"]}**. Complete original risk pass:**{summary["original_record_risk_pass"]}**.',
        f'Development candidate criterion:**{summary["candidate_pass"]}**. Deployment remains unchanged.',
        'Intervals use3,000locality resamples after source-fold/actual-seed averaging.',
        'These are conditional exposed-development intervals, not final confirmation.', '',
        '## Fixed All-Legal SOURCE Bank', '',
        '|Estimator|Known action views|H normalized MSE|EH normalized MSE|R normalized MSE|ER normalized MSE|',
        '|---|---:|---:|---:|---:|---:|']
    for arm, scopes in pools.items():
        c = scopes['all_legal']; m = c['normalized_MSE']
        lines.append(f'|{arm}|{c["known_action_views"]}|{number(m["harm"])}|{number(m["easy_harm"])}|'
                     f'{number(m["reference"])}|{number(m["easy_reference"])}|')
    lines += ['', 'All-legal observed action counts and cost masses match. These squared errors',
        'use the frozen causal scale, uniformly pooled action views; they are not the',
        'query-weighted/RMS training objective or a held-out conditional-risk estimate.', '',
        '### Selected SOURCE Regions', '',
        '|Estimator|Selected known actions|H predicted/true|EH predicted/true|Known all/easy risks %|',
        '|---|---:|---:|---:|---:|']
    for arm,scopes in pools.items():
        c = scopes['selected']; p,y = c['predicted_known_mass'],c['observed_known_mass']
        lines.append(f'|{arm}|{c["known_action_views"]}|{number(ratio(p["harm"],y["harm"]))}|'
            f'{number(ratio(p["easy_harm"],y["easy_harm"]))}|{number(ratio(100*y["harm"],y["reference"]))}/'
            f'{number(ratio(100*y["easy_harm"],y["easy_reference"]))}|')
    lines += ['', 'Selected pools differ; prediction-ratio changes cannot be treated as fixed-set',
        'error improvements. SOURCE fitting resubstitution is not generalization.', '',
        '## Compute And Recovery', '', '|Phase|PID|Seconds|Peak RSS bytes|', '|---|---:|---:|---:|']
    for name in ('pilot','training_freeze','policy_freeze','complete','verification'):
        r = read(name+'.json'); lines.append(f'|{name}|{r["pid"]}|{r["seconds"]:.6f}|{r["peak_RSS_bytes"]}|')
    lines += ['', 'The complete primary budget is2,304trees plus128first-primary refit trees.',
        'Native arm64, four fit threads, single-thread predictions, workers0; no',
        'DataLoader multiprocessing. Atomic checkpoints and the10GiB reserve remain.',
        'Pilot8->16/direct16recovery and full128tree primary refit are exact.',
        'Full18SOURCE and18development cost inferences plus54metric groups replay.', '',
        '## Reproduction And Limits', '', '```sh',
        '.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py pilot',
        '.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py train --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py infer --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py eval --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_selection_cost_forest.py verify',
        '.venv-pytorch/bin/python -m scripts.report_m3w_selection_cost_forest', '```', '',
        'Completed phases are immutable. Private SOURCE/model assets are required;',
        'the public code is not a complete cold-start anonymous reproduction package.',
        'Available-label ADE is not full12ADE. Unknown-row bounds omit missing steps',
        'inside partial futures. Local-image pixels/raw stride12/silver are not metric,',
        'seconds, human gold or physical safety. No true3D/foundation or submission-',
        'ready claim. Calibration private arrays, confirmation and DroneCrowd remain',
        'closed in this study; the older calibration exposure remains disclosed.',
        'Stage5C and SMC remain off.', '']
    (public/'readout.md').write_text('\n'.join(lines))
    (public/'gates.md').write_text('# Matched Cost-Control Gates\n\n'+'\n'.join(
        f'- {k}: {v}' for k,v in dict(complete18fits=True,original_query_coverage=True,
            exact_pilot_and_primary_refit=True,exact_SOURCE_and_selection_inferences=True,exact54metrics=True,
            registered_primary=summary['primary_pass'],complete_original_risk=summary['original_record_risk_pass'],
            development_candidate=summary['candidate_pass'],independent_confirmation=False,
            deployment_changed=False,submission_ready=False,stage5c=False,smc=False).items())+'\n')
    print(json.dumps(dict(fits=18,primary=summary['primary_pass'],risk=summary['original_record_risk_pass'])))


if __name__ == '__main__': main()
