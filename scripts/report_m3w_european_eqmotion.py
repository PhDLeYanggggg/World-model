"""Readable public-only synthesis after all matched public forecasts replay."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_eqmotion as run
from src.evaluation.m3w_european_eqmotion import interval


def main():
    read=lambda name:json.loads((run.PUBLIC/name).read_text())
    cfg=json.loads(run.CONFIG.read_text());summary=read('summary.json');verified=read('verification.json')
    inference=read('inference_verification.json');training=read('training_freeze.json')
    if (not verified['all72_SOURCE_groups_exact'] or verified['summary_sha256']!=run.sha(run.PUBLIC/'summary.json')
            or not inference['all9_new_and9_control_inferences_exact']
            or verified['inference_verification_sha256']!=run.sha(run.PUBLIC/'inference_verification.json')):
        raise ValueError('All nine forecasts and all 72 SOURCE groups must replay first')
    groups=[json.loads(p.read_text()) for p in sorted((run.PUBLIC/'groups').glob('*.json'))]
    if len(groups)!=72:
        raise ValueError('Complete original SOURCE roster required')
    for field,score in summary['effects'].items():
        if interval(groups,field,summary['localities'],cfg['bootstrap_resamples'],cfg['bootstrap_seed'])!=score:
            raise ValueError('Public scalar groups do not reproduce the complete summary')
    fits=[]
    for ref in training['fits']:
        path=ROOT/ref['path']
        if run.sha(path)!=ref['sha256']:raise ValueError('Frozen public fit receipt changed')
        row=json.loads(path.read_text())
        if not row['matched_original_draws_factors_and_SOURCE_exclusions'] or row['fit']['step']!=4000:
            raise ValueError('Complete original training budget required')
        fits.append(row)
    if len(fits)!=9 or sum(r['fit']['step'] for r in fits)!=36000:
        raise ValueError('All nine actual full fits required')
    account=read('parameter_accounting.json')
    if (account['training_freeze_sha256']!=run.sha(run.PUBLIC/'training_freeze.json') or len(account['rows'])!=9
            or any(r['optimizer_step_range']!=[4000,4000] for r in account['rows'])):
        raise ValueError('Actual full-checkpoint parameter accounting required')
    counts={(r['stored_parameters'],r['requires_grad_parameters'],r['optimizer_state_parameters']) for r in account['rows']}
    if len(counts)!=1:raise ValueError('Different parameter counts require explicit per-model reporting')
    stored,flagged,updated=next(iter(counts))
    mean=lambda xs:sum(xs)/len(xs)
    sites=summary['localities']
    def site_average(fn):return mean([mean([fn(r) for r in groups if r['site']==s]) for s in sites])
    fmt=lambda d:f'{d["point"]:+.6f} [{d["CI95"][0]:+.6f}, {d["CI95"][1]:+.6f}]'
    lines=['# Public EqMotion Fixed-K1 SOURCE Comparison','',
        '## What Was Run','',
        'Fresh_run: nine full native-arm64 Torch EqMotion fits, a complete first-model',
        'refit, actual 50->100/direct-100 recovery pilot and complete SOURCE-excluded',
        'inference/replay. Cached_verified: original PCT fits, source roles and arrays.',
        'Every PCT control forecast was also rerun from its original checkpoint.','',
        'The original forecaster cohort contains 318,969 indexed rows at twelve SOURCE',
        'localities. This is not the later 149,510-row controller whitelist or the six',
        'exposed development localities. Each locality has two excluded producer folds',
        'and actual training seeds 17/29/43. Overlapping windows and repeated producer',
        'views are not independent scenes. SOURCE crossfit is not final confirmation.','',
        '## Fixed Adaptation','',
        'Official implementation: [EqMotion](https://github.com/MediaBrain-SJTU/EqMotion),',
        'MIT, pinned commit `5aec2e0b61c511fa93a24138dd90da59a089084b`. All eight pinned',
        'source files were verified before execution. Fixed head 0 emits one deterministic',
        'trajectory; no label-based best-of-20 choice, pretrained weights or author trainer.',
        'The common input conditioner uses past ego/neighbor geometry only and restores',
        'output units before loss/evaluation. Observation 8, prediction 12, raw stride 12.',
        'The original PCT has a motion-bounded output wrapper; EqMotion emits coordinates.',
        'PCT also consumes its causal baseline query, while the EqMotion core ignores it.',
        'This is a family comparison, not isolated equivariance, equal compute or published',
        'best-of-20 reproduction. No new selector or policy was fitted or deployed.','',
        '## Complete Forecast Results','',
        '|Model|Equal-locality available-label ADE gain vs CV %|',
        '|---|---:|']
    for arm in ('eqmotion','PCT'):
        lines.append(f'|{arm}|{site_average(lambda r,a=arm:r["scores"][a]["gain_vs_CV_percent"]):+.6f}|')
    for name in run.source.BASELINES:
        gain=site_average(lambda r,n=name:100*(1-r['baselines'][n]['ADE']/
            (r['scores']['CV_error_mass']/r['scores']['known_rows'])))
        lines.append(f'|{name}|{gain:+.6f}|')
    chosen=lambda r:100*(1-r['baselines'][r['training_selected_strongest']]['ADE']/
        (r['scores']['CV_error_mass']/r['scores']['known_rows']))
    lines.append(f'|Fit-selected causal baseline|{site_average(chosen):+.6f}|')
    derived=[dict(trial=r['trial'],seed=r['seed'],site=r['site'],scores=dict(
        EqMotion_vs_CV=r['scores']['eqmotion']['gain_vs_CV_percent'],
        fit_selected_vs_CV=chosen(r),
        EqMotion_minus_fit_selected=r['scores']['eqmotion']['gain_vs_CV_percent']-chosen(r))) for r in groups]
    baseline_contrasts={key:run.interval(derived,key,sites,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
        for key in ('EqMotion_vs_CV','fit_selected_vs_CV','EqMotion_minus_fit_selected')}
    lines+=['','The fit-selected baseline uses its own four fitting localities only; it',
        'need not be strongest on the excluded SOURCE groups. Fixed-baseline readout',
        'rankings do not retrospectively change the floor.',
        f'Planned known-label EqMotion-minus-fit-selected contrast pp: {fmt(baseline_contrasts["EqMotion_minus_fit_selected"])}.']
    lines+=['','|EqMotion minus PCT|Paired effect pp [95% locality CI]|','|---|---:|']
    for key,value in summary['effects'].items():lines.append(f'|{key}|{fmt(value)}|')
    lines+=['',f'Registered known-ADE primary positive: **{summary["known_primary_positive"]}**.',
        'Intervals use 3,000 locality resamples after averaging fold/actual-seed effects.',
        'Completion bounds cover wholly unknown rows only; they do not impute unobserved',
        'steps inside partial futures or identify full-12-step ADE. They share a fixed known',
        'CV-error denominator, not an inferred full-population error mass. Endpoint FDE',
        'is available only where the twelfth requested endpoint is actually observed.','',
        '## Locality Breakdown','',
        '|SOURCE locality|Known difference pp|Completion-lower pp|Completion-upper pp|',
        '|---|---:|---:|---:|']
    for site in sites:
        values=[summary['effects'][k]['by_locality'][site] for k in ('known_pp','completion_lower_pp','completion_upper_pp')]
        lines.append(f'|{site}|'+ '|'.join(f'{x:+.6f}' for x in values)+'|')
    lines+=['','## Easy, Hard And Endpoint Support','',
        '|Subset|Missing required producer/locality views|PCT gain vs CV %|EqMotion gain vs CV %|',
        '|---|---:|---:|---:|']
    for subset in ('easy','hard','complete'):
        missing=sum(r['subsets'][subset] is None for r in groups)
        vals=[]
        for arm in ('PCT','eqmotion'):
            vals.append('N/A' if missing else f'{site_average(lambda r,a=arm:r["subsets"][subset][a]["gain_vs_CV_percent"]):+.6f}')
        lines.append(f'|{subset}|{missing}|{vals[0]}|{vals[1]}|')
    lines+=['', 'Missing required subgroup support remains N/A, not a safe zero or an excuse',
        'to silently drop views. Per-group available ADE, endpoint FDE, label coverage,',
        'zero-CV harms, all six baselines and fit-selected strongest baseline are retained',
        'in the 72 public scalar group receipts. No physical or complete-policy risk',
        'certificate is inferred from this forecasting comparison.','',
        '## Compute And Exact Reproduction','',
        f'All nine models store {stored:,} parameters, with {flagged:,} gradient-enabled',
        f'parameters and {updated:,} parameters that actually have AdamW state at step 4,000.',
        'These counts are not interchangeable: frozen unused heads remain stored, and',
        'some gradient-enabled core paths receive no gradient under this fixed forward.',
        'That accounting alone identifies no unique performance failure cause.','',
        f'Primary updates: {training["primary_optimizer_updates"]}; full refit: 4,000; pilot: 200.',
        f'Full training/refit process: PID {training["pid"]}, {training["seconds"]:.3f} s,',
        f'peak RSS {training["peak_RSS_bytes"]:,} bytes. Separate inference replay:',
        f'PID {inference["pid"]}, {inference["seconds"]:.3f} s, peak {inference["peak_RSS_bytes"]:,} bytes.',
        'Original query draws, factors, fitting IDs and held-SOURCE exclusions match.',
        'All nine new and nine old model forecasts and all 72 metric groups replay.',
        'Native arm64, CPU4/inter-op1/workers0, atomic checkpoints and heartbeat.',
        'Real recovery and a full refit are reproducibility checks, not independent evidence.','',
        '```sh',
        '.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py pilot --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py train --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py infer --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py eval',
        '.venv-pytorch/bin/python scripts/run_m3w_european_eqmotion.py verify',
        '.venv-pytorch/bin/python scripts/report_m3w_european_eqmotion.py','```','',
        'Completed stage receipts are immutable. Resume incomplete stages only; do not',
        'delete a complete receipt to manufacture a fresh run. Private source arrays and',
        'checkpoints are still required, so anonymous cold-start reproduction is not complete.','',
        '## Verdict And Limits','',
        'No development, calibration, confirmation or DroneCrowd arrays were opened.',
        'The historical calibration remains completed and failed, not untouched data.',
        'No current deployment changes; the original full 2% recording-risk contract',
        'was not retuned or replaced by SOURCE forecast accuracy. This result is not a',
        'JEPA contribution, world-dynamics upgrade or submission-readiness certificate.',
        'Image-local pixels and detector/tracker silver are not metric, seconds, human',
        'gold, physical safety, true 3D or foundation evidence. Stage5C and SMC remain off.','']
    (run.PUBLIC/'readout.md').write_text('\n'.join(lines))
    (run.PUBLIC/'baseline_comparison.json').write_text(json.dumps(dict(
        result_source='cached_verified_public_scalar_summary_not_new_inference',
        contrasts=baseline_contrasts,summary_sha256=run.sha(run.PUBLIC/'summary.json'),
        verification_sha256=run.sha(run.PUBLIC/'verification.json'),
        fitting_only_baseline_selection_unchanged=True,independent_confirmation=False),indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(primary=summary['known_primary_positive'],effects=summary['effects'])))


if __name__=='__main__':main()
