"""Complete width-control report from public scalar receipts only."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_selection_capacity as run


def main():
    read=lambda n:json.loads((run.PUBLIC/n).read_text())
    summary,verified,complete=read('summary.json'),read('verification.json'),read('complete.json')
    if verified['summary_sha256']!=run.frozen.sha(run.PUBLIC/'summary.json') or not all(verified[k]
            for k in ('all9_SOURCE_inferences_and_objectives_exact','all9_development_inferences_exact','all54_metric_groups_exact','old_control_exact')):
        raise ValueError('Complete SOURCE/development replay required')
    rows=[json.loads(run.frozen.checked(ref).read_text()) for ref in complete['groups']]
    if run.summarize(rows,json.loads(run.CONFIG.read_text()))!=summary:
        raise ValueError('All54public metric groups must reproduce the summary')
    fits=[json.loads(run.frozen.checked(ref).read_text()) for ref in read('training_freeze.json')['fits']]
    if len(fits)!=9 or any(not r['original_sampler_preprocess_fit_ids_exact'] or not r['initial_SOURCE_outputs_equal_old64']
            or r['step']!=r['budget'] or r['minimum_query_visits']<4 or r['parameters']!=58501 for r in fits):
        raise ValueError('Complete exact one-factor training required')
    parent=json.loads((run.frozen.PUBLIC/'decoded_SOURCE_objective.json').read_text())
    old=[r for r in parent['rows'] if r['arm']=='mean_absolute']
    if len(old)!=9 or {r['trial'] for r in old}!={r['trial'] for r in fits}:
        raise ValueError('Matching old SOURCE control required')
    source=dict(width64=sum(r['score']['total'] for r in old)/9,width128=sum(r['SOURCE_objective']['total'] for r in fits)/9)
    interval=lambda r:'N/A' if r['CI95'] is None else f'{r["point"]:+.6f} [{r["CI95"][0]:+.6f}, {r["CI95"][1]:+.6f}]'
    lines=['# Matched Neural Cost Width: Complete','', '## Material Passport','',
        'Fresh_run: nine width-128 Torch cost fits, the first full refit, nine frozen',
        'policies, 54 metric groups and separate complete SOURCE/development replay.',
        'Cached_verified: the original width-64 mean critic, forecasters, features,',
        'roles and query exposures.',
        'No new trajectory/JEPA/Transformer training or threshold selection.','',
        '## The Single Change','',
        'The same CostHead grows from 64 to 128 hidden units: 29,253 to 58,501 parameters.',
        'Loss, AdamW, learning rate, training budget, SOURCE rows, normalization, scales,',
        'sampler/draw hashes, initial outputs and inference/risk thresholds match.',
        'Encoder initial weights have different shapes;zero output heads preserve',
        'the exact initial cost predictions. This is not a matched-compute study.',
        'All producers fit their original eight forecaster-excluded SOURCE localities.',
        'The last complete checkpoint is fixed; no epoch or width is picked on test.','',
        '## Complete SOURCE Fitting Objective','',
        f'Original width 64: {source["width64"]:.9f}. New width 128: {source["width128"]:.9f}.',
        'These are full decoded query-visit/RMS-weighted fitting objectives, not',
        'minibatch trace values or validation loss. SOURCE improvement alone is not',
        'safe selection or generalization.','',
        '![Complete width-control result](figures/width_control.svg)','',
        'Gray lines are the nine paired SOURCE producers; teal is their mean.',
        'The primary uses the complete paired lower bound, not observed ADE loss.',
        'Repeated risk views are not independent scenes; neither width passes.','',
        '## Full Development Policies','',
        '|Width|Known ADE gain % [95% CI]|Mean switch %|Supported views|All/easy violations|Undefined easy|Worst signed easy net %|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('width64','width128'):
        s=summary['safety'][arm]; switch=100*sum(r['policies'][arm]['switch_rate'] for r in rows)/54
        lines.append(f'|{arm}|{interval(summary["accuracy"][arm])}|{switch:.6f}|{s["supported_views"]}/{s["required_views"]}|'
            f'{s["known_all_violations"]}/{s["known_easy_violations"]}|{s["undefined_easy_views"]}|{s["worst_signed_easy_net_percent"]:.6f}|')
    lines+=['','## Registered Contrast','',
        '|Width128 minus width64|Paired effect pp [95% CI]|','|---|---:|']
    for k in ('known_pp','completion_lower_pp','completion_upper_pp'):
        lines.append(f'|{k}|{interval(summary["effects"][k])}|')
    lines+=['',f'Primary pass:**{summary["primary_pass"]}**. Complete original risk pass:**{summary["original_record_risk_pass"]}**.',
        f'Development candidate criterion:**{summary["candidate_pass"]}**. Deployment is not changed.',
        'All 38,102 targets, 28 recordings, six exposed development localities and 252',
        'repeated recording/pipeline risk views remain. Intervals use 3,000 locality',
        'resamples after averaging SOURCE folds and actual training seeds 17/29/43.',
        'These are not 252 independent scenes or an independent final confirmation.',
        'The original 2% all/easy positive-harm, easy-net and finite-support rules',
        'remain. Empty selected-reference risk is not a measured safe zero.','',
        '## Secondary Easy And Hard Results','',
        '|Width/subset|Equal-locality gain % [95% CI]|','|---|---:|']
    cfg=json.loads(run.CONFIG.read_text()); sites=sorted({r['site'] for r in rows})
    for arm in ('width64','width128'):
        for subset in ('easy','hard'):
            r=run.prior.metrics.interval(rows,lambda r,a=arm,s=subset:r['policies'][a]['subsets'][s]['gain_percent'],
                expected_sites=sites,resamples=cfg['bootstrap_resamples'],seed=cfg['bootstrap_seed'])
            lines.append(f'|{arm}/{subset}|{interval(r)}|')
    lines+=['','Required-view absence remains N/A, not a favorable-zero subset.','',
        '## Real Compute And Reproduction','', '|Phase|PID|Seconds|Peak RSS bytes|','|---|---:|---:|---:|']
    for phase in ('pilot','training_freeze','policy_freeze','complete','verification'):
        r=read(phase+'.json'); lines.append(f'|{phase}|{r["pid"]}|{r["seconds"]:.6f}|{r["peak_RSS_bytes"]}|')
    budget=read('training_freeze.json')
    lines+=['',f'Primary optimizer updates: {budget["primary_optimizer_updates"]}; first full refit: {budget["first_full_refit_updates"]}; pilot: 200.',
        'The real 50->100/direct-100 pilot and full first-producer model/optimizer/',
        'sampler/trace/RNG refit are exact. All nine SOURCE predictions and objectives,',
        'nine development predictions and 54 metric groups replay. Native arm64 CPU4,',
        'inter-op1, workers0, atomic checkpoints and the 10 GiB/12 h guards are unchanged.','',
        'Stage entry points are immutable after completion; do not delete receipts',
        'to pretend fresh execution. The private SOURCE/forecaster assets are still',
        'needed;this is not complete anonymous cold-start reproduction.','',
        '```sh','.venv-pytorch/bin/python scripts/run_m3w_selection_capacity.py pilot',
        '.venv-pytorch/bin/python scripts/run_m3w_selection_capacity.py train --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_selection_capacity.py infer --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_selection_capacity.py eval --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_selection_capacity.py verify',
        '.venv-pytorch/bin/python scripts/report_m3w_selection_capacity.py','```','',
        'Available-label ADE is not full-12-step ADE; unknown-row bounds do not fill steps',
        'inside partially observed futures. Pixels, raw stride 12 and silver labels: not metric,',
        'seconds, human gold, physical safety, true 3D, foundation or world-dynamics lift.',
        'No private calibration,confirmation or DroneCrowd arrays. Matched public',
        'forecasting and independent confirmation remain research gaps. Not yet',
        'submission-ready. Stage5C and SMC off.','']
    (run.PUBLIC/'readout.md').write_text('\n'.join(lines))
    (run.PUBLIC/'source_objective_comparison.json').write_text(json.dumps(dict(result_source='cached_verified_public_SOURCE_scalar_aggregation',
        values=source,training_freeze_sha256=run.frozen.sha(run.PUBLIC/'training_freeze.json'),
        parent_source_objective_sha256=run.frozen.sha(run.frozen.PUBLIC/'decoded_SOURCE_objective.json'),
        private_arrays_read=False,independent_confirmation=False),indent=2,sort_keys=True)+'\n')
    (run.PUBLIC/'gates.md').write_text('# Width-Control Gates\n\n'+'\n'.join(f'- {k}: {v}' for k,v in dict(
        complete9fits=True,matched_sampler_preprocess_initial_outputs=True,exact_pilot_and_full_refit=True,
        SOURCE_and_development_replay=True,registered_primary=summary['primary_pass'],
        complete_original_risk=summary['original_record_risk_pass'],development_candidate=summary['candidate_pass'],
        independent_confirmation=False,deployment_changed=False,submission_ready=False,stage5c=False,smc=False).items())+'\n')
    print(json.dumps(dict(primary=summary['primary_pass'],risk=summary['original_record_risk_pass'],SOURCE=source)))


if __name__=='__main__': main()
