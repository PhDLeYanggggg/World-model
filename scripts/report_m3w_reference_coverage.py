"""Frozen-endpoint three-arm forecast readout on opened source development."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_reference_coverage as run
from scripts.report_m3w_dimensionless_harm_forecast import metrics
from scripts.report_m3w_reference_harm_forecast import write_text
from src.world_model.m3w_native_forecast import predict
from src.evaluation.m3w_native_metrics import native_errors
from src.evaluation.m3w_reference_harm_readout import harm_summary
from src.evaluation.m3w_reference_coverage_readout import replay_control_metrics, budget_counts
from src.evaluation.m3w_agent_track_refit import paired_localities


def committed_freeze():
    path = run.PUBLIC/'training_freeze.json'
    content = subprocess.check_output(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT)
    if hashlib.sha256(content).hexdigest() != run.digest(path):
        raise ValueError('Commit all nine endpoints before first scoring')
    return run.digest(path)


def bank(ref, ids):
    if run.artifact(ROOT/ref['path']) != ref:
        raise ValueError('Frozen comparator prediction changed')
    with np.load(ROOT/ref['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], ids)
        prediction = z['prediction'].copy()
    if not np.isfinite(prediction).all():
        raise ValueError('Finite frozen forecasts required')
    return prediction


def predictions(job, data, identity, verify=False):
    fit = run.complete(job, identity); ids = job['design']['held_ids']
    path = run.PRIVATE/'predictions'/(job['key']+'.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    rp = path.with_suffix('.json')
    if rp.exists():
        receipt = json.loads(rp.read_text())
        if receipt['checkpoint'] != fit['checkpoint'] or receipt['sha256'] != run.digest(path):
            raise ValueError('Changed current inference cache')
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            new = z['prediction'].copy()
    else:
        if verify:
            raise ValueError('Cached verification requires existing inference')
        run.beat('excluded_source_inference', trial=job['key'], rows=len(ids))
        model = run.make_model(job)
        cp = run.saved(fit['checkpoint'])
        if cp['identity'] != fit['identity'] or cp['step'] != 4000:
            raise ValueError('Wrong frozen endpoint')
        model.load_state_dict(cp['model'])
        new = predict(model, data, ids, 128)
        tmp = path.with_suffix('.tmp.npz')
        np.savez(tmp, ids=ids, prediction=new); os.replace(tmp, path)
        receipt = dict(checkpoint=fit['checkpoint'], sha256=run.digest(path), path=str(path.relative_to(ROOT)),
            result_source='fresh_native_torch_inference', role='previously_opened_source_development')
        run.immutable_json(rp, receipt)
    refs = identity['controls'][job['key']]
    harm = bank(refs['prediction'], ids)
    original = bank(refs['ADE_only_prediction'], ids)
    if fit['radius_unchanged']:
        np.testing.assert_array_equal(new, harm)
    if new.shape != harm.shape or not np.isfinite(new).all() or fit['fit']['parameters'] != 88514:
        raise ValueError('Matched finite parameter-budget predictions required')
    return dict(new=new, harm_control=harm, original_ADE=original), dict(trial=job['key'],
        new=receipt, harm_control=refs['prediction'], original_ADE=refs['ADE_only_prediction'],
        unchanged_radius_predictions_exact=fit['radius_unchanged'])


def evaluate(cfg, pc, data, jobs, identity, verify=False):
    started = time.monotonic(); freeze_sha = committed_freeze()
    rows, records, receipts = [], [], []
    for job in jobs:
        pred, receipt = predictions(job, data, identity, verify); receipts.append(receipt)
        ids = job['design']['held_ids']; sites = data['sites'][ids]; valid = data['valid'][ids]
        errors = {arm:native_errors(p.astype(float)+data['origin'][ids, None], data['target_eval'][ids],
                                   valid, np.ones(len(ids))) for arm,p in pred.items()}
        cv = data['baseline_ade'][ids, 1]
        masks = dict(all=np.ones(len(ids), bool), easy=(cv > 0)&(cv <= job['design']['easy_cut']),
                     hard=cv >= job['design']['hard_cut'], complete=valid.all(1))
        for endpoint, column in (('ADE', 0), ('FDE', 1)):
            baseline = data['baseline_ade' if column == 0 else 'baseline_fde'][ids]
            reference = baseline[:, job['design']['baseline_index']]; ecv = baseline[:, 1]
            for site in sorted(set(sites)):
                for subset, mask in masks.items():
                    take = (sites == site)&mask
                    arguments = errors['new'][column][take], reference[take], ecv[take]
                    rows.append(dict(trial=job['key'], seed=job['old_identity']['seed'], site=str(site),
                        endpoint=endpoint, subset=subset,
                        metric=metrics(arguments[0], errors['harm_control'][column][take], *arguments[1:]),
                        original_ADE_contrast=metrics(arguments[0], errors['original_ADE'][column][take], *arguments[1:])))
        for recording in sorted(set(data['recordings'][ids])):
            use = data['recordings'][ids] == recording
            site = str(np.unique(sites[use]).item())
            for arm in pred:
                for subset, mask in masks.items():
                    for name, reference in (('CV', cv),
                            ('selected', data['baseline_ade'][ids, job['design']['baseline_index']])):
                        records.append(dict(trial=job['key'], seed=job['old_identity']['seed'], source=site,
                            recording=int(recording), arm=arm, subset=subset, reference=name,
                            **harm_summary(errors[arm][0], reference, use&mask)))
    previous_path = run.parent.PUBLIC/'analysis.json'
    previous = json.loads(previous_path.read_text())
    checks = replay_control_metrics(rows, previous)
    roster = sorted(set(data['sites']))
    keys = ('new_gain_CV', 'old_gain_CV', 'new_gain_selected', 'old_gain_selected', 'paired_improvement_pp_CV',
            'new_gain_old', 'new_positive_harm_CV', 'old_positive_harm_CV',
            'new_positive_harm_selected', 'old_positive_harm_selected')
    groups = sorted({(r['endpoint'], r['subset']) for r in rows})

    def summarize(selected, contrast='metric'):
        rr = [dict(row, metric=row[contrast]) for row in selected]
        return {key:paired_localities(rr, roster, key, cfg['bootstrap_resamples'], cfg['bootstrap_seed']) for key in keys}

    summaries = {ep+'_'+sub:summarize([r for r in rows if (r['endpoint'], r['subset']) == (ep, sub)])
                 for ep,sub in groups}
    secondary = {ep+'_'+sub:summarize([r for r in rows if (r['endpoint'], r['subset']) == (ep, sub)],
                                    'original_ADE_contrast') for ep,sub in groups}
    seeds = {str(seed):{ep+'_'+sub:summarize([r for r in rows if r['seed'] == seed and
                                (r['endpoint'], r['subset']) == (ep, sub)]) for ep,sub in groups}
             for seed in cfg['seeds']}
    risks = budget_counts(records)
    result = dict(result_source='fresh_run_native_training_and_inference_source_development',
        controls_source='cached_verified', training_freeze_sha256=freeze_sha, rows=rows,
        summaries=summaries, original_ADE_contrast=secondary, seeds=seeds, recording_diagnostics=risks,
        prediction_receipts=receipts, exact_control_metric_checks=checks,
        original_analysis_sha256=run.digest(previous_path), baselines=previous['baselines'],
        unique_rows=len(data['sites']), unique_recordings=len(set(data['recordings'])),
        unique_localities=len(roster), unknown_future_rows=int((~data['valid'].any(1)).sum()),
        complete_future_rows=int(data['valid'].all(1).sum()), locality_producer_seed_views=72,
        neutral_exact_model_fits=sum(run.complete(j, identity)['radius_unchanged'] for j in jobs),
        affected_model_fits=sum(not run.complete(j, identity)['radius_unchanged'] for j in jobs),
        code_sha256=run.digest(Path(__file__)),
        api_sha256=run.digest(ROOT/'src/evaluation/m3w_reference_coverage_readout.py'),
        independent_roles_opened=False, deployment_changed=False, full_policy_budget_claim=False,
        metric_claim=False, seconds_claim=False, stage5c_executed=False, smc_enabled=False)
    run.assert_identity(identity)
    run.immutable_json(run.PUBLIC/'analysis.json', result)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(records)
    write_text(run.PUBLIC/'record_diagnostics.csv', buffer.getvalue())
    lines = ['# Causal Reference-Coverage Readout', '',
        'All nine endpoints fixed before scoring. Source development only, not independent confirmation.',
        'Six unchanged-radius fits are explicit neutral controls; repeated seed/context errors are averaged within locality.', '',
        '| Endpoint | Harm-aware control vs CV (%) | Coverage vs CV (%) | Primary change (pp) | Conditional 95% CI (pp) | Change vs original ADE-only (pp) |',
        '|---|---:|---:|---:|---|---:|']
    fmt = lambda value:'not_estimable' if value is None else f'{value:.6f}'
    for name, row in summaries.items():
        primary = row['paired_improvement_pp_CV']
        lines.append(f"| {name} | {fmt(row['old_gain_CV']['point'])} | {fmt(row['new_gain_CV']['point'])} | "
            f"{fmt(primary['point'])} | {primary['ci95']} | "
            f"{fmt(secondary[name]['paired_improvement_pp_CV']['point'])} |")
    lines += ['', f"Original comparator metric checks: {checks}; every fixed endpoint/locality/subset retained.",
        'All 163 recordings, including missing-label and zero-reference cases, remain in the diagnostic CSV.',
        'Known positive harm is separate from net error. Forecast diagnostics cannot certify the complete policy.',
        'Changed forecasts invalidate inherited candidate-dependent safety heads. No deployment promotion.',
        'Image pixels/raw frames and released detector tracks only; Stage5C/SMC remain off.', '']
    write_text(run.PUBLIC/'readout.md', '\n'.join(lines))
    runtime = dict(pid=os.getpid(), seconds=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        result_source='cached_verified_recomputed' if verify else 'fresh_run', exact_analysis_replay=verify,
        analysis_sha256=run.digest(run.PUBLIC/'analysis.json'),
        record_csv_sha256=run.digest(run.PUBLIC/'record_diagnostics.csv'))
    run.json_write(run.PRIVATE/('replay_runtime.json' if verify else 'readout_runtime.json'), runtime)
    if verify:
        run.immutable_json(run.PUBLIC/'verification.json', runtime)
    run.beat('readout_complete', verify=verify, seconds=runtime['seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    with (run.PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        cfg, pc, data, jobs, identity = run.load()
        evaluate(cfg, pc, data, jobs, identity, args.verify)


if __name__ == '__main__':
    main()
