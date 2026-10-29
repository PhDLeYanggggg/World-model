"""Fixed-endpoint paired source-development readout; independent roles unopened."""
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
    os.environ.setdefault(key,'4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_reference_harm_forecast as run
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_reference_harm_readout import harm_summary, compare_arms
from src.world_model.m3w_native_forecast import predict
from src.world_model.m3w_european_source_forecast import SourceForecaster, BASELINES


def committed_freeze():
    path = run.PUBLIC/'training_freeze.json'
    rel = str(path.relative_to(ROOT))
    content = subprocess.check_output(['git','show','HEAD:'+rel], cwd=ROOT)
    if hashlib.sha256(content).hexdigest() != run.digest(path):
        raise ValueError('All final endpoints must be committed before readout')
    return run.digest(path)


def predictions(key, design, ti, identity, pc, data, *, verify=False):
    fit = run.complete(key,ti,identity)
    directory = run.PRIVATE/'predictions'
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/(key+'.npz')
    receipt_path = path.with_suffix('.json')
    ids = design['held_ids']
    if receipt_path.exists():
        r = json.loads(receipt_path.read_text())
        if (r['identity'] != fit['identity'] or r['checkpoint_sha256'] != fit['checkpoint_sha256']
                or r['sha256'] != run.digest(path)):
            raise ValueError('Changed cached inference')
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            p = z['prediction'].copy()
    else:
        if verify:
            raise ValueError('Missing predictions cannot be cached_verified')
        run.beat('source_development_inference',trial=key,rows=len(ids))
        torch.manual_seed(ti['seed'])
        model = SourceForecaster(pc['architecture'],design['baseline_index'])
        cp = run.saved(ROOT/fit['checkpoint'])
        if cp['identity'] != fit['identity'] or cp['step'] != 4000:
            raise ValueError('Checkpoint endpoint mismatch')
        model.load_state_dict(cp['model'])
        p = predict(model,data,ids,128)
        temp = path.with_suffix('.tmp.npz')
        np.savez(temp,ids=ids,prediction=p)
        os.replace(temp,path)
        r = dict(identity=fit['identity'],checkpoint_sha256=fit['checkpoint_sha256'],
            path=str(path.relative_to(ROOT)),sha256=run.digest(path),
            input_role='causal_geometry_only_no_future_labels',result_source='fresh_run')
        run.immutable_json(receipt_path,r)
    old_path = run.parent.PRIVATE/'predictions'/(key+'.npz')
    old = json.loads(old_path.with_suffix('.json').read_text())
    if (old['identity'] != ti or old['sha256'] != run.digest(old_path)
            or old['checkpoint_sha256'] != identity['controls'][key]):
        raise ValueError('Original forecast cache not verified')
    with np.load(old_path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], ids)
        original = z['prediction'].copy()
    if p.shape != original.shape or not np.isfinite(p).all() or not np.isfinite(original).all():
        raise ValueError('Aligned finite forecasts required')
    return p,original,dict(trial=key,new=r['sha256'],original=old['sha256'])


def write_text(path, text):
    if path.exists() and path.read_text() != text:
        raise ValueError('Immutable analysis text differs: '+str(path))
    path.write_text(text)


def evaluate(cfg, pc, data, designs, controls, identity, verify):
    started = time.monotonic()
    freeze_sha = committed_freeze()
    n = len(data['sites'])
    cv = np.asarray(data['baseline_ade'][:,1])
    roster = sorted(identity['parent_identity']['folds'])
    seeds, trials, receipts, records = {}, {}, [], []
    def score(m,r,sites,expected,bootstrap=0):
        return paired_scene_metrics(m,r,sites,expected_scenes=expected,
            dataset='EuropeanSquares_released_detector_tracks',coordinate_unit='image_pixel',
            bootstrap_resamples=bootstrap,seed=cfg['bootstrap_seed'])
    for key, design, ti in designs:
        p,old,receipt = predictions(key,design,ti,identity,pc,data,verify=verify)
        receipts.append(receipt)
        ids = design['held_ids']
        sites = data['sites'][ids]
        valid = data['valid'][ids]
        costs = {}
        for arm,pred in (('reference_harm',p),('original_ADE',old)):
            costs[arm] = native_errors(pred.astype(float)+data['origin'][ids,None],
                data['target_eval'][ids],valid,np.ones(len(ids)))
        ref = data['baseline_ade'][ids,design['baseline_index']]
        rf = data['baseline_fde'][ids,design['baseline_index']]
        easy = (cv[ids]>0)&(cv[ids]<=design['easy_cut'])
        hard = cv[ids]>=design['hard_cut']
        tr = {}
        for arm,(ade,fde) in costs.items():
            tr[arm] = dict(ADE_vs_CV=score(ade,cv[ids],sites,sorted(set(sites))),
                ADE_vs_selected=score(ade,ref,sites,sorted(set(sites))),
                FDE_vs_selected=score(fde,rf,sites,sorted(set(sites))),
                all_cv_harm=harm_summary(ade,cv[ids]),easy_cv_harm=harm_summary(ade,cv[ids],easy),
                all_selected_harm=harm_summary(ade,ref))
            for recording in sorted(set(data['recordings'][ids])):
                use = data['recordings'][ids]==recording
                source = str(np.unique(sites[use]).item())
                for reference_name,reference in (('CV',cv[ids]),('training_selected',ref)):
                    for subset,mask in (('all',use),('easy',use&easy),('hard',use&hard)):
                        hs = harm_summary(ade,reference,mask)
                        records.append(dict(trial=key,arm=arm,source=source,recording=int(recording),
                            reference=reference_name,subset=subset,**hs))
        trials[key] = tr
        if ti['kind'] != 'complement':
            continue
        s = seeds.setdefault(str(ti['seed']),dict(seen=np.zeros(n,bool),
            reference=np.full(n,np.nan),reference_fde=np.full(n,np.nan),easy=np.zeros(n,bool),
            hard=np.zeros(n,bool),costs={arm:dict(ade=np.full(n,np.nan),fde=np.full(n,np.nan)) for arm in costs}))
        if s['seen'][ids].any():
            raise ValueError('Repeated complement rows')
        s['seen'][ids]=True
        s['reference'][ids],s['reference_fde'][ids]=ref,rf
        s['easy'][ids],s['hard'][ids]=easy,hard
        for arm,(ade,fde) in costs.items():
            s['costs'][arm]['ade'][ids]=ade
            s['costs'][arm]['fde'][ids]=fde
    def population(s):
        if not s['seen'].all():
            raise ValueError('Incomplete complement source readout')
        result = {}
        for arm,e in s['costs'].items():
            ade,fde=e['ade'],e['fde']
            result[arm]=dict(ADE_vs_CV=score(ade,cv,data['sites'],roster,cfg['bootstrap_resamples']),
                ADE_vs_selected=score(ade,s['reference'],data['sites'],roster,cfg['bootstrap_resamples']),
                FDE_vs_selected=score(fde,s['reference_fde'],data['sites'],roster,cfg['bootstrap_resamples']),
                hard_ADE_vs_CV=score(ade[s['hard']],cv[s['hard']],data['sites'][s['hard']],roster,cfg['bootstrap_resamples']),
                easy_ADE_vs_CV=score(ade[s['easy']],cv[s['easy']],data['sites'][s['easy']],roster,cfg['bootstrap_resamples']),
                complete_ADE_vs_CV=score(ade[data['valid'].all(1)],cv[data['valid'].all(1)],
                    data['sites'][data['valid'].all(1)],roster,cfg['bootstrap_resamples']),
                all_cv_harm=harm_summary(ade,cv), easy_cv_harm=harm_summary(ade,cv,s['easy']),
                all_selected_harm=harm_summary(ade,s['reference']))
        result['new_minus_original_vs_CV_pp']=compare_arms(s['costs']['reference_harm']['ade'],
            s['costs']['original_ADE']['ade'],cv,data['sites'],roster,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
        return result
    seed_results={seed:population(s) for seed,s in seeds.items()}
    first=next(iter(seeds.values()))
    for s in seeds.values():
        np.testing.assert_array_equal(s['easy'],first['easy'])
        np.testing.assert_array_equal(s['hard'],first['hard'])
    mean=dict(first,reference=np.mean([s['reference'] for s in seeds.values()],axis=0),
              reference_fde=np.mean([s['reference_fde'] for s in seeds.values()],axis=0),
              costs={arm:{metric:np.mean([s['costs'][arm][metric] for s in seeds.values()],axis=0)
                            for metric in ('ade','fde')} for arm in first['costs']})
    mean_results=population(mean)
    original_path=run.parent.PUBLIC/'analysis.json'
    original=json.loads(original_path.read_text())
    for key,oldkey in (('ADE_vs_CV','mean_seed_ADE_vs_CV'),
                       ('ADE_vs_selected','mean_seed_ADE_vs_training_selected_strongest'),
                       ('FDE_vs_selected','mean_seed_FDE_vs_training_selected_strongest')):
        if mean_results['original_ADE'][key]!=original[oldkey]:
            raise ValueError('Original primary metric replay differs: '+key)
    baselines={name:score(data['baseline_ade'][:,k],cv,data['sites'],roster,cfg['bootstrap_resamples'])
               for k,name in enumerate(BASELINES)}
    result=dict(result_source='fresh_run_training_and_source_development_inference',
        comparator_source='cached_verified',freeze_sha256=freeze_sha,source_rows=n,
        unique_recordings=len(set(data['recordings'])),source_localities=len(roster),
        supported_rows=int(np.isfinite(cv).sum()),unknown_future_rows=int(np.isnan(cv).sum()),
        complete_future_rows=int(data['valid'].all(1).sum()),prediction_receipts=receipts,
        original_primary_replay_exact=True,original_analysis_sha256=run.digest(original_path),
        trials=trials,seeds=seed_results,mean_seed=mean_results,baselines=baselines,
        public_readout_bindings={p:run.digest(ROOT/p) for p in (
            'scripts/report_m3w_reference_harm_forecast.py','src/evaluation/m3w_reference_harm_readout.py',
            'tests/test_m3w_reference_harm_readout.py')},
        independent_roles_opened=False,previously_opened_development=True,
        full_policy_budget_claim=False,deployment_changed=False,submission_ready=False,
        metric_claim=False,seconds_claim=False,stage5c_executed=False,smc_enabled=False)
    run.assert_identity(identity)
    run.immutable_json(run.PUBLIC/'analysis.json',result)
    buf=io.StringIO()
    writer=csv.DictWriter(buf,fieldnames=list(records[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(records)
    write_text(run.PUBLIC/'record_diagnostics.csv',buf.getvalue())
    summary=['# Reference-Harm Forecast Readout','',
        'Source development only, not independent confirmation. All 18 endpoints fixed before readout.',
        '8 observed / 12 predicted steps; stride 12 raw frames; image pixels, detector-track labels.','',
        '| Seed | Arm | ADE vs CV (%) | ADE vs train-selected (%) | Easy net degradation vs CV (%) | Easy positive harm vs CV (%) |',
        '|---|---|---:|---:|---:|---:|']
    for seed,s in [*seed_results.items(),('mean_seed',mean_results)]:
        for arm in ('original_ADE','reference_harm'):
            r=s[arm];h=r['easy_cv_harm']
            summary.append(f"| {seed} | {arm} | {r['ADE_vs_CV']['equal_scene_gain_percent']:.6f} | "
                f"{r['ADE_vs_selected']['equal_scene_gain_percent']:.6f} | "
                f"{h['known_net_degradation_percent']:.6f} | {h['known_positive_harm_percent']:.6f} |")
    delta=mean_results['new_minus_original_vs_CV_pp']
    summary += ['',f"Mean seed paired locality difference: {delta['equal_locality_gain_difference_pp']:.6f} pp; "
        f"3,000 locality-bootstrap interval {delta['paired_locality_bootstrap_ci95_pp']}.",
        'This interval is conditional exploratory uncertainty, not independent confirmation.',
        'Positive harm and net degradation are separate. No full-policy safety or deployment claim.',
        'Missing future labels and zero reference percentages remain explicit; every recording is retained.',
        'No Stage5C, SMC, metric, seconds, true-3D or foundation-model claim.','']
    write_text(run.PUBLIC/'readout.md','\n'.join(summary))
    runtime=dict(result_source='cached_verified_recomputed' if verify else 'fresh_run',pid=os.getpid(),
        seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        exact_analysis_replay=verify,analysis_sha256=run.digest(run.PUBLIC/'analysis.json'),
        record_csv_sha256=run.digest(run.PUBLIC/'record_diagnostics.csv'),new_gradient_updates=0)
    run.json_write(run.PRIVATE/('replay_runtime.json' if verify else 'readout_runtime.json'),runtime)
    if verify:
        run.immutable_json(run.PUBLIC/'verification.json',runtime)
    run.beat('readout_complete',verify=verify,**{k:runtime[k] for k in ('seconds','analysis_sha256')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    with (run.PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        cfg,pc,data,designs,controls,identity=run.load()
        evaluate(cfg,pc,data,designs,controls,identity,args.verify)


if __name__=='__main__':
    main()
