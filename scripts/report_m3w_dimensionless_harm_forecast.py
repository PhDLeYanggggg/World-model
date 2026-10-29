"""Matched current-producer forecast comparison on opened source development."""
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

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k,'4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_dimensionless_harm_forecast as run
from scripts.report_m3w_reference_harm_forecast import write_text
from src.world_model.m3w_native_forecast import predict
from src.world_model.m3w_european_source_forecast import BASELINES
from src.evaluation.m3w_native_metrics import native_errors
from src.evaluation.m3w_reference_harm_readout import harm_summary
from src.evaluation.m3w_agent_track_refit import paired_localities


def committed_freeze():
    path=run.PUBLIC/'training_freeze.json'
    content=subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)
    if hashlib.sha256(content).hexdigest()!=run.digest(path):raise ValueError('Commit all9endpoints before readout')
    return run.digest(path)


def predictions(job,data,identity,verify=False):
    fit=run.complete(job,identity);ids=job['design']['held_ids']
    path=run.PRIVATE/'predictions'/(job['key']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    rp=path.with_suffix('.json')
    if rp.exists():
        receipt=json.loads(rp.read_text())
        if receipt['checkpoint']!=fit['checkpoint'] or receipt['sha256']!=run.digest(path):
            raise ValueError('Prediction checkpoint/cache mismatch')
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids);p=z['prediction'].copy()
    else:
        if verify:raise ValueError('Missing cached inference')
        run.beat('excluded_source_inference',trial=job['key'],rows=len(ids))
        model=run.parent.make_model(job);cp=run.saved(fit['checkpoint'])
        if cp['identity']!=fit['identity'] or cp['step']!=4000:raise ValueError('Wrong endpoint')
        model.load_state_dict(cp['model']);p=predict(model,data,ids,128)
        tmp=path.with_suffix('.tmp.npz');np.savez(tmp,ids=ids,prediction=p);os.replace(tmp,path)
        receipt=dict(checkpoint=fit['checkpoint'],sha256=run.digest(path),path=str(path.relative_to(ROOT)),
            role='previously_opened_source_development',result_source='fresh_run')
        run.immutable_json(rp,receipt)
    oldref=identity['controls'][job['key']]['prediction']
    if run.parent.artifact(ROOT/oldref['path'])!=oldref:raise ValueError('Original cache changed')
    with np.load(ROOT/oldref['path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],ids);old=z['prediction'].copy()
    return p,old,dict(trial=job['key'],new=receipt,original=oldref)


def metrics(new,old,ref,cv):
    if (new.shape!=old.shape or new.shape!=ref.shape or new.shape!=cv.shape
            or not np.array_equal(np.isnan(new),np.isnan(old))
            or not np.array_equal(np.isnan(new),np.isnan(ref))
            or not np.array_equal(np.isnan(new),np.isnan(cv))):
        raise ValueError('Matched label support required')
    known=np.isfinite(new);n,o,r,b=(a[known] for a in (new,old,ref,cv))
    ratio=lambda numerator,denominator:float(100*numerator/denominator) if denominator>0 else None
    return dict(new_gain_CV=ratio((b-n).sum(),b.sum()),old_gain_CV=ratio((b-o).sum(),b.sum()),
        new_gain_selected=ratio((r-n).sum(),r.sum()),old_gain_selected=ratio((r-o).sum(),r.sum()),
        paired_improvement_pp_CV=ratio((o-n).sum(),b.sum()),
        new_gain_old=ratio((o-n).sum(),o.sum()),
        new_positive_harm_CV=ratio(np.maximum(n-b,0).sum(),b.sum()),
        old_positive_harm_CV=ratio(np.maximum(o-b,0).sum(),b.sum()),
        new_positive_harm_selected=ratio(np.maximum(n-r,0).sum(),r.sum()),
        old_positive_harm_selected=ratio(np.maximum(o-r,0).sum(),r.sum()),
        rows=len(new),supported_rows=int(known.sum()),unknown_rows=int((~known).sum()),
        new_p95=None if not known.any() else float(np.quantile(n,.95)),
        old_p95=None if not known.any() else float(np.quantile(o,.95)),
        zero_CV_rows=int((b==0).sum()),new_zero_CV_harmed=int(((b==0)&(n>0)).sum()),
        old_zero_CV_harmed=int(((b==0)&(o>0)).sum()))


def evaluate(cfg,pc,data,jobs,identity,verify=False):
    start=time.monotonic();freeze=committed_freeze();rows=[];records=[];refs=[];baseline_rows=[]
    for j in jobs:
        new,old,receipt=predictions(j,data,identity,verify);refs.append(receipt)
        ids=j['design']['held_ids'];valid=data['valid'][ids];sites=data['sites'][ids]
        errors={a:native_errors(p.astype(float)+data['origin'][ids,None],data['target_eval'][ids],valid,np.ones(len(ids)))
                for a,p in (('new',new),('old',old))}
        cv=data['baseline_ade'][ids,1]
        easy=(cv>0)&(cv<=j['design']['easy_cut']);hard=cv>=j['design']['hard_cut']
        masks=dict(all=np.ones(len(ids),bool),easy=easy,hard=hard,complete=valid.all(1))
        for endpoint,column in (('ADE',0),('FDE',1)):
            baseline=data['baseline_ade' if column==0 else 'baseline_fde'][ids]
            ref=baseline[:,j['design']['baseline_index']];epcv=baseline[:,1]
            for site in sorted(set(sites)):
                for subset,mask in masks.items():
                    at=(sites==site)&mask
                    rows.append(dict(trial=j['key'],seed=j['old_identity']['seed'],site=str(site),endpoint=endpoint,
                        subset=subset,metric=metrics(errors['new'][column][at],errors['old'][column][at],ref[at],epcv[at])))
                for k,name in enumerate(BASELINES):
                    at=sites==site
                    baseline_rows.append(dict(trial=j['key'],seed=j['old_identity']['seed'],site=str(site),
                        endpoint=endpoint,baseline=name,metric=metrics(baseline[at,k],baseline[at,k],ref[at],epcv[at])))
        for record in sorted(set(data['recordings'][ids])):
            use=data['recordings'][ids]==record;site=str(np.unique(sites[use]).item())
            for arm in ('new','old'):
                for subset,mask in masks.items():
                    for refname,reference in (('CV',cv),('selected',data['baseline_ade'][ids,j['design']['baseline_index']])):
                        records.append(dict(trial=j['key'],seed=j['old_identity']['seed'],source=site,
                            recording=int(record),arm=arm,subset=subset,reference=refname,
                            **harm_summary(errors[arm][0],reference,use&mask)))
    roster=sorted(set(data['sites']))
    keys=['new_gain_CV','old_gain_CV','new_gain_selected','old_gain_selected','paired_improvement_pp_CV',
          'new_gain_old','new_positive_harm_CV','old_positive_harm_CV',
          'new_positive_harm_selected','old_positive_harm_selected']
    def summary(rr):
        return {k:paired_localities(rr,roster,k,cfg['bootstrap_resamples'],cfg['bootstrap_seed']) for k in keys}
    groups=sorted({(r['endpoint'],r['subset']) for r in rows})
    summaries={ep+'_'+sub:summary([r for r in rows if (r['endpoint'],r['subset'])==(ep,sub)]) for ep,sub in groups}
    seed_results={str(seed):{ep+'_'+sub:summary([r for r in rows if r['seed']==seed and (r['endpoint'],r['subset'])==(ep,sub)])
                  for ep,sub in groups} for seed in cfg['seeds']}
    baselines={ep:{name:summary([r for r in baseline_rows if r['baseline']==name and r['endpoint']==ep])
                   for name in BASELINES} for ep in ('ADE','FDE')}
    result=dict(result_source='fresh_run_training_inference_source_development',control_source='cached_verified',
        training_freeze_sha256=freeze,rows=rows,summaries=summaries,seeds=seed_results,baselines=baselines,
        prediction_receipts=refs,unique_rows=len(data['sites']),unique_recordings=len(set(data['recordings'])),
        unique_localities=len(roster),locality_producer_seed_views=72,excluded_producers_per_locality_per_seed=2,
        unknown_rows=int((~data['valid'].any(1)).sum()),complete_rows=int(data['valid'].all(1).sum()),
        independent_roles_opened=False,full_policy_budget_claim=False,deployment_changed=False,
        metric_claim=False,seconds_claim=False,stage5c_executed=False,smc_enabled=False,
        code_sha256=run.digest(Path(__file__)),readout_api_sha256=run.digest(ROOT/'src/evaluation/m3w_reference_harm_readout.py'))
    run.assert_identity(identity)
    run.immutable_json(run.PUBLIC/'analysis.json',result)
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(records[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(records)
    write_text(run.PUBLIC/'record_diagnostics.csv',buffer.getvalue())
    lines=['# Current-Producer Harm-Loss Readout','',
        'All nine fixed endpoints; source-development evidence only, not independent confirmation.',
        'Per-locality averages account for repeated producer contexts before3,000locality bootstrap draws.','',
        '| Endpoint/subset | Original vs CV (%) | New vs CV (%) | Paired difference (pp) | 95% conditional CI (pp) |',
        '|---|---:|---:|---:|---|']
    fmt=lambda x:'not_estimable' if x is None else f'{x:.6f}'
    for name,s in summaries.items():
        lines.append(f"| {name} | {fmt(s['old_gain_CV']['point'])} | {fmt(s['new_gain_CV']['point'])} | "
            f"{fmt(s['paired_improvement_pp_CV']['point'])} | {s['paired_improvement_pp_CV']['ci95']} |")
    lines+=['','Positive-harm budgets are separate from net degradation; see all per-record rows.',
        'A changed forecast invalidates inherited candidate-dependent safety scores. No deployment change.',
        'Image-pixel/raw-frame detector-track evidence, not metric, seconds, human gold, true3D or foundation.','']
    write_text(run.PUBLIC/'readout.md','\n'.join(lines))
    runtime=dict(pid=os.getpid(),seconds=time.monotonic()-start,result_source='cached_verified_recomputed' if verify else 'fresh_run',
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exact_analysis_replay=verify,
        analysis_sha256=run.digest(run.PUBLIC/'analysis.json'),record_csv_sha256=run.digest(run.PUBLIC/'record_diagnostics.csv'))
    run.json_write(run.PRIVATE/('replay_runtime.json' if verify else 'readout_runtime.json'),runtime)
    if verify:run.immutable_json(run.PUBLIC/'verification.json',runtime)
    run.beat('readout_complete',verify=verify,seconds=runtime['seconds'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true');a=p.parse_args()
    with (run.PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        cfg,pc,data,jobs,identity=run.load();evaluate(cfg,pc,data,jobs,identity,a.verify)


if __name__=='__main__':main()
