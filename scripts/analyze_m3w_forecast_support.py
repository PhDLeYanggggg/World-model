"""Exact old-floor TRAIN support comparison; label-aware, never deployed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 required')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k,'4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_dimensionless_harm_forecast as run
from scripts import run_m3w_train_feasibility as old
from scripts.report_m3w_reference_harm_forecast import write_text
from scripts.report_m3w_dimensionless_harm_forecast import committed_freeze
from src.evaluation.m3w_forecast_support import reconstruct_fit_ids,aligned_predictions
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.world_model.m3w_floor_relative import matched_features
from src.world_model.m3w_fixed_floor_probe import targets
from src.evaluation.m3w_native_metrics import native_errors


def cached_bank(job,identity):
    new=run.PRIVATE/'predictions'/(job['key']+'.npz')
    doc=json.loads(new.with_suffix('.json').read_text())
    if run.digest(new)!=doc['sha256'] or doc['checkpoint']!=run.complete(job,identity)['checkpoint']:
        raise ValueError('Changed new prediction')
    ref=identity['controls'][job['key']]['prediction']
    if run.parent.artifact(ROOT/ref['path'])!=ref:raise ValueError('Changed old prediction')
    with np.load(new,allow_pickle=False) as z:ids=z['ids'].copy();newp=z['prediction'].copy()
    with np.load(ROOT/ref['path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(ids,z['ids']);oldp=z['prediction'].copy()
    np.testing.assert_array_equal(ids,job['design']['held_ids'])
    return ids,newp,oldp


def run_diagnostic(verify=False):
    start=time.monotonic();committed_freeze()
    cfg,pc,data,jobs,identity=run.load()
    reg,_,ancestor=old.prior.admission()
    completed=json.loads((old.PUBLIC/'complete.json').read_text())
    if completed['summary_sha256']!=run.digest(old.PUBLIC/'summary.json'):
        raise ValueError('Old support summary changed')
    prior={Path(r['path']).name:r for r in completed['groups']}
    rows=[];alignment=[]
    for context,items in sorted(reg['contexts'].items()):
        key=context.split('_controller')[0]
        job=next(j for j in jobs if j['key']==key)
        bank_ids,newbank,oldbank=cached_bank(job,identity)
        packets=old.fit_packets(items,ancestor,lambda **kw:run.beat('support_packet_read',**{k:v for k,v in kw.items() if k!='state'}))
        for item in items:
            source=item['packet']['identity']['source'];args=packets[source]
            ids=reconstruct_fit_ids(data,item,args)
            npred=aligned_predictions(ids,bank_ids,newbank);opred=aligned_predictions(ids,bank_ids,oldbank)
            switch=args[0][:,-1]
            if not np.isin(switch,[0.,1.]).all():raise ValueError('Frozen floor switch must be binary')
            cvroll=baseline_numpy(data['history'][ids],1)-data['origin'][ids,None]
            droll=baseline_numpy(data['history'][ids],3)-data['origin'][ids,None]
            floor=np.where(switch[:,None,None].astype(bool),droll,cvroll)
            x,env=matched_features(data['geometry'][ids],cvroll,floor,opred,switch.astype(bool))
            np.testing.assert_array_equal(x,args[0]);np.testing.assert_array_equal(env,args[1])
            cost=lambda p:native_errors(p.astype(float)+data['origin'][ids,None],data['target_eval'][ids],
                                       data['valid'][ids],np.ones(len(ids)))[0]
            floorcost=cost(floor);cv=data['baseline_ade'][ids,1]
            oy=targets(cv,floorcost,cost(opred),job['design']['easy_cut'])
            np.testing.assert_array_equal(oy,args[2])
            ny=targets(cv,floorcost,cost(npred),job['design']['easy_cut'])
            _,nenv=matched_features(data['geometry'][ids],cvroll,floor,npred,switch.astype(bool))
            eligible=old.base.shared.eligible(args)
            oldref=prior[f'{context}_{source}_seed17_fit.json']
            if run.digest(ROOT/oldref['path'])!=oldref['sha256']:raise ValueError('Old per-record oracle changed')
            olddoc=json.loads((ROOT/oldref['path']).read_text())
            records=args[5].astype(str)
            for record in sorted(set(records)):
                take=records==record
                old_oracle,_=old.api.solve(oy[take],env[take],eligible[take])
                if old_oracle!=olddoc['result']['recordings'][record]['oracle']:
                    raise ValueError('Exact old oracle replay differs')
                new_oracle,_=old.api.solve(ny[take],nenv[take],eligible[take])
                details={}
                for arm,y in (('old',oy),('new',ny)):
                    mask=take&eligible&np.isfinite(y).all(1)&(y[:,3]>0)
                    details[arm]=dict(known_easy_rows=int(mask.sum()),
                        easy_individual_harm_ratios=(y[mask,4]/y[mask,3]).tolist(),
                        minimum_easy_harm_ratio=float(np.min(y[mask,4]/y[mask,3])) if mask.any() else None)
                rows.append(dict(context=context,source=source,recording=record,
                    old=old_oracle,new=new_oracle,easy_support=details))
            alignment.append(dict(context=context,source=source,rows=len(ids),
                ids_sha256=old.api.hashlib.sha256(ids.tobytes()).hexdigest(),packet_sha256=item['packet']['sha256'],
                x_envelope_labels_reproduced_exactly=True,old_eligibility_frozen=True))
        run.beat('support_context_complete',context=context,record_views=len(rows))
    if len(rows)!=148:raise ValueError('All148required TRAIN record/context views required')
    summary=dict(result_source='fresh_run_label_aware_support_recomputation_cached_verified_original',
        required_record_context_views=148,unique_source_recordings=len({(r['source'],r['recording']) for r in rows}),
        old_budget_feasible=sum(r['old']['budget_feasible'] for r in rows),
        new_budget_feasible=sum(r['new']['budget_feasible'] for r in rows),
        old_positive_feasible=sum(r['old']['safe_positive_selection_exists'] for r in rows),
        new_positive_feasible=sum(r['new']['safe_positive_selection_exists'] for r in rows),
        fixed_views_gained=sum(not r['old']['budget_feasible'] and r['new']['budget_feasible'] for r in rows),
        fixed_views_lost=sum(r['old']['budget_feasible'] and not r['new']['budget_feasible'] for r in rows),
        old_replay_exact=True,alignment=alignment,rows=rows,
        label_aware_not_deployable=True,candidate_dependent_cost_heads_retrained=False,
        independent_roles_opened=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    run.immutable_json(run.PUBLIC/'support_diagnostic.json',summary)
    lines=['# Fixed-Floor Candidate Support','',
        'TRAIN-only label-aware diagnostic, not an inference policy or safety certificate.',
        'Exact frozen CV/damping.97floor, original eligibility and all148record/context views retained.',
        f"Old feasible views: {summary['old_budget_feasible']}/148; new: {summary['new_budget_feasible']}/148.",
        f"Views gained: {summary['fixed_views_gained']}; lost: {summary['fixed_views_lost']}.",'',
        '| Context | Record | Old feasible | New feasible | Old minimum easy H/R | New minimum easy H/R |',
        '|---|---|---|---|---:|---:|']
    for r in rows:
        if not r['old']['budget_feasible'] or not r['new']['budget_feasible']:
            lines.append(f"| {r['context']} | {r['source']}/{r['recording']} | {r['old']['budget_feasible']} | "
                f"{r['new']['budget_feasible']} | {r['easy_support']['old']['minimum_easy_harm_ratio']} | "
                f"{r['easy_support']['new']['minimum_easy_harm_ratio']} |")
    lines+=['','The unchanged budget isH/R<=0.02. Unsupported empty risk is not a safety pass.',
        'Changed forecasts need new cost heads and independent calibration; no deployment promotion.','']
    write_text(run.PUBLIC/'support_diagnostic.md','\n'.join(lines))
    runtime=dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        exact_replay=verify,summary_sha256=run.digest(run.PUBLIC/'support_diagnostic.json'))
    run.json_write(run.PRIVATE/('support_replay_runtime.json' if verify else 'support_runtime.json'),runtime)
    if verify:run.immutable_json(run.PUBLIC/'support_verification.json',runtime)
    run.beat('support_diagnostic_complete',new_feasible=summary['new_budget_feasible'],verify=verify)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    with (run.PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run_diagnostic(a.verify)


if __name__=='__main__':main()
