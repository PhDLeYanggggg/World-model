"""Source-only prefix attribution; no new held-policy optimization/readout."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 required')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ.setdefault(name,'4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_role_separated_cost as parent
from src.evaluation import m3w_fitting_prefix_audit as method
np,torch=parent.np,parent.torch
read,sha,stable,once,artifact,checked=parent.read,parent.sha,parent.stable,parent.once,parent.artifact,parent.checked
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_fitting_prefix_audit_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_fitting_prefix_audit_v1'
CONFIG=ROOT/'configs/m3w_moving_fitting_prefix_audit_v1.json';REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,Path(method.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_fitting_prefix_audit.py')


def register():
    cfg=read(CONFIG);_,old=parent.register()
    if (cfg['prefixes']!=list(method.PREFIXES) or cfg['expected_forecast_banks']!=6
            or cfg['expected_agent_windows']!=125540 or cfg['risk_budget']!=.02
            or cfg['direction_bins']!=list(method.BINS) or cfg['cosine_boundaries']!=[0.,.5,.9]
            or any(cfg[k] for k in ('controller_training','new_own_B_readout','new_outer_C_H_readout',
                'new_image_admission','identity_certification','deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Fixed source-only attribution and original contract required')
    value=stable(dict(config=cfg,files={str(p.relative_to(ROOT)):sha(p) for p in FILES},jobs=old['jobs'],
        feature_names=list(method.NAMES),parent_registration=artifact(parent.REG),
        source_analysis=artifact(parent.source.PUBLIC/'analysis.json'),source_population=artifact(parent.source.PUBLIC/'population.json'),
        previous_negative=artifact(ROOT/'outputs/publication_readiness_2026_09/moving_supported_global_motion_v1/analysis.json'),
        prior_prefix_negative=artifact(ROOT/'outputs/publication_readiness_2026_09/prefix_cost_v1/conclusions.md'),
        prior_temporal_negative=artifact(ROOT/'outputs/publication_readiness_2026_09/temporal_intervention_v1/conclusions.md'),
        role='producer_A_cost_fitting_diagnostic_not_confirmation',independent_confirmation=False))
    once(REG,value);return cfg,value


def committed():
    for p in (*FILES,REG):
        b=subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if hashlib.sha256(b).hexdigest()!=sha(p):raise ValueError('Commit fixed audit before execution')


def run(reg,beat,resume=False,replay=False):
    data=parent.source.load_data();forecasts=read(parent.source.PUBLIC/'analysis.json')['forecasts']
    if len(forecasts)!=6:raise ValueError('All six fitting-only forecast banks required')
    references=[];total=0;by_fold={};cut_by_fold={}
    for fold in (0,1):
        jobs=[n['job'] for n in reg['jobs'] if n['job']['producer_fold']==fold]
        if len({(j['floor_index'],j['easy_cut'],j['hard_cut']) for j in jobs})!=1:raise ValueError('Explicit fixed producer cut required')
        cut_by_fold[fold]=jobs[0]['easy_cut']
        added=parent.added_data(fold);full=added['ids']%7==6
        by_fold[fold]=dict(ids=added['ids'][full]//7,x=added['x'][full])
    for item in forecasts:
        job=item['job'];key=job['key'];out=PUBLIC/'banks'/f'{key}.json'
        if resume and out.exists() and not replay:
            saved=read(out);references.append(saved);total+=saved['rows'];continue
        if set(job['fit_sites'])&set(job['donor_sites']):raise ValueError('Teacher exposure mismatch')
        cached=parent.load_cache(item['forecast']);ix=cached['local_ids']
        np.testing.assert_array_equal(data['global_ids'][ix],cached['global_ids'])
        if set(data['sites'][ix])!=set(job['donor_sites']):raise ValueError('All declared donor sources required')
        prediction=cached['relative_prediction'].astype(float)
        floor=parent.donor.baseline_numpy(data['history'][ix],1)-data['origin'][ix,None]
        cue=method.features(data['history'][ix],prediction,floor)
        bank=by_fold[job['producer_fold']];order=np.argsort(bank['ids']);where=order[np.searchsorted(bank['ids'][order],cached['global_ids'])]
        np.testing.assert_array_equal(bank['ids'][where],cached['global_ids'])
        existing,available=method.first_cos_from_existing120(bank['x'][where])
        match=available&cue[:,6].astype(bool)
        difference=np.abs(existing[match]-cue[match,5])
        parity=dict(comparable_rows=int(match.sum()),all_rows=len(ix),
            maximum_cosine_difference=float(difference.max()) if len(difference) else None,
            rows_above1e_4=int((difference>1e-4).sum()))
        # Causal cue construction and ID joins are complete before these label calls.
        truth=method.labels(prediction,floor,data['target_eval'][ix]-data['origin'][ix,None],data['valid'][ix])
        full_known=truth['known'][:,-1];easy=full_known&(truth['floor'][:,-1]>0)&(truth['floor'][:,-1]<=cut_by_fold[job['producer_fold']])
        group=method.direction_bin(cue[:,5],cue[:,6].astype(bool))
        weights=parent.method.storage.base.balanced_weights(data['sites'][ix],data['queries'][ix])
        rows=[]
        for site in job['donor_sites']:
            at=data['sites'][ix]==site;t={k:v[at] for k,v in truth.items()}
            shared=truth['known'][at,0]&truth['known'][at,-1]
            beneficial=shared&(truth['gain'][at,-1]>0)
            conflict=beneficial&(truth['gain'][at,0]<0)
            rows.append(dict(source=site,rows=int(at.sum()),query_count=len(np.unique(data['queries'][ix][at])),
                label_counts={str(k):int(truth['known'][at,j].sum()) for j,k in enumerate(method.PREFIXES)},
                terminal_beneficial_with_first_label=int(beneficial.sum()),terminal_beneficial_but_first_harm=int(conflict.sum()),
                summary=method.summarize(t,weights[at],easy[at],group[at])))
        value=stable(dict(job=key,producer_fold=job['producer_fold'],rows=len(ix),sources=rows,existing_basis_parity=parity,
            teacher_exposure_disjoint=True,all_inputs_before_labels=True,result_source='fresh_run_from_cached_verified_forecasts'))
        if replay:parent.neural.rel.base.exact(read(out),value)
        else:once(out,value)
        references.append(value);total+=len(ix);beat(state='source_prefix_audit_complete',job=key,rows=len(ix),total_rows=total)
    if total!=125540:raise ValueError('Retain all125540fitting rows')
    result=stable(dict(banks=references,total_rows=total,forecast_banks=6,
        original_rows_retained=True,features_already_in_raw120_all_comparable=max(r['existing_basis_parity']['rows_above1e_4'] for r in references)==0,
        result_source='fresh_run',new_training=False,new_own_B_readout=False,new_outer_C_H_readout=False,
        original_rules_unchanged=True,independent_confirmation=False,deployment_changed=False,submission_ready=False,
        stage5c_executed=False,smc_enabled=False))
    if replay:parent.neural.rel.base.exact(read(PUBLIC/'analysis.json'),result)
    else:once(PUBLIC/'analysis.json',result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('register','run','replay'));parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg,reg=register()
        if args.phase=='register':print('Source-only prefix audit registered.');return
        committed();torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.monotonic()
        def beat(**values):
            now=time.monotonic()-start
            if now>cfg['hard_runtime_limit_seconds']:raise TimeoutError('Resume same audit after12h')
            if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Keep5GiB disk reserve')
            event=dict(pid=os.getpid(),seconds=now,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,phase=args.phase,**values)
            parent.source.atomic(PRIVATE/'heartbeat.json',event)
            with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
        beat(state='ready',machine=platform.machine(),device='cpu',threads=4,workers=0)
        before={str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
        run(reg,beat,args.resume,replay=args.phase=='replay')
        if args.phase=='replay':parent.neural.rel.base.exact(before,{str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()})
        where=PRIVATE/'replays' if args.phase=='replay' else PUBLIC/'executions'
        once(where/f'{args.phase}_{time.time_ns()}_{os.getpid()}.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_training=False,phase=args.phase))
        beat(state='phase_complete')


if __name__=='__main__':main()
