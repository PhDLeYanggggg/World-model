"""Pinned deterministic public-family control on the original European SOURCE jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_source_forecast as source
from src.world_model.m3w_eqmotion_adapter import verified_source
from src.world_model.m3w_native_forecast import fit_trial,predict
from src.world_model.m3w_european_source_forecast import SourceForecaster
from src.world_model.m3w_multiaction_cost import exact
from src.evaluation.m3w_european_eqmotion import paired_group,interval
from src.evaluation.m3w_native_metrics import native_errors

NAME='european_eqmotion_v1'
PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG=ROOT/'configs/m3w_european_eqmotion_v1.json'
REG=PUBLIC/'registration.json'
sha,once,write=source.digest,source.immutable_json,source.json_write
FILES=(Path(__file__),CONFIG,ROOT/'src/evaluation/m3w_european_eqmotion.py',
    ROOT/'tests/test_m3w_european_eqmotion.py',PUBLIC/'protocol.md',
    ROOT/'src/world_model/m3w_eqmotion_adapter.py',ROOT/'configs/m3w_eqmotion_source.json')


def reference_matching(state,reference,design,steps,batch_size):
    for key in ('train_ids','factors','draws','sampler_rng'):
        exact(state[key],reference[key])
    exact(state['train_ids'],design['train_ids'])
    exact(state['factors'],design['factors'])
    if (state['step']!=steps or reference['step']!=steps or
            int(state['draws'].sum())!=steps*batch_size or state['draws'][design['held_ids']].any()):
        raise ValueError('Original full budget, SOURCE exclusions and query draws required')


def scientific_state(left,right):
    for key in left:
        if key!='seconds': exact(left[key],right[key])


def admission(register=False):
    cfg=json.loads(CONFIG.read_text())
    old,manifest,receipts,identity=source.load_source()
    if (cfg['training']!=old['training'] or cfg['seeds']!=[17,29,43] or cfg['kind']!='single'
            or cfg['new_models']!=9 or cfg['architecture']['fixed_head']!=0
            or not cfg['architecture']['prune_unused_heads']):
        raise ValueError('The complete fixed nine-job public-family design is required')
    for key in ('development_arrays_opened','calibration_arrays_opened','confirmation_opened',
            'deployment_changed','stage5c_executed','smc_enabled'):
        if cfg[key]: raise ValueError('No new roles, deployment or generative execution')
    data=source.prepare(manifest,receipts,identity)
    jobs=[(k,d,t) for k,d,t in source.trials(old,data,identity) if t['kind']=='single']
    if len(jobs)!=9: raise ValueError('All original single-fold jobs are required')
    _,author=verified_source()
    controls={key:source.complete(key,ti,old) for key,_,ti in jobs}
    analysis=source.PUBLIC/'analysis.json'
    saved=json.loads(analysis.read_text())
    predictions={r['trial']:r for r in saved['prediction_receipts']}
    for key,_,_ in jobs:
        if sha(ROOT/predictions[key]['path'])!=predictions[key]['sha256']:
            raise ValueError('Original PCT prediction bytes changed')
    registration=dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_identity=identity,parent_analysis_sha256=sha(analysis),
        packed_source_receipt_sha256=sha(source.PRIVATE/'packed/receipt.json'),
        author=author,source_rows=len(data['sites']),source_localities=sorted(set(data['sites'])),
        jobs=[dict(trial=k,identity=t,checkpoint_sha256=controls[k]['checkpoint_sha256'],
            prediction=predictions[k]) for k,_,t in jobs],
        matched_public_family_not_isolated_equivariance=True,published_best_of20_reproduction=False,
        primary=cfg['primary'],new_models=9,primary_optimizer_updates=36000,
        first_full_refit_updates=4000,pilot_updates=200,
        development_or_private_independent_arrays_read=False)
    once(REG,registration)
    if not register:
        for path in (*FILES,REG):
            source_path=str(path.relative_to(ROOT))
            import subprocess
            subprocess.run(['git','ls-files','--error-unmatch',source_path],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
            subprocess.run(['git','diff','--quiet','HEAD','--',source_path],cwd=ROOT,check=True)
    return cfg,old,data,jobs,controls,predictions


def runtime(cfg):
    torch.set_num_threads(cfg['torch_threads']);torch.set_num_interop_threads(cfg['interop_threads'])
    start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume atomic EqMotion checkpoint')))
    def beat(**values):
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12-hour boundary; resume the same complete design')
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:
            raise OSError('Preserve 10 GiB reserve; no budget reduction or data deletion')
        if sum(p.stat().st_size for p in PRIVATE.rglob('checkpoint.pt'))>cfg['checkpoint_cap_bytes']:
            raise OSError('Registered checkpoint storage cap exceeded')
        event=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values)
        write(PRIVATE/'heartbeat.json',event)
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    beat(state='runtime_ready',device='cpu',threads=4,workers=0,architecture=platform.machine(),resource_probing=False)
    return start,beat


def fit(cfg,data,job,directory,beat,resume=False,stop_at=None):
    key,design,old_identity=job
    torch.manual_seed(old_identity['seed'])
    model=SourceForecaster(cfg['architecture'],design['baseline_index'])
    identity=dict(registration_sha256=sha(REG),parent_trial=old_identity)
    result=fit_trial(model,data,design,seed=old_identity['seed'],settings=cfg['training'],identity=identity,
        directory=directory,resume=resume,stop_at=stop_at,heartbeat=lambda **kw:beat(trial=key,**kw))
    state=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
    return result,state


def pilot(cfg,old,data,jobs,controls,predictions,resume):
    if (PUBLIC/'pilot.json').exists():raise ValueError('Completed pilot is immutable')
    start,beat=runtime(cfg);job=jobs[0];home=PRIVATE/'pilot';direct=PRIVATE/'pilot_direct'
    if not (home/'checkpoint.pt').exists():fit(cfg,data,job,home,beat,stop_at=cfg['pilot_resume_at'])
    elif not resume:raise ValueError('Existing pilot requires --resume')
    result,state=fit(cfg,data,job,home,beat,resume=True,stop_at=cfg['pilot_steps'])
    _,other=fit(cfg,data,job,direct,beat,resume=resume and (direct/'checkpoint.pt').exists(),stop_at=cfg['pilot_steps'])
    scientific_state(state,other)
    # The old 4,000-step draw sequence has this exact prefix, independently of architecture RNG.
    from src.world_model.m3w_native_forecast import draw_batch
    rng=torch.Generator().manual_seed(job[2]['seed']+7919);draws=np.zeros(len(data['sites']),np.int64)
    for _ in range(cfg['pilot_steps']):np.add.at(draws,draw_batch(job[1]['groups'],cfg['training']['batch_size'],rng),1)
    exact(state['draws'],draws);exact(state['sampler_rng'],rng.get_state())
    once(PUBLIC/'pilot.json',dict(result_source='fresh_run_actual_SOURCE_training_recovery',pid=os.getpid(),
        seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        real_optimizer_updates=200,exact_resume_direct_state=True,original_query_prefix_exact=True,
        final100step_fit_seconds=result['seconds'],
        projected9primary_plus_first_full_refit_seconds=result['seconds']*40*10,
        projection_excludes_preparation_inference_and_replay=True,engineering_pass=True,
        deployment_changed=False))


def train(cfg,old,data,jobs,controls,predictions,resume):
    if (PUBLIC/'training_freeze.json').exists():raise ValueError('Complete training freeze is immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Actual SOURCE recovery/profile pilot required')
    start,beat=runtime(cfg);refs=[]
    for job in jobs:
        key,design,ti=job;home=PRIVATE/'trials'/key;path=PUBLIC/'fits'/(key+'.json')
        if path.exists():
            if not resume:raise ValueError('Completed fits require explicit --resume')
            row=json.loads(path.read_text())
            if (row['registration_sha256']!=sha(REG) or row['parent_trial']!=ti or row['fit']['step']!=4000
                    or sha(ROOT/row['checkpoint'])!=row['checkpoint_sha256']):
                raise ValueError('Completed checkpoint/role identity differs')
        else:
            result,state=fit(cfg,data,job,home,beat,resume=resume and (home/'checkpoint.pt').exists())
            ref=torch.load(ROOT/controls[key]['checkpoint'],map_location='cpu',weights_only=False)
            reference_matching(state,ref,design,4000,64)
            if key==jobs[0][0]:
                again=PRIVATE/'full_refit'/key
                _,other=fit(cfg,data,job,again,beat,resume=resume and (again/'checkpoint.pt').exists())
                scientific_state(state,other)
                once(PUBLIC/'first_full_refit.json',dict(trial=key,updates=4000,complete_scientific_state_exact=True,
                    checkpoint_sha256=sha(home/'checkpoint.pt'),refit_checkpoint_sha256=sha(again/'checkpoint.pt')))
            row=dict(trial=key,parent_trial=ti,fit=result,checkpoint=str((home/'checkpoint.pt').relative_to(ROOT)),
                checkpoint_sha256=sha(home/'checkpoint.pt'),registration_sha256=sha(REG),
                matched_original_draws_factors_and_SOURCE_exclusions=True,result_source='fresh_native_arm64_Torch_training')
            once(path,row)
            del state,ref
        refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
        beat(state='complete_endpoint',trial=key,models=len(refs),total=9)
    if not json.loads((PUBLIC/'first_full_refit.json').read_text())['complete_scientific_state_exact']:
        raise ValueError('First full refit required')
    once(PUBLIC/'training_freeze.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,fits=refs,
        primary_optimizer_updates=36000,first_full_refit_updates=4000,pilot_updates=200,
        registration_sha256=sha(REG),deployment_changed=False,private_independent_arrays_read=False))


def inference(cfg,old,data,jobs,controls,predictions,resume,verify=False):
    freeze=PUBLIC/'prediction_freeze.json'
    if freeze.exists() and not verify:raise ValueError('Completed prediction freeze is immutable')
    training=json.loads((PUBLIC/'training_freeze.json').read_text())
    if training['registration_sha256']!=sha(REG) or len(training['fits'])!=9:
        raise ValueError('All nine fixed training endpoints required')
    start,beat=runtime(cfg);refs=[]
    for job,ref in zip(jobs,training['fits']):
        key,design,ti=job;rp=ROOT/ref['path'];row=json.loads(rp.read_text());ids=design['held_ids']
        if sha(rp)!=ref['sha256'] or row['trial']!=key or sha(ROOT/row['checkpoint'])!=row['checkpoint_sha256']:
            raise ValueError('Frozen endpoint identity changed')
        torch.manual_seed(ti['seed']);model=SourceForecaster(cfg['architecture'],design['baseline_index'])
        model.load_state_dict(torch.load(ROOT/row['checkpoint'],map_location='cpu',weights_only=False)['model'])
        path=PRIVATE/'predictions'/(key+'.npz');receipt=PUBLIC/'predictions'/(key+'.json')
        if path.exists() and not (resume or verify):raise ValueError('Partial inference requires --resume')
        p=predict(model,data,ids,cfg['inference_batch_size'])
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['prediction'],p)
        else:
            if verify:raise ValueError('Cannot replay missing predictions')
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
            np.savez(tmp,ids=ids,prediction=p);os.replace(tmp,path)
        control=SourceForecaster(old['architecture'],design['baseline_index'])
        control.load_state_dict(torch.load(ROOT/controls[key]['checkpoint'],map_location='cpu',weights_only=False)['model'])
        q=predict(control,data,ids,cfg['inference_batch_size'])
        with np.load(ROOT/predictions[key]['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['prediction'],q)
        once(receipt,dict(trial=key,prediction=str(path.relative_to(ROOT)),sha256=sha(path),rows=len(ids),
            ids_sha256=source.array_hash(ids),original_PCT_fresh_inference_exact=True,checkpoint_sha256=row['checkpoint_sha256']))
        refs.append(dict(path=str(receipt.relative_to(ROOT)),sha256=sha(receipt)))
        beat(state='prediction_replayed' if verify else 'prediction_frozen',trial=key,rows=len(ids))
    result=dict(registration_sha256=sha(REG),training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
        predictions=refs,all9_control_forecasts_exact=True,all_endpoints_before_scoring=True)
    once(freeze,result)
    if verify:
        once(PUBLIC/'inference_verification.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,all9_new_and9_control_inferences_exact=True,
            prediction_freeze_sha256=sha(freeze),independent_confirmation=False))


def evaluate(cfg,old,data,jobs,controls,predictions,verify=False):
    start=time.monotonic();frozen=json.loads((PUBLIC/'prediction_freeze.json').read_text());groups=[]
    if frozen['registration_sha256']!=sha(REG) or len(frozen['predictions'])!=9:
        raise ValueError('Complete pre-scoring prediction freeze required')
    for job,ref in zip(jobs,frozen['predictions']):
        key,design,ti=job;rp=ROOT/ref['path'];r=json.loads(rp.read_text());ids=design['held_ids']
        if sha(rp)!=ref['sha256'] or r['trial']!=key or sha(ROOT/r['prediction'])!=r['sha256']:
            raise ValueError('Prediction receipt changed')
        with np.load(ROOT/r['prediction'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids);p=z['prediction'].astype(float)
        with np.load(ROOT/predictions[key]['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids);q=z['prediction'].astype(float)
        y=np.asarray(data['target_eval'][ids])-data['origin'][ids,None]
        valid=np.asarray(data['valid'][ids]);sites=np.asarray(data['sites'][ids]);cv=np.asarray(data['baseline_ade'][ids,1])
        for site in sorted(set(sites)):
            ix=sites==site;score=paired_group(p[ix],q[ix],y[ix],valid[ix],cv[ix])
            subsets={}
            masks=dict(easy=(cv>0)&(cv<=design['easy_cut']),hard=cv>=design['hard_cut'],complete=valid.all(1))
            for name,mask in masks.items():
                use=ix&mask
                subsets[name]=paired_group(p[use],q[use],y[use],valid[use],cv[use]) if use.any() and cv[use].sum()>0 else None
            baselines={name:dict(ADE=float(np.nanmean(data['baseline_ade'][ids[ix],k])),
                FDE=float(np.nanmean(data['baseline_fde'][ids[ix],k]))) for k,name in enumerate(source.BASELINES)}
            row=dict(trial=key,seed=ti['seed'],site=site,scores=score,subsets=subsets,baselines=baselines,
                training_selected_strongest=source.BASELINES[design['baseline_index']])
            path=PUBLIC/'groups'/(key+'_'+site+'.json')
            if verify and not path.exists():raise ValueError('Cannot verify missing metric group')
            once(path,row);groups.append(row)
    roster=sorted(set(data['sites']))
    summary=dict(result_source='fresh_SOURCE_excluded_forecast_scoring_cached_verified_PCT',groups=len(groups),
        localities=roster,effects={key:interval(groups,key,roster,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
            for key in ('known_pp','completion_lower_pp','completion_upper_pp')},
        primary=cfg['primary'],known_primary_positive=None,independent_confirmation=False,
        deployment_changed=False,full_policy_risk_tested=False,stage5c_executed=False,smc_enabled=False)
    summary['known_primary_positive']=summary['effects']['known_pp']['CI95'][0]>0
    path=PUBLIC/'summary.json'
    if verify and not path.exists():raise ValueError('Cannot verify missing summary')
    once(path,summary)
    if verify:
        if not json.loads((PUBLIC/'inference_verification.json').read_text())['all9_new_and9_control_inferences_exact']:
            raise ValueError('Separate full inference replay required')
        once(PUBLIC/'verification.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
            all72_SOURCE_groups_exact=len(groups)==72,summary_sha256=sha(path),new_optimizer_updates=False,
            inference_verification_sha256=sha(PUBLIC/'inference_verification.json'),independent_confirmation=False))
    else:
        once(PUBLIC/'complete.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,summary_sha256=sha(path),
            registration_sha256=sha(REG),groups=72,development_or_private_independent_arrays_read=False))
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','pilot','train','infer','eval','verify'))
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        values=admission(register=args.phase=='register')
        if args.phase=='register':return
        if args.phase=='verify':
            inference(*values,args.resume,verify=True);evaluate(*values,verify=True)
        elif args.phase=='eval':evaluate(*values)
        else:dict(pilot=pilot,train=train,infer=inference)[args.phase](*values,args.resume)


if __name__=='__main__':main()
