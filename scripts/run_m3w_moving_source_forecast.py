"""Complete three-seed admitted-source controls, with real Torch and recovery."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
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
from scripts.intake_m3w_movingdronecrowd import sha,once,atomic,encoded
from src.data_unification.m3w_moving_forecast_data import prepare,array_hash
from src.evaluation.m3w_moving_development_admission import load_admission
from src.world_model.m3w_european_source_forecast import SourceForecaster,fit_design,BASELINES
from src.world_model.m3w_native_forecast import fit_trial,predict
from src.world_model.m3w_eqmotion_adapter import verified_source
from src.evaluation.m3w_european_eqmotion import paired_group,interval

NAME='moving_source_forecast_v1'
PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG=ROOT/'configs/m3w_moving_source_forecast_v1.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',
    ROOT/'src/data_unification/m3w_moving_forecast_data.py',ROOT/'tests/test_m3w_moving_forecast_data.py',
    ROOT/'src/evaluation/m3w_moving_development_admission.py',
    ROOT/'src/data_unification/m3w_movingdronecrowd_intake.py',ROOT/'src/evaluation/m3w_dronecrowd_windows.py',
    ROOT/'src/world_model/m3w_european_source_forecast.py',ROOT/'src/world_model/m3w_native_forecast.py',
    ROOT/'src/world_model/m3w_supervised_intervention.py',ROOT/'src/world_model/m3w_context_conditioning.py',
    ROOT/'src/world_model/m3w_baseline_relative_forecaster.py',ROOT/'src/world_model/m3w_eqmotion_adapter.py',
    ROOT/'configs/m3w_eqmotion_source.json',ROOT/'src/evaluation/m3w_native_metrics.py',
    ROOT/'src/evaluation/m3w_european_eqmotion.py')


def stable(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def register():
    cfg=json.loads(CONFIG.read_text());admission_path=ROOT/cfg['admission']
    if sha(admission_path)!=cfg['admission_sha256']:raise ValueError('Changed admitted source')
    admission=load_admission(ROOT,admission_path);_,author=verified_source()
    if (cfg['new_primary_models']!=18 or cfg['seeds']!=[17,29,43]
            or set(cfg['families'])!={'pct','eqmotion'} or cfg['training']['steps']!=4000
            or any(cfg[k] for k in ('calibration_opened','confirmation_opened','deployment_changed','stage5c_executed','smc_enabled'))):
        raise ValueError('Keep the complete fixed source-only matrix')
    reference_p=json.loads((ROOT/'configs/m3w_european_source_forecast_v1.json').read_text())
    reference_e=json.loads((ROOT/'configs/m3w_european_eqmotion_v1.json').read_text())
    if (cfg['families']['pct']!=reference_p['architecture'] or cfg['families']['eqmotion']!=reference_e['architecture']
            or cfg['training']!=reference_e['training'] or cfg['training']!=reference_p['training']):
        raise ValueError('Match the original control architectures and complete fitting budget')
    value=dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},configuration=cfg,
        source_admission_sha256=sha(admission_path),source_reservation_sha256=admission['source_reservation_sha256'],
        source_groups=admission['proposed_within_fit_threefold_groups'],recordings=sorted(admission['recordings']),
        data_roles=['supervised_training'],declared_primary=cfg['primary'],
        numpy=np.__version__,torch=str(torch.__version__),machine=platform.machine(),author_core=author,
        predictor_pretraining='none_random_seeded_initialization',test_or_calibration_used=False,
        source_screens_reused_not_rerun=True,new_training_not_yet_claimed=True,
        stage5c_executed=False,smc_enabled=False)
    once(PUBLIC/'registration.json',stable(value))
    return cfg,admission


def check_committed():
    for path in (*FILES,PUBLIC/'registration.json'):
        raw=subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if hashlib.sha256(raw).hexdigest()!=sha(path):raise ValueError('Commit the exact experiment definition first')


def runtime(cfg,phase):
    torch.set_num_threads(cfg['torch_threads']);torch.set_num_interop_threads(cfg['interop_threads'])
    start=time.monotonic();signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    def beat(**values):
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h boundary; resume unchanged science')
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Keep original10GiB disk reserve')
        artifact_bytes=sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())
        if artifact_bytes>cfg['new_artifact_cap_bytes']:raise OSError('Registered private artifact budget exceeded')
        usage=resource.getrusage(resource.RUSAGE_SELF)
        event=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase=phase,
                   elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=usage.ru_maxrss if platform.system()=='Darwin' else usage.ru_maxrss*1024,
                   private_artifact_bytes=artifact_bytes,**values)
        atomic(PRIVATE/'heartbeat.json',event)
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    beat(state='runtime_ready',device='cpu',torch_threads=torch.get_num_threads(),interop_threads=cfg['interop_threads'],
         workers=0,resource_probing=False,machine=platform.machine())
    return start,beat


def data_and_jobs(cfg,admission,beat):
    data,receipt,checked=prepare(ROOT,cfg['admission'],beat)
    once(PUBLIC/'population.json',stable(receipt))
    population_sha=sha(PUBLIC/'population.json');folds=checked['proposed_within_fit_threefold_groups'];designs={}
    for fold in range(3):
        groups=sorted(g for g,k in folds.items() if k==fold)
        designs[fold]=fit_design(data,groups,data['baseline_ade'])
    jobs=[]
    for family in ('pct','eqmotion'):
        for fold in range(3):
            design=designs[fold]
            for seed in cfg['seeds']:
                key=f'{family}_fold{fold}_seed{seed}'
                identity=dict(registration_sha256=sha(PUBLIC/'registration.json'),population_sha256=population_sha,
                    family=family,fold=fold,seed=seed,architecture=cfg['families'][family],
                    fit_groups=sorted(g for g,k in folds.items() if k==fold),
                    excluded_groups=sorted(g for g,k in folds.items() if k!=fold),
                    train_ids_sha256=array_hash(design['train_ids']),held_ids_sha256=array_hash(design['held_ids']),
                    factors_sha256=array_hash(design['factors']),baseline_index=design['baseline_index'],
                    normalizers=design['normalizers'],selection_relative_scores=design['selection_relative_scores'],
                    whole_source_reservations_unchanged=True)
                jobs.append((key,design,stable(identity)))
    once(PUBLIC/'job_roster.json',dict(jobs=[dict(key=k,identity=i) for k,d,i in jobs],primary_models=18))
    return data,jobs


def fit(cfg,data,job,directory,beat,*,resume=False,stop_at=None):
    key,design,identity=job;torch.manual_seed(identity['seed'])
    model=SourceForecaster(identity['architecture'],design['baseline_index'])
    result=fit_trial(model,data,design,seed=identity['seed'],settings=cfg['training'],identity=identity,
        directory=directory,resume=resume,stop_at=stop_at,
        heartbeat=lambda **values:beat(trial=key,**values))
    state=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
    return result,state


def exact(left,right):
    if isinstance(left,torch.Tensor):
        if not isinstance(right,torch.Tensor) or not torch.equal(left,right):raise ValueError('Changed Torch scientific state')
    elif isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
    elif isinstance(left,dict):
        if not isinstance(right,dict) or left.keys()!=right.keys():raise ValueError('Changed state dictionary')
        for key in left:exact(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        if type(left)!=type(right) or len(left)!=len(right):raise ValueError('Changed state sequence')
        for a,b in zip(left,right):exact(a,b)
    elif left!=right:raise ValueError('Changed scalar scientific state')


def same_scientific_checkpoint(left,right):
    if left.keys()!=right.keys():raise ValueError('Changed checkpoint schema')
    exact({k:v for k,v in left.items() if k!='seconds'},{k:v for k,v in right.items() if k!='seconds'})


def pilot(cfg,data,jobs,beat):
    if (PUBLIC/'pilot.json').exists():raise ValueError('Completed pilot is immutable')
    rows=[]
    for family in ('pct','eqmotion'):
        job=next(j for j in jobs if j[2]['family']==family);home=PRIVATE/'pilot'/family/'resumed';direct=home.with_name('direct')
        start=time.monotonic()
        if not (home/'checkpoint.pt').exists():fit(cfg,data,job,home,beat,stop_at=cfg['pilot_resume_at'])
        result,a=fit(cfg,data,job,home,beat,resume=True,stop_at=cfg['pilot_steps'])
        reference,b=fit(cfg,data,job,direct,beat,resume=(direct/'checkpoint.pt').exists(),stop_at=cfg['pilot_steps'])
        same_scientific_checkpoint(a,b)
        ids=job[1]['train_ids'][:16];torch.manual_seed(job[2]['seed'])
        m=SourceForecaster(job[2]['architecture'],job[1]['baseline_index']);m.load_state_dict(a['model'])
        pa=predict(m,data,ids,16);m.load_state_dict(b['model']);pb=predict(m,data,ids,16)
        np.testing.assert_array_equal(pa,pb)
        if not np.isfinite(pa).all():raise FloatingPointError('Nonfinite real-data pilot prediction')
        rows.append(dict(family=family,job=job[0],resumed_step=a['step'],direct_step=b['step'],
            resume_direct_checkpoint_and_outputs_exact=True,seconds=time.monotonic()-start,
            complete100update_fit_seconds=result['seconds'],primary_full_fit_projection_seconds=result['seconds']*4000/100,
            serialized_checkpoint_bytes=(home/'checkpoint.pt').stat().st_size,
            initial_loss=a['losses'][0]['loss'],last_pilot_loss=a['losses'][-1]['loss'],
            prefix_path=str((home/'checkpoint.pt').relative_to(ROOT)),prefix_sha256=sha(home/'checkpoint.pt')))
    once(PUBLIC/'pilot.json',stable(dict(real_Torch_CPU_training=True,device='cpu',rows=rows,
        official_full_training_complete=False,held_metrics_read=False,
        estimated_primary_training_seconds=sum(r['primary_full_fit_projection_seconds']*9 for r in rows),
        checkpoint_prefix_can_seed_same_primary_job=True,pilot_not_scientific_success=True)))


def completed(job):
    key,design,identity=job;path=PRIVATE/'trials'/key/'complete.json'
    receipt=json.loads(path.read_text())
    if (receipt['identity']!=identity or receipt['step']!=4000
            or receipt['checkpoint_sha256']!=sha(ROOT/receipt['checkpoint'])):
        raise ValueError('Missing or changed complete4k producer')
    return receipt


def train(cfg,data,jobs,beat,resume):
    if (PUBLIC/'training.json').exists():raise ValueError('Completed training is immutable')
    pilot_receipt=json.loads((PUBLIC/'pilot.json').read_text());prefix={r['job']:r for r in pilot_receipt['rows']}
    results=[]
    for job in jobs:
        key,design,identity=job;home=PRIVATE/'trials'/key;done=home/'complete.json'
        if done.exists():
            if not resume:raise ValueError('Explicit unchanged matrix resume required')
            results.append(completed(job));continue
        home.mkdir(parents=True,exist_ok=True);checkpoint=home/'checkpoint.pt';copied=None
        if not checkpoint.exists() and key in prefix:
            copied=prefix[key]
            if sha(ROOT/copied['prefix_path'])!=copied['prefix_sha256']:raise ValueError('Changed pilot prefix')
            temporary=home/'checkpoint.importing';shutil.copyfile(ROOT/copied['prefix_path'],temporary);temporary.replace(checkpoint)
            once(home/'prefix_import.json',dict(path=copied['prefix_path'],sha256=copied['prefix_sha256'],updates=100))
        if checkpoint.exists() and not (resume or copied):raise ValueError('Existing partial checkpoint requires --resume')
        result,state=fit(cfg,data,job,home,beat,resume=checkpoint.exists())
        if state['step']!=4000 or not result['complete'] or state['draws'][design['held_ids']].any():
            raise ValueError('Complete source-excluded producer required')
        receipt=stable(dict(identity=identity,step=state['step'],checkpoint=str(checkpoint.relative_to(ROOT)),
            checkpoint_sha256=sha(checkpoint),fit=result,
            primary_updates=4000,pilot_prefix_reused=(home/'prefix_import.json').exists()))
        once(done,receipt);results.append(receipt)
        beat(state='producer_complete',trial=key,complete=len(results),total=18)
    for fold in range(3):
        for seed in cfg['seeds']:
            cp=[]
            for family in ('pct','eqmotion'):
                job=next(j for j in jobs if j[2]['family']==family and j[2]['fold']==fold and j[2]['seed']==seed)
                cp.append(torch.load(ROOT/completed(job)['checkpoint'],map_location='cpu',weights_only=False))
            for key in ('train_ids','factors','draws','sampler_rng'):exact(cp[0][key],cp[1][key])
    once(PUBLIC/'training.json',stable(dict(real_Torch_CPU_training=True,complete_primary_models=len(results),
        primary_optimizer_updates=72000,pilot_prefix_updates_in_primary=200,matched_sampling_all9pairs_exact=True,
        actual_training_seeds=cfg['seeds'],checkpoints=[dict(path=r['checkpoint'],sha256=r['checkpoint_sha256']) for r in results],
        held_group_rows_sampled=0,calibration_opened=False,confirmation_opened=False,device='cpu')))


def prediction(cfg,data,job,beat,verify):
    key,design,identity=job;fit_receipt=completed(job)
    directory=PRIVATE/'predictions';directory.mkdir(exist_ok=True)
    path=directory/(key+'.npz');receipt_path=path.with_suffix('.json');ids=design['held_ids']
    header=stable(dict(identity=identity,checkpoint_sha256=fit_receipt['checkpoint_sha256']))
    cached=None
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if json.loads(str(z['metadata_json']))!=header:raise ValueError('Changed prediction producer/source identity')
            np.testing.assert_array_equal(z['ids'],ids);cached=z['relative_prediction'].copy()
        if cached.shape!=(len(ids),12,2) or cached.dtype!=np.float32 or not np.isfinite(cached).all():
            raise ValueError('Changed complete prediction shape/dtype')
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if receipt['sha256']!=sha(path) or receipt['identity']!=identity:raise ValueError('Changed saved prediction receipt')
    elif verify:raise ValueError('Complete original inference before replay')
    if cached is None or verify:
        beat(state='source_excluded_inference',trial=key,rows=len(ids),fresh_replay=verify)
        torch.manual_seed(identity['seed']);model=SourceForecaster(identity['architecture'],design['baseline_index'])
        state=torch.load(ROOT/fit_receipt['checkpoint'],map_location='cpu',weights_only=False)
        exact(state['identity'],identity);model.load_state_dict(state['model'])
        actual=predict(model,data,ids,cfg['inference_batch_size'])
        if cached is not None:np.testing.assert_array_equal(actual,cached)
        else:
            pending=path.with_suffix('.pending.npz')
            np.savez_compressed(pending,ids=ids,relative_prediction=actual,metadata_json=np.asarray(json.dumps(header,sort_keys=True)))
            pending.replace(path);cached=actual
    receipt=stable(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),identity=identity,
        rows=len(ids),emitted_candidates=1,stored_prediction='relative_to_current_origin_float32',
        checkpoint_sha256=fit_receipt['checkpoint_sha256']))
    once(receipt_path,receipt)
    return cached,receipt


def evaluate(cfg,data,jobs,beat,verify):
    if not (PUBLIC/'training.json').exists():raise ValueError('Complete all18producers before any held readout')
    values={};receipts=[]
    for job in jobs:
        p,r=prediction(cfg,data,job,beat,verify);values[job[0]]=p;receipts.append(r)
    groups=[]
    for fold in range(3):
        for seed in cfg['seeds']:
            pjob=next(j for j in jobs if j[2]['family']=='pct' and j[2]['fold']==fold and j[2]['seed']==seed)
            ejob=next(j for j in jobs if j[2]['family']=='eqmotion' and j[2]['fold']==fold and j[2]['seed']==seed)
            exact(pjob[1]['held_ids'],ejob[1]['held_ids']);ids=pjob[1]['held_ids']
            for site in sorted(set(data['sites'][ids])):
                use=data['sites'][ids]==site;local=ids[use]
                scores=paired_group(values[ejob[0]][use]+data['origin'][local,None],
                    values[pjob[0]][use]+data['origin'][local,None],data['target_eval'][local],
                    data['valid'][local],data['baseline_ade'][local,1])
                known=data['valid'][local].any(1);selected=pjob[1]['baseline_index']
                scores['fitting_selected_baseline']=dict(name=BASELINES[selected],
                    available_ADE=float(data['baseline_ade'][local[known],selected].mean()),
                    endpoint_FDE=float(np.nanmean(data['baseline_fde'][local,selected])) if data['valid'][local,-1].any() else None)
                groups.append(dict(site=str(site),trial=f'producer_fold{fold}',seed=seed,fit_fold=fold,scores=scores))
    sites=sorted(set(data['sites']));primary=interval(groups,'completion_lower_pp',sites,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
    known=interval(groups,'known_pp',sites,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
    scores=stable(dict(scope='admitted_source_development_crossfit_not_independent_confirmation',
        primary=primary,known=known,groups=groups,prediction_receipts=receipts,
        actual_seeds=cfg['seeds'],source_groups_not_certified_IID_sites=len(sites),
        by_seed={str(seed):dict(completion_lower_pp=float(np.mean([r['scores']['completion_lower_pp'] for r in groups if r['seed']==seed])),
                               known_pp=float(np.mean([r['scores']['known_pp'] for r in groups if r['seed']==seed]))) for seed in cfg['seeds']},
        complete_risk_or_policy_gate_evaluated=False,new_policy_deployed=False,
        raw_t50_supplement='not_run_separate_task',stage5c_executed=False,smc_enabled=False))
    once(PUBLIC/'analysis.json',scores)
    return scores


def verify_refits(cfg,data,jobs,beat):
    results=[]
    for family in ('pct','eqmotion'):
        job=next(j for j in jobs if j[2]['family']==family);home=PRIVATE/'full_refit'/family
        result,state=fit(cfg,data,job,home,beat,resume=(home/'checkpoint.pt').exists())
        old=torch.load(ROOT/completed(job)['checkpoint'],map_location='cpu',weights_only=False)
        same_scientific_checkpoint(state,old)
        results.append(dict(family=family,primary_trial=job[0],step=state['step'],all_scientific_checkpoint_fields_exact=True,
                            actual_replay_fit=result,replay_checkpoint=str((home/'checkpoint.pt').relative_to(ROOT))))
    once(PUBLIC/'full_refit_verification.json',stable(dict(rows=results,full_primary_refits=2,new_replay_updates=8000)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','prepare','pilot','train','evaluate','verify'))
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,admission=register()
        if args.phase=='register':return
        check_committed();start,beat=runtime(cfg,args.phase)
        end=PUBLIC/(args.phase+'_execution.json')
        if end.exists():raise ValueError('Completed phase is immutable')
        data,jobs=data_and_jobs(cfg,admission,beat)
        if args.phase=='pilot':pilot(cfg,data,jobs,beat)
        elif args.phase=='train':train(cfg,data,jobs,beat,args.resume)
        elif args.phase in ('evaluate','verify'):
            if args.phase=='verify':verify_refits(cfg,data,jobs,beat)
            evaluate(cfg,data,jobs,beat,verify=args.phase=='verify')
        once(end,stable(dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            phase=args.phase,result_source='fresh_Torch_fits_and_full_saved_prediction_replay' if args.phase=='verify'
                else 'fresh_run' if args.phase in ('pilot','train','prepare') else 'fresh_inference_complete18fitted_models',
            real_Torch_training=args.phase in ('pilot','train','verify'),source_only=True,
            calibration_opened=False,confirmation_opened=False,stage5c_executed=False,smc_enabled=False)))
        beat(state='complete',phase_complete=args.phase)


if __name__=='__main__':main()
