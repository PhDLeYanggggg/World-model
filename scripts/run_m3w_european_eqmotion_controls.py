"""Secondary fixed stronger PCT controls; never replace the EqMotion primary."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_eqmotion as run
from scripts import run_m3w_active_detector_forecaster as producer
from src.evaluation.m3w_european_eqmotion import paired_group,interval
import numpy as np
import torch

PUBLIC=run.PUBLIC/'strong_controls';PRIVATE=run.PRIVATE/'strong_controls';REG=PUBLIC/'registration.json'
FILES=(Path(__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_european_eqmotion_controls.py')


def aligned_rows(ids,subset):
    ids,subset=np.asarray(ids),np.asarray(subset)
    if (ids.ndim!=1 or subset.ndim!=1 or not np.issubdtype(ids.dtype,np.integer)
            or not np.issubdtype(subset.dtype,np.integer) or np.any(np.diff(ids)<=0)
            or np.any(np.diff(subset)<=0)):
        raise ValueError('Sorted unique integer SOURCE IDs required')
    at=np.searchsorted(ids,subset)
    if (at>=len(ids)).any() or not np.array_equal(ids[at],subset):
        raise ValueError('Every secondary row must belong to the original excluded roster')
    return at


def admission(register=False):
    cfg,old,data,jobs,controls,predictions=run.admission()
    pc,current,modern_jobs,features,_=producer.admission();learned=producer.models()
    for key in ('sites','recordings','agents','frames','origin','history','target_eval','valid'):
        np.testing.assert_array_equal(data[key],current[key])
    for (key,design,ti),job in zip(jobs,modern_jobs):
        if key!=job['key'] or ti!=job['old_identity']:
            raise ValueError('All nine original producer identities must match')
        for field in ('train_ids','held_ids','factors'):
            np.testing.assert_array_equal(design[field],job['design'][field])
    oldfreeze=producer.parent.PUBLIC/'prediction_freeze.json'
    dimensionless={r['key']:r['dimensionless'] for r in json.loads(oldfreeze.read_text())['predictions']}
    store=producer.store.old.PUBLIC/'complete.json'
    white=json.loads(store.read_text())['arrays']['ids']
    if run.sha(ROOT/white['path'])!=white['sha256']:raise ValueError('Original whitelist bytes differ')
    whitelist=np.load(ROOT/white['path'],mmap_mode='r',allow_pickle=False)
    if len(whitelist)!=149510:raise ValueError('Complete original SOURCE whitelist required')
    original=json.loads((producer.PUBLIC/'complete.json').read_text())
    refs={Path(r['path']).stem:r for r in original['groups']}
    for key,_,_ in jobs:
        if run.sha(ROOT/dimensionless[key]['path'])!=dimensionless[key]['sha256']:
            raise ValueError('Dimensionless cache identity changed')
        ref=refs[key+'_history']
        if run.sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Actual history-control receipt changed')
    bindings={str(p.relative_to(ROOT)):run.sha(p) for p in (*FILES,run.REG,oldfreeze,
        producer.REG,producer.PUBLIC/'training_freeze.json',producer.PUBLIC/'verification.json',
        producer.PUBLIC/'complete.json',store)}
    value=dict(bindings=bindings,dimensionless=dimensionless,whitelist=white,
        history_models={j['key']:learned[j['key'],'history']['checkpoint'] for j in modern_jobs},
        actual_history_readouts={k:refs[k+'_history'] for k,_,_ in jobs},
        original_primary_changed=False,new_training=False,new_EqMotion_outcomes_scored_at_registration=False,
        development_or_private_independent_arrays_read=False)
    if register and ((run.PUBLIC/'summary.json').exists() or (PUBLIC/'summary.json').exists()):
        raise ValueError('Register secondary controls before any new EqMotion label readout')
    run.once(REG,value)
    if not register:
        for path in (*FILES,REG):producer.committed(path)
    return cfg,data,jobs,pc,current,modern_jobs,features,learned,whitelist,value


def infer(values,resume=False,verify=False):
    cfg,data,jobs,pc,current,modern_jobs,features,learned,whitelist,registration=values
    finish=PUBLIC/'prediction_freeze.json'
    if finish.exists() and not verify:raise ValueError('Completed secondary inference is immutable')
    primary=json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    if len(primary['predictions'])!=9:raise ValueError('Complete primary forecast freeze required')
    start,beat=run.runtime(cfg);refs=[]
    for job in modern_jobs:
        key=job['key'];ids=job['design']['held_ids'];sub=whitelist[np.isin(whitelist,ids)]
        at=aligned_rows(ids,sub)
        dim=registration['dimensionless'][key]
        with np.load(ROOT/dim['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids)
            if z['prediction'].shape!=(len(ids),12,2):raise ValueError('Dimensionless forecast schema differs')
        prepared,norm=producer.prepare_arm(current,features,job,'history');ref=learned[key,'history']
        if norm!=ref['normalizer']:raise ValueError('SOURCE fitting-only detector normalizer changed')
        model=producer.api.ActiveDetectorForecaster(job['design']['baseline_index'],'history',**pc['architecture'])
        model.load_state_dict(torch.load(ROOT/ref['checkpoint']['path'],map_location='cpu',weights_only=False)['model'])
        with producer.api.installed():p=producer.api.native.predict(model,prepared,sub,128)
        old=json.loads((ROOT/registration['actual_history_readouts'][key]['path']).read_text())
        if (run.source.array_hash(sub)!=old['ids_sha256'] or
                run.source.array_hash(p)!=old['prediction_sha256']['history']):
            raise ValueError('Fresh current-bank control does not reproduce its existing readout')
        path=PRIVATE/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if not(resume or verify):raise ValueError('Partial secondary inference requires --resume')
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],sub);np.testing.assert_array_equal(z['prediction'],p)
        else:
            if verify:raise ValueError('Cannot verify absent control predictions')
            temp=path.with_suffix('.tmp.npz');np.savez(temp,ids=sub,prediction=p);run.os.replace(temp,path)
        refs.append(dict(trial=key,dimensionless=dim,history_prediction=str(path.relative_to(ROOT)),
            history_sha256=run.sha(path),history_ids_sha256=run.source.array_hash(sub),
            subset_of_original_excluded_ids_exact=True,current_bank_old_prediction_exact=True))
        beat(state='secondary_control_replayed' if verify else 'secondary_control_frozen',trial=key,rows=len(sub))
        del model,prepared
    run.once(finish,dict(registration_sha256=run.sha(REG),primary_freeze_sha256=run.sha(run.PUBLIC/'prediction_freeze.json'),
        predictions=refs,all9_current_control_arrays_match_old=True,new_outcome_reductions=False))
    if verify:run.once(PUBLIC/'inference_verification.json',dict(pid=run.os.getpid(),seconds=time.monotonic()-start,
        all9_history_control_arrays_exact=True,prediction_freeze_sha256=run.sha(finish),independent_confirmation=False))


def evaluate(values,verify=False):
    cfg,data,jobs,_,_,_,_,_,_,_=values
    freeze=json.loads((PUBLIC/'prediction_freeze.json').read_text())
    primary=json.loads((run.PUBLIC/'prediction_freeze.json').read_text());groups=[]
    if freeze['primary_freeze_sha256']!=run.sha(run.PUBLIC/'prediction_freeze.json'):
        raise ValueError('Primary forecasts must remain frozen before secondary scoring')
    for (key,design,ti),ref,new in zip(jobs,freeze['predictions'],primary['predictions']):
        if key!=ref['trial']:raise ValueError('Secondary producer order differs')
        rp=ROOT/new['path'];r=json.loads(rp.read_text())
        if run.sha(rp)!=new['sha256'] or run.sha(ROOT/r['prediction'])!=r['sha256']:
            raise ValueError('New EqMotion prediction changed')
        with np.load(ROOT/r['prediction'],allow_pickle=False) as z:
            ids=z['ids'];p=z['prediction'].astype(float)
        for arm in ('dimensionless','history'):
            path=ROOT/(ref['dimensionless']['path'] if arm=='dimensionless' else ref['history_prediction'])
            expected=ref['dimensionless']['sha256'] if arm=='dimensionless' else ref['history_sha256']
            if run.sha(path)!=expected:raise ValueError('Fixed control array changed')
            with np.load(path,allow_pickle=False) as z:sub=z['ids'];q=z['prediction'].astype(float)
            at=aligned_rows(ids,sub);sites=np.asarray(data['sites'][sub]);cv=np.asarray(data['baseline_ade'][sub,1])
            y=np.asarray(data['target_eval'][sub])-data['origin'][sub,None];valid=np.asarray(data['valid'][sub])
            for site in sorted(set(sites)):
                ix=sites==site
                row=dict(trial=key,seed=ti['seed'],site=site,control=arm,
                    scores=paired_group(p[at][ix],q[ix],y[ix],valid[ix],cv[ix]))
                path=PUBLIC/'groups'/(key+'_'+site+'_'+arm+'.json')
                if verify and not path.exists():raise ValueError('Missing secondary group')
                run.once(path,row);groups.append(row)
    result=dict(result_source='fresh_secondary_SOURCE_scoring_not_new_training',original_primary_changed=False,
        scopes=dict(dimensionless='all318969_original_SOURCE_rows',history='original149510_SOURCE_controller_whitelist'),
        controls={arm:{f:interval([r for r in groups if r['control']==arm],f,sorted(set(data['sites'])),
            cfg['bootstrap_resamples'],cfg['bootstrap_seed']) for f in ('known_pp','completion_lower_pp','completion_upper_pp')}
            for arm in ('dimensionless','history')},
        matched_internal_features=False,private_independent_arrays_read=False,independent_confirmation=False,
        deployment_changed=False,stage5c=False,smc=False)
    if verify and not (PUBLIC/'summary.json').exists():raise ValueError('Cannot verify missing summary')
    run.once(PUBLIC/'summary.json',result)
    if verify:
        if not json.loads((PUBLIC/'inference_verification.json').read_text())['all9_history_control_arrays_exact']:
            raise ValueError('Complete secondary control replay required')
        run.once(PUBLIC/'verification.json',dict(all144_secondary_groups_exact=len(groups)==144,
            summary_sha256=run.sha(PUBLIC/'summary.json'),original_primary_changed=False))
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','infer','eval','verify'))
    p.add_argument('--resume',action='store_true');a=p.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        values=admission(register=a.phase=='register')
        if a.phase=='register':return
        if a.phase in ('infer','verify'):infer(values,a.resume,a.phase=='verify')
        if a.phase in ('eval','verify'):evaluate(values,a.phase=='verify')


if __name__=='__main__':main()
