"""First frozen-chain outcome readout on the model-selection role only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_selection_forecast as frozen
from src.data_unification.m3w_european_squares_source import build_recording,INPUT_FIELDS
from src.data_unification.m3w_model_selection_inputs import load_record
from src.evaluation import m3w_selection_forecast_readout as api

NAME='european_selection_forecast_readout_v1';PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME;REG=PUBLIC/'registration.json'
CONFIG=ROOT/'configs/m3w_selection_forecast_readout_v1.json';sha,write,once=frozen.sha,frozen.write,frozen.once
FILES=(Path(__file__),CONFIG,ROOT/'src/evaluation/m3w_selection_forecast_readout.py',
    ROOT/'tests/test_m3w_selection_forecast_readout.py',PUBLIC/'protocol.md')


def admission(register=False):
    _,roles,selected,archive,chain=frozen.admission();cfg=json.loads(CONFIG.read_text())
    verification=json.loads((frozen.PUBLIC/'prediction_verification.json').read_text())
    if (not verification['all36fresh_predictions_exact'] or not verification['all9_disabled_parent_predictions_exact']
            or verification['prediction_freeze_sha256']!=sha(frozen.PUBLIC/'prediction_freeze.json')):
        raise ValueError('Freeze and full fresh replay all36forecasts before selection labels')
    if (cfg['expected_target_rows']!=38102 or cfg['expected_groups']!=54 or cfg['positive_harm_budget']!=.02
            or cfg['easy_net_degradation_budget']!=.02 or cfg['bootstrap_resamples']!=3000
            or cfg['training_seeds']!=[17,29,43] or cfg['source_folds']!=[0,1,2]
            or any(cfg[k] for k in ('new_training','normalization_fitted','threshold_search','calibration_access',
                'confirmation_access','stage5c_executed','smc_enabled'))):
        raise ValueError('Original all-row,risk,seed and role contract required')
    forecast=json.loads((frozen.PUBLIC/'prediction_freeze.json').read_text());models=[]
    for ref in forecast['models']:
        path=frozen.checked(ref);row=json.loads(path.read_text());frozen.checked(row['prediction']);models.append(row)
    if len(models)!=36:raise ValueError('All36saved outputs required')
    value=dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        dependencies={p:sha(ROOT/p) for p in ('src/evaluation/m3w_native_metrics.py',
            'src/data_unification/m3w_european_squares_source.py','src/data_unification/m3w_model_selection_inputs.py')},
        frozen_chain_registration=frozen.artifact(frozen.REG),prediction_freeze=frozen.artifact(frozen.PUBLIC/'prediction_freeze.json'),
        prediction_verification=frozen.artifact(frozen.PUBLIC/'prediction_verification.json'),roles_sha256=sha(frozen.inputs.ROLES),
        input_analysis=frozen.artifact(frozen.inputs.PUBLIC/'analysis.json'),members=selected,
        primary=cfg['primary'],outcome_role='model_selection_only',model_selection_outcomes_authorized=True,
        original_role_manifest_unchanged=True,new_training=False,normalization_fitted=False,threshold_search=False,
        calibration_access=False,confirmation_access=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    if register:once(REG,value)
    if json.loads(REG.read_text())!=value:raise ValueError('Outcome contract changed')
    if not register:
        for p in (*FILES,REG):
            blob=subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(blob).hexdigest()!=sha(p):raise ValueError('Commit outcome contract before access')
    return cfg,roles,selected,archive,chain,forecast,models


def run(verify=False,resume=False):
    cfg,roles,selected,archive,chain,forecast,models=admission();start=time.monotonic()
    finish=PUBLIC/('verification.json' if verify else 'complete.json')
    if finish.exists():raise ValueError('Completed selection readout immutable')
    if verify and not (PUBLIC/'complete.json').exists():raise ValueError('Complete readout required before replay')
    def beat(state,**kw):
        if shutil.disk_usage(ROOT).free<10*2**30:raise OSError('Preserve10GiB disk reserve')
        if time.monotonic()-start>43200:raise TimeoutError('12h boundary;resume unchanged readout')
        value=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,verify=verify,**kw)
        write(PRIVATE/'heartbeat.json',value);print(json.dumps(value),flush=True)
    opening=PUBLIC/'selection_outcome_opening.json'
    if not opening.exists():
        once(opening,dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),registration_sha256=sha(REG),
            prediction_freeze_sha256=sha(frozen.PUBLIC/'prediction_freeze.json'),
            scope='first_model_selection_outcomes_only_not_training_calibration_or_confirmation',recordings=28,localities=6,
            full_readout_complete=False,original_reservation_manifest_unchanged=True))
    elif json.loads(opening.read_text())['registration_sha256']!=sha(REG):raise ValueError('Existing selection exposure binding changed')
    data=frozen.load_features();n=len(data['ids']);target=np.empty((n,12,2),float);valid=np.empty((n,12),bool);cursor=0;labelrefs=[]
    if sha(ROOT/archive['path'])!=archive['sha256']:raise ValueError('Raw archive changed')
    feature_refs=json.loads((frozen.PUBLIC/'feature_freeze.json').read_text())['records']
    with zipfile.ZipFile(ROOT/archive['path']) as z:
        for i,row in enumerate(selected):
            member=row['source_member']
            if row['role']!='model_selection_reserved' or row['training_access']:raise PermissionError('No other outcome role admitted')
            home=PRIVATE/'labels'/hashlib.sha256(member.encode()).hexdigest();done=home/'receipt.json'
            input_home=frozen.inputs.PRIVATE/'records'/hashlib.sha256(member.encode()).hexdigest()
            inp,queries=load_record(input_home,roles,member,registration_sha256=sha(frozen.inputs.REG))
            feature=json.loads(frozen.checked(feature_refs[i]).read_text())
            if feature['source_member']!=member:raise ValueError('Original feature record order required')
            local=np.load(frozen.checked(feature['arrays']['input_rows']),mmap_mode='r',allow_pickle=False)
            count=len(local);span=slice(cursor,cursor+count)
            binding=dict(registration_sha256=sha(REG),source_member=member,raw_rows_sha256=row['rows_sha256'],
                input_receipt_sha256=sha(input_home/'receipt.json'),feature_receipt_sha256=feature_refs[i]['sha256'])
            if done.exists() and resume and not verify:
                saved=json.loads(done.read_text())
                if saved['binding']!=binding:raise ValueError('Label checkpoint changed')
                yy=np.load(frozen.checked(saved['target']),mmap_mode='r',allow_pickle=False)
                mm=np.load(frozen.checked(saved['valid']),mmap_mode='r',allow_pickle=False)
                beat('labels_cached_verified',records=i+1,total=28)
            else:
                with z.open(member) as f:raw,_=frozen.inputs.read_raw_csv(f)
                if hashlib.sha256(raw.tobytes()).hexdigest()!=row['rows_sha256']:raise ValueError('Outcome raw row identity changed')
                current,labels=build_recording(raw,queries);del raw
                for key in INPUT_FIELDS:np.testing.assert_array_equal(current[key],inp[key])
                np.testing.assert_array_equal(data['agents'][span],current['agent_id'][local])
                np.testing.assert_array_equal(data['frames'][span],current['query_frame'][local])
                yy=labels['future_xy'][local];mm=labels['future_valid'][local];home.mkdir(parents=True,exist_ok=True)
                if verify:
                    saved=json.loads(done.read_text())
                    if saved['binding']!=binding:raise ValueError('Replay label binding changed')
                    np.testing.assert_array_equal(np.load(frozen.checked(saved['target']),allow_pickle=False),yy)
                    np.testing.assert_array_equal(np.load(frozen.checked(saved['valid']),allow_pickle=False),mm)
                else:
                    frozen.inputs.store_array(home/'target.npy',yy);frozen.inputs.store_array(home/'valid.npy',mm)
                    once(done,dict(binding=binding,target=frozen.artifact(home/'target.npy'),valid=frozen.artifact(home/'valid.npy'),
                        target_rows=count,complete12_rows=int(mm.all(1).sum()),wholly_unknown_rows=int((~mm.any(1)).sum())))
                beat('label_record_complete',records=i+1,total=28,rows=count)
            target[span]=yy;valid[span]=mm;cursor+=count;labelrefs.append(frozen.artifact(done))
    if cursor!=n:raise ValueError('All original38102targets required')
    predictions={(r['trial'],r['representation']):np.load(frozen.checked(r['prediction']),mmap_mode='r',allow_pickle=False) for r in models}
    physical=np.load(frozen.checked(forecast['physical_baselines']),mmap_mode='r',allow_pickle=False)
    rows=[];refs=[];sites=sorted(set(data['sites']));lineage={r['trial']:r for r in chain['lineage'] if r['representation']=='history'}
    for trial,source in sorted(lineage.items()):
        for site in sites:
            take=data['sites']==site;floor=physical[take,source['baseline_index']];cv=physical[take,1]
            trajectories={a:predictions[trial,a][take] for a in frozen.api.REPRESENTATIONS}
            trajectories.update({'physical_'+str(k):physical[take,k] for k in range(6)});trajectories['floor']=floor
            policies={a:api.policy(p,floor,cv,target[take],valid[take],data['recordings'][take],
                easy_cut=source['easy_cut'],hard_cut=source['hard_cut']) for a,p in trajectories.items()}
            pair=api.paired(trajectories['disabled'],trajectories['history'],floor,target[take],valid[take])
            row=dict(trial=trial,seed=source['seed'],source_fold=source['fold'],site=str(site),policies=policies,
                paired_history_minus_disabled=pair,row_ids_sha256=frozen.fingerprint(data['ids'][take]),
                source_floor_index=source['baseline_index'],source_easy_cut=source['easy_cut'],source_hard_cut=source['hard_cut'],
                registration_sha256=sha(REG),prediction_freeze_sha256=sha(frozen.PUBLIC/'prediction_freeze.json'))
            path=PUBLIC/'groups'/(trial+'_'+str(site)+'.json')
            if verify or (path.exists() and resume):
                if json.loads(path.read_text())!=row:raise ValueError('Original complete selection metrics replay differs')
            else:once(path,row)
            rows.append(row);refs.append(frozen.artifact(path));beat('selection_group_complete',groups=len(rows),total=54)
    summary=api.summarize(rows,sites,resamples=cfg['bootstrap_resamples'],seed=cfg['bootstrap_seed'])
    summary.update(known_rows=int(valid.any(1).sum()),complete12_rows=int(valid.all(1).sum()),
        endpoint_rows=int(valid[:,-1].sum()),wholly_unknown_rows=int((~valid.any(1)).sum()))
    runtime=dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if verify:
        complete=json.loads((PUBLIC/'complete.json').read_text())
        if refs!=complete['groups'] or summary!=json.loads((PUBLIC/'summary.json').read_text()):raise ValueError('Complete selection summary differs')
        once(finish,dict(**runtime,all28_raw_labels_reparsed=True,all54_metric_groups_exact=True,
            summary_sha256=sha(PUBLIC/'summary.json'),complete_sha256=sha(PUBLIC/'complete.json'),
            model_inference_reexecuted=False,separate_prior36model_inference_verification=frozen.artifact(frozen.PUBLIC/'prediction_verification.json')))
    else:
        once(PUBLIC/'summary.json',summary);once(finish,dict(**runtime,groups=refs,label_receipts=labelrefs,
            summary_sha256=sha(PUBLIC/'summary.json'),registration_sha256=sha(REG),
            prediction_freeze_sha256=sha(frozen.PUBLIC/'prediction_freeze.json'),model_selection_outcomes_opened=True,
            calibration_access=False,confirmation_access=False,new_training=False,deployment_changed=False))
    beat('all54_selection_metrics_verified' if verify else 'all54_selection_metrics_complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','run','verify'));parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(True)
        else:run(args.phase=='verify',args.resume)
