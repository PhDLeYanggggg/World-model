"""Freeze actual source-trained forecasts before selection outcomes are opened."""
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
import zipfile

if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 before Torch')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import build_m3w_model_selection_inputs as inputs
from scripts import run_m3w_active_detector_forecaster as producers
from src.data_unification import m3w_selection_feature_bridge as api
from src.data_unification.m3w_model_selection_inputs import load_record,build_inputs,require_input_role
from src.evaluation.m3w_european_squares_roles import require_source_training
from src.world_model.m3w_detector_cost import representation
from src.world_model.m3w_detector_forecaster import DetectorGeometry,pack_geometry
from src.world_model.m3w_dimensionless_correction import DimensionlessAgentTrackSourceForecaster
from src.world_model.m3w_native_forecast import pack_geometry as plain_pack

NAME='european_selection_forecast_freeze_v1';PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME;CONFIG=ROOT/'configs/m3w_selection_forecast_freeze_v1.json';REG=PUBLIC/'registration_v2.json'
PACKED=ROOT/'data/stage_cvpr2027_experiments/european_source_forecast_v1/packed'
STORE=ROOT/'outputs/publication_readiness_2026_09/european_producer_detector_store_v1'
sha,write,once=inputs.sha,inputs.write,inputs.once
FILES=(Path(__file__),CONFIG,ROOT/'src/data_unification/m3w_selection_feature_bridge.py',
    ROOT/'tests/test_m3w_selection_feature_bridge.py',ROOT/'tests/test_m3w_selection_forecast_runtime.py',PUBLIC/'protocol.md')
DEPS=('src/world_model/m3w_active_detector_forecaster.py','src/world_model/m3w_detector_forecaster.py',
    'src/world_model/m3w_dimensionless_correction.py','src/world_model/m3w_partial_context.py',
    'src/world_model/m3w_agent_track_context.py','src/world_model/m3w_observation_quality.py',
    'src/world_model/m3w_detector_history_features.py','src/world_model/m3w_european_source_forecast.py',
    'src/world_model/m3w_native_forecast.py','src/world_model/m3w_detector_cost.py',
    'src/world_model/m3w_context_conditioning.py','src/world_model/m3w_baseline_relative_forecaster.py',
    'src/world_model/m3w_supervised_intervention.py','scripts/run_m3w_active_detector_forecaster.py')


def fingerprint(value):
    value=np.asarray(value)
    return hashlib.sha256(str((value.shape,value.dtype.str)).encode()+np.ascontiguousarray(value).tobytes()).hexdigest()


def artifact(path):return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))


def checked(ref):
    path=ROOT/ref['path']
    if sha(path)!=ref['sha256']:raise ValueError('Frozen artifact changed: '+str(path))
    return path


def admission(register=False):
    _,roles,selected,archive=inputs.admission();cfg=json.loads(CONFIG.read_text())
    check=json.loads((inputs.PUBLIC/'verification.json').read_text());cohort=json.loads((inputs.PUBLIC/'analysis.json').read_text())['cohort']
    if not check['all28_raw_reparsed_all_input_arrays_exact'] or check['analysis_sha256']!=sha(inputs.PUBLIC/'analysis.json'):
        raise ValueError('Complete verified input conversion required')
    if (cfg['representations']!=list(api.REPRESENTATIONS) or cfg['seeds']!=[17,29,43] or cfg['producer_folds']!=[0,1,2]
            or cfg['model_count']!=36 or cfg['batch_size']!=128 or cfg['num_workers']!=0
            or any(cfg[k] for k in ('future_targets_constructed','forecast_errors_opened','model_selection_by_outcomes',
                'normalization_fitted','new_training','calibration_access','confirmation_access','stage5c_executed','smc_enabled'))):
        raise ValueError('Complete frozen36family,original batch and no-outcome scope required')
    models=producers.models();rows=list(models.values());api.validate_lineage(rows,roles)
    preg=json.loads(producers.REG.read_text());verification=json.loads((producers.PUBLIC/'verification.json').read_text())
    if not verification['all36groups_exact'] or verification['summary_sha256']!=sha(producers.PUBLIC/'summary.json'):
        raise ValueError('Verified actual source producer family required')
    for rel,digest in preg['bindings'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Original producer definition changed')
    source_receipt=json.loads((PACKED/'receipt.json').read_text());lineage=[]
    for r in sorted(rows,key=lambda r:(r['trial'],api.REPRESENTATIONS.index(r['representation']))):
        ti=r['parent_trial']
        if ti['identity']['source_roles_sha256']!=sha(inputs.ROLES):raise ValueError('Source role lineage changed')
        for rel,digest in ti['identity']['bindings'].items():
            if sha(ROOT/rel)!=digest:raise ValueError('Source training definition changed')
        refpath=producers.PUBLIC/'fits'/(r['trial']+'_'+r['representation']+'.json')
        lineage.append(dict(trial=r['trial'],representation=r['representation'],seed=ti['seed'],fold=ti['fold'],
            fit_sites=ti['fit_sites'],baseline_index=ti['baseline_index'],easy_cut=ti['easy_cut'],hard_cut=ti['hard_cut'],
            train_ids_sha256=ti['train_ids_sha256'],normalizer=r['normalizer'],receipt=artifact(refpath),checkpoint=r['checkpoint']))
    geometry_path=checked(preg['active_geometry']);geometry=json.loads(geometry_path.read_text())
    records=source_receipt['records']
    if len(records)!=163:raise ValueError('Original source recording namespace required')
    value=dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},dependencies={p:sha(ROOT/p) for p in DEPS},
        prior_preexecution_registration_sha256=sha(PUBLIC/'registration.json'),
        pre_execution_amendment='remove_duplicate_fixed_metadata_keyword_before_feature_or_prediction_execution_no_model_data_budget_or_role_change',
        input_analysis=artifact(inputs.PUBLIC/'analysis.json'),input_verification=artifact(inputs.PUBLIC/'verification.json'),
        input_registration=artifact(inputs.REG),roles_sha256=sha(inputs.ROLES),source_packed_receipt=artifact(PACKED/'receipt.json'),
        source_partial_geometry_receipt=preg['active_geometry'],source_partial_geometry=geometry['geometry'],
        producer_feature_complete=artifact(STORE/'complete.json'),producer_feature_verification=artifact(STORE/'verification.json'),
        producer_registration=artifact(producers.REG),producer_training_freeze=artifact(producers.PUBLIC/'training_freeze.json'),
        producer_verification=artifact(producers.PUBLIC/'verification.json'),lineage=lineage,parent_jobs=preg['parent_jobs'],
        selection_rows=cohort['target_rows'],selection_recordings=28,selection_localities=6,
        row_id_namespace='european_selection_v1',recording_id_offset=len(records),
        future_targets_constructed=False,forecast_errors_opened=False,error_readout_authorized=False,
        new_training=False,calibration_access=False,confirmation_access=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    if register:once(REG,value)
    if json.loads(REG.read_text())!=value:raise ValueError('Registered forecast chain changed')
    if not register:
        for path in (*FILES,REG):
            old=subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(old).hexdigest()!=sha(path):raise ValueError('Commit complete chain before feature/prediction access')
    return cfg,roles,selected,archive,value


def runtime(cfg,phase):
    torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume same feature/prediction checkpoints')))
    def guard():
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Keep10GiBfree;do not shrink rows')
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h phase boundary;resume unchanged')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.npy'))>cfg['private_cache_cap_bytes']:raise OSError('1GiBprivate cache cap')
    def beat(state,**kw):
        guard();r=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase=phase,state=state,**kw)
        write(PRIVATE/'heartbeat.json',r);print(json.dumps(r),flush=True)
    def receipt():return dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        registration_sha256=sha(REG),future_targets_constructed=False,forecast_errors_opened=False,new_training=False)
    beat('runtime_ready',architecture=platform.machine(),threads=4,interop=1,workers=0,resource_probing=False)
    return beat,guard,receipt


def source_parity(z,roles,reg):
    path=PUBLIC/'source_feature_parity.json'
    if path.exists():
        if json.loads(path.read_text())['registration_sha256']!=sha(REG):raise ValueError('Source parity identity changed')
        return
    packed=json.loads(checked(reg['source_packed_receipt']).read_text())
    store=json.loads(checked(reg['producer_feature_complete']).read_text())
    verify=json.loads(checked(reg['producer_feature_verification']).read_text())
    if not verify['all_arrays_exact'] or verify['complete_sha256']!=sha(STORE/'complete.json'):raise ValueError('Producer feature seal changed')
    arrays={k:np.load(checked(store['arrays'][k]),mmap_mode='r',allow_pickle=False) for k in ('recordings','agents','frames','features')}
    ri=int(arrays['recordings'].min());member=packed['records'][ri];access=require_source_training(roles,member)
    source_manifest=json.loads((ROOT/'outputs/publication_readiness_2026_09/european_squares_source_v1/analysis.json').read_text())
    ref=next(r for r in source_manifest['record_receipts'] if r['source_member']==member);directory=ROOT/ref['directory']
    if sha(directory/'receipt.json')!=ref['receipt_sha256']:raise ValueError('Source input receipt changed')
    original=json.loads((directory/'receipt.json').read_text());inp={}
    for key in inputs.api.INPUT_FIELDS:
        a=original['arrays']['input_'+key];p=directory/a['name']
        if sha(p)!=a['sha256']:raise ValueError('Source input array changed')
        inp[key]=np.load(p,mmap_mode='r',allow_pickle=False)
    with z.open(member) as f:raw,_=inputs.read_raw_csv(f)
    if hashlib.sha256(raw.tobytes()).hexdigest()!=access['rows_sha256']:raise ValueError('Source parity raw identity changed')
    out=api.pack_record(inp,raw);ids=np.flatnonzero(arrays['recordings']==ri)
    geo=np.load(checked(reg['source_partial_geometry']),mmap_mode='r',allow_pickle=False)
    for key in ('agents','frames'):np.testing.assert_array_equal(out[key],arrays[key][ids])
    np.testing.assert_array_equal(out['geometry'],geo[ids]);np.testing.assert_array_equal(out['detector_history'],arrays['features'][ids])
    if not len(ids):raise ValueError('Nonempty source feature parity required')
    once(path,dict(registration_sha256=sha(REG),source_member=member,source_role='source_training',rows=len(ids),
        actual_partial_geometry_exact=True,actual_detector64_exact=True,agent_frame_alignment_exact=True,
        geometry_sha256=fingerprint(out['geometry']),detector_sha256=fingerprint(out['detector_history']),future_targets_read=False))


def feature_prefix(raw,inp,values,query):
    for q in query:
        prefix=raw[raw['frame']<=q];new=api.pack_record(build_inputs(prefix,np.array([q],np.int64)),prefix)
        at=values['frames']==q
        for key in api.FIELDS:
            if key!='input_rows':np.testing.assert_array_equal(new[key],values[key][at])


def feature_stage(phase,cfg,roles,selected,archive,reg,*,resume=False):
    beat,guard,receipt=runtime(cfg,phase);pilot=phase=='feature_pilot';verify=phase=='feature_verify'
    finish=PUBLIC/('feature_pilot.json' if pilot else 'feature_verification.json' if verify else 'feature_freeze.json')
    if finish.exists():raise ValueError('Completed feature stage immutable')
    if not pilot and not (PUBLIC/'feature_pilot.json').exists():raise ValueError('Actual feature pilot required')
    if verify and not (PUBLIC/'feature_freeze.json').exists():raise ValueError('Complete feature freeze required')
    selected=[min(selected,key=lambda r:(-r['raw_rows'],r['source_member']))] if pilot else selected
    beat('verify_archive',bytes=archive['bytes'])
    if sha(ROOT/archive['path'])!=archive['sha256']:raise ValueError('Raw archive changed')
    refs=[];parsed=0;cached=0
    with zipfile.ZipFile(ROOT/archive['path']) as z:
        source_parity(z,roles,reg)
        for index,row in enumerate(selected):
            guard();member=row['source_member'];require_input_role(roles,member)
            home=PRIVATE/'features'/hashlib.sha256(member.encode()).hexdigest();done=home/'receipt.json'
            input_home=inputs.PRIVATE/'records'/hashlib.sha256(member.encode()).hexdigest()
            inp,queries=load_record(input_home,roles,member,registration_sha256=sha(inputs.REG))
            binding=dict(registration_sha256=sha(REG),input_receipt=artifact(input_home/'receipt.json'),raw_rows_sha256=row['rows_sha256'])
            if done.exists() and not verify:
                if not resume:raise ValueError('Explicit feature resume required')
                old=json.loads(done.read_text())
                if old['binding']!=binding:raise ValueError('Feature checkpoint identity changed')
                for ref in old['arrays'].values():checked(ref)
                refs.append(artifact(done));cached+=1;beat('feature_cached_verified',records=index+1,total=len(selected));continue
            beat('raw_feature_record',record=index+1,total=len(selected),source_member=member)
            with z.open(member) as f:raw,_=inputs.read_raw_csv(f)
            if hashlib.sha256(raw.tobytes()).hexdigest()!=row['rows_sha256']:raise ValueError('Raw feature rows changed')
            out=api.pack_record(inp,raw);probes=queries[sorted({0,len(queries)//2,len(queries)-1})]
            feature_prefix(raw,inp,out,probes);del raw;parsed+=1;home.mkdir(parents=True,exist_ok=True);arrays={}
            for key,value in out.items():
                p=home/(key+'.npy')
                if verify:np.testing.assert_array_equal(np.load(p,mmap_mode='r',allow_pickle=False),value)
                else:guard();inputs.store_array(p,value)
                arrays[key]=dict(**artifact(p),shape=list(value.shape),dtype=str(value.dtype),bytes=p.stat().st_size)
            value=dict(binding=binding,source_member=member,locality_group=row['locality_group'],rows=len(out['frames']),
                arrays=arrays,partial_neighbor_rows=int((out['partial_neighbor_count']>0).sum()),prefix_queries=len(probes),
                future_targets_constructed=False,normalization_fitted=False)
            if verify:
                if json.loads(done.read_text())!=value:raise ValueError('Full feature replay differs')
            else:once(done,value)
            refs.append(artifact(done));beat('feature_record_complete',records=index+1,total=len(selected),rows=len(out['frames']))
    if pilot:
        once(finish,dict(**receipt(),engineering_pass=True,records=refs,parsed_records=parsed,cached_verified_records=cached))
    elif verify:
        frozen=json.loads((PUBLIC/'feature_freeze.json').read_text())
        if refs!=frozen['records']:raise ValueError('Complete feature recording roster differs')
        arrays=load_features()
        cursor=0
        for ref in refs:
            item=json.loads(checked(ref).read_text());n=item['rows']
            for k,v in item['arrays'].items():np.testing.assert_array_equal(arrays[k][cursor:cursor+n],np.load(checked(v),mmap_mode='r',allow_pickle=False))
            cursor+=n
        if cursor!=reg['selection_rows']:raise ValueError('All selection rows required')
        once(finish,dict(**receipt(),all28_raw_reparsed_all_features_exact=True,all_assembled_arrays_exact=True,
            parsed_records=parsed,feature_freeze_sha256=sha(PUBLIC/'feature_freeze.json')))
    else:
        if len(refs)!=28:raise ValueError('All28feature records required')
        records=[json.loads(checked(r).read_text()) for r in refs];total=sum(r['rows'] for r in records)
        if total!=reg['selection_rows']:raise ValueError('Original target membership required')
        store=PRIVATE/'store';store.mkdir(exist_ok=True);arrays={}
        for key in api.FIELDS:
            spec=records[0]['arrays'][key];p=store/(key+'.npy');temp=p.with_suffix('.tmp.npy')
            value=np.lib.format.open_memmap(temp,mode='w+',dtype=spec['dtype'],shape=(total,*spec['shape'][1:]));cursor=0
            for r in records:
                n=r['rows'];value[cursor:cursor+n]=np.load(checked(r['arrays'][key]),mmap_mode='r',allow_pickle=False);cursor+=n
            value.flush();del value;temp.replace(p);arrays[key]=dict(**artifact(p),shape=[total,*spec['shape'][1:]],dtype=spec['dtype'],bytes=p.stat().st_size)
        for key,value in dict(ids=np.arange(total,dtype=np.int64),
                recordings=np.concatenate([np.full(r['rows'],reg['recording_id_offset']+i,np.int32) for i,r in enumerate(records)]),
                sites=np.concatenate([np.full(r['rows'],r['locality_group'],dtype='<U20') for r in records])).items():
            p=store/(key+'.npy');inputs.store_array(p,value);arrays[key]=dict(**artifact(p),shape=list(value.shape),dtype=str(value.dtype),bytes=p.stat().st_size)
        once(finish,dict(**receipt(),records=refs,arrays=arrays,rows=total,recordings=28,localities=6,
            partial_neighbor_rows=sum(r['partial_neighbor_rows'] for r in records),prefix_query_checks=sum(r['prefix_queries'] for r in records),
            row_id_namespace=reg['row_id_namespace'],recording_id_offset=reg['recording_id_offset'],
            parsed_records=parsed,cached_verified_records=cached,normalization_fitted=False))
    beat('feature_phase_complete',records=len(refs))


def load_features():
    frozen=json.loads((PUBLIC/'feature_freeze.json').read_text())
    if frozen['registration_sha256']!=sha(REG) or set(frozen['arrays'])!=set(api.FIELDS)|{'ids','recordings','sites'}:
        raise ValueError('Exact frozen causal bridge required')
    arrays={k:np.load(checked(v),mmap_mode='r',allow_pickle=False) for k,v in frozen['arrays'].items()}
    if len(arrays['ids'])!=38102 or not np.array_equal(arrays['ids'],np.arange(38102)):
        raise ValueError('All explicit new-namespace selection rows required')
    return arrays


def checked_state(ref):
    row=json.loads(checked(ref['receipt']).read_text());state=torch.load(checked(ref['checkpoint']),map_location='cpu',weights_only=False)
    if (state['step']!=4000 or state['seed']!=ref['seed'] or state['identity']['parent_trial']!=row['parent_trial']
            or state['identity']['normalizer']!=ref['normalizer'] or row['normalizer']!=ref['normalizer']
            or state['identity']['registration_sha256']!=sha(producers.REG)):
        raise ValueError('Actual source-trained model/normalizer lineage differs')
    if producers.source.array_hash(state['train_ids'])!=ref['train_ids_sha256']:
        raise ValueError('Actual training row identities differ')
    packed=json.loads((PACKED/'receipt.json').read_text());path=PACKED/'sites.npy'
    if sha(path)!=packed['arrays']['sites']:raise ValueError('Original source site mapping changed')
    sites=np.load(path,mmap_mode='r',allow_pickle=False);ids=np.asarray(state['train_ids'])
    if not np.isin(sites[ids],ref['fit_sites']).all() or set(sites[ids])!=set(ref['fit_sites']):
        raise ValueError('Training row sites do not match source-only fitting roster')
    sampled=np.flatnonzero(state['draws'])
    if not np.isin(sampled,ids).all() or not np.isin(sites[sampled],ref['fit_sites']).all():
        raise ValueError('Held or reserved row sampled during original training')
    return state


def inference_view(ref,data):
    extra=representation(data['detector_history'],data['recordings'],data['agents'],data['frames'],ref['representation'])
    norm=ref['normalizer']
    return DetectorGeometry(data['geometry'],extra,np.asarray(norm['mean']),np.asarray(norm['std']),norm['clip'])


def predict_one(ref,data,path,cfg,beat,guard,*,resume=False,limit=None,stop_at=None):
    n=len(data['ids']) if limit is None else limit;end=n if stop_at is None else stop_at
    if not 0<end<=n:raise ValueError('Fixed positive inference budget required')
    binding=dict(registration_sha256=sha(REG),feature_freeze_sha256=sha(PUBLIC/'feature_freeze.json'),
        checkpoint=ref['checkpoint'],rows=n,batch_size=cfg['batch_size'],coordinate_frame='absolute_local_image_pixel',
        row_id_namespace='european_selection_v1')
    path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_name(path.stem+'.incomplete.npy');progress=path.with_suffix('.progress.json')
    if path.exists():
        if not resume:raise ValueError('Explicit prediction resume required')
        old=json.loads(path.with_suffix('.json').read_text())
        if old['binding']!=binding or old['prediction']!=artifact(path):raise ValueError('Completed prediction changed')
        return old
    done=0
    if progress.exists():
        if not resume:raise ValueError('Explicit partial prediction resume required')
        old=json.loads(progress.read_text())
        if old['binding']!=binding:raise ValueError('Prediction checkpoint inputs changed')
        out=np.load(partial,mmap_mode='r+',allow_pickle=False);done=old['rows_complete']
        if out.shape!=(n,12,2) or not 0<=done<=end or fingerprint(out[:done])!=old['completed_prefix_sha256']:
            raise ValueError('Last committed prediction prefix changed')
    else:
        out=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float64,shape=(n,12,2));out[:]=0.;out.flush()
    state=checked_state(ref);torch.manual_seed(ref['seed'])
    model=producers.api.ActiveDetectorForecaster(ref['baseline_index'],ref['representation'],width=64,heads=4,layers=2)
    model.load_state_dict(state['model']);model.eval();view=inference_view(ref,data);begin=time.monotonic()
    def save(count):
        out.flush();write(progress,dict(binding=binding,rows_complete=count,completed_prefix_sha256=fingerprint(out[:count])))
    if not progress.exists():save(0)
    for lo in range(done,end,cfg['batch_size']):
        guard();hi=min(lo+cfg['batch_size'],end);ids=np.arange(lo,hi)
        with torch.no_grad():relative=model(pack_geometry(view[ids])).numpy()
        prediction=relative.astype(np.float64)+data['origin'][ids,None]
        if prediction.shape!=(len(ids),12,2) or not np.isfinite(prediction).all():raise FloatingPointError('Finite12-step forecasts required')
        out[lo:hi]=prediction
        if (hi//cfg['batch_size'])%cfg['checkpoint_every_batches']==0 or hi==end:
            save(hi);beat('model_inference_progress',trial=ref['trial'],representation=ref['representation'],rows=hi,total=n)
    del out
    if end<n:return dict(incomplete=True,rows_complete=end,binding=binding)
    partial.replace(path);out=np.load(path,mmap_mode='r',allow_pickle=False)
    value=dict(binding=binding,prediction=artifact(path),trial=ref['trial'],representation=ref['representation'],
        source_training_seed=ref['seed'],actual_train_ids_and_sampled_sites_checked=True,source_normalizer_reused=True,
        stored_dtype=str(out.dtype),shape=list(out.shape),seconds=time.monotonic()-begin,
        forecast_errors_opened=False,new_optimizer_updates=0)
    once(path.with_suffix('.json'),value)
    return value


def fresh_prediction(ref,data,cfg,beat,guard,*,limit=None):
    n=len(data['ids']) if limit is None else limit;state=checked_state(ref);torch.manual_seed(ref['seed'])
    model=producers.api.ActiveDetectorForecaster(ref['baseline_index'],ref['representation'],width=64,heads=4,layers=2)
    model.load_state_dict(state['model']);model.eval();view=inference_view(ref,data);out=np.empty((n,12,2),np.float64)
    for lo in range(0,n,cfg['batch_size']):
        guard();hi=min(lo+cfg['batch_size'],n);ids=np.arange(lo,hi)
        with torch.no_grad():out[lo:hi]=model(pack_geometry(view[ids])).numpy().astype(float)+data['origin'][ids,None]
    if not np.isfinite(out).all():raise FloatingPointError('Finite replay required')
    return out


def parent_prediction(job,data,cfg,guard,*,limit=None):
    n=len(data['ids']) if limit is None else limit;identity=job['identity']
    state=torch.load(checked(job['checkpoint']),map_location='cpu',weights_only=False)
    if state['step']!=4000 or state['seed']!=identity['seed']:raise ValueError('Complete actual dimensionless parent required')
    model=DimensionlessAgentTrackSourceForecaster(identity['baseline_index'],width=64,heads=4,layers=2)
    model.load_state_dict(state['model']);model.eval();out=np.empty((n,12,2),np.float64)
    for lo in range(0,n,cfg['batch_size']):
        guard();hi=min(lo+cfg['batch_size'],n);ids=np.arange(lo,hi)
        with torch.no_grad():out[lo:hi]=model(plain_pack(data['geometry'][ids])).numpy().astype(float)+data['origin'][ids,None]
    return out


def prediction_stage(phase,cfg,roles,selected,archive,reg,*,resume=False):
    beat,guard,receipt=runtime(cfg,phase);data=load_features()
    check=json.loads((PUBLIC/'feature_verification.json').read_text())
    if not check['all28_raw_reparsed_all_features_exact'] or check['feature_freeze_sha256']!=sha(PUBLIC/'feature_freeze.json'):
        raise ValueError('Complete raw feature replay required before forecasts')
    finish=PUBLIC/('prediction_pilot.json' if phase=='prediction_pilot' else 'prediction_verification.json' if phase=='verify' else 'prediction_freeze.json')
    if finish.exists():raise ValueError('Completed prediction stage immutable')
    jobs={j['key']:j for j in reg['parent_jobs']}
    if phase=='prediction_pilot':
        ref=next(r for r in reg['lineage'] if r['trial']=='single0_seed17' and r['representation']=='history')
        path=PRIVATE/'pilot/prediction.npy';prefix=cfg['pilot_rows']
        if not path.with_suffix('.progress.json').exists() and not path.exists():
            predict_one(ref,data,path,cfg,beat,guard,limit=prefix,stop_at=cfg['pilot_interrupt_rows'])
        recovered=predict_one(ref,data,path,cfg,beat,guard,resume=True,limit=prefix)
        direct=fresh_prediction(ref,data,cfg,beat,guard,limit=prefix)
        np.testing.assert_array_equal(np.load(path,allow_pickle=False),direct)
        disabled=next(r for r in reg['lineage'] if r['trial']=='single0_seed17' and r['representation']=='disabled')
        dp=fresh_prediction(disabled,data,cfg,beat,guard,limit=prefix)
        np.testing.assert_array_equal(dp,parent_prediction(jobs[ref['trial']],data,cfg,guard,limit=prefix))
        once(finish,dict(**receipt(),engineering_pass=True,rows=prefix,
            exact_interrupted_prediction_recovery=True,disabled_parent_exact=True,
            pilot_model=ref['checkpoint']))
        beat('prediction_pilot_complete',rows=prefix);return
    if not json.loads((PUBLIC/'prediction_pilot.json').read_text())['engineering_pass']:
        raise ValueError('Actual inference resource/recovery pilot required')
    refs=[];neutral=[];verify=phase=='verify'
    if verify:
        frozen=json.loads((PUBLIC/'prediction_freeze.json').read_text())
        if frozen['registration_sha256']!=sha(REG):raise ValueError('Frozen forecast identity changed')
    for ref in reg['lineage']:
        path=PRIVATE/'predictions'/(ref['trial']+'_'+ref['representation']+'.npy')
        if verify:
            record=json.loads(path.with_suffix('.json').read_text())
            if record['prediction']!=artifact(path):raise ValueError('Saved full forecast changed')
            direct=fresh_prediction(ref,data,cfg,beat,guard)
            np.testing.assert_array_equal(np.load(path,allow_pickle=False,mmap_mode='r'),direct)
        else:record=predict_one(ref,data,path,cfg,beat,guard,resume=resume)
        if ref['representation']=='disabled':
            np.testing.assert_array_equal(np.load(path,mmap_mode='r',allow_pickle=False),parent_prediction(jobs[ref['trial']],data,cfg,guard))
            neutral.append(ref['trial'])
        refs.append(artifact(path.with_suffix('.json')));beat('forecast_model_complete',models=len(refs),total=36,trial=ref['trial'],representation=ref['representation'])
    baseline=PRIVATE/'physical_baselines.npy'
    physical=np.stack([producers.source.baseline_numpy(data['history'],k) for k in range(6)],1)
    if verify:np.testing.assert_array_equal(np.load(baseline,mmap_mode='r',allow_pickle=False),physical)
    elif baseline.exists():np.testing.assert_array_equal(np.load(baseline,mmap_mode='r',allow_pickle=False),physical)
    else:inputs.store_array(baseline,physical)
    if len(refs)!=36 or len(neutral)!=9:raise ValueError('All36models/nine parents required')
    value=dict(**receipt(),models=refs,rows=len(data['ids']),actual_training_seeds=cfg['seeds'],producer_folds=cfg['producer_folds'],
        disabled_parent_trials_exact=neutral,physical_baselines=artifact(baseline),
        feature_freeze_sha256=sha(PUBLIC/'feature_freeze.json'),row_id_namespace='european_selection_v1',
        coordinate_frame='absolute_local_image_pixel',normalization_fitted=False,model_selection_by_outcomes=False,
        calibration_access=False,confirmation_access=False,deployment_changed=False)
    if verify:
        if refs!=frozen['models'] or value['physical_baselines']!=frozen['physical_baselines']:raise ValueError('Complete forecast roster changed')
        value.update(all36fresh_predictions_exact=True,all9_disabled_parent_predictions_exact=True,
            prediction_freeze_sha256=sha(PUBLIC/'prediction_freeze.json'))
    once(finish,value);beat('all36_forecasts_verified' if verify else 'all36_forecasts_frozen',rows=len(data['ids']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','feature_pilot','features','feature_verify','prediction_pilot','predict','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        values=admission(args.phase=='register')
        if args.phase.startswith('feature'):feature_stage(args.phase,*values,resume=args.resume)
        elif args.phase!='register':prediction_stage(args.phase,*values,resume=args.resume)
