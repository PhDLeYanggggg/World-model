"""Registered nested source-only neural forecasting; no reserved readout."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.fetch_m3w_european_squares import digest,save
from scripts.run_m3w_native_forecast import array_hash,json_write,immutable_json
from src.evaluation.m3w_european_squares_roles import require_source_training
from src.evaluation.m3w_native_metrics import native_errors,paired_scene_metrics
from src.world_model.m3w_european_source_forecast import (
    BASELINES,source_folds,pack_scene,fit_design,SourceForecaster,baseline_numpy)
from src.world_model.m3w_native_forecast import fit_trial,predict

CONFIG = 'configs/m3w_european_source_forecast_v1.json'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_source_forecast_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_source_forecast_v1'
FILES = (CONFIG,'scripts/run_m3w_european_source_forecast.py',
    'src/world_model/m3w_european_source_forecast.py','tests/test_m3w_european_source_forecast.py',
    'src/world_model/m3w_native_forecast.py','scripts/run_m3w_native_forecast.py',
    'src/world_model/m3w_supervised_intervention.py','src/world_model/m3w_context_conditioning.py',
    'src/world_model/m3w_baseline_relative_forecaster.py','src/evaluation/m3w_native_metrics.py',
    'src/evaluation/m3w_european_squares_roles.py')


def beat(state,**values):
    event = dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,**values)
    json_write(PRIVATE/'heartbeat.json',event)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(event)+'\n')
    print(json.dumps(event),flush=True)


def load_source():
    reg = json.loads((ROOT/CONFIG).read_text())
    if reg['baselines'] != list(BASELINES) or reg['seeds'] != [17,29,43] or reg['independent_reserved_readout']:
        raise ValueError('Registered source-only matrix differs')
    for path,sha in [(reg['source_manifest'],reg['source_manifest_sha256']),(reg['roles'],reg['roles_sha256'])]:
        if digest(ROOT/path) != sha:
            raise ValueError('Source/role identity differs')
    manifest = json.loads((ROOT/reg['source_manifest']).read_text())
    verified = json.loads((ROOT/reg['source_manifest']).with_name('verification.json').read_text())
    if not verified['all_arrays_exact'] or verified['analysis_sha256'] != reg['source_manifest_sha256']:
        raise ValueError('Full raw source replay missing')
    roles = json.loads((ROOT/reg['roles']).read_text())
    for entry in roles['dependency_bindings'].values():
        if digest(ROOT/entry['path']) != entry['sha256']:
            raise ValueError('Frozen source-role dependency changed')
    receipts = []
    support = {}
    for r in manifest['record_receipts']:
        access = require_source_training(roles,r['source_member'])
        path = ROOT/r['directory']/'receipt.json'
        if digest(path) != r['receipt_sha256']:
            raise ValueError('Source receipt differs')
        receipt = json.loads(path.read_text())
        if receipt['rows_sha256'] != access['rows_sha256'] or receipt['role'] != 'source_training':
            raise ValueError('Unapproved source receipt')
        for v in receipt['arrays'].values():
            if digest(path.parent/v['name']) != v['sha256']:
                raise ValueError('Source array hash changed')
        receipts.append((path.parent,receipt))
        support[r['locality_group']] = support.get(r['locality_group'],0)+receipt['support']['all_past_eligible_targets']
    folds = source_folds(support,reg['fold_salt'])
    bindings = {p:digest(ROOT/p) for p in (*FILES,reg['registration'])}
    identity = dict(bindings=bindings,source_manifest_sha256=reg['source_manifest_sha256'],
        source_roles_sha256=reg['roles_sha256'],folds=folds,numpy=np.__version__,torch=torch.__version__,
        architecture=platform.machine(),runtime=reg['runtime'])
    immutable_json(PRIVATE/'identity.json',identity)
    return reg,manifest,receipts,identity


def prepare(manifest,receipts,identity):
    directory = PRIVATE/'packed'
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/'receipt.json'
    if path.exists():
        receipt = json.loads(path.read_text())
        if receipt['identity'] != identity:
            raise ValueError('Packed population identity differs')
        for k,v in receipt['arrays'].items():
            if digest(directory/(k+'.npy')) != v:
                raise ValueError('Packed array differs')
        return {k:np.load(directory/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in receipt['arrays']}
    n = manifest['totals']['target_rows']
    specs = dict(geometry=('float32',(n,476)),target=('float32',(n,12,2)),target_eval=('float64',(n,12,2)),
        valid=('bool',(n,12)),history=('float64',(n,8,2)),origin=('float64',(n,2)),width=('float64',(n,)),
        sites=('<U20',(n,)),recordings=('int32',(n,)),frames=('int64',(n,)),agents=('int64',(n,)),
        baseline_ade=('float64',(n,6)),baseline_fde=('float64',(n,6)))
    out = {k:np.lib.format.open_memmap(directory/(k+'.npy'),mode='w+',dtype=dtype,shape=shape)
           for k,(dtype,shape) in specs.items()}
    cursor = 0
    for ri,(source,receipt) in enumerate(receipts):
        beat('packing_source',record=ri+1,total=len(receipts))
        arrays = {k:np.load(source/v['name'],mmap_mode='r',allow_pickle=False) for k,v in receipt['arrays'].items()}
        inputs = {k.removeprefix('input_'):v for k,v in arrays.items() if k.startswith('input_')}
        offsets = inputs['query_offsets']
        start_cursor = cursor
        for start,end in zip(offsets[:-1],offsets[1:]):
            scene = {k:v[start:end] for k,v in inputs.items() if k != 'query_offsets'}
            g,local = pack_scene(scene)
            num = len(g)
            dest = slice(cursor,cursor+num)
            ids = start+local
            out['geometry'][dest] = g
            h = inputs['history_xy'][ids]
            y = arrays['label_future_xy'][ids]
            valid = arrays['label_future_valid'][ids]
            out['history'][dest] = h
            out['origin'][dest] = h[:,-1]
            out['target'][dest] = np.where(valid[...,None],y-h[:,-1,None],0).astype(np.float32)
            out['target_eval'][dest] = y
            out['valid'][dest] = valid
            boxes = inputs['history_boxes'][ids,-1]
            out['width'][dest] = boxes[:,2]-boxes[:,0]
            out['sites'][dest] = receipt['locality_group']
            out['recordings'][dest] = ri
            out['frames'][dest] = inputs['query_frame'][ids]
            out['agents'][dest] = inputs['agent_id'][ids]
            for index in range(6):
                ade,fde = native_errors(baseline_numpy(h,index),y,valid,np.ones(num))
                out['baseline_ade'][dest,index] = ade
                out['baseline_fde'][dest,index] = fde
            cursor += num
        if cursor-start_cursor != receipt['target_rows']:
            raise ValueError('Source target count differs')
    if cursor != n:
        raise ValueError('Full source cohort not packed')
    for value in out.values():
        value.flush()
    immutable_json(path,dict(identity=identity,arrays={k:digest(directory/(k+'.npy')) for k in out},
        records=[r['source_member'] for _,r in receipts],rows=n,source_only=True))
    return out


def trials(reg,data,identity):
    designs = []
    for kind in ('single','complement'):
        for fold in range(3):
            sites = [s for s,v in identity['folds'].items() if (v==fold)==(kind=='single')]
            design = fit_design(data,sites,data['baseline_ade'])
            for seed in reg['seeds']:
                key = f'{kind}{fold}_seed{seed}'
                ti = dict(identity=identity,kind=kind,fold=fold,seed=seed,fit_sites=sorted(sites),
                    train_ids_sha256=array_hash(design['train_ids']),held_ids_sha256=array_hash(design['held_ids']),
                    factor_sha256=array_hash(design['factors']),normalizers=design['normalizers'],
                    baseline_index=design['baseline_index'],selection_relative_scores=design['selection_relative_scores'],
                    easy_cut=design['easy_cut'],hard_cut=design['hard_cut'])
                designs.append((key,design,ti))
    return designs


def complete(key,ti,reg):
    path = PRIVATE/'trials'/key/'complete.json'
    if not path.exists():
        raise ValueError('All eighteen fixed training endpoints required before readout')
    r = json.loads(path.read_text())
    if r['identity'] != ti or not r['fit']['complete'] or r['fit']['step'] != reg['training']['steps']:
        raise ValueError('Incomplete or changed trial')
    if digest(ROOT/r['checkpoint']) != r['checkpoint_sha256']:
        raise ValueError('Checkpoint differs')
    return r


def assert_identity(identity):
    for path,sha in identity['bindings'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen experiment dependency changed: '+path)


def evaluate(reg,data,identity,designs,verify):
    training = {key:complete(key,ti,reg) for key,_,ti in designs}
    prediction_receipts = []
    scores = {}
    roster = sorted(identity['folds'])
    n = len(data['sites'])
    cv = np.asarray(data['baseline_ade'][:,1])
    for key,design,ti in designs:
        ids = design['held_ids']
        directory = PRIVATE/'predictions'
        directory.mkdir(parents=True,exist_ok=True)
        path = directory/(key+'.npz')
        receipt_path = path.with_suffix('.json')
        fit = training[key]
        if receipt_path.exists():
            r = json.loads(receipt_path.read_text())
            if r['identity'] != ti or r['checkpoint_sha256'] != fit['checkpoint_sha256'] or digest(path) != r['sha256']:
                raise ValueError('Cached predictions differ')
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                p = z['prediction'].copy()
        else:
            if verify:
                raise ValueError('Cannot verify missing predictions')
            beat('source_excluded_inference',trial=key,rows=len(ids))
            torch.manual_seed(ti['seed'])
            model = SourceForecaster(reg['architecture'],design['baseline_index'])
            cp = torch.load(ROOT/fit['checkpoint'],map_location='cpu',weights_only=False)
            if cp['identity'] != ti or cp['step'] != reg['training']['steps']:
                raise ValueError('Checkpoint identity differs')
            model.load_state_dict(cp['model'])
            p = predict(model,data,ids,128)
            temporary = path.with_suffix('.tmp.npz')
            np.savez(temporary,ids=ids,prediction=p)
            os.replace(temporary,path)
            r = dict(identity=ti,checkpoint_sha256=fit['checkpoint_sha256'],sha256=digest(path),
                     path=str(path.relative_to(ROOT)),input_source='query_prefix_and_fixed_causal_rollout_only')
            immutable_json(receipt_path,r)
        prediction_receipts.append(dict(trial=key,path=r['path'],sha256=r['sha256']))
        if ti['kind'] != 'complement':
            continue
        ade,fde = native_errors(p.astype(float)+data['origin'][ids,None],data['target_eval'][ids],
                               data['valid'][ids],np.ones(len(ids)))
        s = scores.setdefault(ti['seed'],dict(ade=np.full(n,np.nan),fde=np.full(n,np.nan),
            reference_ade=np.full(n,np.nan),reference_fde=np.full(n,np.nan),easy=np.zeros(n,bool),
            hard=np.zeros(n,bool),seen=np.zeros(n,bool)))
        if s['seen'][ids].any():
            raise ValueError('Duplicate complement prediction')
        s['seen'][ids] = True
        s['ade'][ids],s['fde'][ids] = ade,fde
        s['reference_ade'][ids] = data['baseline_ade'][ids,design['baseline_index']]
        s['reference_fde'][ids] = data['baseline_fde'][ids,design['baseline_index']]
        s['easy'][ids] = (cv[ids]>0)&(cv[ids]<=design['easy_cut'])
        s['hard'][ids] = cv[ids]>=design['hard_cut']
    def score(a,b,mask=None):
        mask = np.ones(n,bool) if mask is None else mask
        return paired_scene_metrics(a[mask],b[mask],data['sites'][mask],expected_scenes=roster,
            dataset='EuropeanSquares_released_detector_tracks',coordinate_unit='image_pixel',
            bootstrap_resamples=reg['bootstrap_resamples'],seed=reg['bootstrap_seed'])
    result_seeds = {}
    for seed,s in scores.items():
        if not s['seen'].all():
            raise ValueError('Missing complement readout')
        zero = np.isfinite(cv)&(cv==0)
        result_seeds[str(seed)] = dict(ADE_vs_CV=score(s['ade'],cv),
            ADE_vs_training_selected_strongest=score(s['ade'],s['reference_ade']),
            FDE_vs_training_selected_strongest=score(s['fde'],s['reference_fde']),
            hard_ADE_vs_CV=score(s['ade'],cv,s['hard']),
            positive_easy_ADE_vs_CV=score(s['ade'],cv,s['easy']),
            complete_ADE_vs_CV=score(s['ade'],cv,data['valid'].all(1)),
            zero_CV=dict(rows=int(zero.sum()),harmed_rows=int((s['ade'][zero]>0).sum()),
                         total_absolute_added_error=float(s['ade'][zero].sum())))
    average = lambda key:np.mean([s[key] for s in scores.values()],axis=0)
    baselines = {name:score(data['baseline_ade'][:,k],cv) for k,name in enumerate(BASELINES)}
    result = dict(result_source='fresh_training_and_source_excluded_inference',identity=identity,
        training=list(training.values()),prediction_receipts=prediction_receipts,baselines=baselines,seeds=result_seeds,
        mean_seed_ADE_vs_CV=score(average('ade'),cv),
        mean_seed_ADE_vs_training_selected_strongest=score(average('ade'),average('reference_ade')),
        mean_seed_FDE_vs_training_selected_strongest=score(average('fde'),average('reference_fde')),
        total_models=len(training),total_optimizer_updates=sum(v['fit']['step'] for v in training.values()),
        summed_fit_seconds=sum(v['fit']['seconds'] for v in training.values()),
        source_targets=n,locality_groups=len(roster),independent_reserved_readout=False,
        scene_bootstrap_is_conditional_exploratory=True,seed_averaging='errors_not_predictions',
        deployment_changed=False,metric_claim=False,seconds_claim=False,stage5c_executed=False,smc_enabled=False)
    assert_identity(identity)
    immutable_json(PUBLIC/'analysis.json',result)
    if verify:
        immutable_json(PUBLIC/'verification.json',dict(result_source='cached_verified',
            analysis_sha256=digest(PUBLIC/'analysis.json'),prediction_bytes_verified=True,metrics_recomputed=True,
            new_gradient_updates=False,new_inference=False))
    beat('source_readout_complete',models=len(training),updates=result['total_optimizer_updates'],
         ADE_gain_vs_CV=result['mean_seed_ADE_vs_CV']['equal_scene_gain_percent'],
         ADE_gain_vs_strongest=result['mean_seed_ADE_vs_training_selected_strongest']['equal_scene_gain_percent'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    if not any([args.prepare,args.pilot,args.train,args.evaluate,args.verify]):
        parser.error('Explicit phase required')
    for path in (PRIVATE,PUBLIC):
        if any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError('Symlinked experiment destination')
        path.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        reg,manifest,receipts,identity = load_source()
        beat('verified_source',records=len(receipts),architecture=platform.machine(),
             torch=torch.__version__,threads=torch.get_num_threads(),workers=0)
        data = prepare(manifest,receipts,identity)
        designs = trials(reg,data,identity)
        immutable_json(PUBLIC/'matrix.json',dict(identity=identity,trials=[ti for _,_,ti in designs],
            total_models=len(designs),updates_per_model=reg['training']['steps'],reserved_outcomes_opened=False))
        if args.train or args.pilot:
            for key,design,ti in designs[:1] if args.pilot else designs:
                directory = PRIVATE/'trials'/key
                if (directory/'complete.json').exists():
                    complete(key,ti,reg)
                    beat('completed_trial_verified',trial=key)
                    continue
                beat('begin_trial',trial=key,training_sites=ti['fit_sites'],baseline=BASELINES[design['baseline_index']])
                torch.manual_seed(ti['seed'])
                model = SourceForecaster(reg['architecture'],design['baseline_index'])
                fit = fit_trial(model,data,design,seed=ti['seed'],settings=reg['training'],identity=ti,
                    directory=directory,resume=args.resume,stop_at=100 if args.pilot else None,
                    heartbeat=lambda **kw:beat(trial=key,**kw))
                cp = directory/'checkpoint.pt'
                output = dict(identity=ti,fit=fit,checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=digest(cp))
                immutable_json(directory/('complete.json' if fit['complete'] else 'pilot.json'),output)
                assert_identity(identity)
                beat('trial_endpoint',trial=key,complete=fit['complete'],step=fit['step'],fit_seconds=fit['seconds'])
        if args.evaluate or args.verify:
            evaluate(reg,data,identity,designs,args.verify)
        beat('phase_complete')


if __name__ == '__main__':
    main()
