"""Frozen-source rare-event support and annotation provenance; no policy fit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time
if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import torch
from scripts import run_m3w_zero_atom as parent
from scripts.diagnose_m3w_zero_reference_support import native_history
from scripts.audit_m3w_sdd_state_support import load_source
from src.world_model.m3w_causal_motion_support import motion_profile
from src.world_model.m3w_sdd_state_support import frame_lookup,control_provenance,validate_track
from src.world_model.m3w_native_gain_harm import standardized
from src.evaluation.m3w_moving_zero_support import fixed_controls,neighbors,describe_neighbors

CONFIG='configs/m3w_moving_zero_support_v1.json'
CODE=('scripts/audit_m3w_moving_zero_support.py','src/evaluation/m3w_moving_zero_support.py',
      'tests/test_m3w_moving_zero_support.py','scripts/diagnose_m3w_zero_reference_support.py',
      'scripts/audit_m3w_sdd_state_support.py','src/world_model/m3w_causal_motion_support.py',
      'src/world_model/m3w_sdd_state_support.py')
base=parent.base


def load():
    cfg=parent.read(ROOT/CONFIG); ctx=parent.load()
    assert base.file_digest(ROOT/cfg['parent_analysis'])==cfg['parent_analysis_sha256']
    assert base.file_digest(ROOT/cfg['source_manifest'])==cfg['source_manifest_sha256']
    assert not any(cfg[k] for k in ('new_training','new_policy','threshold_search','external_readout','deployment','stage5c_executed','smc_enabled'))
    assert cfg['controls_per_view']==32 and cfg['neighbor_counts']==[32,128,512]
    assert cfg['feature_arms']==['history','full_risk']
    bindings=dict(ctx['identity']['source_bindings'])
    for p in (CONFIG,cfg['parent_analysis'],cfg['source_manifest'],cfg['reports']+'/registration.md',*CODE):
        bindings[p]=base.file_digest(ROOT/p)
    root=ROOT/ctx['cfg']['output']; assert ctx['identity']==parent.read(root/'identity.json')
    a=parent.read(ROOT/cfg['parent_analysis'])
    assert base.file_digest(root/'decisions_complete.json')==a['decisions_sha256']
    for path in (root/'identity.json',root/'decisions_complete.json'):
        bindings[str(path.relative_to(ROOT))]=base.file_digest(path)
    decisions={}
    for ref in parent.read(root/'decisions_complete.json')['archives']:
        assert base.file_digest(ROOT/ref['path'])==ref['sha256']
        r=parent.read(ROOT/ref['path']); assert base.file_digest(ROOT/r['path'])==r['sha256']
        bindings[ref['path']],bindings[r['path']]=ref['sha256'],r['sha256']
        decisions[r['view'],r['action']]=r
    raw=parent.read(ROOT/cfg['source_manifest'])
    for r in raw['records']:
        if r['scene_id'] in ctx['cfg']['sites']: bindings[r['annotations_path']]=r['annotations_sha256']
    identity=dict(config=cfg,source_bindings=bindings,runtime=ctx['identity']['runtime'],
        role='descriptive_design_exposed_source_only_not_calibration')
    base.assert_current(identity)
    return cfg,ctx,a,decisions,raw,identity


def annotation_cases(ids,data,history,offsets,manifest):
    profile=motion_profile(history,offsets); rows=[]
    for rec in np.unique(data['recordings'][ids]):
        ref=next(r for r in manifest['records'] if r['annotation_key']==rec)
        assert base.file_digest(ROOT/ref['annotations_path'])==ref['annotations_sha256']
        raw,labels=load_source(ROOT/ref['annotations_path'])
        for i in ids[data['recordings'][ids]==rec]:
            aid=int(data['tracks'][i].rsplit(':',1)[1]); q=int(data['frames'][i])
            take=raw[:,0]==aid; track=validate_track(raw[take]); lab=labels[take]
            hi=frame_lookup(track,q+np.arange(-7,1)*12)
            fi=frame_lookup(track,q+np.arange(1,13)*12)
            assert (hi>=0).all() and (fi>=0).all()
            assert not track[hi,6].any() and not track[fi,6].any()
            xy=(track[:,1:3]+track[:,3:5])/2
            np.testing.assert_array_equal(xy[hi],history[i])
            assert lab[hi[-1]]=='Pedestrian'
            cv=history[i,-1]+np.arange(1,13)[:,None]*(history[i,-1]-history[i,-2])
            error=np.linalg.norm(cv-xy[fi],axis=1)
            provenance=control_provenance(track)
            nxt=provenance['next_control_row'][hi]; generated=track[hi,8]==1
            later=generated&(nxt>=0)&(track[np.maximum(nxt,0),5]>q)
            raw_future=frame_lookup(track,np.arange(q+1,q+145))
            assert (raw_future>=0).all() and not track[raw_future,6].any()
            dense_cv=history[i,-1]+np.arange(1,145)[:,None]*(history[i,-1]-history[i,-2])/12
            dense_error=np.linalg.norm(dense_cv-xy[raw_future],axis=1)
            futgen=track[fi,8]==1; bracket=provenance['bracketed'][fi]
            pe=provenance['max_box_interpolation_error'][fi]
            # These provenance fields are retrospective audit labels, not features.
            rows.append(dict(row=int(i),site=str(data['sites'][i]),recording=str(rec),
                scoped_track=str(data['tracks'][i]),frame=q,
                history_hash=base.array_hash(history[i],offsets[i]),
                **{k:bool(v[i]) if v.dtype==bool else float(v[i]) for k,v in profile.items()},
                raw_sampled_cv_ADE=float(error.mean()),raw_sampled_cv_max_error=float(error.max()),
                dense_raw_cv_ADE=float(dense_error.mean()),dense_raw_cv_max_error=float(dense_error.max()),
                past_generated_count=int(generated.sum()),future_generated_count=int(futgen.sum()),
                past_rows_with_post_query_control=int(later.sum()),
                future_bracketed_generated=int((futgen&bracket).sum()),
                future_interpolation_box_error_max=float(np.nanmax(pe)) if np.isfinite(pe).any() else None,
                past_occluded_count=int(track[hi,7].sum()),future_occluded_count=int(track[fi,7].sum())))
    return rows


def calculate(cfg,ctx,decisions,key,action,zero,controls,data):
    pc=ctx['parent']; ids,raw,d,pr,q,draws,scale=base.prepare(pc['pack'],key,action,pc['refs'])
    ref=ctx['refs'][key,action]; cp=joblib.load(ROOT/ref['checkpoint'])
    assert cp['identity']==ref['identity']
    x=parent.parent.features(raw,d,scale,ref['identity']['cutoff'])
    assert base.array_hash(ids,x,d)==ref['identity']['inputs_sha256']
    assert base.array_hash(q)==ref['identity']['targets_sha256']
    event=parent.atom.zero_event(q,pr['known']); w=parent.atom.source_weights(pr['known'],draws,d)
    np.testing.assert_array_equal(w,cp['sample_weight'])
    h=data['geometry'][ids,:16].reshape(-1,8,2)
    moving=np.any(h[:,-1]!=h[:,-2],axis=1); use=(w>0)&moving
    z=standardized(x,cp['preprocess'])[use]; source_ids=ids[use]
    assert key.rsplit('_seed',1)[0] not in data['sites'][source_ids]
    values,p,_,cutoff=base.causal_view(pc['pack'][1],key,action)
    held=values['ids']; hh=data['geometry'][held]; ss=data['scale'][held]
    raw,dd,_=base.native_features(hh,p,ss)
    xx=parent.parent.features(raw,dd,ss,cutoff)
    zz=standardized(xx,cp['preprocess'])
    selected=np.unique(np.r_[held[zero[held]],controls])
    pos=np.searchsorted(held,selected); np.testing.assert_array_equal(held[pos],selected)
    with np.load(ROOT/decisions[key,action]['path'],allow_pickle=False) as a:
        np.testing.assert_array_equal(a['ids'],held); probability=a['probability'][pos]
    hc=np.ravel(np.column_stack((np.arange(8)*3,np.arange(8)*3+1)))
    # Confirm the frozen risk schema's history channels before subsetting.
    np.testing.assert_array_equal(x[:,hc],data['geometry'][ids,:16])
    out=[]
    for arm in cfg['feature_arms']:
        train=z[:,hc] if arm=='history' else z
        queries=zz[pos][:,hc] if arm=='history' else zz[pos]
        for i,v,prob in zip(selected,queries,probability):
            r=neighbors(train,v,source_ids,k=512)
            s=describe_neighbors(r,event[use],q[use,0],q[use,1],data['tracks'][source_ids],ks=cfg['neighbor_counts'])
            out.append(dict(view=key,action=action,arm=arm,row=int(i),
                zero_case=bool(zero[i]),metadata_control=bool(i in controls),atom_probability=float(prob),**s))
    return dict(view=key,action=action,source_rows=int(use.sum()),source_tracks=len(np.unique(data['tracks'][source_ids])),
        source_zero_rows=int(event[use].sum()),source_zero_tracks=len(np.unique(data['tracks'][source_ids][event[use]])),
        source_inputs_sha256=ref['identity']['inputs_sha256'],source_targets_sha256=ref['identity']['targets_sha256'],
        query_ids_sha256=base.array_hash(selected),records=out)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--verify',action='store_true'); parser.add_argument('--view'); parser.add_argument('--action')
    args=parser.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg=parent.read(ROOT/CONFIG); root=ROOT/cfg['output']; root.mkdir(parents=True,exist_ok=True)
    def beat(**values):
        row=dict(pid=os.getpid(),unix=time.time(),**values); base.json_write(root/'heartbeat.json',row)
        with (root/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); beat(state='verifying_sources')
        cfg,ctx,old,decisions,manifest,identity=load(); base.immutable_json(root/'identity.json',identity)
        data=ctx['parent']['pack'][1][1]; history,offsets=native_history(ctx['parent']['pack'])
        profile=motion_profile(history,offsets)
        h=data['geometry'][:,:16].reshape(-1,8,2); moving=np.any(h[:,-1]!=h[:,-2],axis=1)
        np.testing.assert_array_equal(moving,~profile['last_stop'])
        ref=next(r for r in ctx['parent']['pack'][1][3]['outcome_archives'] if r['path'].endswith('transformer_seed17.npz'))
        assert base.file_digest(ROOT/ref['path'])==ref['sha256']
        with np.load(ROOT/ref['path'],allow_pickle=False) as z: zero=z['zero_CV']&moving
        ids=np.flatnonzero(zero); assert len(ids)==7
        cases=annotation_cases(ids,data,history,offsets,manifest)
        base.immutable_json(root/'raw_cases.json',cases)
        beat(state='raw_cases_verified',rows=len(ids))
        receipts=[]
        for key in ctx['parent']['pack'][1][4]:
            if args.view and key!=args.view: continue
            site=key.rsplit('_seed',1)[0]
            controls=fixed_controls(np.flatnonzero((data['sites']==site)&moving),data['recordings'],data['frames'],data['tracks'],32)
            for action in ctx['cfg']['actions']:
                if args.action and action!=args.action: continue
                path=root/'views'/f'{key}_{action}.json'; start=time.monotonic()
                if path.exists() and args.resume and not args.verify:
                    r=parent.read(path); assert r['identity_sha256']==base.file_digest(root/'identity.json')
                    beat(state='cached_verified_view',view=key,action=action)
                else:
                    r=calculate(cfg,ctx,decisions,key,action,zero,controls,data)
                    r['identity_sha256']=base.file_digest(root/'identity.json')
                    if args.verify and not path.exists(): raise ValueError('Replay needs existing record')
                    base.immutable_json(path,r)
                    beat(state='view_complete',view=key,action=action,seconds=time.monotonic()-start)
                receipts.append(dict(path=str(path.relative_to(ROOT)),sha256=base.file_digest(path)))
        if args.view or args.action:
            beat(state='pilot_complete_not_full_matrix',views=len(receipts)); return
        assert len(receipts)==36
        results=[parent.read(ROOT/r['path']) for r in receipts]
        public=ROOT/cfg['reports']; allrecords=[r for v in results for r in v['records']]
        control={}
        for action in ctx['cfg']['actions']:
            for arm in cfg['feature_arms']:
                rr=[r for r in allrecords if r['action']==action and r['arm']==arm and r['metadata_control']]
                control[action+'__'+arm]=dict(rows=len(rr),
                    nearest_distance_quantiles=np.quantile([r['nearest_distance'] for r in rr],[0,.25,.5,.75,1]).tolist(),
                    nearest_zero_rank_fraction_quantiles=np.quantile([r['nearest_zero_rank_fraction'] for r in rr],[0,.25,.5,.75,1]).tolist(),
                    exact_rows=sum(r['exact_rows'] for r in rr),exact_zero_rows=sum(r['exact_zero_rows'] for r in rr),
                    neighborhoods={str(k):dict(mean_zero_rows=float(np.mean([r['neighborhoods'][str(k)]['zero_rows'] for r in rr])),
                        mean_tracks=float(np.mean([r['neighborhoods'][str(k)]['tracks'] for r in rr]))) for k in cfg['neighbor_counts']})
        report=dict(identity_sha256=base.file_digest(root/'identity.json'),parent_analysis_sha256=cfg['parent_analysis_sha256'],
            result_sources=dict(distances_and_annotation_checks='fresh_run',forests_and_forecasts='cached_verified',
                new_training='not_run',external_readout='not_run'),views=receipts,
            support=[{k:v for k,v in r.items() if k!='records'} for r in results],
            raw_cases=cases,case_neighborhoods=[r for r in allrecords if r['zero_case']],control_summary=control,
            full_source_histories_checked=len(data['sites']),all_native_moving_flags_match=True,
            controls_are_outcome_blind=True,no_policy_change=True,independent_confirmation=False,
            stage5c_executed=False,smc_enabled=False)
        base.assert_current(identity); base.immutable_json(public/'analysis.json',report)
        if args.verify: base.immutable_json(public/'replay.json',dict(exact=True,analysis_sha256=base.file_digest(public/'analysis.json')))
        beat(state='full_diagnosis_complete',views=36,cases=len(cases))


if __name__=='__main__': main()
