"""Admitted native 8/12 source data; causal inputs and future labels stay separate."""
from pathlib import Path
import hashlib
import json

import numpy as np

from src.data_unification.m3w_movingdronecrowd_intake import mot_record
from src.evaluation.m3w_dronecrowd_windows import WindowSpec,past_inputs,target_labels
from src.evaluation.m3w_moving_development_admission import load_admission,require_fit_record
from src.evaluation.m3w_native_metrics import native_errors
from src.world_model.m3w_european_source_forecast import baseline_numpy,BASELINES

SPEC=WindowSpec('moving_obs8_pred12_native_stride1',8,1,tuple(range(1,13)))
SHARED_NAMES=tuple(f'past_step_{t}_{name}' for t in range(-6,1)
                   for name in ('mean_dx','mean_dy','median_dx','median_dy','motion_RMS_spread','supported_fraction'))+(
    'log_current_agents','fraction_complete_history','crowd_mean_dx','crowd_mean_dy',
    'mean_neighbor_radius','nearest_neighbor_radius')


def encode_scene(history,mask,agent_ids):
    h=np.asarray(history,float);m=np.asarray(mask);agents=np.asarray(agent_ids)
    if (h.shape!=(len(agents),8,2) or m.shape!=h.shape[:2] or m.dtype!=bool
            or len(np.unique(agents))!=len(agents) or not m[:,-1].all()
            or not np.isfinite(h[m]).all()):
        raise ValueError('Unique current-visible agents, past history and masks required')
    targets=np.flatnonzero(m.all(1));g=np.zeros((len(targets),476),np.float32)
    context=np.zeros((len(targets),len(SHARED_NAMES)),np.float32)
    times=np.arange(-7,1,dtype=np.float32)/12
    shared=np.zeros((7,6),float)
    for step in range(1,8):
        use=m[:,step]&m[:,step-1]
        v=h[use,step]-h[use,step-1]
        if len(v):
            mean=v.mean(0)
            shared[step-1,:2]=mean;shared[step-1,2:4]=np.median(v,axis=0)
            shared[step-1,4]=np.sqrt(np.mean(np.sum((v-mean)**2,axis=1)))
        shared[step-1,5]=float(use.mean()) if len(use) else 0.
    for j,i in enumerate(targets):
        origin=h[i,-1]
        g[j,:16]=(h[i]-origin).reshape(-1);g[j,16:24]=times
        # Both public-family controls use the same complete-past neighbor rule.
        candidates=targets[targets!=i]
        order=np.lexsort((agents[candidates],np.sum((h[candidates,-1]-origin)**2,axis=1)))
        near=candidates[order[:8]]
        nxy=np.zeros((8,8,2),np.float32);nt=np.zeros((8,8),np.float32);nm=np.zeros((8,8),np.float32)
        nxy[:len(near)]=h[near]-origin;nt[:len(near)]=times;nm[:len(near)]=1
        g[j,38:166]=nxy.reshape(-1);g[j,166:230]=nt.reshape(-1);g[j,230:294]=nm.reshape(-1)
        g[j,332:356]=(baseline_numpy(h[i:i+1],1)[0]-origin).reshape(-1)
        scale=max(float(np.linalg.norm(np.diff(h[i],axis=0),axis=1).mean()),1.)
        local=shared.copy();local[:,:5]/=scale
        others=np.arange(len(agents))!=i;radii=np.linalg.norm(h[others,-1]-origin,axis=1)/scale
        context[j,:42]=local.reshape(-1)
        context[j,42:]=[np.log1p(len(agents)),float(m.all(1).mean()),
            *((h[:,-1].mean(0)-origin)/scale),float(radii.mean()) if len(radii) else 0.,
            float(radii.min()) if len(radii) else 0.]
    return g,targets,context


def array_hash(value):
    a=np.ascontiguousarray(value);digest=hashlib.sha256()
    digest.update(str((a.dtype.str,a.shape)).encode());digest.update(memoryview(a).cast('B'))
    return digest.hexdigest()


def prepare(root,admission_path,heartbeat=lambda **_:None):
    root=Path(root);admission=load_admission(root,root/admission_path)
    structure=json.loads((root/'outputs/publication_readiness_2026_09/movingdronecrowd_intake_v1/analysis.json').read_text())
    lookup={r['clip']:r for r in structure['recordings']}
    records=admission['recordings'];rows=sum(lookup[r['clip']]['window_support']['obs8_pred12_native_stride1']['history_complete_agent_instances']
                                            for r in records.values())
    specs=dict(geometry=(np.float32,(rows,476)),shared_context=(np.float32,(rows,48)),
        target=(np.float32,(rows,12,2)),target_eval=(np.float64,(rows,12,2)),valid=(bool,(rows,12)),
        history=(np.float64,(rows,8,2)),origin=(np.float64,(rows,2)),scale=(np.float64,(rows,)),
        sites=('<U40',(rows,)),recordings=(np.int32,(rows,)),frames=(np.int32,(rows,)),agents=(np.int64,(rows,)),
        queries=(np.int32,(rows,)),baseline_ade=(np.float64,(rows,len(BASELINES))),baseline_fde=(np.float64,(rows,len(BASELINES))))
    out={k:np.empty(shape,dtype=dtype) for k,(dtype,shape) in specs.items()}
    cursor=0;query_id=0;inventory_hash=hashlib.sha256();packets=[]
    for ri,node in enumerate(sorted(records)):
        meta=records[node];source=root/meta['source_annotation_path']
        require_fit_record(admission,node,'fit',source)
        rec=mot_record(source.read_bytes(),meta['clip'],range(1,lookup[meta['clip']]['image_frames']+1))
        begin=cursor
        for q in range(SPEC.history_span,rec.valid.shape[1]-12):
            inputs=past_inputs(rec,q,SPEC)
            g,local,shared=encode_scene(inputs['positions'],inputs['history_mask'],inputs['agent_ids'])
            n=len(local);dest=slice(cursor,cursor+n)
            for key in ('agent_ids','positions','history_mask'):
                inventory_hash.update(array_hash(inputs[key]).encode())
            packets.append(dict(node=node,query=q,current_agents=len(inputs['agent_ids']),
                                past_eligible_agents=n,row_start=cursor,row_end=cursor+n))
            if n:
                labels=target_labels(rec,inputs,SPEC)
                h=inputs['positions'][local];y=labels['future_positions'][local];valid=labels['future_mask'][local]
                out['geometry'][dest]=g;out['shared_context'][dest]=shared
                out['history'][dest]=h;out['origin'][dest]=h[:,-1];out['scale'][dest]=1.
                out['target'][dest]=np.where(valid[...,None],y-h[:,-1,None],0).astype(np.float32)
                out['target_eval'][dest]=y;out['valid'][dest]=valid
                out['sites'][dest]=meta['exclusion_group'];out['recordings'][dest]=ri
                out['frames'][dest]=q;out['agents'][dest]=inputs['agent_ids'][local];out['queries'][dest]=query_id
                for k in range(len(BASELINES)):
                    a,f=native_errors(baseline_numpy(h,k),y,valid,np.ones(n))
                    out['baseline_ade'][dest,k]=a;out['baseline_fde'][dest,k]=f
            cursor+=n;query_id+=1
        expected=lookup[meta['clip']]['window_support']['obs8_pred12_native_stride1']['history_complete_agent_instances']
        if cursor-begin!=expected:raise ValueError('Past-eligible recording population differs from audited intake')
        heartbeat(state='packing_admitted_source',record=ri+1,total=len(records),rows=cursor)
    if cursor!=rows:raise ValueError('Complete admitted cohort not packed')
    receipt=dict(rows=rows,recordings=len(records),groups=sorted(set(out['sites'])),queries=query_id,
        known_rows=int(out['valid'].any(1).sum()),complete_rows=int(out['valid'].all(1).sum()),
        wholly_unknown_future_rows=int((~out['valid'].any(1)).sum()),endpoint_rows=int(out['valid'][:,-1].sum()),
        all_current_visible_agent_instances=sum(r['current_agents'] for r in packets),
        all_current_input_inventory_sha256=inventory_hash.hexdigest(),
        arrays={k:dict(dtype=v.dtype.str,shape=list(v.shape),bytes=v.nbytes,sha256=array_hash(v)) for k,v in out.items()},
        packed_bytes=sum(v.nbytes for v in out.values()),feature_arrays_written_to_disk=False,
        geometry_features=476,shared_context_features=list(SHARED_NAMES),
        forecaster_neighbor_rule='nearest8complete_past_current_agents_same_for_both_families',
        shared_context_rule='all_current_visible_agents_past_masks_only_not_physical_egomotion',
        coordinate_unit='image_pixel',time_unit='native_annotation_step_not_verified_raw_video_frame',
        no_future_or_label_masks_in_model_inputs=True,source_time_causality_proved=False)
    return out,receipt,admission
