"""Fixed source annotation-quality audit and past-box probability probes."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_motion_candidate import load_config as load_parent, context
from scripts.run_m3w_source_crossfit import save_arrays
from scripts.run_m3w_source_continuation import immutable_json
from scripts.audit_m3w_sdd_state_support import load_source
from src.world_model.m3w_sdd_state_support import frame_lookup, control_provenance
from src.world_model.m3w_offline_visual_data import json_write
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_source_motion_quality import (
    BOX_EDGES, PIXEL_EDGES, bins, past_box_features, trajectory_quality,
    neighbor_directions, direction_scores,
)
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import torch


def load_config(path):
    r = json.loads(path.read_text())
    if (str(path) != r['registration_path'] or r['role'] != 'training_only_motion_quality_probability_probe'
            or r['sites'] != ['coupa', 'deathCircle', 'gates', 'hyang'] or r['seeds'] != [17, 29, 43]
            or r['arms'] != ['geometry', 'geometry_past_box']
            or r['labels'] != ['nonzero', 'half_box_excursion'] or r['models'] != 48
            or r['selection'] or r['new_deployment']):
        raise ValueError('Fixed training-side diagnostic required')
    for name, sha in r['bindings'].items():
        if file_digest(ROOT/name) != sha:
            raise ValueError('Frozen dependency changed: '+name)
    return r


def positive_probability(model, x):
    # Serial reduction makes saved probability replays bitwise reproducible.
    workers = model.n_jobs
    try:
        model.n_jobs = 1
        p = model.predict_proba(x)
    finally:
        model.n_jobs = workers
    loc = np.flatnonzero(model.classes_ == 1)
    return p[:, loc[0]] if len(loc) else np.zeros(len(x))


def probability_metrics(y, p, reference):
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Aligned finite probabilities required')
    ece = 0.; curve = []
    for index in range(10):
        m = np.minimum((p*10).astype(int), 9) == index
        if m.any():
            confidence, frequency = float(p[m].mean()), float(y[m].mean())
            ece += m.mean()*abs(confidence-frequency)
            curve.append(dict(bin=index, rows=int(m.sum()), probability=confidence, frequency=frequency))
    brier, ref = float(np.mean((p-y)**2)), float(np.mean((reference-y)**2))
    return dict(rows=len(y), positives=int(y.sum()), positive_rate=float(y.mean()),
        brier=brier, reference_brier=ref, brier_lift=ref-brier,
        auroc=float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else None,
        auprc=float(average_precision_score(y,p)) if y.any() else None,
        ece=float(ece), calibration_curve=curve)


def prepare(reg, data, public, private, beat):
    loc = np.flatnonzero(data.source_sites != 'bookstore')
    ids = loc+data.nmain
    if len(ids) != 15430:
        raise ValueError('Frozen source cohort changed')
    path = private/'rows.npz'; receipt_path = public/'preparation.json'
    identity = dict(registration_sha256=file_digest(ROOT/reg['registration_path']),
                    source=data.identity, source_assignment=data.assignment_hash)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['identity'] == identity and file_digest(path) == receipt['rows_sha256']
        for item in receipt['raw_sources']:
            assert file_digest(ROOT/item['path']) == item['sha256']
        return receipt
    raw_links = {r['annotation_key']:r for r in json.loads((ROOT/reg['source_manifest']).read_text())['records']}
    entries = json.loads(data.manifest_path.read_text())['records']
    future = np.zeros((len(ids),12,2)); boxes = np.zeros((len(ids),8,4))
    flags = np.zeros((len(ids),20,2),bool)
    later = np.zeros(len(ids),bool); unbracketed = later.copy()
    keys = np.zeros((len(ids),2),np.int64); sources=[]; poison_checks=0
    sites, records, tracks = data.source_sites[loc], data.source_records[loc], data.source_tracks[loc]
    for record in sorted(set(records)):
        selected = np.flatnonzero(records == record)
        source_ids = data.sid[loc[selected]]
        rid = int(data.record_ids[source_ids[0]])
        receipt, entry = entries[rid], raw_links[record]
        ap = ROOT/entry['annotations_path']
        assert file_digest(ap) == entry['annotations_sha256']
        qp = data.manifest_path.parent/record/'query_keys.npy'
        assert file_digest(qp) == receipt['arrays']['query_keys.npy']
        query = np.load(qp,allow_pickle=False)[data.local_ids[source_ids]]
        keys[selected] = query
        raw, labels = load_source(ap)
        starts = np.r_[0,np.flatnonzero(np.diff(raw[:,0]))+1]; ends = np.r_[starts[1:],len(raw)]
        lookup = {int(raw[a,0]):(a,b) for a,b in zip(starts,ends)}
        for agent in np.unique(query[:,1]):
            positions = selected[query[:,1] == agent]
            q = keys[positions,0]
            start,end = lookup[int(agent)]; track = raw[start:end]
            history = frame_lookup(track,q[:,None]-np.arange(7,-1,-1)*12)
            label_ids = frame_lookup(track,q[:,None]+np.arange(1,13)*12)
            assert np.all(history >= 0) and np.all(label_ids >= 0)
            assert not track[history,6].any() and not track[label_ids,6].any()
            assert np.all(labels[start:end][history[:,-1]] == 'Pedestrian')
            past = track[history,1:5]; boxes[positions] = past
            xy = (past[:,:,:2]+past[:,:,2:])/2
            assert np.all(xy == xy[:,-1:, :])
            native = (track[label_ids,1:3]+track[label_ids,3:5])/2-xy[:,-1:, :]
            future[positions] = native
            np.testing.assert_allclose(native, data.target[loc[positions]]*data.native_scale[loc[positions],None,None],
                                       rtol=1e-6,atol=1e-5)
            flags[positions] = np.concatenate((track[history,7:9],track[label_ids,7:9]),axis=1).astype(bool)
            provenance = control_provenance(track)
            following = provenance['next_control_row'][history]
            generated = track[history,8] == 1
            later[positions] = (generated & (following >= 0) & (track[np.maximum(following,0),5] > q[:,None])).any(1)
            unbracketed[positions] = (generated & ~provenance['bracketed'][history]).any(1)
            # Mutation is per query: no later row may alter its box features.
            for i in [0,len(q)-1] if len(q) > 1 else [0]:
                changed = track.copy(); changed[changed[:,5] > q[i],1:5] += 10000
                np.testing.assert_array_equal(past_box_features(changed[history[i:i+1],1:5]),
                                              past_box_features(track[history[i:i+1],1:5]))
                poison_checks += 1
        sources.append(dict(recording=record,path=entry['annotations_path'],sha256=entry['annotations_sha256'],rows=len(selected)))
        beat(state='raw_annotation_alignment',recording=record,rows=len(selected),completed_records=len(sources))
    scale = np.median(np.linalg.norm(boxes[:,:,2:]-boxes[:,:,:2],axis=-1),axis=1)
    box_features = past_box_features(boxes)
    geometry = data.aux['geometry'][data.sid[loc]].copy()
    arrays = dict(ids=ids,loc=loc,sites=sites,records=records,tracks=tracks,query_keys=keys,
        future_native=future,box_features=box_features,past_box_scale=scale,annotation_flags=flags,
        past_has_later_control=later,past_unbracketed_generated=unbracketed,geometry=geometry)
    save_arrays(path,arrays)
    receipt = dict(identity=identity,result_source='fresh_run_raw_alignment_with_cached_verified_cohort',
        rows=len(ids),records=len(sources),sites=4,tracks=len(set(tracks)),raw_sources=sources,
        rows_sha256=file_digest(path),raw_future_box_poison_checks=poison_checks,
        all_rows_aligned_to_raw=True,main_rows_scored=0,outer_rows_scored=0,
        sensor_asof_certified=False,new_deployment=False)
    immutable_json(receipt_path,receipt)
    return receipt


def probe(reg, data, a, public, private, beat, replay):
    quality = trajectory_quality(a['future_native'],a['past_box_scale'])
    trials=[]; replayed=[]; new_fits=0
    for site in reg['sites']:
        train,_,held,_,_ = data.configure(site)
        tr = np.flatnonzero(a['sites'] != site); ho = np.flatnonzero(a['sites'] == site)
        np.testing.assert_array_equal(a['ids'][tr],train); np.testing.assert_array_equal(a['ids'][ho],held)
        assert not set(a['tracks'][tr]) & set(a['tracks'][ho])
        for label in reg['labels']:
            y = quality[label].astype(np.int64)
            for seed in reg['seeds']:
                for arm in reg['arms']:
                    name=f'{site}_{label}_{arm}_seed{seed}'
                    x = data.z[a['ids']].copy()
                    if arm == 'geometry_past_box': x = np.column_stack((x,a['box_features']))
                    identity=dict(registration=file_digest(ROOT/reg['registration_path']),rows=file_digest(private/'rows.npz'),
                        site=site,label=label,seed=seed,arm=arm,sklearn=sklearn.__version__)
                    cp,pp,rp=private/'models'/f'{name}.joblib',private/'predictions'/f'{name}.npz',private/'trials'/f'{name}.json'
                    old=json.loads(rp.read_text()) if rp.exists() else None
                    if old:
                        assert old['identity']==identity and file_digest(cp)==old['model_sha256'] and file_digest(pp)==old['prediction_sha256']
                        if not replay: trials.append(old); continue
                        model=joblib.load(cp)['model']
                    elif replay: raise ValueError('Replay needs complete receipts')
                    else:
                        started=time.monotonic(); beat(state='fitting_probe',trial=name,training_rows=len(tr))
                        model=ExtraTreesClassifier(**reg['forest'],random_state=seed)
                        model.fit(x[tr],y[tr]); seconds=time.monotonic()-started
                        cp.parent.mkdir(parents=True,exist_ok=True); temp=cp.with_suffix('.tmp')
                        joblib.dump(dict(identity=identity,model=model,train_ids=train),temp);os.replace(temp,cp)
                        new_fits+=1
                    probability=positive_probability(model,x[ho])
                    if replay:
                        with np.load(pp,allow_pickle=False) as saved:
                            np.testing.assert_array_equal(saved['ids'],held)
                            np.testing.assert_array_equal(saved['probability'],probability)
                        replayed.append(name); continue
                    reference=float(y[tr].mean())
                    save_arrays(pp,dict(ids=held,probability=probability,target=y[ho],reference=np.full(len(ho),reference)))
                    receipt=dict(identity=identity,result_source='fresh_run_fixed_probability_probe',trial=name,
                        training_rows=len(tr),training_positive_rate=reference,fit_seconds=seconds,
                        metrics=probability_metrics(y[ho],probability,reference),
                        model_path=str(cp.relative_to(ROOT)),model_sha256=file_digest(cp),
                        prediction_path=str(pp.relative_to(ROOT)),prediction_sha256=file_digest(pp))
                    immutable_json(rp,receipt);trials.append(receipt)
                    beat(state='probe_complete',trial=name,seconds=seconds,brier_lift=receipt['metrics']['brier_lift'])
    if replay:
        assert len(replayed)==reg['models']
        immutable_json(public/'replay.json',dict(replayed_models=replayed,all_bitwise_exact=True,new_fits=0,
            probability_reduction='serial_tree_order_training_threads_remain_four'))
    else:
        assert len(trials)==reg['models']
        immutable_json(public/'probes.json',dict(models=len(trials),trials=trials,selection=False,new_deployment=False))
    beat(state='replay_complete' if replay else 'probes_complete',models=reg['models'],new_fits=new_fits)


def analyze(reg, data, a, public):
    quality=trajectory_quality(a['future_native'],a['past_box_scale']); n=len(a['ids'])
    parent=load_parent(Path(reg['parent_registration'])); _,control,_=context(parent)
    motion=json.loads((ROOT/parent['reports']/'report.json').read_text())
    errors={}; cv=np.linalg.norm(data.target[a['loc']].astype(float),axis=-1).mean(1)
    for name,report in [('unconditional',control),('motion_loss',motion)]:
        values=[]
        for receipt in report['oof_labels']:
            path=ROOT/receipt['path'];assert file_digest(path)==receipt['sha256']
            with np.load(path,allow_pickle=False) as pred:
                np.testing.assert_array_equal(pred['ids'],a['ids']);values.append(pred['ade'].copy())
        errors[name]=np.asarray(values)
    def summarize(mask):
        if not mask.any(): return dict(rows=0)
        flags=a['annotation_flags'][mask]
        row=dict(rows=int(mask.sum()),tracks=len(set(a['tracks'][mask])),records=len(set(a['records'][mask])),
            sites=len(set(a['sites'][mask])),cv_error_share=float(cv[mask].sum()/cv.sum()),
            median_max_pixel=float(np.median(quality['max_pixel_displacement'][mask])),
            median_endpoint_path_ratio=float(np.median(quality['endpoint_over_path'][mask])),
            returned_to_origin=int(quality['returned_to_origin'][mask].sum()),
            final_four_outside_half_box=int(quality['final_four_outside_half_box'][mask].sum()),
            median_first_changed_step=float(np.median(quality['first_changed_step'][mask])),
            future_any_generated=int(flags[:,8:,1].any(1).sum()),future_all_generated=int(flags[:,8:,1].all(1).sum()),
            future_any_occluded=int(flags[:,8:,0].any(1).sum()),
            past_has_later_control=int(a['past_has_later_control'][mask].sum()),
            past_unbracketed_generated=int(a['past_unbracketed_generated'][mask].sum()),
            half_pixel_grid_fraction=float(quality['half_pixel_grid_fraction'][mask].mean()),candidate={})
        for name,e in errors.items():
            costs=e[:,mask].mean(0); ref=cv[mask]
            row['candidate'][name]=dict(gain_percent=float(100*(1-costs.sum()/ref.sum())) if ref.sum()>0 else None,
                oracle_gain_percent=float(100*(1-np.minimum(e[:,mask],ref).mean(0).sum()/ref.sum())) if ref.sum()>0 else None,
                native_pixel_excess=float(((costs-ref)*data.native_scale[a['loc'][mask]]).mean()))
        return row
    strata={}
    for field,edges in [('max_pixel_displacement',PIXEL_EDGES),('displacement_over_past_box',BOX_EDGES)]:
        strata[field]={name:summarize(mask) for name,mask in bins(quality[field],edges).items()}
    direction=[]
    for name,v in neighbor_directions(a['geometry']).items():
        supported,cosine=direction_scores(v,a['future_native'][:,-1])
        for subset,mask in [('nonzero',quality['nonzero']),('half_box_excursion',quality['half_box_excursion'])]:
            for site in reg['sites']:
                m=mask & (a['sites']==site); s=m & supported
                direction.append(dict(hint=name,label_subset=subset,site=site,rows=int(m.sum()),supported_rows=int(s.sum()),
                    mean_endpoint_cosine=float(cosine[s].mean()) if s.any() else None))
    probes=json.loads((public/'probes.json').read_text()); summary={};contrasts=[]
    draws=np.random.default_rng(reg['bootstrap_seed']).integers(4,size=(2000,4))
    for label in reg['labels']:
        for arm in reg['arms']:
            ts=[t for t in probes['trials'] if t['identity']['label']==label and t['identity']['arm']==arm]
            site_lift=np.array([np.mean([t['metrics']['brier_lift'] for t in ts if t['identity']['site']==s]) for s in reg['sites']])
            summary[label+'_'+arm]=dict(equal_site_brier_lift=float(site_lift.mean()),
                conditional_four_site_ci95=np.quantile(site_lift[draws].mean(1),[.025,.975]).tolist(),
                positive_fits=sum(t['metrics']['brier_lift']>0 for t in ts),total_fits=len(ts),site_lift=site_lift.tolist())
        delta=np.array(summary[label+'_geometry_past_box']['site_lift'])-summary[label+'_geometry']['site_lift']
        contrasts.append(dict(label=label,past_box_minus_geometry_brier_lift=float(delta.mean()),
            conditional_four_site_ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()))
    result=dict(result_source='fresh_run_fixed_quality_and_probability_analysis_cached_verified_candidates',
        rows=n,all_rows_retained=True,overall=summarize(np.ones(n,bool)),strata=strata,
        per_site={s:summarize(a['sites']==s) for s in reg['sites']},direction_hints=direction,
        probability_probe_summary=summary,past_box_contrasts=contrasts,bootstrap_draws=2000,
        uncertainty='four_explored_sites_shared_fit_populations_conditional_not_confirmation',
        labels_are_human_gold=False,annotation_jitter_proven=False,forecasting_gain_established=False,
        main_rows_scored=0,outer_rows_scored=0,new_deployment=False,stage5c_executed=False,smc_enabled=False)
    immutable_json(public/'analysis.json',result)
    print(json.dumps(dict(overall=result['overall'],probe_summary=summary,past_box_contrasts=contrasts),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','probe','replay','analyze'],required=True)
    args=parser.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    reg=load_config(args.registration);public,private=ROOT/reg['reports'],ROOT/reg['output']
    def beat(**v):
        event=dict(pid=os.getpid(),timestamp_unix=time.time(),**v)
        json_write(private/'heartbeat.json',event);print(json.dumps(event),flush=True)
    beat(state='verifying_assets',phase=args.phase)
    parent=load_parent(Path(reg['parent_registration']));_,_,data=context(parent)
    prepare(reg,data,public,private,beat)
    if args.phase=='prepare': beat(state='prepared_no_probe_fit');return
    with np.load(private/'rows.npz',allow_pickle=False) as stored:
        a={k:stored[k].copy() for k in stored.files}
    if args.phase in ('probe','replay'): probe(reg,data,a,public,private,beat,args.phase=='replay')
    else: analyze(reg,data,a,public)


if __name__=='__main__': main()
