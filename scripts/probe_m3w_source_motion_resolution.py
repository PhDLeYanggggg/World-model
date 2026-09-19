"""Prespecified probability probes for source-only resolution/window controls."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required')
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_pretrained_temporal import load_config, context, FeatureCorpus
from scripts.run_m3w_source_crossfit import immutable_json, save_arrays, array_hash
from scripts.probe_m3w_source_box_motion import probability_metrics
from src.world_model.m3w_source_box_motion_head import BoxMotionCorpus
from src.world_model.m3w_source_motion_resolution import registration, REGISTRATION, VARIANTS, raw_labels
from src.world_model.m3w_offline_visual_data import json_write
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from scipy.special import expit
from threadpoolctl import threadpool_limits


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--replay',action='store_true')
    args=parser.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    plan=registration(ROOT); private,public=ROOT/plan['output'],ROOT/plan['reports']
    reg=load_config(Path(plan['source_registration'])); data=context(reg)
    cached=FeatureCorpus(data,ROOT/reg['output'],json.loads((ROOT/reg['reports']/'preparation.json').read_text()))
    ids=cached.ids; extraction=json.loads((public/'extraction.json').read_text())
    quality=json.loads((ROOT/plan['raw_label_receipt']).read_text()); rawpath=ROOT/plan['raw_label_archive']
    assert quality['rows_sha256']==file_digest(rawpath)
    with np.load(rawpath,allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw['ids'],ids); targets=raw['future_native'].copy()
    labels={label:raw_labels(targets,label) for label in plan['targets']}
    assert labels['over10_annotation_pixels'].sum()==728
    flows={}
    for variant,artifact in extraction['variants'].items():
        assert file_digest(ROOT/artifact['path'])==artifact['sha256']
        flows[variant]=BoxMotionCorpus(data,cached,ROOT/artifact['path'])
    identity=dict(registration_sha256=file_digest(ROOT/REGISTRATION),source=data.identity,
        assignment=data.assignment_hash,extraction_sha256=file_digest(public/'extraction.json'),
        raw_labels_sha256=file_digest(rawpath),sklearn=sklearn.__version__,numpy=np.__version__,
        torch=torch.__version__,machine=platform.machine(),threads=4,num_workers=0)
    trials=[]; replays=0; fresh=0; poison_checks=0; role_checks=0
    def beat(**value):
        event=dict(pid=os.getpid(),time=time.time(),**value)
        json_write(private/'probe_heartbeat.json',event); print(json.dumps(event),flush=True)
    with threadpool_limits(limits=4),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for site in reg['sites']:
            train,_,held,outer,_=data.configure(site)
            ti,hi=np.searchsorted(ids,train),np.searchsorted(ids,held)
            np.testing.assert_array_equal(ids[ti],train); np.testing.assert_array_equal(ids[hi],held)
            for variant,flow in flows.items():
                norm=flow.configure(train)
                for illegal in (np.array([0]),outer[:1],held[:1]):
                    try: flow.inputs(illegal,'motion',training=True)
                    except ValueError: role_checks+=1
                    else: raise AssertionError('Illegal training role admitted')
                before=flow.inputs(held[:8],'motion'); saved=data.target.copy()
                try: data.target[:]=np.nan; after=flow.inputs(held[:8],'motion')
                finally: data.target[:]=saved
                assert all(torch.equal(a,b) for x,y in zip(before,after) for a,b in zip(x,y))
                poison_checks+=len(held[:8])
                for arm in plan['arms']:
                    def design(q,training=False):
                        features,_=flow.inputs(q,arm,training=training)
                        return np.concatenate((features[0].numpy(),features[1].numpy()[:,1:,:19].reshape(len(q),-1)),1).astype(float)
                    x,z=design(train,True),design(held)
                    for label in plan['targets']:
                        y,yh=labels[label][ti],labels[label][hi]
                        if len(np.unique(y))!=2: raise ValueError('Both training classes required')
                        key=f'{site}_{variant}_{arm}_{label}'; cp=private/'probes'/(key+'.npz'); rp=cp.with_suffix('.json')
                        trial_identity=dict(identity,site=site,variant=variant,arm=arm,label=label,
                            train_ids_sha256=array_hash(train),held_ids_sha256=array_hash(held),
                            train_label_sha256=array_hash(y),held_label_sha256=array_hash(yh),
                            normalizer_sha256=array_hash(*norm),design_sha256=array_hash(x,z))
                        if rp.exists():
                            receipt=json.loads(rp.read_text()); assert receipt['identity']==trial_identity
                            assert file_digest(cp)==receipt['artifact']['sha256']
                            with np.load(cp,allow_pickle=False) as a:
                                np.testing.assert_array_equal(a['train_ids'],train); np.testing.assert_array_equal(a['held_ids'],held)
                                np.testing.assert_array_equal(a['train_label'],y); np.testing.assert_array_equal(a['held_label'],yh)
                                if args.replay:
                                    np.testing.assert_array_equal(a['train_probability'],expit(x@a['coefficient'].T+a['intercept'])[:,0])
                                    np.testing.assert_array_equal(a['held_probability'],expit(z@a['coefficient'].T+a['intercept'])[:,0])
                                    replays+=1
                            trials.append(receipt); continue
                        if args.replay: raise ValueError('Incomplete probe cannot be replayed')
                        beat(state='fit',trial=key); start=time.monotonic()
                        model=LogisticRegression(C=plan['C'],solver='lbfgs',max_iter=plan['max_iter'],
                            tol=plan['tol'],class_weight=None,random_state=17)
                        model.fit(x,y); pt,ph=model.predict_proba(x)[:,1],model.predict_proba(z)[:,1]
                        save_arrays(cp,dict(train_ids=train,held_ids=held,train_label=y,held_label=yh,
                            train_probability=pt,held_probability=ph,coefficient=model.coef_,intercept=model.intercept_))
                        receipt=dict(identity=trial_identity,trial=key,site=site,variant=variant,arm=arm,label=label,
                            seconds=time.monotonic()-start,iterations=model.n_iter_.tolist(),
                            training=probability_metrics(y,pt),held=probability_metrics(yh,ph),
                            prior=probability_metrics(yh,np.full(len(yh),y.mean())),
                            artifact=dict(path=str(cp.relative_to(ROOT)),sha256=file_digest(cp)))
                        immutable_json(rp,receipt); trials.append(receipt); fresh+=1
                        beat(state='fit_complete',trial=key,seconds=receipt['seconds'],new_fits=fresh)
    assert len(trials)==64
    if args.replay:
        immutable_json(public/'probe_replay.json',dict(identity=identity,exact_coefficient_replays=replays,
            future_poison_queries=poison_checks,illegal_training_roles_rejected=role_checks,new_fits=0))
        beat(state='replay_complete',exact_probes=replays); return
    draws=np.random.default_rng(plan['bootstrap_seed']).integers(4,size=(plan['bootstrap_resamples'],4))
    contrasts=[]
    comparisons=[(v+'_motion_vs_quality',(v,'motion'),(v,'quality')) for v in VARIANTS]
    comparisons += [('native_minus_lowpass_w45',('native_w45','motion'),('lowpass_w45','motion')),
                    ('native_minus_lowpass_w15',('native_w15','motion'),('lowpass_w15','motion')),
                    ('w15_minus_w45_lowpass',('lowpass_w15','motion'),('lowpass_w45','motion')),
                    ('w15_minus_w45_native',('native_w15','motion'),('native_w45','motion'))]
    for name,a,b in comparisons:
        for label in plan['targets']:
            for metric in ('brier','log_loss','auroc','auprc'):
                def collect(pair):
                    return [next(t for t in trials if t['site']==s and t['variant']==pair[0]
                        and t['arm']==pair[1] and t['label']==label)['held'][metric] for s in reg['sites']]
                va,vb=collect(a),collect(b)
                if None in va or None in vb: raise ValueError('Undefined preregistered contrast')
                delta=np.array(vb)-va if metric in ('brier','log_loss') else np.array(va)-vb
                contrasts.append(dict(name=name,label=label,metric=metric,positive_means_improvement=True,
                    equal_site_difference=float(delta.mean()),site_differences=delta.tolist(),
                    conditional_four_site_ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()))
    immutable_json(public/'probes.json',dict(identity=identity,result_source='fresh_run_fixed_probability_probes',
        models=64,trials=trials,contrasts=contrasts,raw_label_positives={k:int(v.sum()) for k,v in labels.items()},
        future_poison_queries=poison_checks,illegal_training_roles_rejected=role_checks,
        selection=False,new_deployment=False,main_outer_rows_scored=0,
        uncertainty='four_explored_sites_shared_training_unadjusted_exploratory_not_confirmation',
        stage5c_executed=False,smc_enabled=False))
    beat(state='complete',new_fits=fresh,models=64)


if __name__=='__main__': main()
