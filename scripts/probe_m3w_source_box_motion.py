"""Fixed source motion-information probability probes, never a deployment gate."""
import argparse
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_box_motion import registration
from scripts.run_m3w_source_importance_sampling import load_config, context
from scripts.run_m3w_source_crossfit import immutable_json, save_arrays, array_hash
from src.world_model.m3w_source_box_motion_head import BoxMotionCorpus
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


def probability_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or not len(y) or not np.isin(y, [0,1]).all() or not np.isfinite(p).all():
        raise ValueError('Aligned binary labels and finite probabilities required')
    p = np.clip(p, 1e-12, 1-1e-12)
    both = len(np.unique(y)) == 2
    return dict(rows=len(y), positives=int(y.sum()), positive_rate=float(y.mean()),
        brier=float(np.mean((p-y)**2)), log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
        auroc=float(roc_auc_score(y,p)) if both else None,
        auprc=float(average_precision_score(y,p)) if both else None)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--replay', action='store_true'); args = p.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    plan = json.loads(args.registration.read_text())
    for name, digest in plan['bindings'].items():
        if file_digest(ROOT/name) != digest: raise ValueError('Changed probe registration: '+name)
    assert plan['targets'] == ['any_nonzero', 'over10_annotation_pixels'] and plan['arms'] == ['quality','motion']
    reg = registration(Path(plan['source_registration'])); base = load_config(Path(reg['source_registration']))
    _, data, cached, _, _, ids, _, _ = context(base)
    extraction = json.loads((ROOT/reg['extraction_report']).read_text()); artifact = extraction['artifact']
    assert file_digest(ROOT/artifact['path']) == artifact['sha256']
    flow = BoxMotionCorpus(data, cached, ROOT/artifact['path'])
    private, public = ROOT/plan['output'], ROOT/plan['reports']
    identity = dict(registration_sha256=file_digest(args.registration), source=data.identity,
                    assignment=data.assignment_hash, feature_sha256=artifact['sha256'])
    trials = []; replays = 0
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for site in base['sites']:
            train, _, held, _, _ = data.configure(site); norm = flow.configure(train)
            tr = data.loss_targets(train).numpy(); held_target = data.target[held-data.nmain]
            def labels(target, q, name):
                if name == 'any_nonzero': return np.any(target != 0, axis=(1,2)).astype(int)
                excursion = np.linalg.norm(target.astype(float), axis=-1).max(1)*data.native_scale[q-data.nmain]
                return (excursion > 10).astype(int)
            for arm in plan['arms']:
                def design(q, training=False):
                    features, _ = flow.inputs(q, arm, training=training)
                    return np.concatenate((features[0].numpy(), features[1].numpy()[:,1:,:19].reshape(len(q),-1)),1).astype(float)
                x, z = design(train, True), design(held)
                for label in plan['targets']:
                    y, yh = labels(tr, train, label), labels(held_target, held, label)
                    assert len(np.unique(y)) == 2
                    key = f'{site}_{arm}_{label}'; cp = private/(key+'.npz'); rp = private/(key+'.json')
                    ti = dict(identity, site=site, arm=arm, label=label, train_hash=array_hash(train), held_hash=array_hash(held),
                              normalizer_hash=array_hash(*norm), design_hash=array_hash(x,z))
                    if rp.exists():
                        receipt = json.loads(rp.read_text()); assert receipt['identity'] == ti
                        assert file_digest(cp) == receipt['artifact']['sha256']
                        with np.load(cp,allow_pickle=False) as a:
                            if args.replay:
                                from scipy.special import expit
                                np.testing.assert_array_equal(a['train_probability'], expit(x @ a['coefficient'].T + a['intercept'])[:,0])
                                np.testing.assert_array_equal(a['held_probability'], expit(z @ a['coefficient'].T + a['intercept'])[:,0])
                                replays += 1
                        trials.append(receipt); continue
                    if args.replay: raise ValueError('Cannot replay incomplete probe')
                    print(json.dumps(dict(state='fit', trial=key)), flush=True); start = time.monotonic()
                    model = LogisticRegression(C=plan['C'], solver='lbfgs', max_iter=plan['max_iter'],
                        tol=plan['tol'], class_weight=None, random_state=17)
                    model.fit(x,y); pt, ph = model.predict_proba(x)[:,1], model.predict_proba(z)[:,1]
                    save_arrays(cp, dict(train_ids=train, held_ids=held, coefficient=model.coef_, intercept=model.intercept_,
                                         train_probability=pt, held_probability=ph, train_label=y, held_label=yh))
                    receipt = dict(identity=ti, trial=key, site=site, arm=arm, label=label, seconds=time.monotonic()-start,
                        iterations=model.n_iter_.tolist(), training=probability_metrics(y,pt), held=probability_metrics(yh,ph),
                        prior=probability_metrics(yh,np.full(len(yh),y.mean())),
                        artifact=dict(path=str(cp.relative_to(ROOT)),sha256=file_digest(cp)))
                    immutable_json(rp,receipt); trials.append(receipt)
    assert len(trials) == 16
    if args.replay:
        immutable_json(public/'replay.json', dict(result_source='fresh_run_coefficient_replay', exact_probes=replays, new_fits=0))
        print(json.dumps(dict(state='replay_complete', exact_probes=replays))); return
    draws = np.random.default_rng(base['bootstrap_seed']).integers(4,size=(base['bootstrap_resamples'],4))
    contrasts = []
    for label in plan['targets']:
        for metric in ('brier','log_loss','auroc','auprc'):
            a = [next(t for t in trials if t['site']==s and t['arm']=='motion' and t['label']==label)['held'][metric] for s in base['sites']]
            b = [next(t for t in trials if t['site']==s and t['arm']=='quality' and t['label']==label)['held'][metric] for s in base['sites']]
            if None in a or None in b:
                contrasts.append(dict(target=label,metric=metric,status='undefined_for_single_class_site')); continue
            delta = (np.array(b)-a) if metric in ('brier','log_loss') else (np.array(a)-b)
            contrasts.append(dict(target=label,metric=metric,positive_means_improvement=True,
                equal_site_difference=float(delta.mean()),conditional_four_site_ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
                site_differences=delta.tolist()))
    immutable_json(public/'report.json',dict(identity=identity,result_source='fresh_run_fixed_logistic_probes',
        models=16,trials=trials,contrasts=contrasts,selection=False,new_deployment=False,main_outer_rows_scored=0,
        uncertainty='four_explored_sites_shared_training_not_confirmation',stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(state='complete',models=16,contrasts=contrasts),indent=2))


if __name__ == '__main__': main()
