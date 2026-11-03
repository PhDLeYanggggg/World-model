"""Full SOURCE crossfit release-support probe, with resumable fixed forest budgets."""
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

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before importing the training libraries')
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
from scripts import run_m3w_selection_controller as prior
from src.evaluation import m3w_source_label_support as api
from src.evaluation.m3w_eqmotion_amplitude import locality_interval

NAME = 'european_source_label_support_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_source_label_support_v1.json'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, ROOT/'src/evaluation/m3w_source_label_support.py',
         ROOT/'tests/test_m3w_source_label_support.py', PUBLIC/'protocol.md')
sha, once, write, artifact, checked = prior.sha, prior.once, prior.write, prior.artifact, prior.checked


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); _, jobs, ids = prior.admission()
    amplitude = ROOT/'outputs/publication_readiness_2026_09/european_eqmotion_amplitude_v1'
    v = json.loads((amplitude/'verification.json').read_text())
    if not v['metric_replay_exact'] or v['summary_sha256'] != sha(amplitude/'summary.json'):
        raise ValueError('Complete amplitude diagnosis and replay required')
    if (cfg['arms'] != list(api.ARMS) or cfg['new_models'] != 18 or cfg['seeds'] != [17,29,43]
            or cfg['forest']['n_estimators'] != 128 or cfg['source_whitelist_rows'] != len(ids)
            or any(cfg[k] for k in ('new_neural_or_trajectory_training', 'threshold_selection',
                'development_arrays_opened', 'calibration_arrays_opened', 'confirmation_opened',
                'deployment_changed', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Complete eighteen-fit SOURCE-only design required')
    src = prior.source_inputs(json.loads(prior.frozen.REG.read_text()))
    sites = np.asarray(src['sites'][ids])
    roster = sorted(set(sites))
    designs = []
    for j in jobs:
        train = np.asarray(ids[np.isin(sites, j['fit_sites'])]); held = np.asarray(ids[~np.isin(sites, j['fit_sites'])])
        if len(set(src['sites'][train])) != 4 or len(set(src['sites'][held])) != 8:
            raise ValueError('Original four-to-eight SOURCE producer exclusions required')
        designs.append(dict(trial=j['trial'], seed=j['seed'], fitting_sites=j['fit_sites'],
            train_rows=len(train), held_rows=len(held), train_ids_sha256=prior.frozen.fingerprint(train),
            held_ids_sha256=prior.frozen.fingerprint(held)))
    bind = (*FILES, prior.REG, amplitude/'summary.json', amplitude/'verification.json',
            prior.SOURCE/'receipt.json', prior.SOURCE_STORE, prior.CONTROLLER_STORE,
            ROOT/'src/evaluation/m3w_eqmotion_amplitude.py')
    value = dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in bind},
        source_rows=len(ids), source_recordings=len(set(src['recordings'][ids])), source_localities=roster,
        designs=designs, sklearn=sklearn.__version__, source_is_exposed=True,
        future_mask_in_inference=False, threshold_selection=False, independent_roles_opened=False,
        new_neural_or_trajectory_training=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    if register: once(REG, value)
    if json.loads(REG.read_text()) != value: raise ValueError('Source support contract changed')
    if not register:
        for path in (*FILES, REG):
            blob = subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(blob).hexdigest() != sha(path): raise ValueError('Commit definition before fitting')
    return cfg, jobs, ids, src, value


def runtime(cfg, phase):
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume unchanged tree prefix')))
    def guard():
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12-hour boundary; resume')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('Preserve10GiB free')
    def beat(**kw):
        guard(); value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase=phase,**kw)
        write(PRIVATE/'heartbeat.json', value)
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    def receipt(): return dict(pid=os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,registration_sha256=sha(REG))
    beat(state='runtime_ready',architecture=platform.machine(),fit_threads=4,prediction_threads=1,workers=0,
         algorithm='ExtraTrees_not_new_Torch_or_trajectory_training')
    return beat,guard,receipt


def view(src, ids, arm):
    return api.features(geometry=src['geometry'][ids], detector_history=src['detector_history'][ids], arm=arm)


def labels(ids):
    mask = np.load(prior.SOURCE/'valid.npy',mmap_mode='r',allow_pickle=False)
    return ~np.asarray(mask[ids]).any(1)


def fit_one(cfg, src, ids, job, arm, path, beat, guard, resume=False, stop_at=None):
    train = ids[np.isin(src['sites'][ids],job['fit_sites'])]
    return api.fit(view(src,train,arm), labels(train), src['sites'][train],
        settings=cfg['forest'], seed=job['seed'], clip=cfg['feature_clip'],
        chunk=cfg['checkpoint_every_trees'], identity=dict(registration_sha256=sha(REG),arm=arm,
            train_ids_sha256=prior.frozen.fingerprint(train)),path=path,resume=resume,stop_at=stop_at,
        heartbeat=lambda **kw:beat(trial=job['trial'],arm=arm,**kw),guard=guard)


def pilot(cfg,jobs,ids,src,reg):
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Complete pilot immutable')
    beat,guard,receipt = runtime(cfg,'pilot'); job=jobs[0]; arm=cfg['arms'][0]
    path=PRIVATE/'pilot_resume.joblib'
    fit_one(cfg,src,ids,job,arm,path,beat,guard,stop_at=cfg['pilot_stop'])
    a=fit_one(cfg,src,ids,job,arm,path,beat,guard,resume=True,stop_at=cfg['pilot_target'])
    b=fit_one(cfg,src,ids,job,arm,PRIVATE/'pilot_direct.joblib',beat,guard,stop_at=cfg['pilot_target'])
    if a['tree_fingerprint']!=b['tree_fingerprint']: raise ValueError('Recovery tree states differ')
    held=ids[~np.isin(src['sites'][ids],job['fit_sites'])]
    x=view(src,held,arm);np.testing.assert_array_equal(api.predict(a,x),api.predict(b,x))
    once(PUBLIC/'pilot.json',dict(**receipt(),actual_trees=2*cfg['pilot_target'],engineering_pass=True,
        tree_states_and_full_held_probabilities_exact=True,
        projected18_full_fits_plus_first_refit_seconds=19*b['seconds']/cfg['pilot_target']*cfg['forest']['n_estimators'],
        projection_excludes_preparation_and_inference=True,scientific_efficacy=False))


def train(cfg,jobs,ids,src,reg,resume):
    if (PUBLIC/'training_freeze.json').exists(): raise ValueError('Complete fits immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']: raise ValueError('Real recovery pilot required')
    beat,guard,receipt=runtime(cfg,'train'); refs=[]; trees=refit_trees=0
    for job in jobs:
        for arm in cfg['arms']:
            path=PRIVATE/'fits'/(job['trial']+'_'+arm+'.joblib')
            state=fit_one(cfg,src,ids,job,arm,path,beat,guard,resume=resume and path.exists())
            if state['trees'] != cfg['forest']['n_estimators']: raise ValueError('Full128-tree endpoints required')
            if job==jobs[0] and arm==cfg['arms'][0]:
                rp=PRIVATE/'full_refit.joblib'
                other=fit_one(cfg,src,ids,job,arm,rp,beat,guard,resume=resume and rp.exists())
                if other['tree_fingerprint']!=state['tree_fingerprint']: raise ValueError('Full refit differs')
                held=ids[~np.isin(src['sites'][ids],job['fit_sites'])];x=view(src,held,arm)
                np.testing.assert_array_equal(api.predict(state,x),api.predict(other,x));refit_trees+=other['trees']
                once(PUBLIC/'first_full_refit.json',dict(tree_states_and_full_held_probabilities_exact=True,
                    trees=state['trees'],checkpoint=artifact(path),refit=artifact(rp)))
            train_ids=ids[np.isin(src['sites'][ids],job['fit_sites'])]
            fitting=api.score(labels(train_ids),api.predict(state,view(src,train_ids,arm)))
            out=PUBLIC/'fits'/(job['trial']+'_'+arm+'.json')
            value=dict(trial=job['trial'],seed=job['seed'],arm=arm,checkpoint=artifact(path),trees=state['trees'],
                tree_fingerprint=state['tree_fingerprint'],seconds=state['seconds'],fitting_sites=job['fit_sites'],
                fitting_rows=len(train_ids),fitting_scores=fitting,registration_sha256=sha(REG))
            once(out,value);refs.append(artifact(out));trees+=state['trees']
            beat(state='fit_complete',trial=job['trial'],arm=arm,fits=len(refs),total=18)
    once(PUBLIC/'training_freeze.json',dict(**receipt(),fits=refs,primary_trees=trees,first_full_refit_trees=refit_trees,
        new_neural_or_trajectory_training=False,held_outcomes_scored=False))


def fitted():
    value=json.loads((PUBLIC/'training_freeze.json').read_text());out={}
    for ref in value['fits']:
        row=json.loads(checked(ref).read_text());checked(row['checkpoint'])
        if row['trees']!=128:raise ValueError('Complete endpoints required')
        out[row['trial'],row['arm']]=row
    if len(out)!=18:raise ValueError('All18fits required')
    return out


def infer(cfg,jobs,ids,src,reg,resume=False,verify=False):
    end=PUBLIC/('inference_verification.json' if verify else 'prediction_freeze.json')
    if end.exists():raise ValueError('Complete prediction stage immutable')
    beat,guard,receipt=runtime(cfg,'infer_verify' if verify else 'infer');models=fitted();refs=[]
    for job in jobs:
        held=ids[~np.isin(src['sites'][ids],job['fit_sites'])]
        for arm in cfg['arms']:
            row=models[job['trial'],arm];state=joblib.load(checked(row['checkpoint']))
            if api.tree_fingerprint(state['model'])!=row['tree_fingerprint']:raise ValueError('Tree state changed')
            p=api.predict(state,view(src,held,arm))
            path=PRIVATE/'predictions'/(job['trial']+'_'+arm+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                if not(verify or resume):raise ValueError('Partial inference requires --resume')
                with np.load(path,allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],held);np.testing.assert_array_equal(z['probability'],p)
            elif verify: raise ValueError('Cannot verify absent predictions')
            else:
                temp=path.with_suffix('.tmp.npz');np.savez(temp,ids=held,probability=p);os.replace(temp,path)
            refs.append(dict(trial=job['trial'],arm=arm,seed=job['seed'],prediction=artifact(path),checkpoint=row['checkpoint']))
            beat(state='probabilities_replayed' if verify else 'probabilities_frozen',trial=job['trial'],arm=arm,count=len(refs),total=18)
    freeze=dict(registration_sha256=sha(REG),training_freeze=artifact(PUBLIC/'training_freeze.json'),
                predictions=refs,held_labels_used_for_prediction=False)
    once(PUBLIC/'prediction_freeze.json',freeze)
    if verify:once(end,dict(**receipt(),all18_probability_arrays_exact=True,prediction_freeze=artifact(PUBLIC/'prediction_freeze.json')))


def evaluate(cfg,jobs,ids,src,reg,verify=False):
    end=PUBLIC/('verification.json' if verify else 'complete.json')
    if end.exists():raise ValueError('Complete scalar stage immutable')
    start=time.monotonic();freeze=json.loads((PUBLIC/'prediction_freeze.json').read_text());rows=[]
    if freeze['training_freeze']!=artifact(PUBLIC/'training_freeze.json') or len(freeze['predictions'])!=18:
        raise ValueError('All eighteen probability arrays must freeze before scores')
    for ref in freeze['predictions']:
        with np.load(checked(ref['prediction']),allow_pickle=False) as z:held,p=z['ids'],z['probability']
        y=labels(held);sites=src['sites'][held];rec=src['recordings'][held]
        for site in sorted(set(sites)):
            use=sites==site
            row=dict(trial=ref['trial'],seed=ref['seed'],arm=ref['arm'],site=site,scores=api.score(y[use],p[use]),
                recordings={str(r):api.score(y[use&(rec==r)],p[use&(rec==r)]) for r in sorted(set(rec[use]))})
            path=PUBLIC/'groups'/(ref['trial']+'_'+site+'_'+ref['arm']+'.json')
            if verify and not path.exists():raise ValueError('Cannot verify absent group')
            once(path,row);rows.append(row)
    roster=reg['source_localities'];summaries={}
    for arm in cfg['arms']:
        arm_rows=[r for r in rows if r['arm']==arm]
        summaries[arm]={field:locality_interval(arm_rows,lambda g,f=field:g['scores'][f],roster,
            cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
            for field in ('AUROC','AUPRC','positive_rate','AUPRC_minus_prevalence','Brier','ECE')}
    controls={(r['trial'],r['site']):r for r in rows if r['arm']=='geometry'}
    paired=[]
    for r in rows:
        if r['arm']=='geometry_detector':
            c=controls[r['trial'],r['site']]
            a,b=r['scores']['AUROC'],c['scores']['AUROC']
            paired.append(dict(trial=r['trial'],site=r['site'],seed=r['seed'],delta=None if a is None or b is None else a-b))
    contrast=locality_interval(paired,lambda g:g['delta'],roster,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
    summary=dict(result_source='fresh18_forest_fits_SOURCE_excluded_release_label_scores',groups=len(rows),arms=summaries,
        primary=cfg['primary'],primary_effect=contrast,independent_confirmation=False,
        new_neural_or_trajectory_training=False,new_policy_evaluated=False,deployment_changed=False,
        submission_ready=False,stage5c_executed=False,smc_enabled=False)
    if len(rows)!=144:raise ValueError('All144fixed model/locality score groups required')
    if verify and not (PUBLIC/'summary.json').exists():raise ValueError('No primary summary')
    once(PUBLIC/'summary.json',summary)
    if verify and not json.loads((PUBLIC/'inference_verification.json').read_text())['all18_probability_arrays_exact']:
        raise ValueError('Separate complete forest inference replay required')
    once(end,dict(pid=os.getpid(),seconds=time.monotonic()-start,all144_groups_exact=verify,
        summary_sha256=sha(PUBLIC/'summary.json'),registration_sha256=sha(REG),independent_confirmation=False))
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','pilot','train','infer','eval','verify'))
    p.add_argument('--resume',action='store_true');args=p.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);values=admission(args.phase=='register')
        if args.phase=='pilot':pilot(*values)
        elif args.phase=='train':train(*values,args.resume)
        elif args.phase=='infer':infer(*values,resume=args.resume)
        elif args.phase=='eval':evaluate(*values)
        elif args.phase=='verify':infer(*values,verify=True);evaluate(*values,verify=True)


if __name__=='__main__':main()
