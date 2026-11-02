"""Complete matched SOURCE forest control; no independent test or threshold search."""
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
    raise RuntimeError('Native arm64 required before runtime imports')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
import torch
from scripts import run_m3w_selection_cardinal_cost as parent
from scripts import run_m3w_selection_cost_transport as transport
from src.world_model import m3w_selection_cost_forest as api

prior = parent.prior
NAME = 'european_selection_cost_forest_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_selection_cost_forest_v1.json'; REG = PUBLIC/'registration.json'
sha, artifact, checked, once, write = parent.sha, parent.artifact, parent.checked, parent.once, parent.write
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_selection_cost_forest.py',
         ROOT/'tests/test_m3w_selection_cost_forest.py', PUBLIC/'protocol.md')


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); _, producers, ids = parent.admission()
    preceding = ROOT/'outputs/publication_readiness_2026_09/european_selection_harm_deviance_v1'
    verified = json.loads((preceding/'verification.json').read_text())
    if (not verified['all54_metric_groups_exact'] or verified['summary_sha256'] != sha(preceding/'summary.json')
            or cfg['new_arms'] != list(api.ARMS) or cfg['new_fits'] != 18
            or cfg['positive_harm_budget'] != .02 or cfg['easy_net_budget'] != .02
            or any(cfg[k] for k in ('selection_labels_for_training','threshold_search','calibration_array_access',
                'confirmation_access','new_forecast_training','stage5c_executed','smc_enabled'))):
        raise ValueError('Complete negative loss readout and unchanged roles/risk required')
    definition = dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        dependencies={p:sha(ROOT/p) for p in ('src/world_model/m3w_selection_controller.py',
            'src/world_model/m3w_source_forest.py','src/world_model/m3w_selection_cardinal_cost.py',
            'scripts/run_m3w_selection_cost_transport.py','src/evaluation/m3w_selection_forecast_readout.py')},
        parent_registration=artifact(parent.REG), parent_training=artifact(parent.PUBLIC/'training_freeze.json'),
        parent_policy=artifact(parent.PUBLIC/'policy_freeze.json'), previous_negative=artifact(preceding/'summary.json'),
        sklearn=sklearn.__version__, actual_forecaster_seeds=[17,29,43], new_cost_fits=18,
        source_whitelist_rows=len(ids), primary=cfg['primary'], selection_already_exposed=True,
        calibration_arrays_read=False, confirmation_access=False, deployment_changed=False,
        new_forecast_training=False, threshold_search=False, stage5c_executed=False, smc_enabled=False)
    if register: once(REG, definition)
    if json.loads(REG.read_text()) != definition:
        raise ValueError('Registered estimator control changed')
    if not register:
        for p in (*FILES, REG):
            blob = subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT)
            if hashlib.sha256(blob).hexdigest() != sha(p):
                raise ValueError('Commit the matched control before real execution')
    return cfg, producers, ids


def runtime(cfg, phase):
    torch.set_num_threads(4); torch.set_num_interop_threads(1); began = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume unchanged tree checkpoint')))
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Preserve10GiB reserve; keep existing checkpoints')
        if time.monotonic()-began > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume unchanged experiment')
    def beat(**values):
        guard(); row = dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase=phase,**values)
        write(PRIVATE/'heartbeat.json',row); print(json.dumps(row),flush=True)
    def receipt():
        return dict(pid=os.getpid(),seconds=time.monotonic()-began,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,registration_sha256=sha(REG))
    beat(state='runtime_ready',architecture=platform.machine(),sklearn=sklearn.__version__,
         device='cpu',fit_threads=4,prediction_threads=1,workers=0)
    return beat, guard, receipt


def fit_path(row, arm, family='fits'):
    return PRIVATE/family/(row['trial']+'_'+arm+'.joblib.gz')


def fitting_data(row, source, ids, controls):
    case = transport.source_case(row,source,ids)
    data = prior.api.training_data(case['causal'],case['target'])
    control = prior.api.read_checkpoint(checked(controls[row['trial'],'mean_absolute']['checkpoint']))
    prior.api.exact(data['preprocess'],control['preprocess'])
    np.testing.assert_array_equal(data['causal']['ids'],control['fit_ids'])
    if control['step'] != control['budget'] or control['sampler']['counts'].min() < 4:
        raise ValueError('Complete original cost-head query coverage required')
    return data, case, control['sampler']['counts'].copy()


def one(matrix, data, row, arm, cfg, beat, guard, *, family='fits', resume=False, stop_at=None):
    return api.fit(matrix,preprocess=data['preprocess'],arm=arm,settings=cfg['estimator'],seed=row['seed'],
        identity=dict(registration_sha256=sha(REG),producer=row['checkpoint'],
                      source_bank=artifact(prior.PRIVATE/'source_banks'/row['trial']/'complete.json')),
        path=fit_path(row,arm,family),resume=resume,stop_at=stop_at,guard=guard,
        heartbeat=lambda **v:beat(trial=row['trial'],**v))


def pilot():
    cfg, producers, ids = admission(); beat, guard, receipt = runtime(cfg,'pilot')
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Completed pilot immutable')
    source = prior.source_inputs(json.loads(prior.frozen.REG.read_text())); row = producers[0]
    data, _, counts = fitting_data(row,source,ids,parent.fitted()); cells = []
    for arm in api.ARMS:
        guard(); matrix = api.training_matrix(data,counts,arm)
        one(matrix,data,row,arm,cfg,beat,guard,family='pilot_resume',stop_at=cfg['pilot_resume_trees'])
        a = one(matrix,data,row,arm,cfg,beat,guard,family='pilot_resume',resume=True,stop_at=cfg['pilot_complete_trees'])
        b = one(matrix,data,row,arm,cfg,beat,guard,family='pilot_direct',stop_at=cfg['pilot_complete_trees'])
        api.exact_forest(a,b)
        cells.append(dict(arm=arm,trees=a['trees'],seconds=a['seconds'],matrix_bytes=matrix['matrix_bytes'],
            candidate_rows=matrix['positive_candidate_rows'],minimum_query_visits=int(counts.min()),
            checkpoint_bytes=fit_path(row,arm,'pilot_direct').stat().st_size,exact_resume=True))
        del matrix, a, b
    once(PUBLIC/'pilot.json',dict(**receipt(),engineering_pass=True,cells=cells,actual_trees=64,
        projected18_fit_seconds=sum(c['seconds']/c['trees']*cfg['estimator']['trees']*9 for c in cells),
        projection_excludes_preparation_prediction_and_full_refit=True,selection_outcomes_read=False))


def train(resume):
    cfg, producers, ids = admission(); beat, guard, receipt = runtime(cfg,'train')
    if (PUBLIC/'training_freeze.json').exists(): raise ValueError('Completed training immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Actual matched-data recovery pilot required')
    source = prior.source_inputs(json.loads(prior.frozen.REG.read_text())); controls = parent.fitted(); refs = []
    for row in producers:
        guard(); data, case, counts = fitting_data(row,source,ids,controls)
        for arm in api.ARMS:
            guard(); matrix = api.training_matrix(data,counts,arm)
            state = one(matrix,data,row,arm,cfg,beat,guard,resume=resume and fit_path(row,arm).exists())
            if state['trees'] != cfg['estimator']['trees']: raise ValueError('Full original forest budget required')
            if row == producers[0] and arm == 'context_forest':
                other = one(matrix,data,row,arm,cfg,beat,guard,family='full_refit',
                    resume=resume and fit_path(row,arm,'full_refit').exists())
                api.exact_forest(state,other)
                once(PUBLIC/'first_full_refit.json',dict(full128_tree_state_exact=True,
                    model=artifact(fit_path(row,arm)),refit=artifact(fit_path(row,arm,'full_refit'))))
                del other
            del matrix
            p = api.predict(state,data['causal'])
            choice,_ = prior.allocator.select_actions(p,data['causal']['allowed'],arm='physical_neural')
            diagnosis = transport.api.diagnose(p,case['target'],data['causal']['allowed'],choice,data['causal']['scales'])
            path = PUBLIC/'fits'/(row['trial']+'_'+arm+'.json')
            once(path,dict(trial=row['trial'],arm=arm,checkpoint=artifact(fit_path(row,arm)),trees=state['trees'],
                nodes=state['nodes'],seconds=state['seconds'],matrix_bytes=state['matrix_bytes'],
                feature_dimension=state['feature_dimension'],input_hashes=state['input_hashes'],
                original_query_counts_sha256=api.fingerprint(counts),minimum_query_visits=int(counts.min()),
                unknown_supervised_rows=0,source_sites=sorted(set(data['causal']['sites'])),
                fit_ids_sha256=api.fingerprint(data['causal']['ids']),trace=state['trace'],
                SOURCE_fit_diagnosis=diagnosis,SOURCE_prediction_sha256=api.fingerprint(p),registration_sha256=sha(REG)))
            refs.append(artifact(path)); beat(state='fit_complete',trial=row['trial'],arm=arm,fits=len(refs),total=18)
            del state, p
        del data, case
    once(PUBLIC/'training_freeze.json',dict(**receipt(),fits=refs,primary_trees=18*cfg['estimator']['trees'],
        full_refit_trees=cfg['estimator']['trees'],source_only=True,selection_labels_used=False,new_forecast_training=False))


def fitted():
    freeze = json.loads((PUBLIC/'training_freeze.json').read_text()); output = {}
    for ref in freeze['fits']:
        r = json.loads(checked(ref).read_text()); checked(r['checkpoint']); output[r['trial'],r['arm']] = r
    if len(output) != 18: raise ValueError('All18matched forests required')
    return output


def infer(resume):
    cfg, producers, _ = admission(); beat, guard, receipt = runtime(cfg,'infer')
    if (PUBLIC/'policy_freeze.json').exists(): raise ValueError('Completed action freeze immutable')
    models = fitted(); refs = []
    for row in producers:
        _, _, causal = prior.selection_bank(row)
        for arm in api.ARMS:
            guard(); path = PUBLIC/'policies'/(row['trial']+'_'+arm+'.json')
            if resume and path.exists():
                r = json.loads(path.read_text()); checked(r['prediction']); checked(r['choices'])
                if r['checkpoint'] != models[row['trial'],arm]['checkpoint'] or r['registration_sha256'] != sha(REG):
                    raise ValueError('Frozen action binding changed')
            else:
                state = joblib.load(checked(models[row['trial'],arm]['checkpoint'])); p = api.predict(state,causal)
                choices,_ = prior.allocator.select_actions(p,causal['allowed'],arm='physical_neural')
                home = PRIVATE/'policies'/(row['trial']+'_'+arm); home.mkdir(parents=True,exist_ok=True)
                prior.frozen.inputs.store_array(home/'prediction.npy',p); prior.frozen.inputs.store_array(home/'choices.npy',choices)
                once(path,dict(trial=row['trial'],arm=arm,prediction=artifact(home/'prediction.npy'),
                    choices=artifact(home/'choices.npy'),checkpoint=models[row['trial'],arm]['checkpoint'],
                    registration_sha256=sha(REG),selection_already_exposed=True,labels_read_for_this_inference=False))
            refs.append(artifact(path)); beat(state='policy_frozen',policies=len(refs),total=18)
    once(PUBLIC/'policy_freeze.json',dict(**receipt(),policies=refs,training_freeze=artifact(PUBLIC/'training_freeze.json'),
        threshold_search=False,selection_already_exposed=True))


def summarize(rows,cfg):
    sites = sorted({r['site'] for r in rows}); arms = ('mean_absolute',*api.ARMS)
    if len(sites) != 6 or len(rows) != 54 or {(r['trial'],r['site']) for r in rows} != {
            (f'single{f}_seed{s}',site) for f in range(3) for s in (17,29,43) for site in sites}:
        raise ValueError('Complete nine-producer/six-locality development roster required')
    interval = lambda fn:prior.metrics.interval(rows,fn,expected_sites=sites,
        resamples=cfg['bootstrap_resamples'],seed=cfg['bootstrap_seed'])
    accuracy = {a:interval(lambda r,a=a:r['policies'][a]['known_ADE_gain_floor_percent']) for a in arms}
    effects = {a:{k:interval(lambda r,a=a,k=k:r['effects'][a][k])
        for k in ('known_pp','completion_lower_pp','completion_upper_pp')} for a in api.ARMS}
    safety = {}
    for a in arms:
        views = [v for r in rows for v in r['policies'][a]['recordings'].values()]
        easy = [v['known_easy_net_degradation'] for v in views if v['known_easy_net_degradation'] is not None]
        safety[a] = dict(required_record_views=len(views),supported_record_views=sum(v['supported'] for v in views),
            known_all_risk_violations=sum(v['known_all_risk'] is not None and v['known_all_risk'] > .02 for v in views),
            known_easy_risk_violations=sum(v['known_easy_risk'] is not None and v['known_easy_risk'] > .02 for v in views),
            undefined_easy_views=sum(v['completion_easy_risk_upper'] is None for v in views),
            worst_known_easy_net_degradation_percent=100*max(easy) if easy else None)
    ci = effects['context_forest']['completion_lower_pp']['CI95']; primary = ci is not None and ci[0] > 0
    risk = safety['context_forest']['supported_record_views'] == safety['context_forest']['required_record_views']
    return dict(accuracy=accuracy,effects=effects,safety=safety,groups=54,primary_pass=primary,
        original_record_risk_pass=risk,candidate_pass=primary and risk,selection_development_not_confirmation=True,
        new_forecast_training=False,threshold_search=False,calibration_access=False,confirmation_access=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)


def evaluate(verify,resume):
    cfg, producers, ids = admission(); beat, guard, receipt = runtime(cfg,'verify' if verify else 'eval')
    finish = PUBLIC/('verification.json' if verify else 'complete.json')
    if finish.exists(): raise ValueError('Completed evaluation immutable')
    freeze = json.loads((PUBLIC/'policy_freeze.json').read_text())
    if len(freeze['policies']) != 18 or freeze['training_freeze'] != artifact(PUBLIC/'training_freeze.json'):
        raise ValueError('All18new policies must freeze before scoring')
    models = fitted(); saved = {}
    for ref in freeze['policies']:
        r = json.loads(checked(ref).read_text()); checked(r['prediction']); checked(r['choices']); saved[r['trial'],r['arm']] = r
    if verify:
        source = prior.source_inputs(json.loads(prior.frozen.REG.read_text())); controls = parent.fitted()
        for row in producers:
            data,case,_ = fitting_data(row,source,ids,controls)
            for arm in api.ARMS:
                guard(); r = models[row['trial'],arm]; p = api.predict(joblib.load(checked(r['checkpoint'])),data['causal'])
                choice,_ = prior.allocator.select_actions(p,data['causal']['allowed'],arm='physical_neural')
                d = transport.api.diagnose(p,case['target'],data['causal']['allowed'],choice,data['causal']['scales'])
                if d != r['SOURCE_fit_diagnosis'] or api.fingerprint(p) != r['SOURCE_prediction_sha256']:
                    raise ValueError('SOURCE forest cost inference replay differs')
                del p
            beat(state='SOURCE_replay_exact',trial=row['trial'])
            del data,case
        del source
    rows, refs = [], []
    for row in producers:
        guard(); data,paths,causal = prior.selection_bank(row); target,valid = prior.selection_labels(data)
        floor = paths[:,row['baseline_index']]
        old = json.loads((parent.PUBLIC/'policies'/(row['trial']+'_mean_absolute.json')).read_text())
        cp = np.load(checked(old['choices']),allow_pickle=False)
        policies = {'mean_absolute':prior.selected_paths(paths,cp,row['baseline_index'])}
        for arm in api.ARMS:
            r = saved[row['trial'],arm]; p = np.load(checked(r['prediction']),mmap_mode='r',allow_pickle=False)
            choices = np.load(checked(r['choices']),allow_pickle=False)
            if verify:
                np.testing.assert_array_equal(p,api.predict(joblib.load(checked(r['checkpoint'])),causal))
            replay,_ = prior.allocator.select_actions(p,causal['allowed'],arm='physical_neural')
            np.testing.assert_array_equal(choices,replay)
            policies[arm] = prior.selected_paths(paths,choices,row['baseline_index'])
        for site in sorted(set(data['sites'])):
            at = data['sites'] == site
            scores = {a:prior.metrics.policy(p[at],floor[at],paths[at,1],target[at],valid[at],data['recordings'][at],
                easy_cut=row['easy_cut'],hard_cut=row['hard_cut']) for a,p in policies.items()}
            oldgroup = json.loads((parent.PUBLIC/'groups'/(row['trial']+'_'+str(site)+'.json')).read_text())
            if scores['mean_absolute'] != oldgroup['policies']['mean_absolute']:
                raise ValueError('Matched original mean-control metrics changed')
            effects = {a:prior.metrics.paired(policies['mean_absolute'][at],policies[a][at],floor[at],target[at],valid[at]) for a in api.ARMS}
            value = dict(trial=row['trial'],site=str(site),policies=scores,effects=effects,
                registration_sha256=sha(REG),policy_freeze_sha256=sha(PUBLIC/'policy_freeze.json'))
            path = PUBLIC/'groups'/(row['trial']+'_'+str(site)+'.json')
            if verify or resume and path.exists():
                if json.loads(path.read_text()) != value: raise ValueError('Complete metric replay differs')
            else: once(path,value)
            rows.append(value); refs.append(artifact(path))
        beat(state='metric_groups',groups=len(rows),total=54)
    summary = summarize(rows,cfg)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()): raise ValueError('Summary replay differs')
        once(finish,dict(**receipt(),all18_SOURCE_cost_inferences_exact=True,all18_selection_cost_inferences_exact=True,
            all54_metric_groups_exact=True,mean_control_exact=True,summary_sha256=sha(PUBLIC/'summary.json')))
    else:
        once(PUBLIC/'summary.json',summary); once(finish,dict(**receipt(),groups=refs,summary_sha256=sha(PUBLIC/'summary.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','pilot','train','infer','eval','verify'))
    parser.add_argument('--resume',action='store_true'); args = parser.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase == 'register': admission(True)
        elif args.phase == 'pilot': pilot()
        elif args.phase == 'train': train(args.resume)
        elif args.phase == 'infer': infer(args.resume)
        else: evaluate(args.phase == 'verify',args.resume)
