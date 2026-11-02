"""One-factor width128 cost learning; frozen predictors,roles,loss and risk."""
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

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_selection_cost_forest as frozen
from src.evaluation.m3w_selection_cost_fit_objective import score

base,prior,api,np = frozen.parent,frozen.prior,frozen.parent.api,frozen.np
NAME = 'european_selection_capacity_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_selection_capacity_v1.json'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__),CONFIG,ROOT/'tests/test_m3w_selection_capacity.py',PUBLIC/'protocol.md')


def width_only_settings(old,new):
    if old.get('width') != 64 or new.get('width') != 128 or dict(new,width=64) != old:
        raise ValueError('Only the registered64to128width change is allowed')


def assert_common_training(old,new):
    width_only_settings(old['settings'],new['settings'])
    for key in ('preprocess','budget','sampler','draw_hash','fit_ids','unknown_draws','seed','source_sites','arm','objective_arm','step'):
        prior.api.exact(old[key],new[key])
    for key in ('action.bias','reference.bias'):
        prior.api.exact(old['initial_model'][key],new['initial_model'][key])
    for state in (old,new):
        if any(frozen.torch.count_nonzero(state['initial_model'][key]) for key in ('action.weight','reference.weight')):
            raise ValueError('Same zero-head initialization required')


def initial_prediction(state,causal):
    initial = dict(state,model=state['initial_model'])
    return prior.api.predict(initial,causal)


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); old,producers,ids = base.admission()
    width_only_settings(old['training'],cfg['training'])
    diagnostic = ROOT/'outputs/publication_readiness_2026_09/european_cost_decode_decomposition_v1'
    verified = json.loads((diagnostic/'verification.json').read_text())
    if (not verified['all18_SOURCE_inferences_and54_stage_scores_exact']
            or verified['summary_sha256'] != frozen.sha(diagnostic/'summary.json') or cfg['new_fits'] != 9
            or cfg['positive_harm_budget'] != .02 or cfg['easy_net_degradation_budget'] != .02
            or any(cfg[k] for k in ('model_selection_labels_for_training','threshold_search','calibration_access',
                'confirmation_access','new_trajectory_training','stage5c_executed','smc_enabled'))):
        raise ValueError('Verified SOURCE cause diagnosis and unchanged roles/risk required')
    definition = dict(bindings={str(p.relative_to(ROOT)):frozen.sha(p) for p in FILES},
        dependencies={p:frozen.sha(ROOT/p) for p in ('src/world_model/m3w_selection_controller.py',
            'src/world_model/m3w_selection_cardinal_cost.py','src/evaluation/m3w_selection_cost_fit_objective.py')},
        parent_registration=frozen.artifact(base.REG),parent_training=frozen.artifact(base.PUBLIC/'training_freeze.json'),
        parent_policy=frozen.artifact(base.PUBLIC/'policy_freeze.json'),
        diagnosis=frozen.artifact(diagnostic/'summary.json'),diagnosis_replay=frozen.artifact(diagnostic/'verification.json'),
        source_rows=len(ids),new_fits=9,width=128,parameters=58501,actual_seeds=[17,29,43],
        primary=cfg['primary'],selection_already_exposed=True,new_trajectory_training=False,
        threshold_search=False,calibration_access=False,confirmation_access=False,stage5c_executed=False,smc_enabled=False)
    if register: frozen.once(REG,definition)
    if json.loads(REG.read_text()) != definition: raise ValueError('Registered width study changed')
    if not register:
        for path in (*FILES,REG):
            blob = subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)
            if hashlib.sha256(blob).hexdigest() != frozen.sha(path): raise ValueError('Commit capacity study before execution')
    return cfg,producers,ids


def runtime(cfg,phase):
    frozen.torch.set_num_threads(4); frozen.torch.set_num_interop_threads(1); start = time.monotonic()
    def interrupt(*_): raise KeyboardInterrupt('Preserve atomic same-binding checkpoint')
    signal.signal(signal.SIGTERM,interrupt)
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('Keep10GiB disk reserve')
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12hboundary;resume unchanged')
    def beat(**values):
        guard(); value = dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase=phase,**values)
        frozen.write(PRIVATE/'heartbeat.json',value); print(json.dumps(value),flush=True)
    def receipt():
        return dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,registration_sha256=frozen.sha(REG))
    beat(state='runtime_ready',architecture=platform.machine(),device='cpu',threads=4,interop=1,workers=0)
    return beat,guard,receipt


def model_path(row,family='fits'): return PRIVATE/family/(row['trial']+'_width128.pt.gz')


def one(data,row,cfg,beat,guard,*,family='fits',resume=False,stop_at=None):
    return api.fit(data,arm='mean_absolute',seed=row['seed'],settings=cfg['training'],
        identity=dict(registration_sha256=frozen.sha(REG),producer=row['checkpoint'],
            source_bank=frozen.artifact(prior.PRIVATE/'source_banks'/row['trial']/'complete.json')),
        path=model_path(row,family),resume=resume,stop_at=stop_at,guard=guard,
        heartbeat=lambda **v:beat(trial=row['trial'],**v))


def pilot():
    cfg,producers,ids = admission(); beat,guard,receipt = runtime(cfg,'pilot')
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Completed pilot immutable')
    source = prior.source_inputs(json.loads(prior.frozen.REG.read_text())); row = producers[0]
    data,_,_ = frozen.fitting_data(row,source,ids,base.fitted())
    one(data,row,cfg,beat,guard,family='pilot_resume',stop_at=50)
    recovered = one(data,row,cfg,beat,guard,family='pilot_resume',resume=True,stop_at=100)
    direct = one(data,row,cfg,beat,guard,family='pilot_direct',stop_at=100)
    prior.assert_recovery(recovered,direct)
    old = prior.api.read_checkpoint(base.fit_path(row,'mean_absolute','pilot_direct'))
    assert_common_training(old,direct)
    np.testing.assert_array_equal(initial_prediction(old,data['causal']),initial_prediction(direct,data['causal']))
    frozen.once(PUBLIC/'pilot.json',dict(**receipt(),engineering_pass=True,actual_optimizer_updates=200,
        exact50to100_resume=True,old64_vs_new128_initial_SOURCE_outputs_exact=True,
        original100step_query_draws_and_preprocess_exact=True,parameters=direct['parameters'],
        fit100_seconds=direct['seconds'],first_budget=direct['budget'],
        projected9_primary_fit_seconds=direct['seconds']/100*direct['budget']*9,
        projection_excludes_preparation_inference_and_full_refit=True,development_labels_read=False))


def train(resume):
    cfg,producers,ids = admission(); beat,guard,receipt = runtime(cfg,'train')
    if (PUBLIC/'training_freeze.json').exists(): raise ValueError('Completed training immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']: raise ValueError('Real recovery pilot required')
    controls = base.fitted(); source = prior.source_inputs(json.loads(prior.frozen.REG.read_text())); refs=[]; updates=refit_updates=0
    for row in producers:
        guard(); data,case,_ = frozen.fitting_data(row,source,ids,controls)
        old = prior.api.read_checkpoint(frozen.checked(controls[row['trial'],'mean_absolute']['checkpoint']))
        state = one(data,row,cfg,beat,guard,resume=resume and model_path(row).exists())
        assert_common_training(old,state)
        if state['step'] != state['budget'] or state['sampler']['counts'].min() < 4:
            raise ValueError('Original complete budget and query exposure required')
        np.testing.assert_array_equal(initial_prediction(old,data['causal']),initial_prediction(state,data['causal']))
        if row == producers[0]:
            other = one(data,row,cfg,beat,guard,family='full_refit',resume=resume and model_path(row,'full_refit').exists())
            prior.assert_recovery(state,other); refit_updates += other['step']
            frozen.once(PUBLIC/'first_full_refit.json',dict(scientific_state_exact=True,updates=other['step'],
                model=frozen.artifact(model_path(row)),refit=frozen.artifact(model_path(row,'full_refit'))))
            del other
        c = data['causal']; p = prior.api.predict(state,c)
        objective = score(p,data['truth'].numpy(),c['allowed'],data['known'],c['scales'],data['preprocess']['rms'],
            c['groups'],state['sampler']['counts'],c['sites'])
        choice,_ = prior.allocator.select_actions(p,c['allowed'],arm='physical_neural')
        diagnosis = frozen.transport.api.diagnose(p,case['target'],c['allowed'],choice,c['scales'])
        path = PUBLIC/'fits'/(row['trial']+'.json')
        frozen.once(path,dict(trial=row['trial'],checkpoint=frozen.artifact(model_path(row)),
            step=state['step'],budget=state['budget'],parameters=state['parameters'],seconds=state['seconds'],
            source_sites=state['source_sites'],draw_hash=state['draw_hash'],original_sampler_preprocess_fit_ids_exact=True,
            initial_SOURCE_outputs_equal_old64=True,minimum_query_visits=int(state['sampler']['counts'].min()),
            unknown_supervised_rows=0,SOURCE_objective=objective,SOURCE_diagnosis=diagnosis,
            SOURCE_prediction_sha256=frozen.api.fingerprint(p),trace=state['trace'],registration_sha256=frozen.sha(REG)))
        refs.append(frozen.artifact(path)); updates+=state['step']; beat(state='fit_complete',fits=len(refs),total=9,trial=row['trial'])
        del data,case,c,old,state,p
    frozen.once(PUBLIC/'training_freeze.json',dict(**receipt(),fits=refs,primary_optimizer_updates=updates,
        first_full_refit_updates=refit_updates,source_only=True,new_trajectory_training=False))


def fitted():
    result = {}
    for ref in json.loads((PUBLIC/'training_freeze.json').read_text())['fits']:
        value = json.loads(frozen.checked(ref).read_text()); frozen.checked(value['checkpoint']); result[value['trial']] = value
    if len(result) != 9: raise ValueError('All9wide critics required')
    return result


def infer(resume):
    cfg,producers,_ = admission(); beat,guard,receipt = runtime(cfg,'infer'); models=fitted(); refs=[]
    if (PUBLIC/'policy_freeze.json').exists(): raise ValueError('Completed policy freeze immutable')
    for row in producers:
        guard(); _,_,c = prior.selection_bank(row); path = PUBLIC/'policies'/(row['trial']+'.json')
        if resume and path.exists():
            value = json.loads(path.read_text()); frozen.checked(value['prediction']); frozen.checked(value['choices'])
            if value['checkpoint'] != models[row['trial']]['checkpoint']: raise ValueError('Same checkpoint required')
        else:
            state = prior.api.read_checkpoint(frozen.checked(models[row['trial']]['checkpoint'])); p=prior.api.predict(state,c)
            choices,_ = prior.allocator.select_actions(p,c['allowed'],arm='physical_neural')
            home = PRIVATE/'policies'/row['trial']; home.mkdir(parents=True,exist_ok=True)
            prior.frozen.inputs.store_array(home/'prediction.npy',p); prior.frozen.inputs.store_array(home/'choices.npy',choices)
            frozen.once(path,dict(trial=row['trial'],checkpoint=models[row['trial']]['checkpoint'],
                prediction=frozen.artifact(home/'prediction.npy'),choices=frozen.artifact(home/'choices.npy'),
                registration_sha256=frozen.sha(REG),labels_read_for_this_inference=False,selection_already_exposed=True))
        refs.append(frozen.artifact(path)); beat(state='policy_frozen',policies=len(refs),total=9)
    frozen.once(PUBLIC/'policy_freeze.json',dict(**receipt(),policies=refs,training_freeze=frozen.artifact(PUBLIC/'training_freeze.json')))


def summarize(rows,cfg):
    sites = sorted({r['site'] for r in rows})
    expected={(f'single{f}_seed{s}',site) for f in range(3) for s in (17,29,43) for site in sites}
    if len(sites)!=6 or len(rows)!=54 or {(r['trial'],r['site']) for r in rows}!=expected:
        raise ValueError('All54fixed development groups required')
    interval=lambda fn:prior.metrics.interval(rows,fn,expected_sites=sites,resamples=cfg['bootstrap_resamples'],seed=cfg['bootstrap_seed'])
    accuracy={arm:interval(lambda r,a=arm:r['policies'][a]['known_ADE_gain_floor_percent']) for arm in ('width64','width128')}
    effect={k:interval(lambda r,k=k:r['effect'][k]) for k in ('known_pp','completion_lower_pp','completion_upper_pp')}
    safety={}
    for arm in accuracy:
        views=[v for r in rows for v in r['policies'][arm]['recordings'].values()]
        easy=[v['known_easy_net_degradation'] for v in views if v['known_easy_net_degradation'] is not None]
        safety[arm]=dict(required_views=len(views),supported_views=sum(v['supported'] for v in views),
            known_all_violations=sum(v['known_all_risk'] is not None and v['known_all_risk']>.02 for v in views),
            known_easy_violations=sum(v['known_easy_risk'] is not None and v['known_easy_risk']>.02 for v in views),
            undefined_easy_views=sum(v['completion_easy_risk_upper'] is None for v in views),
            worst_signed_easy_net_percent=100*max(easy) if easy else None)
    ci=effect['completion_lower_pp']['CI95']; primary=ci is not None and ci[0]>0
    risk=safety['width128']['supported_views']==safety['width128']['required_views']
    return dict(accuracy=accuracy,effects=effect,safety=safety,primary_pass=primary,original_record_risk_pass=risk,
        candidate_pass=primary and risk,groups=54,selection_development_not_confirmation=True,
        new_trajectory_training=False,threshold_search=False,calibration_access=False,confirmation_access=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)


def evaluate(verify,resume):
    cfg,producers,ids=admission(); beat,guard,receipt=runtime(cfg,'verify' if verify else 'eval')
    finish=PUBLIC/('verification.json' if verify else 'complete.json')
    if finish.exists(): raise ValueError('Completed evaluation immutable')
    models=fitted(); freeze=json.loads((PUBLIC/'policy_freeze.json').read_text())
    if len(freeze['policies'])!=9 or freeze['training_freeze']!=frozen.artifact(PUBLIC/'training_freeze.json'):
        raise ValueError('All9new policies must freeze before scoring')
    policies={}
    for ref in freeze['policies']:
        v=json.loads(frozen.checked(ref).read_text()); frozen.checked(v['prediction']); frozen.checked(v['choices']); policies[v['trial']]=v
    if verify:
        source=prior.source_inputs(json.loads(prior.frozen.REG.read_text())); controls=base.fitted()
        for row in producers:
            guard(); data,_,counts=frozen.fitting_data(row,source,ids,controls)
            state=prior.api.read_checkpoint(frozen.checked(models[row['trial']]['checkpoint'])); p=prior.api.predict(state,data['causal'])
            if frozen.api.fingerprint(p)!=models[row['trial']]['SOURCE_prediction_sha256']: raise ValueError('SOURCE inference replay differs')
            measured=score(p,data['truth'].numpy(),data['causal']['allowed'],data['known'],data['causal']['scales'],
                data['preprocess']['rms'],data['causal']['groups'],counts,data['causal']['sites'])
            if measured!=models[row['trial']]['SOURCE_objective']: raise ValueError('SOURCE objective replay differs')
            beat(state='SOURCE_replay_exact',trial=row['trial']); del data,state,p
        del source
    rows=[]; refs=[]
    for row in producers:
        guard(); data,paths,c=prior.selection_bank(row); target,valid=prior.selection_labels(data); floor=paths[:,row['baseline_index']]
        old=json.loads((base.PUBLIC/'policies'/(row['trial']+'_mean_absolute.json')).read_text())
        control=np.load(frozen.checked(old['choices']),allow_pickle=False)
        saved=policies[row['trial']]; p=np.load(frozen.checked(saved['prediction']),allow_pickle=False)
        choices=np.load(frozen.checked(saved['choices']),allow_pickle=False)
        if verify:
            fresh=prior.api.predict(prior.api.read_checkpoint(frozen.checked(saved['checkpoint'])),c); np.testing.assert_array_equal(p,fresh)
        replay,_=prior.allocator.select_actions(p,c['allowed'],arm='physical_neural'); np.testing.assert_array_equal(choices,replay)
        forecasts=dict(width64=prior.selected_paths(paths,control,row['baseline_index']),width128=prior.selected_paths(paths,choices,row['baseline_index']))
        for site in sorted(set(data['sites'])):
            at=data['sites']==site
            scores={a:prior.metrics.policy(v[at],floor[at],paths[at,1],target[at],valid[at],data['recordings'][at],
                easy_cut=row['easy_cut'],hard_cut=row['hard_cut']) for a,v in forecasts.items()}
            original=json.loads((base.PUBLIC/'groups'/(row['trial']+'_'+str(site)+'.json')).read_text())
            if scores['width64']!=original['policies']['mean_absolute']: raise ValueError('Old control changed')
            effect=prior.metrics.paired(forecasts['width64'][at],forecasts['width128'][at],floor[at],target[at],valid[at])
            value=dict(trial=row['trial'],site=str(site),policies=scores,effect=effect,
                registration_sha256=frozen.sha(REG),policy_freeze_sha256=frozen.sha(PUBLIC/'policy_freeze.json'))
            path=PUBLIC/'groups'/(row['trial']+'_'+str(site)+'.json')
            if verify or resume and path.exists():
                if json.loads(path.read_text())!=value: raise ValueError('Metric replay differs')
            else: frozen.once(path,value)
            rows.append(value); refs.append(frozen.artifact(path))
        beat(state='metric_groups',groups=len(rows),total=54)
    summary=summarize(rows,cfg)
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()): raise ValueError('Summary replay differs')
        frozen.once(finish,dict(**receipt(),all9_SOURCE_inferences_and_objectives_exact=True,
            all9_development_inferences_exact=True,all54_metric_groups_exact=True,old_control_exact=True,summary_sha256=frozen.sha(PUBLIC/'summary.json')))
    else:
        frozen.once(PUBLIC/'summary.json',summary); frozen.once(finish,dict(**receipt(),groups=refs,summary_sha256=frozen.sha(PUBLIC/'summary.json')))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','pilot','train','infer','eval','verify')); parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register': admission(True)
        elif args.phase=='pilot': pilot()
        elif args.phase=='train': train(args.resume)
        elif args.phase=='infer': infer(args.resume)
        else: evaluate(args.phase=='verify',args.resume)
