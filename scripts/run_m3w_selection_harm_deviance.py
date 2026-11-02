"""Complete SOURCE-only harm-loss experiment; exposed selection is not a test."""
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
    raise RuntimeError('Native arm64 required before Torch imports')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_selection_cardinal_cost as parent
from scripts import run_m3w_selection_cost_transport as transport
from src.world_model import m3w_selection_harm_deviance as api

prior = parent.prior
NAME = 'european_selection_harm_deviance_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_selection_harm_deviance_v1.json'; REG = PUBLIC/'registration.json'
sha, artifact, checked, once, write = parent.sha, parent.artifact, parent.checked, parent.once, parent.write
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_selection_harm_deviance.py',
         ROOT/'tests/test_m3w_selection_harm_deviance.py', PUBLIC/'protocol.md')


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); old, producers, ids = parent.admission()
    source = ROOT/'outputs/publication_readiness_2026_09/european_calibration_source_fit_v1'
    verification = json.loads((source/'verification.json').read_text())
    if (not verification['all18_SOURCE_model_inferences_exact']
            or verification['summary_sha256'] != sha(source/'summary.json')
            or cfg['training'] != old['training'] or cfg['new_arms'] != list(api.NEW_ARMS)
            or cfg['new_fits'] != 18 or cfg['positive_harm_budget'] != .02 or cfg['easy_net_budget'] != .02
            or any(cfg[k] for k in ('selection_labels_for_training', 'threshold_search',
                'calibration_array_access', 'confirmation_access', 'new_forecast_training', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Verified SOURCE failure and unchanged complete training design required')
    definition = dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_registration=artifact(parent.REG), parent_training=artifact(parent.PUBLIC/'training_freeze.json'),
        parent_policy=artifact(parent.PUBLIC/'policy_freeze.json'),
        SOURCE_diagnosis=artifact(source/'summary.json'), SOURCE_diagnosis_replay=artifact(source/'verification.json'),
        historical_easy_deviance_failure=artifact(ROOT/'outputs/publication_readiness_2026_09/european_easy_harm_deviance_v1/readout_verified_v3/conclusions.md'),
        dependencies={p:sha(ROOT/p) for p in ('src/world_model/m3w_selection_cardinal_cost.py',
            'src/world_model/m3w_selection_controller.py', 'src/world_model/m3w_easy_harm_deviance.py',
            'src/evaluation/m3w_selection_forecast_readout.py', 'scripts/run_m3w_selection_cost_transport.py')},
        actual_seeds=[17, 29, 43], new_SOURCE_fits=18, source_whitelist_rows=len(ids),
        primary=cfg['primary'], selection_already_exposed=True, calibration_arrays_read=False,
        threshold_search=False, confirmation_access=False, deployment_changed=False,
        new_forecast_training=False, stage5c_executed=False, smc_enabled=False)
    if register: once(REG, definition)
    if json.loads(REG.read_text()) != definition:
        raise ValueError('Registered complete experiment changed')
    if not register:
        for p in (*FILES, REG):
            content = subprocess.check_output(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT)
            if hashlib.sha256(content).hexdigest() != sha(p):
                raise ValueError('Commit training contract before real execution')
    return cfg, producers, ids


def runtime(cfg, phase):
    torch.set_num_threads(4); torch.set_num_interop_threads(1); start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume atomic checkpoint')))
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('Preserve10GiB reserve')
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12h boundary; resume unchanged')
    def beat(**v):
        guard(); row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), phase=phase, **v)
        write(PRIVATE/'heartbeat.json', row); print(json.dumps(row), flush=True)
    def receipt():
        return dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, registration_sha256=sha(REG))
    beat(state='runtime_ready', architecture=platform.machine(), device='cpu', threads=4, interop=1, workers=0)
    return beat, guard, receipt


def fit_path(row, arm, family='fits'):
    return PRIVATE/family/(row['trial']+'_'+arm+'.pt.gz')


def fitting_data(row, source, ids):
    case = transport.source_case(row, source, ids)
    return prior.api.training_data(case['causal'], case['target']), case


def one(data, row, arm, cfg, beat, guard, family='fits', resume=False, stop_at=None):
    return api.fit(data, arm=arm, seed=row['seed'], settings=cfg['training'],
        identity=dict(registration_sha256=sha(REG), source_bank=artifact(prior.PRIVATE/'source_banks'/row['trial']/'complete.json'),
                      producer=row['checkpoint']), path=fit_path(row, arm, family),
        heartbeat=lambda **v:beat(trial=row['trial'], **v), guard=guard, resume=resume, stop_at=stop_at)


def pilot():
    cfg, producers, ids = admission(); beat, guard, receipt = runtime(cfg, 'pilot')
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Completed pilot immutable')
    row = producers[0]; source = prior.source_inputs(json.loads(prior.frozen.REG.read_text()))
    data, _ = fitting_data(row, source, ids); cells = []
    for arm in api.ARMS:
        one(data, row, arm, cfg, beat, guard, family='pilot_resume', stop_at=50)
        a = one(data, row, arm, cfg, beat, guard, family='pilot_resume', resume=True, stop_at=100)
        b = one(data, row, arm, cfg, beat, guard, family='pilot_direct', stop_at=100)
        prior.assert_recovery(a, b)
        original_exact = None
        if arm == 'mean_absolute':
            old = prior.api.read_checkpoint(parent.fit_path(row, arm, 'pilot_direct'))
            for key in (*parent.MATCHED, 'model', 'optimizer', 'trace', 'step'): prior.api.exact(b[key], old[key])
            original_exact = True
        cells.append(dict(arm=arm, updates=100, budget=b['budget'], seconds=b['seconds'],
            parameters=b['parameters'], exact_resume=True, original_mean_control_exact=original_exact))
    once(PUBLIC/'pilot.json', dict(**receipt(), engineering_pass=True, cells=cells,
        actual_updates=600, projection_excludes_data_preparation_and_refits=True,
        projected18_primary_fit_seconds=sum(c['seconds']/100*c['budget'] for c in cells[1:])*9))


def train(resume):
    cfg, producers, ids = admission(); beat, guard, receipt = runtime(cfg, 'train')
    if (PUBLIC/'training_freeze.json').exists(): raise ValueError('Completed training immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']: raise ValueError('Real recovery pilot required')
    source = prior.source_inputs(json.loads(prior.frozen.REG.read_text())); originals = parent.fitted()
    refs, updates, refit_updates = [], 0, 0
    for row in producers:
        data, case = fitting_data(row, source, ids)
        control = prior.api.read_checkpoint(checked(originals[row['trial'], 'mean_absolute']['checkpoint']))
        if row == producers[0]:
            exact = one(data, row, 'mean_absolute', cfg, beat, guard, family='full_control_refit',
                resume=resume and fit_path(row, 'mean_absolute', 'full_control_refit').exists())
            for key in (*parent.MATCHED, 'model', 'optimizer', 'trace', 'step'): prior.api.exact(exact[key], control[key])
            once(PUBLIC/'full_mean_control_refit.json', dict(model_optimizer_sampler_trace_exact=True,
                updates=exact['step'], checkpoint=artifact(fit_path(row, 'mean_absolute', 'full_control_refit'))))
            refit_updates += exact['step']; del exact
        for arm in api.NEW_ARMS:
            state = one(data, row, arm, cfg, beat, guard, resume=resume and fit_path(row, arm).exists())
            for key in parent.MATCHED: prior.api.exact(state[key], control[key])
            if state['step'] != state['budget'] or state['sampler']['counts'].min() < 4:
                raise ValueError('Every fitting query requires full budget')
            if row == producers[0]:
                other = one(data, row, arm, cfg, beat, guard, family='full_refit',
                    resume=resume and fit_path(row, arm, 'full_refit').exists())
                prior.assert_recovery(state, other); refit_updates += other['step']
                once(PUBLIC/('first_full_refit_'+arm+'.json'), dict(scientific_state_exact=True,
                    checkpoint=artifact(fit_path(row, arm)), refit=artifact(fit_path(row, arm, 'full_refit'))))
            prediction = prior.api.predict(state, data['causal'])
            choice, _ = prior.allocator.select_actions(prediction, data['causal']['allowed'], arm='physical_neural')
            diagnostic = transport.api.diagnose(prediction, case['target'], data['causal']['allowed'], choice, data['causal']['scales'])
            path = PUBLIC/'fits'/(row['trial']+'_'+arm+'.json')
            result = dict(trial=row['trial'], arm=arm, checkpoint=artifact(fit_path(row, arm)),
                step=state['step'], budget=state['budget'], parameters=state['parameters'], seconds=state['seconds'],
                source_sites=state['source_sites'], fit_ids_sha256=prior.frozen.fingerprint(state['fit_ids']),
                draw_hash=state['draw_hash'], minimum_query_visits=int(state['sampler']['counts'].min()),
                unknown_supervised_rows=0, matched_to_original_control=True, SOURCE_fit_diagnosis=diagnostic,
                SOURCE_fit_prediction_sha256=prior.frozen.fingerprint(prediction), trace=state['trace'],
                registration_sha256=sha(REG))
            once(path, result); refs.append(artifact(path)); updates += state['step']
            beat(state='fit_complete', trial=row['trial'], arm=arm, fits=len(refs), total=18)
            del state, prediction
        del data, case, control
    once(PUBLIC/'training_freeze.json', dict(**receipt(), fits=refs,
        primary_optimizer_updates=updates, complete_refit_updates=refit_updates,
        source_only=True, selection_labels_used=False, new_forecast_training=False))


def fitted():
    freeze = json.loads((PUBLIC/'training_freeze.json').read_text()); rows = {}
    for ref in freeze['fits']:
        r = json.loads(checked(ref).read_text()); checked(r['checkpoint']); rows[r['trial'], r['arm']] = r
    if len(rows) != 18: raise ValueError('All18complete cost fits required')
    return rows


def infer(resume):
    cfg, producers, _ = admission(); beat, guard, receipt = runtime(cfg, 'infer')
    if (PUBLIC/'policy_freeze.json').exists(): raise ValueError('Full policy freeze immutable')
    models = fitted(); refs = []
    for row in producers:
        _, _, causal = prior.selection_bank(row)
        for arm in api.NEW_ARMS:
            guard(); path = PUBLIC/'policies'/(row['trial']+'_'+arm+'.json')
            if path.exists() and resume:
                r = json.loads(path.read_text()); checked(r['prediction']); checked(r['choices'])
                if r['registration_sha256'] != sha(REG) or r['checkpoint'] != models[row['trial'], arm]['checkpoint']:
                    raise ValueError('Policy resume binding changed')
                refs.append(artifact(path)); continue
            state = prior.api.read_checkpoint(checked(models[row['trial'], arm]['checkpoint']))
            p = prior.api.predict(state, causal); choices, _ = prior.allocator.select_actions(p, causal['allowed'], arm='physical_neural')
            home = PRIVATE/'policies'/(row['trial']+'_'+arm); home.mkdir(parents=True, exist_ok=True)
            prior.frozen.inputs.store_array(home/'prediction.npy', p); prior.frozen.inputs.store_array(home/'choices.npy', choices)
            once(path, dict(trial=row['trial'], arm=arm, prediction=artifact(home/'prediction.npy'),
                choices=artifact(home/'choices.npy'), checkpoint=models[row['trial'], arm]['checkpoint'],
                registration_sha256=sha(REG), labels_read_for_this_comparison=False,
                selection_previously_exposed=True))
            refs.append(artifact(path)); beat(state='policy_frozen', policies=len(refs), total=18)
    once(PUBLIC/'policy_freeze.json', dict(**receipt(), policies=refs, training_freeze=artifact(PUBLIC/'training_freeze.json'),
        threshold_search=False, selection_previously_exposed=True))


def summarize(rows, cfg):
    sites = sorted({r['site'] for r in rows})
    if len(rows) != 54 or len(sites) != 6 or {(r['trial'], r['site']) for r in rows} != {
            (f'single{f}_seed{s}', site) for f in range(3) for s in (17,29,43) for site in sites}:
        raise ValueError('All9pipelines and6selection localities required')
    ci = lambda fn:prior.metrics.interval(rows, fn, expected_sites=sites, resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'])
    accuracy = {a:ci(lambda r,a=a:r['policies'][a]['known_ADE_gain_floor_percent']) for a in api.ARMS}
    effects = {a:{k:ci(lambda r,a=a,k=k:r['effects'][a][k]) for k in ('known_pp','completion_lower_pp','completion_upper_pp')} for a in api.NEW_ARMS}
    safety = {}
    for arm in api.ARMS:
        views = [v for r in rows for v in r['policies'][arm]['recordings'].values()]
        easy = [v['known_easy_net_degradation'] for v in views if v['known_easy_net_degradation'] is not None]
        safety[arm] = dict(required_record_views=len(views), supported_record_views=sum(v['supported'] for v in views),
            known_all_risk_violations=sum(v['known_all_risk'] is not None and v['known_all_risk'] > .02 for v in views),
            known_easy_risk_violations=sum(v['known_easy_risk'] is not None and v['known_easy_risk'] > .02 for v in views),
            undefined_easy_views=sum(v['completion_easy_risk_upper'] is None for v in views),
            worst_known_easy_net_degradation_percent=100*max(easy) if easy else None)
    interval = effects['dual_deviance']['completion_lower_pp']['CI95']
    primary = interval is not None and interval[0] > 0
    risk = safety['dual_deviance']['supported_record_views'] == safety['dual_deviance']['required_record_views']
    return dict(accuracy=accuracy, effects=effects, safety=safety, groups=54, primary_pass=primary,
        original_record_risk_pass=risk, candidate_pass=primary and risk, selection_development_not_confirmation=True,
        threshold_search=False, calibration_access=False, confirmation_access=False,
        new_forecast_training=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)


def evaluate(verify, resume):
    cfg, producers, _ = admission(); beat, guard, receipt = runtime(cfg, 'verify' if verify else 'eval')
    finish = PUBLIC/('verification.json' if verify else 'complete.json')
    if finish.exists(): raise ValueError('Completed evaluation immutable')
    freeze = json.loads((PUBLIC/'policy_freeze.json').read_text())
    if len(freeze['policies']) != 18 or freeze['training_freeze'] != artifact(PUBLIC/'training_freeze.json'):
        raise ValueError('All new policies must freeze first')
    saved = {}
    for ref in freeze['policies']:
        r = json.loads(checked(ref).read_text()); checked(r['prediction']); checked(r['choices']); saved[r['trial'], r['arm']] = r
    rows, refs = [], []
    for row in producers:
        guard(); data, paths, causal = prior.selection_bank(row); target, valid = prior.selection_labels(data)
        floor = paths[:, row['baseline_index']]
        control = json.loads((parent.PUBLIC/'policies'/(row['trial']+'_mean_absolute.json')).read_text())
        cp = np.load(checked(control['choices']), allow_pickle=False)
        policies = dict(mean_absolute=prior.selected_paths(paths, cp, row['baseline_index']))
        for arm in api.NEW_ARMS:
            ref = saved[row['trial'], arm]; p = np.load(checked(ref['prediction']), allow_pickle=False)
            choices = np.load(checked(ref['choices']), allow_pickle=False)
            if verify:
                fresh = prior.api.predict(prior.api.read_checkpoint(checked(ref['checkpoint'])), causal)
                np.testing.assert_array_equal(p, fresh)
            replay, _ = prior.allocator.select_actions(p, causal['allowed'], arm='physical_neural')
            np.testing.assert_array_equal(choices, replay)
            policies[arm] = prior.selected_paths(paths, choices, row['baseline_index'])
        for site in sorted(set(data['sites'])):
            at = data['sites'] == site
            scores = {a:prior.metrics.policy(p[at], floor[at], paths[at, 1], target[at], valid[at], data['recordings'][at],
                easy_cut=row['easy_cut'], hard_cut=row['hard_cut']) for a,p in policies.items()}
            old = json.loads((parent.PUBLIC/'groups'/(row['trial']+'_'+str(site)+'.json')).read_text())
            if scores['mean_absolute'] != old['policies']['mean_absolute']: raise ValueError('Original mean control changed')
            effects = {a:prior.metrics.paired(policies['mean_absolute'][at], policies[a][at], floor[at], target[at], valid[at]) for a in api.NEW_ARMS}
            result = dict(trial=row['trial'], site=str(site), policies=scores, effects=effects,
                registration_sha256=sha(REG), policy_freeze_sha256=sha(PUBLIC/'policy_freeze.json'))
            path = PUBLIC/'groups'/(row['trial']+'_'+str(site)+'.json')
            if verify or resume and path.exists():
                if json.loads(path.read_text()) != result: raise ValueError('Full scalar replay differs')
            else: once(path, result)
            rows.append(result); refs.append(artifact(path))
        beat(state='metric_groups', groups=len(rows), total=54)
    summary = summarize(rows, cfg)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()): raise ValueError('Summary replay differs')
        once(finish, dict(**receipt(), all18_new_cost_inferences_exact=True, all54_metric_groups_exact=True,
            mean_control_exact=True, summary_sha256=sha(PUBLIC/'summary.json')))
    else:
        once(PUBLIC/'summary.json', summary); once(finish, dict(**receipt(), groups=refs, summary_sha256=sha(PUBLIC/'summary.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register','pilot','train','infer','eval','verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase == 'register': admission(True)
        elif args.phase == 'pilot': pilot()
        elif args.phase == 'train': train(args.resume)
        elif args.phase == 'infer': infer(args.resume)
        else: evaluate(args.phase == 'verify', args.resume)
