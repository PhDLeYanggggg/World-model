"""Registered source-development forecast loss experiment; no independent readout."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts import run_m3w_european_source_forecast as parent
from src.world_model.m3w_european_source_forecast import SourceForecaster
from src.world_model.m3w_native_forecast import fit_trial as native_fit
from src.world_model.m3w_reference_harm_forecast import fit_trial, assert_training_equivalent

CONFIG = 'configs/m3w_reference_harm_forecast_v1.json'
NAME = 'european_reference_harm_forecast_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
FILES = (CONFIG, 'scripts/run_m3w_reference_harm_forecast.py',
         'src/world_model/m3w_reference_harm_forecast.py',
         'tests/test_m3w_reference_harm_forecast.py',
         str((PUBLIC/'protocol.md').relative_to(ROOT)))
digest, immutable_json, json_write = parent.digest, parent.immutable_json, parent.json_write


def beat(state, **values):
    event = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 state=state, **values)
    json_write(PRIVATE/'heartbeat.json', event)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(event)+'\n')
    print(json.dumps(event), flush=True)


def assert_identity(identity):
    parent.assert_identity(identity['parent_identity'])
    for path, sha in identity['bindings'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Registered dependency changed: '+path)


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    if (cfg['objective'] != dict(selected_weight=2., cv_weight=2.)
            or cfg['models'] != 18 or cfg['steps_per_model'] != 4000
            or cfg['independent_reserved_readout'] or cfg['new_candidate_deployment_allowed']):
        raise ValueError('Registration scope differs')
    if shutil.disk_usage(ROOT).free < cfg['minimum_free_disk_GiB']*1024**3:
        raise RuntimeError('Disk reserve insufficient; do not downgrade or delete data')
    pc, manifest, receipts, pi = parent.load_source()
    data = parent.prepare(manifest, receipts, pi)
    designs = parent.trials(pc, data, pi)
    controls = {key:parent.complete(key, ti, pc) for key, _, ti in designs}
    identity = dict(parent_identity=pi, bindings={p:digest(ROOT/p) for p in FILES},
        controls={k:v['checkpoint_sha256'] for k,v in controls.items()},
        source_role='previously_opened_source_development_not_confirmation',
        objective=cfg['objective'], runtime=pc['runtime'])
    immutable_json(PRIVATE/'identity.json', identity)
    immutable_json(PUBLIC/'registration.json', dict(identity=identity, trials=[ti for _,_,ti in designs],
        planned_primary_updates=72000, planned_neutral_updates=4000, planned_engineering_updates=80,
        result_source='cached_verified_input_assets_new_training_not_run_at_registration',
        source_rows=len(data['sites']), unique_recordings=len(set(data['recordings'])),
        source_localities=len(set(data['sites'])), independent_roles_opened=False))
    return cfg, pc, data, designs, controls, identity


def trial_identity(identity, ti):
    return dict(experiment=identity, parent_trial=ti)


def fit_one(key, design, ti, pc, data, identity, directory, *, weights, resume=False,
            stop_at=None, native=False):
    torch.manual_seed(ti['seed'])
    model = SourceForecaster(pc['architecture'], design['baseline_index'])
    fn = native_fit if native else fit_trial
    extra = {} if native else weights
    fit = fn(model, data, design, seed=ti['seed'], settings=pc['training'],
        identity=trial_identity(identity, ti), directory=directory, resume=resume, stop_at=stop_at,
        heartbeat=lambda **kw:beat(trial=key, **kw), **extra)
    cp = directory/'checkpoint.pt'
    output = dict(identity=trial_identity(identity, ti), fit=fit,
                  checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=digest(cp), objective=weights)
    assert_identity(identity)
    return output


def saved(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def verify_receipt(path):
    result = json.loads(path.read_text())
    for cp, sha in result['checkpoints'].items():
        if digest(ROOT/cp) != sha:
            raise ValueError('Receipt checkpoint changed')
    return result


def pilot(cfg, pc, data, designs, identity):
    path = PUBLIC/'pilot.json'
    if path.exists():
        verify_receipt(path)
        return
    started = time.monotonic()
    key, design, ti = designs[0]
    outputs = {}
    for mode in ('direct', 'resumed', 'neutral', 'native'):
        directory = PRIVATE/'pilot'/mode
        weights = dict(selected_weight=0., cv_weight=0.) if mode in ('neutral','native') else cfg['objective']
        if mode == 'resumed':
            fit_one('pilot_'+mode, design, ti, pc, data, identity, directory,
                weights=weights, stop_at=cfg['pilot_resume_at'])
        outputs[mode] = fit_one('pilot_'+mode, design, ti, pc, data, identity, directory,
            weights=weights, stop_at=cfg['pilot_steps'], resume=mode=='resumed', native=mode=='native')
    get = lambda mode:saved(ROOT/outputs[mode]['checkpoint'])
    assert_training_equivalent(get('direct'), get('resumed'))
    assert_training_equivalent(get('neutral'), get('native'))
    seconds = time.monotonic()-started
    result = dict(result_source='fresh_run', pid=os.getpid(), updates=80, seconds=seconds,
        exact_resume=True, exact_native_neutral=True, held_rows_sampled=0,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        checkpoints={v['checkpoint']:v['checkpoint_sha256'] for v in outputs.values()},
        estimated_primary_fit_seconds=outputs['direct']['fit']['seconds']/20*72000,
        estimate_is_not_completion=True, independent_roles_opened=False)
    immutable_json(path, result)
    beat('real_pilot_complete', seconds=seconds, updates=80)


def neutral(cfg, pc, data, designs, controls, identity, resume):
    path = PUBLIC/'neutral_replay.json'
    if path.exists():
        verify_receipt(path)
        return
    key, design, ti = next(v for v in designs if v[0] == cfg['full_neutral_control'])
    output = fit_one('neutral_'+key, design, ti, pc, data, identity, PRIVATE/'neutral'/key,
        weights=dict(selected_weight=0., cv_weight=0.), resume=resume)
    reference = controls[key]
    assert_training_equivalent(saved(ROOT/output['checkpoint']), saved(ROOT/reference['checkpoint']))
    immutable_json(path, dict(result_source='fresh_run_replay_against_cached_verified', pid=os.getpid(),
        control_key=key, fit=output['fit'], exact_full_neutral=True,
        checkpoints={v['checkpoint']:v['checkpoint_sha256'] for v in (output,reference)}))
    beat('full_neutral_replay_exact', trial=key, updates=output['fit']['step'])


def complete(key, ti, identity):
    directory = PRIVATE/'trials'/key
    result = json.loads((directory/'complete.json').read_text())
    if (result['identity'] != trial_identity(identity,ti) or not result['fit']['complete']
            or result['fit']['step'] != 4000 or result['fit']['held_rows_sampled'] != 0
            or digest(ROOT/result['checkpoint']) != result['checkpoint_sha256']):
        raise ValueError('Changed or incomplete final checkpoint: '+key)
    return result


def train(cfg, pc, data, designs, identity, resume):
    verify_receipt(PUBLIC/'pilot.json')
    verify_receipt(PUBLIC/'neutral_replay.json')
    started = time.monotonic()
    fresh_updates = 0
    finals = {}
    for key, design, ti in designs:
        directory = PRIVATE/'trials'/key
        if (directory/'complete.json').exists():
            finals[key] = complete(key, ti, identity)
            beat('verified_completed_fit', trial=key)
            continue
        beat('begin_primary_fit', trial=key, train_rows=len(design['train_ids']),
             excluded_rows=len(design['held_ids']))
        output = fit_one(key, design, ti, pc, data, identity, directory,
                        weights=cfg['objective'], resume=resume)
        immutable_json(directory/'complete.json', output)
        finals[key] = complete(key, ti, identity)
        fresh_updates += output['fit']['new_updates']
        beat('primary_fit_complete', trial=key, seconds=output['fit']['seconds'])
    freeze = dict(result_source='fresh_run', identity=identity, fits=finals,
        completed_models=len(finals), primary_updates=sum(v['fit']['step'] for v in finals.values()),
        summed_fit_seconds=sum(v['fit']['seconds'] for v in finals.values()),
        readout_not_yet_run=True, independent_roles_opened=False, deployment_changed=False)
    immutable_json(PUBLIC/'training_freeze.json', freeze)
    json_write(PRIVATE/'training_runtime.json', dict(pid=os.getpid(), seconds=time.monotonic()-started,
        new_updates_this_invocation=fresh_updates, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat('primary_training_complete', models=len(finals), updates=freeze['primary_updates'],
         freeze_sha256=digest(PUBLIC/'training_freeze.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for phase in ('prepare', 'pilot', 'neutral', 'train', 'resume'):
        parser.add_argument('--'+phase, action='store_true')
    args = parser.parse_args()
    if not any((args.prepare,args.pilot,args.neutral,args.train)):
        parser.error('Explicit phase required')
    for path in (PRIVATE,PUBLIC):
        if any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError('Symlinked experiment destination')
        path.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        cfg, pc, data, designs, controls, identity = load()
        beat('assets_verified', rows=len(data['sites']), models=len(designs), workers=0,
             device='cpu', torch=torch.__version__, architecture=platform.machine())
        if args.pilot:
            pilot(cfg,pc,data,designs,identity)
        if args.neutral:
            neutral(cfg,pc,data,designs,controls,identity,args.resume)
        if args.train:
            train(cfg,pc,data,designs,identity,args.resume)
        beat('phase_complete')


if __name__ == '__main__':
    main()
