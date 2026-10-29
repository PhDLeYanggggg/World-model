"""Matched causal output-radius experiment; all preceding results remain frozen."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before importing Torch')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from scripts import run_m3w_dimensionless_harm_forecast as parent
from src.world_model.m3w_reference_coverage import ReferenceCompleteAgentTrackForecaster, reference_motion_budget
from src.world_model.m3w_reference_harm_forecast import fit_trial, assert_training_equivalent
from src.world_model.m3w_native_forecast import pack_geometry
from src.world_model.m3w_baseline_relative_forecaster import past_motion_budget
from src.world_model.m3w_european_source_forecast import baseline_torch

NAME = 'european_reference_coverage_v1'
CONFIG = 'configs/m3w_reference_coverage_v1.json'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
FILES = (CONFIG, 'scripts/run_m3w_reference_coverage.py', 'src/world_model/m3w_reference_coverage.py',
         'tests/test_m3w_reference_coverage.py', str((PUBLIC/'protocol.md').relative_to(ROOT)))
digest, immutable_json, artifact = parent.digest, parent.immutable_json, parent.parent.artifact
json_write, saved = parent.json_write, parent.saved


def beat(state, **kw):
    event = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    json_write(PRIVATE/'heartbeat.json', event)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(event)+'\n')
    print(json.dumps(event), flush=True)


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    if (cfg['models'] != 9 or cfg['steps'] != 4000 or cfg['reference_margin'] != .05
            or cfg['objective'] != dict(selected_weight=2., cv_weight=2.)
            or cfg['seeds'] != [17, 29, 43] or cfg['risk_budget'] != .02):
        raise ValueError('Registered single-factor comparison changed')
    if shutil.disk_usage(ROOT).free < cfg['minimum_free_disk_GiB']*2**30:
        raise OSError('Preserve10GiB and resume existing checkpoints')
    _, pc, data, jobs, pidentity = parent.load()
    controls = {}
    for job in jobs:
        old = parent.complete(job, pidentity)
        path = parent.PRIVATE/'predictions'/(job['key']+'.npz')
        receipt = json.loads(path.with_suffix('.json').read_text())
        if receipt['checkpoint'] != old['checkpoint'] or receipt['sha256'] != digest(path):
            raise ValueError('Frozen harm-aware prediction changed')
        controls[job['key']] = dict(checkpoint=old['checkpoint'], prediction=artifact(path),
            complete=artifact(parent.PRIVATE/'trials'/job['key']/'complete.json'),
            ADE_only_prediction=pidentity['controls'][job['key']]['prediction'])
    verification = json.loads((parent.PUBLIC/'verification.json').read_text())
    if (not verification['exact_analysis_replay']
            or verification['analysis_sha256'] != digest(parent.PUBLIC/'analysis.json')):
        raise ValueError('Verified preceding readout required')
    identity = dict(parent_identity=pidentity, controls=controls,
        parent_freeze=artifact(parent.PUBLIC/'training_freeze.json'),
        parent_registration=artifact(parent.PUBLIC/'registration.json'),
        parent_analysis=artifact(parent.PUBLIC/'analysis.json'),
        bindings={p:digest(ROOT/p) for p in FILES}, source_role='previously_opened_source_development',
        objective=cfg['objective'], reference_margin=cfg['reference_margin'], runtime=pc['runtime'])
    immutable_json(PRIVATE/'identity.json', identity)
    immutable_json(PUBLIC/'registration.json', dict(identity=identity,
        jobs=[dict(key=j['key'], trial=j['old_identity']) for j in jobs],
        primary_updates_planned=36000, neutral_updates_planned=4000, engineering_updates_planned=80,
        result_source='cached_verified_assets_no_new_training_at_registration',
        independent_roles_opened=False, deployment_changed=False))
    return cfg, pc, data, jobs, identity


def assert_identity(identity):
    for path, sha in identity['bindings'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen comparison code changed: '+path)
    for key in ('parent_freeze', 'parent_registration', 'parent_analysis'):
        ref = identity[key]
        if artifact(ROOT/ref['path']) != ref:
            raise ValueError('Frozen comparator changed')


def preflight(cfg, data, jobs, identity):
    masks, checks = {}, {}
    for index in sorted({job['design']['baseline_index'] for job in jobs}):
        changed, counts = [], dict(original_CV_outside=0, new_CV_outside=0,
                                   original_D97_outside=0, new_D97_outside=0)
        for start in range(0, len(data['sites']), 4096):
            inputs = pack_geometry(data['geometry'][start:start+4096])
            inputs['baseline'] = baseline_torch(inputs['history'], index)
            original = past_motion_budget(inputs)
            new = reference_motion_budget(inputs, margin=cfg['reference_margin'])
            if (new < original).any():
                raise ValueError('Reference coverage cannot reduce original support')
            changed.append((new > original).any(1).numpy())
            for name, ref_index in (('CV', 1), ('D97', 3)):
                distance = (baseline_torch(inputs['history'], ref_index)-inputs['baseline']).norm(dim=-1)
                tol = 1e-6*torch.maximum(torch.ones_like(new), new)
                counts['original_'+name+'_outside'] += int((distance > original+tol).any(1).sum())
                counts['new_'+name+'_outside'] += int((distance > new+tol).any(1).sum())
                if ((distance > 0) & (distance >= new)).any():
                    raise ValueError('Finite nonzero references must lie inside the new bound')
        masks[index] = np.concatenate(changed)
        checks[str(index)] = dict(rows=len(data['sites']), radius_changed_rows=int(masks[index].sum()),
            changed_mask_sha256=hashlib.sha256(masks[index].tobytes()).hexdigest(), **counts)
    views = {j['key']:dict(baseline_index=j['design']['baseline_index'],
                train_radius_changed=int(masks[j['design']['baseline_index']][j['design']['train_ids']].sum()),
                excluded_radius_changed=int(masks[j['design']['baseline_index']][j['design']['held_ids']].sum()))
             for j in jobs}
    assert_identity(identity)
    result = dict(result_source='fresh_run_past_only_geometry_preflight', by_baseline=checks, by_trial=views,
        labels_used_for_coverage=False, independent_roles_opened=False, safety_guarantee=False)
    immutable_json(PUBLIC/'coverage_preflight.json', result)
    beat('coverage_preflight_complete', models=len(jobs), baseline_checks=checks)
    return result


def make_model(job, enabled=True):
    torch.manual_seed(job['old_identity']['seed'])
    return ReferenceCompleteAgentTrackForecaster(job['design']['baseline_index'], width=64, heads=4, layers=2,
                                                 enabled=enabled, margin=.05)


def trial_identity(job, identity, enabled):
    return dict(experiment=identity, parent_trial=job['old_identity'], coverage_enabled=enabled)


def fit(job, pc, data, identity, home, *, enabled=True, control_model=False, resume=False, stop_at=None):
    model = parent.parent.make_model(job) if control_model else make_model(job, enabled)
    ti = trial_identity(job, identity, enabled)
    result = fit_trial(model, data, job['design'], seed=job['old_identity']['seed'],
        settings=pc['training'], identity=ti, directory=home, resume=resume, stop_at=stop_at,
        heartbeat=lambda **kw:beat(trial=job['key'], **kw), **identity['objective'])
    assert_identity(identity)
    return dict(identity=ti, fit=result, checkpoint=artifact(home/'checkpoint.pt'))


def checked(path):
    doc = json.loads(path.read_text())
    for ref in doc['checkpoints']:
        if artifact(ROOT/ref['path']) != ref:
            raise ValueError('Changed pilot/neutral state')
    return doc


def pilot(cfg, pc, data, jobs, identity):
    path = PUBLIC/'pilot.json'
    if path.exists():
        return checked(path)
    start = time.monotonic()
    job = next(j for j in jobs if j['key'] == cfg['pilot_control'])
    out = {}
    for arm in ('direct', 'resumed', 'neutral', 'control'):
        home = PRIVATE/'pilot'/arm
        enabled = arm in ('direct', 'resumed')
        if arm == 'resumed':
            fit(job, pc, data, identity, home, enabled=True, stop_at=cfg['pilot_resume_at'])
        out[arm] = fit(job, pc, data, identity, home, enabled=enabled, control_model=arm == 'control',
            resume=arm == 'resumed', stop_at=cfg['pilot_steps'])
    assert_training_equivalent(saved(out['direct']['checkpoint']), saved(out['resumed']['checkpoint']))
    assert_training_equivalent(saved(out['neutral']['checkpoint']), saved(out['control']['checkpoint']))
    immutable_json(path, dict(result_source='fresh_native_torch_run', pid=os.getpid(), updates=80,
        exact_resume=True, exact_coverage_off_control=True, trial=job['key'],
        seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        checkpoints=[row['checkpoint'] for row in out.values()]))
    beat('pilot_complete', updates=80)


def neutral(cfg, pc, data, jobs, identity, resume):
    path = PUBLIC/'neutral_replay.json'
    if path.exists():
        return checked(path)
    job = next(j for j in jobs if j['key'] == cfg['full_neutral_control'])
    result = fit(job, pc, data, identity, PRIVATE/'neutral'/job['key'], enabled=False, resume=resume)
    control = identity['controls'][job['key']]['checkpoint']
    assert_training_equivalent(saved(result['checkpoint']), saved(control))
    immutable_json(path, dict(result_source='fresh_run_against_cached_verified_control', pid=os.getpid(),
        exact_full_neutral=True, trial=job['key'], fit=result['fit'], checkpoints=[result['checkpoint'], control]))
    beat('full_neutral_exact', updates=4000, trial=job['key'])


def complete(job, identity):
    path = PRIVATE/'trials'/job['key']/'complete.json'
    doc = json.loads(path.read_text())
    if (doc['identity'] != trial_identity(job, identity, True) or not doc['fit']['complete']
            or doc['fit']['step'] != 4000 or doc['fit']['held_rows_sampled'] != 0
            or artifact(ROOT/doc['checkpoint']['path']) != doc['checkpoint']):
        raise ValueError('Changed or incomplete primary endpoint')
    return doc


def train(cfg, pc, data, jobs, identity, resume):
    checked(PUBLIC/'pilot.json'); checked(PUBLIC/'neutral_replay.json')
    coverage = json.loads((PUBLIC/'coverage_preflight.json').read_text())
    start = time.monotonic(); finals = {}; new = 0
    for job in jobs:
        if shutil.disk_usage(ROOT).free < cfg['minimum_free_disk_GiB']*2**30:
            raise OSError('Preserve10GiB; resume remaining endpoints')
        home = PRIVATE/'trials'/job['key']
        if not (home/'complete.json').exists():
            beat('primary_fit_started', trial=job['key'])
            result = fit(job, pc, data, identity, home, resume=resume)
            current, control = saved(result['checkpoint']), saved(identity['controls'][job['key']]['checkpoint'])
            for key in ('draws', 'train_ids', 'factors'):
                np.testing.assert_array_equal(current[key], control[key])
            for key in ('sampler_rng', 'torch_rng'):
                torch.testing.assert_close(current[key], control[key], rtol=0, atol=0)
            view = coverage['by_trial'][job['key']]
            unchanged = view['train_radius_changed'] == view['excluded_radius_changed'] == 0
            if unchanged:
                assert_training_equivalent(current, control)
            result.update(matched_draws_rng_exact=True, radius_unchanged=unchanged,
                          unchanged_radius_full_state_exact=unchanged)
            immutable_json(home/'complete.json', result)
            new += result['fit']['new_updates']
        finals[job['key']] = complete(job, identity)
        beat('primary_fit_complete', trial=job['key'])
    immutable_json(PUBLIC/'training_freeze.json', dict(result_source='fresh_run', identity=identity,
        fits=finals, models=9, primary_updates=36000, readout_not_yet_run=True,
        summed_fit_seconds=sum(v['fit']['seconds'] for v in finals.values()),
        independent_roles_opened=False, deployment_changed=False))
    json_write(PRIVATE/'training_runtime.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        new_updates_this_invocation=new, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat('training_complete', models=9, updates=36000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('prepare', 'pilot', 'neutral', 'train', 'resume'):
        parser.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    if not any((args.prepare, args.pilot, args.neutral, args.train)):
        parser.error('Explicit experiment phase required')
    for path in (PUBLIC, PRIVATE):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Symlinked experiment destination')
        path.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        cfg, pc, data, jobs, identity = load()
        if args.prepare:
            preflight(cfg, data, jobs, identity)
        if args.pilot:
            pilot(cfg, pc, data, jobs, identity)
        if args.neutral:
            neutral(cfg, pc, data, jobs, identity, args.resume)
        if args.train:
            train(cfg, pc, data, jobs, identity, args.resume)
        beat('phase_complete')


if __name__ == '__main__':
    main()
