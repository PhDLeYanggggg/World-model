"""Local-first paired warm-start training with fitting-only label accounting."""
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
    raise RuntimeError('Use native arm64 before importing Torch')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import build_m3w_fitting_gain_labels as labels
from scripts.export_m3w_easy_hurdle_create import closure
from src.world_model import m3w_fixed_occurrence as api
from src.world_model.m3w_easy_component_diagnostic import diagnose, weights

old = labels.old
NAME = 'european_fixed_occurrence_v1'
PUBLIC = labels.PUBLIC.parent/NAME; PRIVATE = labels.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
digest, immutable = labels.digest, labels.immutable


def registration():
    cfg = json.loads(CONFIG.read_text())
    for name, key in [('european_easy_component_diagnostic_v1', 'diagnostic_seal_sha256'),
                      ('european_fitting_gain_labels_v1', 'gain_label_seal_sha256')]:
        home = PUBLIC.parent/name; path = home/'verification.json'
        assert digest(path) == cfg[key]; seal = json.loads(path.read_text())
        for p, h in seal['source_bindings'].items(): assert digest(ROOT/p) == h, p
        for p, h in seal['artifacts'].items(): assert digest(home/p) == h, p
    assert cfg['groups'] == 108 and cfg['heads'] == 216 and cfg['arms'] == list(api.ARMS)
    assert cfg['risk_budget'] == .02 and cfg['disk_reserve_bytes'] == 10*2**30
    assert not any(cfg[k] for k in ('independent_roles_read', 'threshold_search', 'deployment_changed',
                                  'formal_primary_replaced', 'stage5c_executed', 'smc_enabled'))
    files = closure(ROOT, ['scripts.train_m3w_fixed_occurrence'])
    files += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_fixed_occurrence.py']
    return cfg, dict(bindings={str(p.relative_to(ROOT)): digest(p) for p in files},
        groups=108, heads=216, arms=list(api.ARMS), parameter_updates=432000,
        primary_development_contrast=['fixed_matched', 'trainable_matched'],
        source_roles_unchanged=True, independent_roles_read=False, deployment_changed=False)


def beat(state, **kw):
    row = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    old.base.inter.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def guard(extra=0):
    if shutil.disk_usage(PRIVATE).free < 10*2**30+extra:
        raise OSError('Keep10GiB reserve and all completed checkpoints')


def fitting_screen(pred, gains, sites, recordings, frames):
    known = gains['known']; w, _ = weights(sites, recordings, frames, known)
    at = np.flatnonzero(known); take = pred[at, 3] <= 0; ww = w[at]
    b, h, g = (gains[k][at] for k in ('positive_benefit', 'positive_harm', 'signed_gain'))
    total = float(ww@b)
    return dict(screen='fitting_easy_risk_sign_only_not_deployment', admitted_mass=float(ww@take),
        retained_benefit=float(ww@(b*take)), missed_benefit=float(ww@(b*~take)),
        retained_harm=float(ww@(h*take)), net_gain=float(ww@(g*take)),
        benefit_retention=float(ww@(b*take))/total if total > 0 else None)


def run(cfg, reg, *, pilot=False, resume=False, replay=False):
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    if not pilot and not replay:
        p = json.loads((PUBLIC/'pilot.json').read_text())
        assert p['registration_sha256'] == digest(PUBLIC/'registration.json')
        assert p['storage_sufficient'] and p['local_runtime_feasible'], 'Use explicit CREATE placement if local guard fails'
    if (PUBLIC/('fit_replay.json' if replay else 'training_freeze.json')).exists():
        raise ValueError('Completed receipt already exists')
    guard(); start = time.monotonic(); beat('loading_frozen_source_assets', pilot=pilot, replay=replay)
    _, _, data, jobs, oid, pid, fits, _ = old.load()
    warm_refs = {r['group']: r for r in json.loads((labels.PARENT/'create_training_freeze.json').read_text())['receipt']['artifacts'] if r['arm'] == 'uncapped'}
    records = []; causal = {k: data[k] for k in old.parent.CAUSAL_KEYS}
    for c in old.base.floor_api.contexts(causal, jobs, oid):
        for pair in range(6):
            name = c['name']+f'_pair{pair}'; home = PRIVATE/'heads'/name; rec = home/'complete.json'
            guard()
            if rec.exists() and resume and not replay:
                doc = json.loads(rec.read_text()); assert doc['identity']['registration_sha256'] == digest(PUBLIC/'registration.json')
                for ref in doc['artifacts'].values(): assert old.base.artifact(ROOT/ref['path']) == ref
            else:
                if rec.exists() and not replay: raise ValueError('Existing fit requires resume or replay')
                fit, ids, u, y, pr, fid = old.fitting(c, data, pair, fits[name], pid)
                wr = warm_refs[name]; wp = old.PRIVATE.parent/'european_easy_risk_priority_v1'/wr['path']
                assert digest(wp) == wr['sha256']; warm = old.api.head.read_checkpoint(wp)
                old.api.sampling.exact(warm['identity']['source_identity'], fid)
                z, env = old.api.inputs(c['x'][fit], u, c['env'][fit], warm['norm'])
                z, env = z.numpy(), env.numpy()
                gr = json.loads((labels.PRIVATE/'labels'/(name+'.json')).read_text())
                assert gr['source_identity'] == fid and old.base.artifact(ROOT/gr['labels']['path']) == gr['labels']
                with np.load(ROOT/gr['labels']['path'], allow_pickle=False) as f: gains = {k: f[k].copy() for k in f.files}
                np.testing.assert_array_equal(gains['ids'], ids); assert np.array_equal(gains['known'], pr['known'])
                identity = dict(registration_sha256=digest(PUBLIC/'registration.json'), source_identity=fid,
                                warm_start_sha256=wr['sha256'], gain_label_ref=gr['labels'])
                states = {}; before = time.monotonic()
                for arm in api.ARMS:
                    path = (PRIVATE/'fit_replay'/name if replay else home)/arm/'checkpoint.pt.gz'
                    states[arm] = api.fit(z, env, y, data['sites'][ids], data['recordings'][ids], data['frames'][ids],
                        warm, arm=arm, settings=cfg['head_training'], identity=identity, path=path,
                        heartbeat=lambda **kw: beat(group=name, arm=arm, **kw),
                        resume=resume and not replay, stop_at=cfg['pilot_updates'] if pilot else None)
                    if replay:
                        existing = old.api.head.read_checkpoint(home/arm/'checkpoint.pt.gz')
                        for key in states[arm]:
                            if key != 'seconds': old.api.sampling.exact(existing[key], states[arm][key])
                elapsed_fit = time.monotonic()-before; api.assert_matched(states['trainable'], states['fixed'])
                if pilot:
                    size = sum((home/a/'checkpoint.pt.gz').stat().st_size for a in api.ARMS)
                    estimated = size*108*3+128*2**20; free = shutil.disk_usage(PRIVATE).free
                    seconds = time.monotonic()-start
                    projected = (elapsed_fit*2000/cfg['pilot_updates']+max(0, seconds-elapsed_fit))*108
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    if platform.system() != 'Darwin': rss *= 1024
                    immutable(PUBLIC/'pilot.json', dict(registration_sha256=digest(PUBLIC/'registration.json'),
                        group=name, updates_per_head=cfg['pilot_updates'], seconds=seconds, fit_seconds=elapsed_fit,
                        paired_checkpoint_bytes=size, projected_bytes_3x_plus128MiB=estimated,
                        storage_sufficient=free-estimated > 10*2**30, free_disk_bytes=free, peak_RSS_bytes=rss,
                        projected_full_seconds=projected, projection_not_full_runtime=True,
                        local_runtime_feasible=projected < 12*3600 and rss < 40*2**30,
                        runtime_machine=platform.machine(), torch=api.torch.__version__, workers=0, cpu_threads=4,
                        frozen_probability_exact=states['fixed']['occurrence_frozen_exact']))
                    beat('pilot_complete', fit_seconds=elapsed_fit, projected_full_seconds=projected); return
                if replay:
                    immutable(PUBLIC/'fit_replay.json', dict(group=name, heads=2, updates=4000,
                        registration_sha256=digest(PUBLIC/'registration.json'), exact_except_elapsed=True,
                        all108_pairs_retrained=False, seconds=time.monotonic()-start))
                    beat('first_pair_replay_complete'); return
                predictions = {arm: api.predict(s, z, env) for arm, s in states.items()}
                baseline = old.api.predict(warm, c['x'][fit], u, c['env'][fit])
                np.testing.assert_array_equal(predictions['fixed'][:, 0], baseline[:, 0])
                accounting = diagnose(predictions['trainable'][:, :3], predictions['fixed'][:, :3], y,
                                      data['sites'][ids], data['recordings'][ids], data['frames'][ids])
                doc = dict(identity=identity, artifacts={a: old.base.artifact(home/a/'checkpoint.pt.gz') for a in api.ARMS},
                    fitting_accounting_arm_map={'uncapped': 'trainable', 'risk_priority': 'fixed'},
                    fitting_accounting=accounting,
                    fitting_sign_screen={a: fitting_screen(p, gains, data['sites'][ids], data['recordings'][ids], data['frames'][ids]) for a, p in predictions.items()},
                    frozen_occurrence_predictions_exact=True, matched_initializations_and_queries=True,
                    held_outcomes_used=False, independent_roles_read=False)
                immutable(rec, doc)
            records.append(old.base.artifact(rec)); beat('pair_complete', group=name, groups=len(records))
    assert len(records) == 108
    immutable(PUBLIC/'training_freeze.json', dict(registration_sha256=digest(PUBLIC/'registration.json'), groups=records,
        heads=216, parameter_updates=432000, seconds=time.monotonic()-start, pilot_resumed=True,
        held_outcomes_used=False, independent_roles_read=False, deployment_changed=False))
    beat('training_complete', groups=108, seconds=time.monotonic()-start)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=['register', 'pilot', 'train', 'replay'])
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True); cfg, reg = registration()
    if a.phase == 'register': immutable(PUBLIC/'registration.json', reg); print(json.dumps(dict(registered=True, files=len(reg['bindings'])))); return
    assert reg == json.loads((PUBLIC/'registration.json').read_text()); old.base.inter.committed(PUBLIC/'registration.json')
    with (PRIVATE/'train.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(cfg, reg, pilot=a.phase == 'pilot', resume=a.resume, replay=a.phase == 'replay')


if __name__ == '__main__': main()
