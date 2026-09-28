"""Build immutable label-only signed gains on existing source-excluded fitting rows."""
import argparse
import fcntl
import io
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 runtime required before Torch import')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_easy_hurdle as old
from scripts.export_m3w_easy_hurdle_create import closure
from src.world_model.m3w_fitting_gain_labels import gain_labels, verify_alignment

NAME = 'european_fitting_gain_labels_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
PARENT = PUBLIC.parent/'european_easy_risk_priority_v1'
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
digest, immutable = old.base.digest, old.base.immutable_json


def registration():
    cfg = json.loads(CONFIG.read_text()); seal = json.loads((PARENT/'verification.json').read_text())
    assert digest(PARENT/'verification.json') == cfg['parent_seal_sha256']
    for p, h in seal['source_bindings'].items(): assert digest(ROOT/p) == h, p
    for p, h in seal['artifacts'].items(): assert digest(PARENT/p) == h, p
    assert cfg['groups'] == 108 and cfg['disk_reserve_bytes'] == 10*2**30
    assert not any(cfg[k] for k in ('new_parameter_updates', 'held_outcomes_used', 'independent_roles_read',
                                  'threshold_search', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    files = closure(ROOT, ['scripts.build_m3w_fitting_gain_labels'])
    files += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_fitting_gain_labels.py']
    return cfg, dict(parent_seal_sha256=cfg['parent_seal_sha256'],
                     bindings={str(p.relative_to(ROOT)): digest(p) for p in files}, groups=108,
                     role='supervision_labels_only_not_inference_features', parameter_updates=0)


def beat(state, **kw):
    row = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    old.base.inter.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def guard(cfg, extra=0):
    free = shutil.disk_usage(PRIVATE).free
    if free < cfg['disk_reserve_bytes']+extra:
        raise OSError('Preserve10GiB and completed sidecars; no group reduction or deletion')
    return free


def construct(c, data, pair, ref, pid, trained):
    fit, ids, u, y, pr, identity = old.fitting(c, data, pair, ref, pid)
    name = c['name']+f'_pair{pair}'
    checkpoint_ref = trained[name]
    checkpoint = old.PRIVATE.parent/'european_easy_risk_priority_v1'/checkpoint_ref['path']
    assert digest(checkpoint) == checkpoint_ref['sha256']
    state = old.api.head.read_checkpoint(checkpoint)
    old.api.sampling.exact(state['identity']['source_identity'], identity)
    cv, _, (floor, _), (neural, _) = old.base.floor_api.costs(c, data, fit)
    labels = gain_labels(cv, floor, neural, easy_cut=c['job']['design']['easy_cut'], scale=pr['cost_scale'])
    verify_alignment(ids, data['sites'][ids], labels, y, identity['roles'])
    assert old.base.inter.array_hash(ids) == identity['fitting_ids_hash']
    assert old.base.inter.array_hash(y) == identity['input_hashes']['targets']
    known = labels['known']; g = labels['signed_gain']
    stats = dict(rows=len(ids), known=int(known.sum()), unknown=int((~known).sum()),
        beneficial=int((g > 0).sum()), harmful=int((g < 0).sum()), tied=int((g == 0).sum()),
        easy=int((labels['easy'] == 1).sum()))
    assert stats['beneficial']+stats['harmful']+stats['tied'] == stats['known']
    meta = dict(group=name, source_identity=identity, checkpoint_sha256=checkpoint_ref['sha256'],
        cost_scale=float(pr['cost_scale']), fitting_sources=list(pr['training_sites']), stats=stats,
        role='supervision_labels_only_not_inference_features', held_outcomes_used=False,
        independent_roles_read=False, policy_actions_computed=False)
    return dict(ids=ids, **labels), meta


def write_or_compare(path, arrays, *, replay):
    if path.exists():
        if not replay: raise ValueError('Existing target sidecar requires resume or replay')
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == set(arrays)
            for k, v in arrays.items(): np.testing.assert_array_equal(z[k], v)
        return
    if replay: raise FileNotFoundError(path)
    buf = io.BytesIO(); np.savez_compressed(buf, **arrays)
    temp = path.with_suffix('.partial')
    with temp.open('xb') as f: f.write(buf.getvalue())
    temp.replace(path)


def build(cfg, reg, *, pilot, resume, replay):
    old.base.torch.set_num_threads(cfg['cpu_threads']); old.base.torch.set_num_interop_threads(cfg['interop_threads'])
    if replay:
        previous = json.loads((PUBLIC/'completion.json').read_text())
        assert previous['registration_sha256'] == digest(PUBLIC/'registration.json') and previous['groups'] == 108
    elif not pilot:
        p = json.loads((PUBLIC/'pilot.json').read_text())
        assert p['registration_sha256'] == digest(PUBLIC/'registration.json') and p['storage_sufficient']
    if (PUBLIC/('replay.json' if replay else 'completion.json')).exists():
        raise ValueError('Complete receipt exists; do not overwrite')
    guard(cfg); started = time.monotonic(); beat('loading_registered_parent', replay=replay)
    _, _, data, jobs, oid, pid, fits, _ = old.load()
    all_refs = json.loads((PARENT/'create_training_freeze.json').read_text())['receipt']['artifacts']
    trained = {r['group']: r for r in all_refs if r['arm'] == 'uncapped'}
    assert len(trained) == 108
    causal = {k: data[k] for k in old.parent.CAUSAL_KEYS}
    records = []; totals = dict(rows=0, known=0, unknown=0, beneficial=0, harmful=0, tied=0, easy=0)
    for c in old.base.floor_api.contexts(causal, jobs, oid):
        for pair in range(6):
            name = c['name']+f'_pair{pair}'; dest = PRIVATE/'labels'/(name+'.npz'); record = dest.with_suffix('.json')
            guard(cfg); tick = time.monotonic()
            if record.exists() and resume and not replay:
                doc = json.loads(record.read_text())
                assert doc['registration_sha256'] == digest(PUBLIC/'registration.json')
                assert doc['source_identity']['parent_fit'] == fits[name]
                assert doc['checkpoint_sha256'] == trained[name]['sha256']
                assert old.base.artifact(dest) == doc['labels']
            else:
                if record.exists() and not replay: raise ValueError('Existing group requires resume or replay')
                arrays, meta = construct(c, data, pair, fits[name], pid, trained)
                # Bound peak on-disk write by the uncompressed arrays, not a favorable first group.
                guard(cfg, sum(v.nbytes for v in arrays.values())+2**20)
                write_or_compare(dest, arrays, replay=replay)
                doc = dict(registration_sha256=digest(PUBLIC/'registration.json'),
                           labels=old.base.artifact(dest), **meta)
                immutable(record, doc)
            records.append(old.base.artifact(record))
            for k in totals: totals[k] += doc['stats'][k]
            beat('labels_replayed' if replay else 'labels_verified', group=name, groups=len(records),
                 group_seconds=time.monotonic()-tick, **doc['stats'])
            if pilot:
                size = dest.stat().st_size+record.stat().st_size
                estimate = size*108*3; free = shutil.disk_usage(PRIVATE).free
                immutable(PUBLIC/'pilot.json', dict(group=name, registration_sha256=digest(PUBLIC/'registration.json'),
                    seconds=time.monotonic()-started, bytes=size, projected_full_bytes_3x=estimate,
                    free_disk_bytes=free, storage_sufficient=free-estimate > cfg['disk_reserve_bytes'],
                    projected_full_seconds_upper_from_loading_inclusive_pilot=(time.monotonic()-started)*108,
                    projection_not_measured_full_run=True, stats=doc['stats'],
                    peak_RSS_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, runtime_platform=platform.system()))
                beat('pilot_complete'); return
    assert len(records) == 108
    out = dict(registration_sha256=digest(PUBLIC/'registration.json'), groups=108, records=records,
        counts_repeated_fitting_views=totals, seconds=time.monotonic()-started, pid=os.getpid(),
        label_bytes=sum((ROOT/json.loads((ROOT/r['path']).read_text())['labels']['path']).stat().st_size for r in records),
        replay_exact=replay, result_source='fresh_run_label_replay' if replay else 'fresh_run_label_completion',
        input_source='cached_verified_frozen_predictions_and_fitting_roles', held_outcomes_used=False,
        independent_roles_read=False, parameter_updates=0, policy_actions_computed=False,
        peak_RSS_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, runtime_platform=platform.system())
    immutable(PUBLIC/('replay.json' if replay else 'completion.json'), out)
    beat('complete', groups=108, replay=replay, seconds=out['seconds'])


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=['register', 'pilot', 'build', 'replay'])
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True); (PRIVATE/'labels').mkdir(exist_ok=True)
    cfg, reg = registration()
    if a.phase == 'register':
        immutable(PUBLIC/'registration.json', reg); print(json.dumps(dict(registered=True, bindings=len(reg['bindings'])))); return
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    old.base.inter.committed(PUBLIC/'registration.json')
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build(cfg, reg, pilot=a.phase == 'pilot', resume=a.resume, replay=a.phase == 'replay')


if __name__ == '__main__': main()
