"""Recompute frozen utility/risk on fitting sources only; never select a new policy."""
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
    raise RuntimeError('Native arm64 required before Torch import')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import train_m3w_fixed_occurrence as source
from src.world_model.m3w_fitting_switch_diagnostic import diagnose

old, base = source.old, source.old.base
NAME = 'european_fitting_switch_diagnostic_v1'
PUBLIC = source.PUBLIC.parent/NAME; PRIVATE = source.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
digest, immutable = source.digest, source.immutable


def registration():
    cfg = json.loads(CONFIG.read_text()); sp = source.PUBLIC/'verification.json'
    assert digest(sp) == cfg['parent_verification_sha256']
    seal = json.loads(sp.read_text())
    for p, h in seal['source_bindings'].items(): assert digest(ROOT/p) == h, p
    for p, h in seal['artifacts'].items(): assert digest(source.PUBLIC/p) == h, p
    assert cfg['groups'] == 108 and cfg['risk_budget'] == .02 and cfg['arms'] == ['raw', *source.api.ARMS]
    assert not any(cfg[k] for k in ('new_parameter_updates', 'held_outcomes_used', 'independent_roles_read',
        'threshold_search', 'deployment_changed', 'formal_primary_replaced', 'stage5c_executed', 'smc_enabled'))
    files = source.closure(ROOT, ['scripts.diagnose_m3w_fitting_switches'])
    files += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_fitting_switch_diagnostic.py']
    return cfg, dict(parent_seal=base.artifact(sp),
        bindings={str(p.relative_to(ROOT)): digest(p) for p in files},
        groups=108, arms=cfg['arms'], role='fitting_only_descriptive_diagnostic',
        parameter_updates=0, independent_roles_read=False, held_outcomes_used=False)


def beat(state, **kw):
    row = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    base.inter.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def checked(ref):
    assert base.artifact(ROOT/ref['path']) == ref
    return json.loads((ROOT/ref['path']).read_text())


def scores(c, data, fit, ids, u, pr, parent_fit, trained):
    # No outcome/target arrays are accepted by this prediction path.
    if set(data) != set(old.parent.CAUSAL_KEYS): raise ValueError('Causal whitelist required')
    raw_ref = parent_fit['artifacts']['subset_aggregate']
    assert base.artifact(ROOT/raw_ref['path']) == raw_ref
    raw = old.api.head.read_checkpoint(ROOT/raw_ref['path'])
    model = old.parent.previous.descriptor.model_from(raw)
    p = old.api.head.predict(model, c['x'][fit], u, c['env'][fit], pr, raw['descriptor_preprocess'])
    risks = dict(raw=old.parent.api.head.parent.signed(p.astype(float))/pr['cost_scale'])
    states = {}
    for arm, ref in trained['artifacts'].items():
        assert base.artifact(ROOT/ref['path']) == ref
        state = old.api.head.read_checkpoint(ROOT/ref['path']); states[arm] = state
        assert state['step'] == 2000 and state['identity'] == trained['identity']
        z, env = old.api.inputs(c['x'][fit], u, c['env'][fit], state['warm_start']['norm'])
        pred = source.api.predict(state, z.numpy(), env.numpy())
        if arm == 'fixed':
            teacher = old.api.predict(state['warm_start'], c['x'][fit], u, c['env'][fit])
            np.testing.assert_array_equal(pred[:, 0], teacher[:, 0])
        risks[arm] = np.column_stack((risks['raw'][:, 0], pred[:, 3].astype(float)))
    source.api.assert_matched(states['trainable'], states['fixed'])
    return risks, [raw_ref, *trained['artifacts'].values()]


def construct(c, data, pair, fit_ref, pid, trained_ref):
    name = c['name']+f'_pair{pair}'
    fit, ids, u, y, pr, fid = old.fitting(c, data, pair, fit_ref, pid)
    trained = checked(trained_ref); parent_fit = checked(fit_ref)
    assert trained['identity']['source_identity'] == fid
    assert set(data['sites'][ids]) == set(fid['roles']['training_sites'])
    assert not set(data['sites'][ids]) & set(fid['roles']['held_sites'])
    gr = json.loads((source.labels.PRIVATE/'labels'/(name+'.json')).read_text())
    assert gr['source_identity'] == fid and base.artifact(ROOT/gr['labels']['path']) == gr['labels']
    with np.load(ROOT/gr['labels']['path'], allow_pickle=False) as z: g = {k: z[k].copy() for k in z.files}
    np.testing.assert_array_equal(g['ids'], ids); np.testing.assert_array_equal(g['known'], pr['known'])
    np.testing.assert_array_equal(g['easy'], y[:, 0]); np.testing.assert_array_equal(g['reference'], y[:, 1])
    risks, refs = scores(c, {k: data[k] for k in old.parent.CAUSAL_KEYS}, fit, ids, u, pr, parent_fit, trained)
    ridge_doc = json.loads((base.floor_api.PRIVATE/'fits'/name/'complete.json').read_text())
    ridge_ref = ridge_doc['artifacts']['checkpoint']; assert base.artifact(ROOT/ridge_ref['path']) == ridge_ref
    ridge = base.torch.load(ROOT/ridge_ref['path'], map_location='cpu', weights_only=False)['model']
    assert set(ridge['training_sites']) == set(fid['roles']['training_sites'])
    # Match the original float32 score export before converting to normalized utility.
    a = base.floor_api.score(ridge, c, fit); np.testing.assert_array_equal(a['ids'], ids)
    utility = (a['scores'][:, 5].astype(float)-a['scores'][:, 6].astype(float))/pr['cost_scale']
    features = dict(ids=ids, sites=data['sites'][ids], recordings=data['recordings'][ids], frames=data['frames'][ids],
        utility=utility, moving=c['moving'][fit], supported=a['support'], risks=risks)
    result = diagnose(**features, known=g['known'], easy=g['easy'], reference=g['reference'],
                      benefit=g['positive_benefit'], harm=g['positive_harm'])
    return dict(group=name, source_identity=fid, input_source='cached_verified', result_source='fresh_run_fitting_only',
        upstream_refs=[fit_ref, trained_ref, gr['labels'], ridge_ref, *refs],
        causal_score_hashes={k: base.inter.array_hash(v) for k, v in dict(utility=utility, support=a['support'], **risks).items()},
        result=result, held_outcomes_used=False, independent_roles_read=False, policy_changed=False)


def run(cfg, reg, *, pilot=False, replay=False, resume=False):
    base.torch.set_num_threads(cfg['cpu_threads']); base.torch.set_num_interop_threads(cfg['interop_threads'])
    if not pilot and not replay:
        p = json.loads((PUBLIC/'pilot.json').read_text()); assert p['registration_sha256'] == digest(PUBLIC/'registration.json')
        assert p['local_feasible'] and p['storage_sufficient']
    receipt = PUBLIC/('replay.json' if replay else 'completion.json')
    if receipt.exists(): raise ValueError('Completed receipt exists; do not overwrite')
    start = time.monotonic(); beat('loading_verified_fitting_assets', pilot=pilot, replay=replay)
    _, _, data, jobs, oid, pid, fits, _ = old.load()
    trained = {Path(r['path']).parent.name: r for r in json.loads((source.PUBLIC/'training_freeze.json').read_text())['groups']}
    records = []
    for c in base.floor_api.contexts({k: data[k] for k in old.parent.CAUSAL_KEYS}, jobs, oid):
        for pair in range(6):
            if shutil.disk_usage(PRIVATE).free < cfg['disk_reserve_bytes']+32*2**20:
                raise OSError('Keep10GiB reserve plus32MiB; do not drop groups')
            name = c['name']+f'_pair{pair}'; path = PRIVATE/'groups'/(name+'.json')
            if path.exists() and resume and not replay:
                doc = json.loads(path.read_text()); assert doc['registration_sha256'] == digest(PUBLIC/'registration.json')
                for r in doc['upstream_refs']: assert base.artifact(ROOT/r['path']) == r
            else:
                if path.exists() and not replay: raise ValueError('Existing group needs resume or replay')
                doc = construct(c, data, pair, fits[name], pid, trained[name])
                doc['registration_sha256'] = digest(PUBLIC/'registration.json')
                if replay:
                    assert path.exists() and doc == json.loads(path.read_text()), name
                else: immutable(path, doc)
            records.append(base.artifact(path)); beat('group_replayed' if replay else 'group_complete', groups=len(records), group=name)
            if pilot:
                elapsed = time.monotonic()-start; free = shutil.disk_usage(PRIVATE).free
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if platform.system() != 'Darwin': rss *= 1024
                immutable(PUBLIC/'pilot.json', dict(registration_sha256=digest(PUBLIC/'registration.json'),
                    group=name, seconds=elapsed, peak_RSS_bytes=rss, group_bytes=path.stat().st_size,
                    projected_seconds=elapsed*108, projection_not_full_runtime=True,
                    projected_bytes=path.stat().st_size*108*3+32*2**20, free_disk_bytes=free,
                    storage_sufficient=free-path.stat().st_size*108*3-32*2**20 > cfg['disk_reserve_bytes'],
                    local_feasible=elapsed*108 < 12*3600 and rss < 40*2**30,
                    runtime_machine=platform.machine(), torch=base.torch.__version__, cpu_threads=4, workers=0))
                return
    assert len(records) == 108
    immutable(receipt, dict(registration_sha256=digest(PUBLIC/'registration.json'), groups=records,
        exact=replay, seconds=time.monotonic()-start, parameter_updates=0,
        held_outcomes_used=False, independent_roles_read=False))
    beat('replay_complete' if replay else 'diagnosis_complete', groups=108, seconds=time.monotonic()-start)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=['register', 'pilot', 'run', 'replay'])
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True); cfg, reg = registration()
    if a.phase == 'register': immutable(PUBLIC/'registration.json', reg); print(json.dumps(dict(registered=True, files=len(reg['bindings'])))); return
    assert reg == json.loads((PUBLIC/'registration.json').read_text()); base.inter.committed(PUBLIC/'registration.json')
    with (PRIVATE/'run.lock').open('w') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(cfg, reg, pilot=a.phase == 'pilot', replay=a.phase == 'replay', resume=a.resume)


if __name__ == '__main__': main()
