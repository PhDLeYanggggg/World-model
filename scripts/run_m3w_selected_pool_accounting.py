"""Registered frozen selected-pool accounting with source OOF and transfer replay."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_component_calibration as parent
from src.world_model import m3w_selected_pool_accounting as api

NAME = 'european_selected_pool_v1'
PUBLIC = parent.PUBLIC.parent / NAME
PRIVATE = parent.PRIVATE.parent / NAME
CONFIG = ROOT / 'configs' / ('m3w_' + NAME + '.json')
base, core, inner, forest = parent.base, parent.core, parent.inner, parent.forest
digest, immutable = parent.digest, parent.immutable


def registration():
    cfg = json.loads(CONFIG.read_text()); parent.registration()
    seal = parent.PUBLIC / 'verification.json'
    assert digest(seal) == cfg['parent_seal_sha256']
    v = json.loads(seal.read_text())
    for k, h in v['source_bindings'].items(): assert digest(ROOT/k) == h
    for k, h in v['artifacts'].items(): assert digest(parent.PUBLIC/k) == h
    paths = forest.closure(ROOT, ['scripts.run_m3w_selected_pool_accounting'])
    paths += [CONFIG, PUBLIC/'protocol.md', PUBLIC/'asset_and_method_note.md',
              ROOT/'tests/test_m3w_selected_pool_accounting.py']
    return cfg, dict(parent_seal_sha256=digest(seal),
        bindings={str(p.relative_to(ROOT)): digest(p) for p in paths},
        independent_roles_read=False, new_parameter_updates=0, policy_changed=False)


def beat(**kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    base.inter.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def guard(cfg):
    if shutil.disk_usage(PRIVATE).free < cfg['disk_reserve_bytes'] + 32*2**20:
        raise OSError('Keep10GiB reserve; migrate bounded diagnostic rather than delete unrelated assets')


def analyze(y, p, q, env, raw, action, recordings, meta, calibration_support):
    stats = api.account(y, p, q, env, raw, action, audit=True)
    assert not (action & ~calibration_support).any()
    supported = api.account(y, p, q, env, raw & calibration_support, action)
    records = []
    for recording in np.unique(recordings):
        ix = recordings == recording
        s = api.account(y[ix], p[ix], q[ix], env[ix], raw[ix], action[ix])
        records.append(dict(recording=str(recording), raw_rows=s['raw']['rows'],
            calibration_supported=bool(calibration_support[ix].all()),
            kept_rows=s['kept']['rows'], unknown_kept=s['kept']['unknown'],
            raw_easy_risk=s['raw']['easy']['observed_risk'],
            kept_easy_risk=s['kept']['easy']['observed_risk'],
            easy_signed_bias_shift=s['easy_contrast']['signed_bias_shift'],
            new_known_violation=s['easy_contrast']['new_known_violation']))
    return dict(**meta, statistics=stats, recordings=records,
        unsupported_raw_rows=int((raw & ~calibration_support).sum()),
        supported_pool_bias={prefix:dict(raw=supported['raw'][prefix]['signed_bias'],
            kept=supported['kept'][prefix]['signed_bias'], shift=supported[prefix+'_contrast']['signed_bias_shift'])
            for prefix in ('all', 'easy')},
        action_hash=base.inter.array_hash(action), raw_action_hash=base.inter.array_hash(raw),
        target_hash=base.inter.array_hash(y), adjusted_prediction_hash=base.inter.array_hash(q))


def source(data, jobs, oid, cfg, *, pilot=False, resume=False, replay=False):
    fitted, original = parent.docs(), parent.parent.docs()
    refs = []; started = time.monotonic()
    for c in parent.parent.contexts(data, jobs, oid):
        for site in inner.sources(c):
            group = c['name']+'_fit_'+site
            at, ids, x, env, y, _, upstream = inner.training_arrays(c, data, site)
            _, val, partition = forest.parent.api.source_partition(data['recordings'][ids], data['frames'][ids], site)
            ids, x, env, y = ids[val], x[val], env[val], y[val]
            moving, rec = c['moving'][at][val], data['recordings'][ids].astype(str)
            for seed in parent.parent.api.SEEDS:
                guard(cfg); doc = fitted[group, seed]; old = original[group, seed]
                assert doc['identity']['partition'] == partition == old['partition']
                identity = dict(registration_sha256=digest(PUBLIC/'registration.json'),
                    calibration=base.artifact(ROOT / next(r['path'] for r in
                        json.loads((parent.PUBLIC/'calibration_freeze.json').read_text())['groups']
                        if Path(r['path']).stem == group+'_head'+str(seed))),
                    checkpoint=old['checkpoint'], ids_hash=base.inter.array_hash(ids))
                path = PRIVATE/'source'/(group+'_head'+str(seed)+'.json')
                if path.exists() and resume and not replay:
                    value = json.loads(path.read_text()); assert value['identity'] == identity
                    refs.append(base.artifact(path)); continue
                state = joblib.load(ROOT/old['checkpoint']['path'])
                assert state['identity']['upstream'] == upstream
                p, support = forest.api.predict(state, x, env)
                raw = parent.api.eligible(p, moving, support)
                assert base.inter.array_hash(p) == doc['identity']['prediction_hash']
                assert base.inter.array_hash(y) == doc['identity']['target_hash']
                assert base.inter.array_hash(raw) == old['validation']['action_hashes']['forest']
                folds = {f['held_recording']: f['calibration'] for f in doc['folds']}
                assert set(folds) == set(rec)
                rows = []
                for mode in parent.api.MODES:
                    q = np.zeros_like(p)
                    calibration_support = np.zeros(len(ids), bool)
                    for name, cal in folds.items():
                        assert name not in cal['recordings']
                        ix = rec == name
                        q[ix] = parent.api.adjust(p[ix], env[ix], cal, mode)
                        calibration_support[ix] = cal['supported']
                    action = parent.api.eligible(q, moving, support)
                    assert base.inter.array_hash(action) == doc['oof_action_hashes'][mode]
                    meta = dict(view=group+'_head'+str(seed), group=group, site=site, source=site,
                                head_seed=seed, mode=mode)
                    rows.append(analyze(y, p, q, env, raw, action, rec,
                        dict(**meta, role='source_oof', source_screen=doc['source'][mode]['oof']['finite_completion_supported']), calibration_support))
                    q = parent.api.adjust(p, env, doc['final'], mode)
                    action = parent.api.eligible(q, moving, support)
                    rows.append(analyze(y, p, q, env, raw, action, rec,
                        dict(**meta, role='source_resubstitution', source_screen=doc['source'][mode]['full_fit_resubstitution']['finite_completion_supported']),
                        np.full(len(ids), doc['final']['supported'], bool)))
                immutable(path, dict(identity=identity, rows=rows, no_parameter_updates=True))
                refs.append(base.artifact(path)); beat(state='source_accounting', heads=len(refs), group=group, head_seed=seed)
                if pilot:
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
                    estimate = path.stat().st_size*72*4 + 16*2**20
                    immutable(PUBLIC/'pilot.json', dict(seconds=time.monotonic()-started,
                        peak_RSS_bytes=rss, one_group_bytes=path.stat().st_size,
                        estimated_storage_bytes=estimate, memory_feasible=rss<40*2**30,
                        storage_feasible=shutil.disk_usage(PRIVATE).free-estimate>cfg['disk_reserve_bytes'],
                        real_source_group=True))
                    return
    assert len(refs) == 72
    immutable(PUBLIC/'source_manifest.json', dict(groups=refs, source_heads=72))
    immutable(PUBLIC/('source_replay.json' if replay else 'source_runtime.json'),
        dict(seconds=time.monotonic()-started, all72_exact_replay=replay))


def transfer(data, jobs, oid, cfg, *, replay=False, resume=False):
    fitted = parent.docs()
    frozen = {r['view']: r for r in json.loads((parent.PUBLIC/'decision_freeze.json').read_text())['rows']}
    old_metrics = {(r['view'], r['policy']): r['metric'] for r in json.loads((parent.PUBLIC/'readout.json').read_text())['rows']}
    refs = []; started = time.monotonic(); risk_checks = 0
    for c, at, ids, predictions, _, pr, oldmeta in parent.parent.views(data, jobs, oid):
        support = forest.api.causal_inputs(c['x'][at], c['env'][at], pr)[1]
        env, moving, rec = c['env'][at], c['moving'][at], data['recordings'][ids].astype(str)
        for seed in parent.parent.api.SEEDS:
            guard(cfg); group = c['name']+'_fit_'+oldmeta['source']; view = oldmeta['view']+'_head'+str(seed)
            doc, fr = fitted[group, seed], frozen[view]
            identity = dict(registration_sha256=digest(PUBLIC/'registration.json'),
                            parent_frozen_action=fr, ids_hash=base.inter.array_hash(ids))
            path = PRIVATE/'transfer'/(view+'.json')
            if path.exists() and resume and not replay:
                value = json.loads(path.read_text()); assert value['identity'] == identity
                refs.append(base.artifact(path)); risk_checks += value['parent_risk_checks']; continue
            p = predictions[seed]; raw = parent.api.eligible(p, moving, support)
            assert base.inter.array_hash(raw) == fr['action_hashes']['raw']
            adjusted, actions = {}, {}
            for mode in parent.api.MODES:
                adjusted[mode] = parent.api.adjust(p, env, doc['final'], mode)
                actions[mode] = parent.api.eligible(adjusted[mode], moving, support)
                assert base.inter.array_hash(adjusted[mode]) == fr['adjusted_hashes'][mode]
                assert base.inter.array_hash(actions[mode]) == fr['action_hashes'][mode]
            # Outcomes are loaded only after all three causal action hashes agree.
            cv, _, (floor, _), (neural, _) = base.floor_api.costs(c, data, at)
            y = core.targets(cv, floor, neural, c['job']['design']['easy_cut'])
            rows = []; checks = 0
            for mode in parent.api.MODES:
                row = analyze(y, p, adjusted[mode], env, raw, actions[mode], rec,
                    dict(view=view, group=group, source=oldmeta['source'], site=oldmeta['site'], head_seed=seed,
                         mode=mode, role='transfer', source_screen=doc['source'][mode]['oof']['finite_completion_supported']),
                    np.full(len(ids), doc['final']['supported'], bool))
                for pool, name in (('raw', 'raw'), ('kept', mode)):
                    for prefix, field in (('all', 'selected_positive_harm_ratio'), ('easy', 'selected_easy_positive_harm_ratio')):
                        actual = row['statistics'][pool][prefix]['observed_risk']; expected = old_metrics[view, name][field]
                        assert (actual is None) == (expected is None)
                        if actual is not None: np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
                        checks += 1
                rows.append(row)
            immutable(path, dict(identity=identity, rows=rows, parent_risk_checks=checks))
            risk_checks += checks; refs.append(base.artifact(path))
            if len(refs) % 18 == 0: beat(state='transfer_accounting', heads=len(refs))
    assert len(refs) == 216 and risk_checks == 2592
    immutable(PUBLIC/'transfer_manifest.json', dict(groups=refs, head_views=216, parent_risk_checks=risk_checks))
    immutable(PUBLIC/('transfer_replay.json' if replay else 'transfer_runtime.json'),
        dict(seconds=time.monotonic()-started, all216_exact_replay=replay))


def read_rows():
    rows = []
    for name in ('source', 'transfer'):
        for ref in json.loads((PUBLIC/(name+'_manifest.json')).read_text())['groups']:
            assert base.artifact(ROOT/ref['path']) == ref
            rows.extend(json.loads((ROOT/ref['path']).read_text())['rows'])
    assert len(rows) == 1080
    return rows


def interval(pairs, cfg):
    defined = [(s, v) for s, v in pairs if v is not None and np.isfinite(v)]
    return dict(total_views=len(pairs), defined_views=len(defined), undefined_views=len(pairs)-len(defined),
        all_views=forest.locality_interval(pairs, cfg['bootstrap_resamples'], cfg['bootstrap_seed']),
        defined_only_descriptive=forest.locality_interval(defined, cfg['bootstrap_resamples'], cfg['bootstrap_seed']) if defined else None)


def report(cfg, reg):
    assert json.loads((PUBLIC/'source_replay.json').read_text())['all72_exact_replay']
    assert json.loads((PUBLIC/'transfer_replay.json').read_text())['all216_exact_replay']
    rows = read_rows(); summaries = {}; paired = {}
    source = {(r['group'], r['head_seed'], r['mode'], r['role']): r for r in rows if r['role'] != 'transfer'}
    for mode in parent.api.MODES:
        for role in ('source_oof', 'source_resubstitution', 'transfer'):
            rr = [r for r in rows if r['mode'] == mode and r['role'] == role]
            z = {}
            for prefix in ('all', 'easy'):
                z[prefix] = dict(
                    bias_shift=interval([(r['site'], r['statistics'][prefix+'_contrast']['signed_bias_shift']) for r in rr], cfg),
                    supported_pool_bias_shift=interval([(r['site'], r['supported_pool_bias'][prefix]['shift']) for r in rr], cfg),
                    raw_bias=interval([(r['site'], r['statistics']['raw'][prefix]['signed_bias']) for r in rr], cfg),
                    kept_bias=interval([(r['site'], r['statistics']['kept'][prefix]['signed_bias']) for r in rr], cfg),
                    risk_increased=sum(r['statistics'][prefix+'_contrast']['risk_delta'] is not None and r['statistics'][prefix+'_contrast']['risk_delta']>1e-12 for r in rr),
                    new_known_violations=sum(r['statistics'][prefix+'_contrast']['new_known_violation'] for r in rr),
                    kept_violations=sum(r['statistics']['kept'][prefix]['observed_risk'] is not None and r['statistics']['kept'][prefix]['observed_risk']>.02 for r in rr),
                    defined_kept=sum(r['statistics']['kept'][prefix]['observed_risk'] is not None for r in rr))
            z.update(views=len(rr), source_screen_pass=sum(r['source_screen'] for r in rr),
                known_kept_occurrences=sum(r['statistics']['kept']['known'] for r in rr),
                unknown_kept_occurrences=sum(r['statistics']['kept']['unknown'] for r in rr),
                unique_recordings=len({v['recording'] for r in rr for v in r['recordings']}),
                record_view_count=sum(len(r['recordings']) for r in rr))
            summaries[mode+'_'+role] = z
        pairs = {}
        for prefix in ('all', 'easy'):
            for subset in ('raw', 'kept'):
                for source_role in ('source_oof', 'source_resubstitution'):
                    key = prefix+'_'+subset+'_transfer_minus_'+source_role
                    pp = []
                    for r in rows:
                        if r['mode'] != mode or r['role'] != 'transfer': continue
                        src = source[r['group'], r['head_seed'], mode, source_role]
                        pp.append((r['site'], api.difference(r['statistics'][subset][prefix]['signed_bias'],
                            src['statistics'][subset][prefix]['signed_bias'])))
                    pairs[key] = interval(pp, cfg)
        paired[mode] = pairs
    summary = dict(groups=summaries, paired_bias_contrasts=paired,
        scalar_and_identity_checks=sum(r['statistics']['scalar_and_identity_checks'] for r in rows),
        parent_risk_checks=2592, group_rows=len(rows), source_heads=72, transfer_head_views=216,
        new_parameter_updates=0, result_source='fresh_run_on_cached_verified_frozen_assets',
        independent_confirmation=False, policy_changed=False)
    immutable(PUBLIC/'readout.json', dict(rows=rows))
    immutable(PUBLIC/'summary.json', summary)
    result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_m3w_selected_pool_accounting.py',
        'tests/test_m3w_component_calibration.py', 'tests/test_m3w_unknown_outcome_bounds.py', '-q'],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode: raise RuntimeError(result.stdout+result.stderr)
    (PUBLIC/'scoped_tests.txt').write_text(result.stdout+result.stderr)
    immutable(PUBLIC/'verification.json', dict(source_bindings=reg['bindings'],
        artifacts={p.name:digest(p) for p in PUBLIC.iterdir() if p.is_file() and p.name != 'verification.json'},
        source_and_transfer_exact_replay=True, independent_roles_read=False,
        policy_changed=False, full_legacy_suite='not_run'))
    print(json.dumps(dict(verified=True, scalar_checks=summary['scalar_and_identity_checks'], group_rows=len(rows))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'pilot', 'source', 'replay_source', 'transfer', 'replay_transfer', 'report'])
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    cfg, reg = registration()
    if a.phase == 'register': immutable(PUBLIC/'registration.json', reg); print('Registered selected-pool accounting'); return
    assert json.loads((PUBLIC/'registration.json').read_text()) == reg
    base.inter.committed(PUBLIC/'registration.json')
    core.torch.set_num_threads(cfg['cpu_threads']); core.torch.set_num_interop_threads(cfg['interop_threads'])
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guard(cfg); beat(state='starting', phase=a.phase)
        if a.phase == 'report': report(cfg, reg); beat(state='completed', phase=a.phase); return
        if a.phase not in ('pilot', 'replay_source', 'replay_transfer'):
            pilot = json.loads((PUBLIC/'pilot.json').read_text())
            assert pilot['memory_feasible'] and pilot['storage_feasible']
        _, _, data, jobs, oid, _, _, _ = inner.old.load()
        if a.phase in ('pilot', 'source', 'replay_source'):
            source(data, jobs, oid, cfg, pilot=a.phase=='pilot', resume=a.resume, replay=a.phase=='replay_source')
        else: transfer(data, jobs, oid, cfg, replay=a.phase=='replay_transfer', resume=a.resume)
        beat(state='completed', phase=a.phase)


if __name__ == '__main__': main()
