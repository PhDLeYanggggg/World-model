"""Nested fitting-only temporal probes; outer and independent roles stay closed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
from scripts import run_m3w_european_cost_shape as parent
from src.world_model import m3w_temporal_support as method
import numpy as np
import torch

magnitude, nested = parent.magnitude, parent.magnitude.nested
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_temporal_support_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_temporal_support_v1'
CONFIG = 'configs/m3w_european_temporal_support_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_temporal_support.py',
    'scripts/run_m3w_european_temporal_support.py', 'scripts/report_m3w_european_temporal_support.py',
    'tests/test_m3w_temporal_support.py', 'tests/test_m3w_temporal_support_report.py',
    'src/data_unification/m3w_european_squares_source.py',
    str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json, array_hash = parent.artifact, parent.digest, parent.immutable_json, parent.array_hash


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    parent.strong.risk.base.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); _, pid = parent.registration()
    seal = magnitude.previous.checked_seal(parent.PUBLIC/'verification.json')
    for ref in json.loads((parent.PUBLIC/'verification.json').read_text())['local_detailed_metrics']:
        assert artifact(ROOT/ref['path']) == ref
    assert cfg['arms'] == list(method.ARMS) and cfg['ridge'] == .1
    assert (cfg['outer_views'], cfg['inner_views'], cfg['fresh_ridge_fits']) == (144, 432, 1728)
    assert not any(cfg[k] for k in ('new_neural_training', 'threshold_refit', 'selection_access',
        'reserved_calibration_access', 'confirmation_access', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    identity = dict(parent=pid, parent_verification=seal, bindings={p:digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity; parent.require_committed(path)
    return cfg, identity


def checked_reference(v, site):
    _, pr, _, _, record = magnitude.ref_input(v, site)
    home = magnitude.reference_home(v, site)
    receipt = magnitude.checked_receipt(home)
    assert receipt['identity']['input'] == record
    model, state = magnitude.method.restore(home)
    assert state['arm'] == 'cost_only' and state['step'] == 2000
    assert state['preprocess']['training_sites'] == [site]
    for key in ('mean', 'std', 'known', 'weights'):
        np.testing.assert_array_equal(pr[key], state['preprocess'][key])
    assert state['preprocess']['positive_easy_cut'] == pr['positive_easy_cut']
    return model, pr, artifact(home/'complete.json')


def score_bank(v, inner, refs):
    take, pr, y, _ = nested.method.inner_inputs(v['x'], v['raw'], v['cv'], v['sites'], v['outer'], inner)
    fit_sites = v['sites'][take]; p = np.zeros((take.sum(), 4)); lineage = []
    for destination in sorted(set(fit_sites)):
        source = next(s for s in set(fit_sites) if s != destination)
        model, rp, receipt = refs[source]
        magnitude.method.check_lineage([destination, inner], rp['training_sites'], v['outer'], v['g']['producer_roster'])
        use = fit_sites == destination
        p[use], _ = magnitude.neural.predict(model, v['x'][take][use], v['env'][take][use], rp)
        lineage.append(dict(prediction_site=str(destination), training_sites=rp['training_sites'], reference=receipt))
    home = nested.PRIVATE/'heads'/v['tag']/inner
    receipt = magnitude.checked_receipt(home)
    expected = nested.input_record(v, inner, take, pr, y)
    assert receipt['input'] == expected
    with np.load(home/'scores.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], v['ids']); held = z['scores'][~take].astype(float)
    return take, pr, y, p, held, lineage, artifact(home/'complete.json')


def fitting(cfg, identity, *, pilot=False, verify=False):
    magnitude.checked_inner(); receipts = []; started = time.monotonic()
    for v in parent.views(identity['parent']):
        if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Preserve10GiB; completed fits resume unchanged')
        refs = {s:checked_reference(v, s) for s in sorted(set(v['sites']))}
        history, neighbor = method.temporal_features(v['data']['geometry'][v['ids']], v['data']['width'][v['ids']])
        for inner in sorted(set(v['sites'])):
            take, pr, y, p, hp, lineage, href = score_bank(v, inner, refs)
            env, he = v['env'][take], v['env'][~take]
            held_ids, fit_ids = v['ids'][~take], v['ids'][take]
            models, scores, fits = {}, {}, {}
            for arm in cfg['arms']:
                x = method.design_inputs(p, env, v['context'][take], history[take], neighbor[take], arm)
                hx = method.design_inputs(hp, he, v['context'][~take], history[~take], neighbor[~take], arm)
                model = method.fit(x, p, y, env, v['sites'][take], {inner, v['outer']}, ridge=cfg['ridge'])
                score = method.predict(model, hx, hp, he)
                fitted = method.predict(model, x, p, env)
                models[arm] = model; scores[arm] = score[:, 3]
                fits[arm] = dict(input_sha256=array_hash(x), scoring_features_sha256=array_hash(hx),
                    training_metrics=method.metrics(fitted, y, env))
            input_record = dict(tag=v['tag'], inner=inner, outer=v['outer'], pair=v['pair'],
                group=v['g']['group'], seed=magnitude.seed_of(v), producer=v['g']['producer'],
                controller=v['g']['controller'], training_sites=sorted(set(v['sites'][take])),
                producer_sites=v['g']['producer_roster'], training_ids_sha256=array_hash(fit_ids),
                scoring_ids_sha256=array_hash(held_ids), target_sha256=array_hash(y),
                nuisance_prediction_sha256=array_hash(p), scoring_nuisance=href,
                nuisance_lineage=lineage, training_easy_cut=pr['positive_easy_cut'],
                outer_labels_used=False, scoring_labels_used_for_fit=False)
            support = method.support(y, env, v['sites'][take], v['data']['recordings'][fit_ids],
                v['data']['agents'][fit_ids], v['data']['frames'][fit_ids])
            record = dict(registration=artifact(PUBLIC/'registration_lock.json'), input=input_record,
                models=models, fitting=fits, support=support, new_neural_training=False)
            home = PRIVATE/'probes'/v['tag']/inner
            if verify or (home/'complete.json').exists():
                magnitude.checked_receipt(home)
                assert json.loads((home/'models.json').read_text()) == record
                with np.load(home/'scores.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['ids'], held_ids)
                    np.testing.assert_array_equal(saved['raw'], hp)
                    for arm in cfg['arms']: np.testing.assert_array_equal(saved[arm], scores[arm])
            else:
                home.mkdir(parents=True, exist_ok=True)
                immutable_json(home/'models.json', record)
                magnitude.atomic_npz(home/'scores.npz', ids=held_ids, raw=hp, **scores)
                immutable_json(home/'complete.json', dict(registration=artifact(PUBLIC/'registration_lock.json'),
                    artifacts=dict(models=artifact(home/'models.json'), scores=artifact(home/'scores.npz'))))
            receipts.append(artifact(home/'complete.json'))
            beat('fit_replayed' if verify else 'fit_frozen', inner_views=len(receipts), fits=4*len(receipts), tag=v['tag'], inner=inner)
            if pilot:
                immutable_json(PRIVATE/'pilot.json', dict(complete=True, receipt=receipts[0],
                    fits=4, seconds_excluding_ancestry=time.monotonic()-started,
                    scoring_labels_used=False, free_GiB=shutil.disk_usage(PRIVATE).free/2**30))
                return
    assert len(receipts) == cfg['inner_views']
    freeze = dict(registration=artifact(PUBLIC/'registration_lock.json'), probes=receipts,
        fresh_fits=4*len(receipts), scoring_labels_used=False, outer_labels_used=False,
        independent_roles_read=False, new_neural_training=False)
    if verify:
        assert json.loads((PUBLIC/'prediction_freeze.json').read_text()) == freeze
        immutable_json(PUBLIC/'fitting_replay.json', dict(exact=True, inner_views=432, ridge_fits=1728))
    else: immutable_json(PUBLIC/'prediction_freeze.json', freeze)


def evaluate(cfg, identity, verify=False):
    path = PUBLIC/'prediction_freeze.json'; parent.require_committed(path)
    frozen = json.loads(path.read_text()); assert frozen['registration'] == artifact(PUBLIC/'registration_lock.json')
    for ref in frozen['probes']: assert artifact(ROOT/ref['path']) == ref
    rows = []; cohort = {}
    for v in parent.views(identity['parent']):
        for inner in sorted(set(v['sites'])):
            home = PRIVATE/'probes'/v['tag']/inner; magnitude.checked_receipt(home)
            record = json.loads((home/'models.json').read_text()); take = v['sites'] == inner
            ids = v['ids'][take]; env = v['env'][take]
            assert record['input']['scoring_ids_sha256'] == array_hash(ids)
            y = nested.method.labels(v['raw'][take], v['cv'][take], record['input']['training_easy_cut'])
            registry = cohort.setdefault(str(inner), set())
            valid = np.isfinite(y).all(1) & (env > 0)
            registry.update(zip(v['data']['recordings'][ids][valid].tolist(),
                v['data']['agents'][ids][valid].tolist(), v['data']['frames'][ids][valid].tolist()))
            with np.load(home/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], ids)
                scores = dict(raw=z['raw'].copy())
                for arm in cfg['arms']:
                    scores[arm] = scores['raw'].copy(); scores[arm][:, 3] = z[arm]
                metrics = {k:method.metrics(s, y, env) for k, s in scores.items()}
            support = method.support(y, env, v['sites'][take], v['data']['recordings'][ids],
                v['data']['agents'][ids], v['data']['frames'][ids])
            rows.append(dict(input=record['input'], metrics=metrics, scoring_support=support,
                fitting_support=record['support'], fitting_metrics=record['fitting'],
                evaluation_target_sha256=array_hash(y)))
        beat('inner_scored', inner_views=len(rows), tag=v['tag'])
    assert len(rows) == 432
    counts = {}
    for site, keys in cohort.items():
        counts[site] = dict(unique_supported_agent_queries=len(keys),
            unique_tracks=len({k[:2] for k in keys}), unique_recordings=len({k[0] for k in keys}),
            unique_scene_queries=len({(k[0], k[2]) for k in keys}))
    doc = dict(rows=rows, cohort=counts, result_source='fresh_run_inner_fitting_locality_screen_cached_verified_nuisance_heads',
        outer_source_evaluation=False, independent_roles_read=False, deployment_changed=False)
    if verify:
        assert json.loads((PRIVATE/'readout.json').read_text()) == doc
        immutable_json(PUBLIC/'evaluation_replay.json', dict(exact=True, inner_views=432))
    else: immutable_json(PRIVATE/'readout.json', doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['register', 'pilot', 'fit', 'evaluate', 'verify_fit', 'verify_eval'])
    args = parser.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True); PUBLIC.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        cfg, identity = registration(args.phase == 'register')
        beat('started', phase=args.phase, threads=4, workers=0, architecture=platform.machine())
        if args.phase in ('pilot', 'fit', 'verify_fit'):
            fitting(cfg, identity, pilot=args.phase == 'pilot', verify=args.phase == 'verify_fit')
        if args.phase in ('evaluate', 'verify_eval'): evaluate(cfg, identity, verify=args.phase == 'verify_eval')
        beat('complete', phase=args.phase)


if __name__ == '__main__': main()
