"""Attribution using frozen fitting-development predictions, never reserved roles."""
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
    raise RuntimeError('Native arm64 required before importing Torch')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
from scripts import run_m3w_european_temporal_support as parent
from src.world_model import m3w_cap_attribution as method
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_cap_attribution_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_cap_attribution_v1'
CONFIG = 'configs/m3w_european_cap_attribution_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_cap_attribution.py', 'tests/test_m3w_cap_attribution.py',
    'scripts/run_m3w_european_cap_attribution.py', 'scripts/report_m3w_european_cap_attribution.py',
    'tests/test_m3w_cap_attribution_report.py', str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json, array_hash = parent.artifact, parent.digest, parent.immutable_json, parent.array_hash


def beat(state, **kwargs):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kwargs)
    parent.parent.strong.risk.base.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); _, pid = parent.registration()
    seal_path = parent.PUBLIC/'verification.json'
    seal = parent.magnitude.previous.checked_seal(seal_path)
    assert digest(seal_path) == cfg['parent_verification_sha256']
    for ref in json.loads(seal_path.read_text())['local_detailed_metrics']:
        assert artifact(ROOT/ref['path']) == ref
    assert cfg['arms'] == list(parent.method.ARMS) and cfg['projections'] == list(method.PROJECTIONS)
    assert cfg['inner_views'] == 432 and cfg['seeds'] == [17, 29, 43]
    assert not any(cfg[k] for k in ('new_neural_training', 'refit', 'threshold_refit', 'selection_access',
        'reserved_calibration_access', 'confirmation_access', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    identity = dict(parent=pid, parent_seal=seal, bindings={p:digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity; parent.parent.require_committed(path)
    return cfg, identity


def views(identity):
    yield from parent.parent.views(identity['parent']['parent'])


def infer(v, inner, history, neighbor, arms):
    home = parent.PRIVATE/'probes'/v['tag']/inner
    parent.magnitude.checked_receipt(home)
    record = json.loads((home/'models.json').read_text()); take = v['sites'] == inner
    ids, env = v['ids'][take], v['env'][take]
    assert record['input']['scoring_ids_sha256'] == array_hash(ids)
    assert record['input']['outer'] == v['outer']
    assert {inner, v['outer']}.isdisjoint(record['input']['training_sites'])
    scores, feature_hashes = {}, {}
    with np.load(home/'scores.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], ids); p = z['raw'].copy()
        for arm in arms:
            x = parent.method.design_inputs(p, env, v['context'][take], history[take], neighbor[take], arm)
            feature_hashes[arm] = array_hash(x)
            assert feature_hashes[arm] == record['fitting'][arm]['scoring_features_sha256']
            scores[arm] = method.unprojected(record['models'][arm], x, p, env)
            np.testing.assert_array_equal(method.project(scores[arm], p, env)['frozen_cap'][:, 3], z[arm])
    receipt = dict(parent=artifact(home/'complete.json'), input=record['input'],
        features=feature_hashes, raw_prediction_sha256=array_hash(p), envelope_sha256=array_hash(env),
        unprojected_scores={arm:array_hash(q) for arm,q in scores.items()},
        target_access_for_inference=False, new_fit=False)
    return take, env, p, scores, receipt


def execute(cfg, identity, phase):
    scoring = phase in ('evaluate', 'verify_eval'); verify = phase.startswith('verify')
    pilot = phase == 'pilot'; refs = []; rows = []; start = time.monotonic()
    frozen = PUBLIC/'prediction_freeze.json'
    if scoring:
        parent.parent.require_committed(frozen)
        for ref in json.loads(frozen.read_text())['receipts']: assert artifact(ROOT/ref['path']) == ref
    old_rows = { (r['input']['tag'], r['input']['inner']):r
        for r in json.loads((parent.PRIVATE/'readout.json').read_text())['rows']} if scoring else {}
    for v in views(identity):
        if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Preserve10GiB; verified completed receipts resume')
        history, neighbor = parent.method.temporal_features(v['data']['geometry'][v['ids']], v['data']['width'][v['ids']])
        for inner in sorted(set(v['sites'])):
            take, env, p, scores, receipt = infer(v, inner, history, neighbor, cfg['arms'])
            dest = PRIVATE/'receipts'/v['tag']/inner/'complete.json'
            if verify or scoring:
                assert json.loads(dest.read_text()) == receipt
            else:
                dest.parent.mkdir(parents=True, exist_ok=True); immutable_json(dest, receipt)
            refs.append(artifact(dest))
            if scoring:
                y = parent.nested.method.labels(v['raw'][take], v['cv'][take], receipt['input']['training_easy_cut'])
                old = old_rows[v['tag'], inner]
                assert array_hash(y) == old['evaluation_target_sha256']
                metrics, diagnostics = {}, {}
                for arm, q in scores.items():
                    ms = method.evaluate(q, p, env, y)
                    assert ms['frozen_cap'] == old['metrics'][arm]
                    metrics.update({arm+'_'+name:val for name,val in ms.items()})
                    diagnostics[arm] = method.diagnose(q, p, env, y)
                rows.append(dict(input=receipt['input'], metrics=metrics, diagnostics=diagnostics,
                    target_sha256=array_hash(y), prior_scores_exact=True))
            beat('scored' if scoring else 'replayed' if verify else 'prediction_bound', views=len(refs), tag=v['tag'], inner=inner)
            if pilot:
                immutable_json(PRIVATE/'pilot.json', dict(complete=True, receipt=refs[0],
                    seconds_excluding_ancestry=time.monotonic()-start, scoring_labels_used=False,
                    free_GiB=shutil.disk_usage(PRIVATE).free/2**30, new_fits=0))
                return
    assert len(refs) == cfg['inner_views']
    freeze = dict(registration=artifact(PUBLIC/'registration_lock.json'), receipts=refs,
        unprojected_vectors=4*len(refs), new_fits=0, independent_roles_read=False)
    if scoring:
        doc = dict(rows=rows, result_source='fresh_attribution_cached_verified_models_no_refit',
            independent_roles_read=False, deployment_changed=False)
        if verify:
            assert json.loads((PRIVATE/'readout.json').read_text()) == doc
            immutable_json(PUBLIC/'evaluation_replay.json', dict(exact=True, views=len(rows), prior_scores_exact=True))
        else: immutable_json(PRIVATE/'readout.json', doc)
    elif verify:
        assert json.loads(frozen.read_text()) == freeze
        immutable_json(PUBLIC/'prediction_replay.json', dict(exact=True, views=len(refs), unprojected_vectors=4*len(refs)))
    else: immutable_json(frozen, freeze)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['register', 'pilot', 'freeze', 'evaluate', 'verify_predictions', 'verify_eval'])
    args = parser.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True); PUBLIC.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        cfg, identity = registration(args.phase == 'register')
        beat('started', phase=args.phase, threads=4, workers=0, architecture=platform.machine())
        if args.phase != 'register': execute(cfg, identity, args.phase)
        beat('complete', phase=args.phase)


if __name__ == '__main__': main()
