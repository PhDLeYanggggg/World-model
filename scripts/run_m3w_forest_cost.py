"""Frozen ExtraTrees cost comparator on existing nested source-only forecasts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before importing numerical runtimes')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
import torch
from scripts import run_m3w_temporal_intervention as parent
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write, array_hash
from src.world_model.m3w_forest_cost_head import fit, ARMS, POLICIES
from src.evaluation.m3w_experiment_contract import file_digest

CONFIG = 'configs/m3w_forest_cost_v1.json'
CODE = ('scripts/run_m3w_forest_cost.py', 'src/world_model/m3w_forest_cost_head.py',
        'src/evaluation/m3w_forest_cost_eval.py', 'src/evaluation/m3w_fixed_choice_context.py',
        'tests/test_m3w_forest_cost_head.py')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    old, data, views, predictions, scalar, refs, pid = parent.load()
    if (sklearn.__version__ != '1.8.0' or cfg['sites'] != old['sites'] or cfg['seeds'] != old['seeds']
            or tuple(cfg['arms']) != ARMS or tuple(cfg['policies']) != POLICIES
            or any(cfg[k] for k in ('threshold_search', 'model_selection', 'closed_role_readout',
                'risk_calibration', 'independent_confirmation', 'deployment', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Registered runtime and source-only comparator required')
    records, states = parent.load_states(old, views, pid)
    bindings = dict(pid['source_bindings'])
    def bind(path, expected=None):
        actual = file_digest(ROOT/path)
        if (expected is not None and actual != expected) or (path in bindings and bindings[path] != actual):
            raise ValueError('Changed source: '+path)
        bindings[path] = actual
    for path in (CONFIG, cfg['registration'], *CODE): bind(path)
    public = ROOT/old['reports']; report = json.loads((public/'analysis.json').read_text())
    assert report['identity'] == pid
    for name in ('analysis.json', 'replay.json', 'separate_verification.json',
                 'fit_support_diagnosis.json'):
        bind(str((public/name).relative_to(ROOT)))
    for name in ('replay.json', 'separate_verification.json'):
        r = json.loads((public/name).read_text())
        assert r['all_checks_passed'] and r['analysis_sha256'] == file_digest(public/'analysis.json')
    diagnostic = json.loads((public/'fit_support_diagnosis.json').read_text())
    assert diagnostic['analysis_sha256'] == file_digest(public/'analysis.json')
    for path, key in (('scripts/audit_m3w_temporal_fit_support.py', 'code_sha256'),
                      ('src/evaluation/m3w_temporal_fit_support.py', 'helper_sha256'),
                      ('tests/test_m3w_temporal_fit_support.py', 'tests_sha256')):
        bind(path, diagnostic[key])
    for r in records.values(): bind(r['checkpoint'], r['checkpoint_sha256'])
    for r in report['archives']: bind(r['path'], r['sha256'])
    identity = dict(source_bindings=bindings, config=cfg, parent_identity=pid,
        sklearn=sklearn.__version__, numpy=np.__version__, architecture=platform.machine(),
        runtime=dict(fit_threads=4, prediction_threads=1, num_workers=0),
        all_four_sites_design_exposed=True, closed_role_readout=False)
    assert_current(identity)
    return cfg, old, data, views, predictions, refs, states, report, identity


def train(cfg, old, data, views, refs, previous, identity, args, beat):
    for key, meta in views.items():
        if args.view and args.view != key: continue
        for arm in ARMS:
            if args.arm and args.arm != arm: continue
            ids, x, y, d, pr, bits, w, q = parent.training_arrays(meta, data, refs, key, old, arm)
            prior = previous[key, arm]
            parent.check_state(prior, refs[key, 'frozen_region'], pr, w, old)
            ti = dict(identity=identity, view=key, arm=arm, inputs_sha256=array_hash(ids, x, d, q),
                labels_sha256=array_hash(y), draws_sha256=array_hash(prior['draws']),
                weights_sha256=array_hash(w), training_sites=pr['training_sites'],
                producers=meta['training_producers'])
            folder = ROOT/cfg['output']/'trials'/key/arm; receipt = folder/'complete.json'
            if receipt.exists():
                r = json.loads(receipt.read_text())
                assert r['identity'] == ti and r['fit']['complete']
                assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
                beat(state='cached_verified_complete', view=key, arm=arm); continue
            _, result = fit(x, y, d, pr, prior['draws'], w, seed=meta['seed'],
                settings=cfg['forest'], identity=ti, directory=folder,
                heartbeat=lambda **v: beat(view=key, arm=arm, **v),
                resume=args.resume, stop_at=args.stop_at)
            if not result['complete']:
                beat(state='pilot_complete_not_matrix', view=key, arm=arm, trees=result['trees']); return
            path = folder/'checkpoint.joblib'; assert_current(identity)
            immutable_json(receipt, dict(identity=ti, view=key, arm=arm, fit=result,
                checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=file_digest(path)))
    beat(state='training_call_complete')


def load_states(cfg, views, identity):
    records, states = {}, {}
    for key in views:
        for arm in ARMS:
            r = json.loads((ROOT/cfg['output']/'trials'/key/arm/'complete.json').read_text())
            assert r['identity']['identity'] == identity and r['fit']['complete']
            assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
            cp = joblib.load(ROOT/r['checkpoint'])
            assert cp['identity'] == r['identity'] and len(cp['model'].estimators_) == cfg['forest']['trees']
            records[key, arm], states[key, arm] = r, cp
    return records, states


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-only', 'resume', 'evaluate', 'verify'): p.add_argument('--'+name, action='store_true')
    p.add_argument('--view'); p.add_argument('--arm', choices=ARMS); p.add_argument('--stop-at', type=int)
    args = p.parse_args()
    modes = sum((args.audit_only, args.evaluate, args.verify))
    if modes > 1 or (modes and (args.view or args.arm or args.stop_at is not None or args.resume)):
        raise ValueError('Whole fixed readout and disjoint execution modes required')
    if args.stop_at and (not args.view or not args.arm): raise ValueError('Pilot requires one view and arm')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg = json.loads((ROOT/CONFIG).read_text()); root = ROOT/cfg['output']; root.mkdir(parents=True, exist_ok=True)
    def beat(**v):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **v)
        json_write(root/'heartbeat.json', event)
        with (root/'events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last saved forest')))
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, old, data, views, predictions, refs, previous, report, identity = load()
        if args.view and args.view not in views: raise ValueError('Unregistered view')
        immutable_json(root/'identity.json', identity)
        if args.audit_only:
            beat(state='preflight_pass', bindings=len(identity['source_bindings']), planned_forests=24)
        elif args.evaluate or args.verify:
            from src.evaluation.m3w_forest_cost_eval import evaluate
            records, states = load_states(cfg, views, identity)
            evaluate(cfg, data, views, predictions, report, identity, records, states, beat, args.verify)
        else:
            train(cfg, old, data, views, refs, previous, identity, args, beat)


if __name__ == '__main__': main()
