"""Registered, resumable matched prefix-risk experiment on existing source roles."""
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
    raise RuntimeError('Native arm64 required before Torch import')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_log_cost as parent
from scripts.run_m3w_bounded_cost import read_arrays
from scripts.run_m3w_conditional_cost import weighted_data, validate_args
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write, array_hash
from src.world_model.m3w_prefix_cost_head import fit, ARMS, OBJECTIVE
from src.world_model.m3w_prefix_cost_targets import supervised_prefix_costs
from src.evaluation.m3w_prefix_cost_policy import POLICIES
from src.evaluation.m3w_experiment_contract import file_digest

CONFIG = 'configs/m3w_prefix_cost_v1.json'
CODE = ('scripts/run_m3w_prefix_cost.py', 'scripts/verify_m3w_prefix_cost.py',
        'src/world_model/m3w_prefix_cost_head.py', 'src/world_model/m3w_prefix_cost_targets.py',
        'src/evaluation/m3w_prefix_cost_policy.py', 'src/evaluation/m3w_prefix_cost_eval.py',
        'tests/test_m3w_prefix_cost_head.py')


def validate_config(cfg, old):
    for k in ('sites', 'seeds', 'training', 'region_multiplier', 'bootstrap_resamples'):
        if cfg[k] != old[k]:
            raise ValueError('Changed matched setting: ' + k)
    if (tuple(cfg['arms']) != ARMS or tuple(cfg['policies']) != POLICIES
            or cfg['primary_policy'] != 'profile_guard' or cfg['primary_reference'] != 'control_terminal'
            or any(cfg[k] for k in ('threshold_search', 'model_selection', 'closed_role_readout',
                'risk_calibration', 'independent_confirmation', 'deployment', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed matched source-only experiment required')


def load():
    cfg = json.loads((ROOT / CONFIG).read_text())
    old, data, views, predictions, prior, refs, pid = parent.load()
    validate_config(cfg, old)
    bindings = dict(pid['source_bindings'])
    def bind(path, expected=None):
        actual = file_digest(ROOT / path)
        if (expected is not None and actual != expected) or (path in bindings and bindings[path] != actual):
            raise ValueError('Changed source: ' + path)
        bindings[path] = actual
    for path in (CONFIG, cfg['registration'], *CODE):
        bind(path)
    public = Path(old['reports'])
    scalar = json.loads((ROOT / public / 'analysis.json').read_text())
    assert scalar['identity'] == pid
    bind(str(public / 'analysis.json'))
    for name in ('replay.json', 'independent_verification.json', 'prefix_target_verification.json'):
        r = json.loads((ROOT / public / name).read_text())
        assert r['all_checks_passed'] and r['analysis_sha256'] == bindings[str(public / 'analysis.json')]
        bind(str(public / name))
    for name in ('conditional_support_diagnosis.md', 'horizon_cost_support.json', 'harm_concentration.json'):
        bind(str(public / name))
    records, _ = parent.load_states(old, views, pid)
    for r in records.values():
        bind(r['checkpoint'], r['checkpoint_sha256'])
    for r in scalar['archives']:
        bind(r['path'], r['sha256'])
    identity = dict(source_bindings=bindings, config=cfg, parent_identity=pid,
                    runtime=dict(torch_threads=4, interop_threads=1, num_workers=0),
                    torch=torch.__version__, numpy=np.__version__, architecture=platform.machine(),
                    all_four_sites_design_exposed=True, closed_role_readout=False)
    assert_current(identity)
    return cfg, data, views, predictions, scalar, refs, identity


def training_arrays(meta, data, refs, key, cfg):
    ids, x, y, d, pr, bits, w, _ = weighted_data(meta, data, refs[key, 'tempered'], cfg)
    with np.load(ROOT / meta['inputs_path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids']); p = z['prediction'].copy()
    b = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
    labels = supervised_prefix_costs(b, p, read_arrays(data, ids, 'target'),
                                     read_arrays(data, ids, 'valid'), data['scale'][ids])
    np.testing.assert_array_equal(labels['available'][:, -1], pr['known'])
    np.testing.assert_allclose(labels['costs'][pr['known'], -1], y[pr['known']], rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(labels['disagreement'][:, -1], d, rtol=1e-10, atol=1e-8)
    return ids, x, labels['costs'], labels['disagreement'], pr, bits, w


def check_state(cp, reference, w, cfg):
    assert cp['settings'] == reference['settings'] == cfg['training']
    assert cp['seed'] == reference['seed'] and cp['step'] == reference['step'] == 12000
    assert cp['objective'] == OBJECTIVE and cp['arm'] in ARMS
    np.testing.assert_array_equal(cp['loss_weights'], w)
    np.testing.assert_array_equal(cp['draws'], reference['draws'])
    for k in ('mean', 'std', 'known', 'weights', 'constant'):
        np.testing.assert_array_equal(cp['preprocess'][k], reference['preprocess'][k])
    for k in ('cost_scale', 'hard_cut', 'positive_easy_cut'):
        assert cp['preprocess'][k] == reference['preprocess'][k]
    assert cp['draws'].sum() == 12000*256 and cp['draws'][~cp['preprocess']['known']].sum() == 0


def train(cfg, data, views, refs, identity, args, beat):
    for key, meta in views.items():
        if args.view and args.view != key:
            continue
        ids, x, costs, d, pr, bits, w = training_arrays(meta, data, refs, key, cfg)
        ti = dict(identity=identity, view=key, inputs_sha256=array_hash(ids, x, d),
                  labels_sha256=array_hash(costs), region_sha256=array_hash(ids, bits, w),
                  complete_mask_sha256=array_hash(pr['known']), training_sites=pr['training_sites'],
                  producers=meta['training_producers'])
        for arm in ARMS:
            if args.arm and args.arm != arm:
                continue
            folder = ROOT / cfg['output'] / 'trials' / key / arm
            receipt = folder / 'complete.json'
            if receipt.exists():
                r = json.loads(receipt.read_text())
                assert r['identity'] == ti and r['fit']['complete'] and r['arm'] == arm
                assert file_digest(ROOT / r['checkpoint']) == r['checkpoint_sha256']
                beat(state='cached_verified_complete', view=key, arm=arm); continue
            _, result = fit(x, costs, d, data['sites'][ids], pr, w, arm=arm, seed=meta['seed'],
                            settings=cfg['training'], identity=ti, directory=folder,
                            resume=args.resume, stop_at=args.stop_at,
                            heartbeat=lambda **v: beat(view=key, arm=arm, **v))
            if not result['complete']:
                beat(state='pilot_complete_not_matrix', view=key, arm=arm, step=result['step']); return
            path = folder / 'checkpoint.pt'
            cp = torch.load(path, map_location='cpu', weights_only=False)
            check_state(cp, refs[key, 'frozen_region'], w, cfg)
            assert_current(identity)
            immutable_json(receipt, dict(identity=ti, arm=arm, fit=result, rows=len(ids),
                           supported_rows=int(pr['known'].sum()), checkpoint=str(path.relative_to(ROOT)),
                           checkpoint_sha256=file_digest(path)))
    beat(state='training_call_complete')


def load_states(cfg, views, identity):
    records, states = {}, {}
    for key in views:
        for arm in ARMS:
            r = json.loads((ROOT / cfg['output'] / 'trials' / key / arm / 'complete.json').read_text())
            assert r['identity']['identity'] == identity and r['fit']['complete'] and r['arm'] == arm
            assert file_digest(ROOT / r['checkpoint']) == r['checkpoint_sha256']
            cp = torch.load(ROOT / r['checkpoint'], map_location='cpu', weights_only=False)
            assert cp['identity'] == r['identity'] and cp['arm'] == arm and cp['step'] == 12000
            records[key, arm], states[key, arm] = r, cp
    return records, states


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-only', 'evaluate', 'verify', 'resume'):
        p.add_argument('--' + name, action='store_true')
    p.add_argument('--view'); p.add_argument('--arm', choices=ARMS); p.add_argument('--stop-at', type=int)
    args = p.parse_args(); validate_args(args)
    if args.arm and (args.audit_only or args.evaluate or args.verify):
        raise ValueError('No arm-restricted audit or readout')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg = json.loads((ROOT / CONFIG).read_text()); root = ROOT / cfg['output']; root.mkdir(parents=True, exist_ok=True)
    def beat(**v):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **v)
        json_write(root / 'heartbeat.json', event)
        with (root / 'events.jsonl').open('a') as f:
            f.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    with (root / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, data, views, predictions, scalar, refs, identity = load()
        if args.view and args.view not in views:
            raise ValueError('Unregistered view: ' + args.view)
        immutable_json(root / 'identity.json', identity)
        if args.audit_only:
            beat(state='preflight_pass', bindings=len(identity['source_bindings']), new_fits=24, views=list(views))
        elif args.evaluate or args.verify:
            from src.evaluation.m3w_prefix_cost_eval import evaluate
            records, states = load_states(cfg, views, identity)
            evaluate(cfg, data, views, predictions, scalar, refs, identity, records, states, beat, args.verify)
        else:
            train(cfg, data, views, refs, identity, args, beat)


if __name__ == '__main__':
    main()
