"""Fixed temporal candidate versus matched uniform shrinkage, resumable source study."""
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
from scripts import run_m3w_prefix_cost as parent
from scripts.run_m3w_bounded_cost import features, read_arrays
from scripts.run_m3w_conditional_cost import weighted_data, validate_args
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write, array_hash
from src.world_model.m3w_log_cost_head import fit, OBJECTIVE
from src.world_model.m3w_native_gain_harm import preprocess
from src.world_model.m3w_temporal_intervention import candidates, ARMS, POLICIES
from src.evaluation.m3w_native_metrics import native_errors
from src.evaluation.m3w_experiment_contract import file_digest

CONFIG = 'configs/m3w_temporal_intervention_v1.json'
CODE = ('scripts/run_m3w_temporal_intervention.py', 'scripts/verify_m3w_temporal_intervention.py',
        'src/world_model/m3w_temporal_intervention.py', 'src/evaluation/m3w_temporal_intervention_eval.py',
        'src/evaluation/m3w_temporal_interaction_audit.py', 'tests/test_m3w_temporal_intervention.py')


def validate_config(cfg, old):
    for k in ('sites', 'seeds', 'training', 'region_multiplier', 'bootstrap_resamples'):
        if cfg[k] != old[k]:
            raise ValueError('Changed matched setting: ' + k)
    if (tuple(cfg['arms']) != ARMS or tuple(cfg['policies']) != POLICIES or any(cfg[k] for k in (
            'threshold_search', 'model_selection', 'closed_role_readout', 'risk_calibration',
            'independent_confirmation', 'deployment', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed source-only temporal intervention study required')


def load():
    cfg = json.loads((ROOT / CONFIG).read_text())
    old, data, views, predictions, scalar, refs, pid = parent.load()
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
    prior = json.loads((ROOT / public / 'analysis.json').read_text())
    assert prior['identity'] == pid
    bind(str(public / 'analysis.json'))
    for name in ('replay.json', 'separate_verification.json'):
        r = json.loads((ROOT / public / name).read_text())
        assert r['all_checks_passed'] and r['analysis_sha256'] == bindings[str(public / 'analysis.json')]
        bind(str(public / name))
    veto = json.loads((ROOT / public / 'veto_diagnosis.json').read_text())
    assert veto['analysis_sha256'] == bindings[str(public / 'analysis.json')] and not veto['changed_decisions']
    for name in ('veto_diagnosis.json', 'conclusions.md'):
        bind(str(public / name))
    bind('scripts/verify_m3w_prefix_cost_readout.py', json.loads((ROOT / public / 'separate_verification.json').read_text())['verifier_sha256'])
    bind('scripts/audit_m3w_prefix_cost_veto.py', veto['code_sha256'])
    bind('src/evaluation/m3w_prefix_veto_audit.py', veto['helper_sha256'])
    for r in prior['training']:
        bind(r['checkpoint'], r['checkpoint_sha256'])
    identity = dict(source_bindings=bindings, config=cfg, parent_identity=pid,
                    runtime=dict(torch_threads=4, interop_threads=1, num_workers=0),
                    torch=torch.__version__, numpy=np.__version__, architecture=platform.machine(),
                    all_four_sites_design_exposed=True, closed_role_readout=False)
    assert_current(identity)
    return cfg, data, views, predictions, scalar, refs, identity


def training_arrays(meta, data, refs, key, cfg, arm):
    ids, _, _, _, oldpr, bits, w, _ = weighted_data(meta, data, refs[key, 'tempered'], cfg)
    with np.load(ROOT / meta['inputs_path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids']); p = z['prediction'].copy()
    b = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
    q = candidates(b, p)[0][arm]
    x, d, _ = features(data['geometry'][ids], q, data['scale'][ids])
    y, valid = read_arrays(data, ids, 'target'), read_arrays(data, ids, 'valid')
    cv, _ = native_errors(b, y, valid, data['scale'][ids])
    error, _ = native_errors(q, y, valid, data['scale'][ids])
    labels = np.column_stack((np.maximum(cv-error, 0), np.maximum(error-cv, 0)))
    known = valid.all(1); labels[~known] = np.nan
    with np.load(ROOT / meta['targets_path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids']); storedcv = z['baseline_ade'].copy()
    np.testing.assert_allclose(cv[known], storedcv[known], rtol=1e-10, atol=1e-8)
    storedcv[~known] = np.nan
    pr = preprocess(x, labels, storedcv, data['sites'][ids], meta['outer_site'])
    for k in ('known', 'weights'):
        np.testing.assert_array_equal(pr[k], oldpr[k])
    for k in ('cost_scale', 'hard_cut', 'positive_easy_cut'):
        assert pr[k] == oldpr[k]
    return ids, x, labels, d, pr, bits, w, q


def check_state(cp, reference, pr, w, cfg):
    assert cp['settings'] == reference['settings'] == cfg['training']
    assert cp['seed'] == reference['seed'] and cp['step'] == reference['step'] == 12000
    assert cp['objective'] == OBJECTIVE
    np.testing.assert_array_equal(cp['loss_weights'], w)
    np.testing.assert_array_equal(cp['draws'], reference['draws'])
    for k in ('mean', 'std', 'constant', 'weights', 'known'):
        np.testing.assert_array_equal(cp['preprocess'][k], pr[k])
    for k in ('weights', 'known'):
        np.testing.assert_array_equal(cp['preprocess'][k], reference['preprocess'][k])
    for k in ('cost_scale', 'hard_cut', 'positive_easy_cut'):
        assert cp['preprocess'][k] == pr[k] == reference['preprocess'][k]
    assert cp['draws'].sum() == 3072000 and cp['draws'][~pr['known']].sum() == 0


def train(cfg, data, views, refs, identity, args, beat):
    for key, meta in views.items():
        if args.view and args.view != key:
            continue
        for arm in ARMS:
            if args.arm and args.arm != arm:
                continue
            ids, x, labels, d, pr, bits, w, q = training_arrays(meta, data, refs, key, cfg, arm)
            ti = dict(identity=identity, view=key, arm=arm, inputs_sha256=array_hash(ids, x, d, q),
                      labels_sha256=array_hash(labels), region_sha256=array_hash(ids, bits, w),
                      complete_mask_sha256=array_hash(pr['known']), training_sites=pr['training_sites'],
                      producers=meta['training_producers'])
            folder = ROOT / cfg['output'] / 'trials' / key / arm; receipt = folder / 'complete.json'
            if receipt.exists():
                r = json.loads(receipt.read_text())
                assert r['identity'] == ti and r['fit']['complete'] and r['arm'] == arm
                assert file_digest(ROOT / r['checkpoint']) == r['checkpoint_sha256']
                beat(state='cached_verified_complete', view=key, arm=arm); continue
            _, result = fit(x, labels, d, data['sites'][ids], pr, w, seed=meta['seed'],
                            settings=cfg['training'], identity=ti, directory=folder,
                            resume=args.resume, stop_at=args.stop_at,
                            heartbeat=lambda **v: beat(view=key, arm=arm, **v))
            if not result['complete']:
                beat(state='pilot_complete_not_matrix', view=key, arm=arm, step=result['step']); return
            path = folder / 'checkpoint.pt'; cp = torch.load(path, map_location='cpu', weights_only=False)
            check_state(cp, refs[key, 'frozen_region'], pr, w, cfg); assert_current(identity)
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
            assert cp['identity'] == r['identity'] and cp['step'] == 12000
            records[key, arm], states[key, arm] = r, cp
    return records, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-only', 'evaluate', 'verify', 'resume'):
        parser.add_argument('--' + name, action='store_true')
    parser.add_argument('--view'); parser.add_argument('--arm', choices=ARMS); parser.add_argument('--stop-at', type=int)
    args = parser.parse_args(); validate_args(args)
    if args.arm and (args.audit_only or args.evaluate or args.verify):
        raise ValueError('No arm-restricted audit/readout')
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
            raise ValueError('Unregistered view')
        immutable_json(root / 'identity.json', identity)
        if args.audit_only:
            beat(state='preflight_pass', bindings=len(identity['source_bindings']), new_fits=24)
        elif args.evaluate or args.verify:
            from src.evaluation.m3w_temporal_intervention_eval import evaluate
            records, states = load_states(cfg, views, identity)
            evaluate(cfg, data, views, predictions, scalar, identity, records, states, beat, args.verify)
        else:
            train(cfg, data, views, refs, identity, args, beat)


if __name__ == '__main__':
    main()
