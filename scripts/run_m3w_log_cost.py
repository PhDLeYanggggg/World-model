"""One-factor proper compositional loss experiment on existing source roles."""
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
from scripts import run_m3w_adaptive_region_cost as parent
from scripts.run_m3w_conditional_cost import weighted_data, check_state as reference_check, load_states, validate_args
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write, array_hash
from src.world_model.m3w_log_cost_head import fit, OBJECTIVE
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch

CONFIG = 'configs/m3w_log_cost_v1.json'
CODE = ('scripts/run_m3w_log_cost.py', 'scripts/verify_m3w_log_cost.py',
        'src/world_model/m3w_log_cost_head.py', 'src/evaluation/m3w_log_cost_eval.py',
        'tests/test_m3w_log_cost.py')


def validate_config(cfg, old):
    for k in ('sites', 'seeds', 'training', 'loss_exponent', 'region_multiplier', 'region_reference',
              'primary_reference', 'policies', 'matched_count_reference', 'bootstrap_resamples'):
        if cfg[k] != old[k]:
            raise ValueError('Changed matched setting: ' + k)
    if cfg['objective'] != OBJECTIVE or any(cfg[k] for k in ('threshold_search', 'model_selection',
            'closed_role_readout', 'risk_calibration', 'independent_confirmation', 'deployment',
            'stage5c_executed', 'smc_enabled')):
        raise ValueError('Fixed log-loss training experiment only')


def load():
    cfg = json.loads((ROOT / CONFIG).read_text())
    old, data, views, predictions, prior, refs, pid = parent.load()
    validate_config(cfg, old)
    bindings = dict(pid['source_bindings'])
    for path in (CONFIG, cfg['registration'], cfg['motivation'], *CODE):
        sha = file_digest(ROOT / path)
        if path in bindings and sha != bindings[path]:
            raise ValueError('Changed parent source')
        bindings[path] = sha
    motivation = json.loads((ROOT / cfg['motivation']).read_text())
    source = str(Path(cfg['motivation']).parent / 'analysis.json')
    assert file_digest(ROOT / source) == motivation['source_analysis_sha256']
    bindings[source] = motivation['source_analysis_sha256']
    for name in ('replay.json', 'independent_verification.json'):
        path = str(Path(source).parent / name); r = json.loads((ROOT / path).read_text())
        assert r['all_checks_passed'] and r['analysis_sha256'] == bindings[source]
        bindings[path] = file_digest(ROOT / path)
    identity = dict(source_bindings=bindings, config=cfg, parent_identity=pid,
        runtime=dict(torch_threads=4, interop_threads=1, num_workers=0),
        torch=torch.__version__, numpy=np.__version__, architecture=platform.machine(),
        closed_role_readout=False, all_four_sites_design_exposed=True)
    assert_current(identity)
    return cfg, data, views, predictions, prior, refs, identity


def check_state(cp, reference, weights, cfg):
    reference_check(cp, reference, weights, cfg)
    assert cp['objective'] == cfg['objective'] == OBJECTIVE


def train(cfg, data, views, refs, identity, args, beat):
    for key, meta in views.items():
        if args.view and args.view != key:
            continue
        ids, x, y, d, pr, bits, weights, _ = weighted_data(meta, data, refs[key, 'tempered'], cfg)
        ti = dict(identity=identity, view=key, inputs_sha256=array_hash(ids, x, d),
            labels_sha256=array_hash(y), region_sha256=array_hash(ids, bits, weights),
            complete_mask_sha256=array_hash(pr['known']), training_sites=pr['training_sites'],
            producers=meta['training_producers'])
        folder = ROOT / cfg['output'] / 'trials' / key
        receipt = folder / 'complete.json'
        if receipt.exists():
            r = json.loads(receipt.read_text())
            assert r['identity'] == ti and r['fit']['complete']
            assert file_digest(ROOT / r['checkpoint']) == r['checkpoint_sha256']
            beat(state='cached_verified_complete', view=key); continue
        _, result = fit(x, y, d, data['sites'][ids], pr, weights, seed=meta['seed'], settings=cfg['training'],
            identity=ti, directory=folder, resume=args.resume, stop_at=args.stop_at,
            heartbeat=lambda **v: beat(view=key, **v))
        if not result['complete']:
            beat(state='pilot_complete_not_matrix', view=key, step=result['step']); return
        path = folder / 'checkpoint.pt'; cp = torch.load(path, map_location='cpu', weights_only=False)
        check_state(cp, refs[key, 'frozen_region'], weights, cfg)
        assert_current(identity)
        immutable_json(receipt, dict(identity=ti, fit=result, rows=len(ids), supported_rows=int(pr['known'].sum()),
            emphasized_complete_rows=int((bits & pr['known']).sum()),
            mean_weight_under_sampler=float(np.dot(pr['weights'], weights)),
            checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=file_digest(path)))
    beat(state='training_call_complete')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-only', 'evaluate', 'verify', 'resume'):
        p.add_argument('--' + name, action='store_true')
    p.add_argument('--view'); p.add_argument('--stop-at', type=int)
    args = p.parse_args(); validate_args(args)
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
        cfg, data, views, predictions, prior, refs, identity = load()
        if args.view and args.view not in views:
            raise ValueError('Unregistered view')
        immutable_json(root / 'identity.json', identity)
        if args.audit_only:
            beat(state='preflight_pass', bindings=len(identity['source_bindings']), new_fits=12)
        elif args.evaluate or args.verify:
            from src.evaluation.m3w_log_cost_eval import evaluate
            records, states = load_states(cfg, views, identity)
            evaluate(cfg, data, views, predictions, prior, refs, identity, records, states, beat, args.verify)
        else:
            train(cfg, data, views, refs, identity, args, beat)


if __name__ == '__main__':
    main()
