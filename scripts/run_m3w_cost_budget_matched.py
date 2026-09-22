"""Frozen matched-capacity/budget native and fraction controls."""
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
from scripts import run_m3w_cost_capacity as parent
from scripts.run_m3w_bounded_cost import training_data
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write, array_hash
from src.world_model.m3w_bounded_cost_head import fit
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch

CONFIG = 'configs/m3w_cost_budget_matched_v1.json'
CODE = ('scripts/run_m3w_cost_budget_matched.py', 'src/evaluation/m3w_cost_budget_matched_eval.py',
        'scripts/verify_m3w_cost_budget_matched.py', 'tests/test_m3w_cost_budget_matched.py')
ARMS = ('native', 'fraction', 'tempered')
FORWARD = dict(native='bounded_native', fraction='bounded_fraction')


def validate_config(cfg, pcfg, tcfg):
    if (cfg['sites'] != pcfg['sites'] or cfg['seeds'] != pcfg['seeds']
            or cfg['training'] != tcfg['training'] | {'width':128, 'steps':12000}
            or cfg['training_arms'] != ['native', 'fraction'] or cfg['reference_arm'] != 'tempered'
            or cfg['primary_contrasts'] != ['tempered_minus_native', 'tempered_minus_fraction']
            or cfg['primary_policy'] != 'strict_stop'
            or cfg['policies'] != ['net_stop', 'strict_stop', 'matched_count']
            or cfg['matched_count_reference'] != pcfg['matched_count_reference']
            or cfg['bootstrap_resamples'] != 3000 or any(cfg[k] for k in (
                'threshold_search', 'model_selection', 'closed_role_readout', 'risk_calibration',
                'independent_confirmation', 'deployment', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed matched-budget controls and unchanged readout required')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    pcfg, tcfg, data, views, predictions, controls, prior, pid = parent.load()
    validate_config(cfg, pcfg, tcfg)
    reference_records, reference_states = parent.load_states(pcfg, tcfg, views, prior, pid)
    report = json.loads((ROOT/cfg['parent_analysis']).read_text())
    assert report['identity'] == pid and report['new_updates'] == 252000
    bindings = dict(pid['source_bindings'])
    def bind(path, expected=None):
        actual = file_digest(ROOT/path)
        if (expected is not None and actual != expected) or (path in bindings and bindings[path] != actual):
            raise ValueError('Changed dependency: '+path)
        bindings[path] = actual
    for path in (CONFIG, cfg['registration'], cfg['parent_analysis'], *CODE):
        bind(path)
    for name in ('replay.json', 'independent_verification.json'):
        path = str(Path(cfg['parent_analysis']).parent/name)
        r = json.loads((ROOT/path).read_text())
        assert r['all_checks_passed'] and r['analysis_sha256'] == bindings[cfg['parent_analysis']]
        bind(path)
    for (key, width), r in reference_records.items():
        bind(str(Path(pcfg['output'])/'trials'/key/width/'complete.json'))
        bind(r['checkpoint'], r['checkpoint_sha256'])
        if width == 'wide':
            bind(r['prefix_checkpoint'], r['prefix_checkpoint_sha256'])
    for r in report['archives']:
        bind(r['path'], r['sha256'])
    identity = dict(config=cfg, source_bindings=bindings, parent_identity=pid,
        torch=torch.__version__, numpy=np.__version__, architecture=platform.machine(),
        runtime=dict(torch_threads=4, interop_threads=1, num_workers=0), closed_role_readout=False)
    refs = {(key, 'tempered'): reference_states[key, 'wide_long'] for key in views}
    assert_current(identity)
    return cfg, data, views, predictions, controls, report, refs, identity


def check_matched(cp, reference, cfg, seed, arm):
    if (cp['settings'] != cfg['training'] or cp['step'] != 12000 or cp['seed'] != seed
            or cp['arm'] != FORWARD[arm] or reference['settings'] != cp['settings']
            or reference['seed'] != seed or reference['step'] != cp['step']):
        raise ValueError('Matched configuration and completed endpoint required')
    np.testing.assert_array_equal(cp['draws'], reference['draws'])
    for f in ('mean', 'std', 'known', 'weights', 'constant'):
        np.testing.assert_array_equal(cp['preprocess'][f], reference['preprocess'][f])
    for f in ('cost_scale', 'hard_cut', 'positive_easy_cut'):
        assert cp['preprocess'][f] == reference['preprocess'][f]
    assert cp['draws'].sum() == 12000*256 and cp['draws'][~cp['preprocess']['known']].sum() == 0


def train(cfg, data, views, refs, identity, args, beat):
    for key, meta in views.items():
        if args.view and key != args.view:
            continue
        ids, x, y, d, pr = training_data(meta, data)
        for arm in cfg['training_arms']:
            if args.arm and arm != args.arm:
                continue
            ti = dict(identity=identity, view=key, arm=arm, inputs_sha256=array_hash(ids, x, d),
                labels_sha256=array_hash(y), complete_mask_sha256=array_hash(pr['known']),
                training_sites=pr['training_sites'], producers=meta['training_producers'])
            folder = ROOT/cfg['output']/'trials'/key/arm
            receipt = folder/'complete.json'
            if receipt.exists():
                r = json.loads(receipt.read_text())
                assert r['identity'] == ti and r['fit']['complete']
                assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
                beat(state='cached_verified_complete', view=key, arm=arm)
                continue
            _, result = fit(x, y, d, data['sites'][ids], pr, arm=FORWARD[arm], seed=meta['seed'],
                settings=cfg['training'], identity=ti, directory=folder, resume=args.resume,
                stop_at=args.stop_at, heartbeat=lambda **v:beat(view=key, arm=arm, **v))
            if not result['complete']:
                beat(state='pilot_complete_not_matrix', view=key, arm=arm, step=result['step'])
                return
            path = folder/'checkpoint.pt'
            cp = torch.load(path, map_location='cpu', weights_only=False)
            check_matched(cp, refs[key, 'tempered'], cfg, meta['seed'], arm)
            assert_current(identity)
            immutable_json(receipt, dict(identity=ti, fit=result, rows=len(ids),
                supported_rows=int(pr['known'].sum()), checkpoint=str(path.relative_to(ROOT)),
                checkpoint_sha256=file_digest(path), matched_reference_draws=True))
    beat(state='training_call_complete')


def load_states(cfg, views, refs, identity):
    records, states = {}, dict(refs)
    for key, meta in views.items():
        for arm in cfg['training_arms']:
            r = json.loads((ROOT/cfg['output']/'trials'/key/arm/'complete.json').read_text())
            assert r['identity']['identity'] == identity and r['fit']['complete']
            assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
            cp = torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=False)
            assert cp['identity'] == r['identity']
            check_matched(cp, refs[key, 'tempered'], cfg, meta['seed'], arm)
            records[key, arm], states[key, arm] = r, cp
    return records, states


def validate_args(args):
    if (sum((args.audit_only, args.evaluate, args.verify)) > 1
            or ((args.audit_only or args.evaluate or args.verify)
                and (args.resume or args.view or args.arm or args.stop_at is not None))
            or (args.stop_at is not None and (not args.view or not args.arm or not 0 < args.stop_at <= 12000))):
        raise ValueError('Separate phases and explicit bounded pilot required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-only', 'evaluate', 'verify', 'resume'):
        p.add_argument('--'+name, action='store_true')
    p.add_argument('--view')
    p.add_argument('--arm', choices=('native', 'fraction'))
    p.add_argument('--stop-at', type=int)
    args = p.parse_args()
    validate_args(args)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg = json.loads((ROOT/CONFIG).read_text())
    root = ROOT/cfg['output']
    root.mkdir(parents=True, exist_ok=True)
    def beat(**v):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **v)
        json_write(root/'heartbeat.json', event)
        with (root/'events.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    signal.signal(signal.SIGTERM, lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, data, views, predictions, controls, prior, refs, identity = load()
        if args.view and args.view not in views:
            raise ValueError('Unregistered view')
        immutable_json(root/'identity.json', identity)
        if args.audit_only:
            beat(state='preflight_pass', bindings=len(identity['source_bindings']), new_paths=24, cached_references=12)
        elif args.evaluate or args.verify:
            from src.evaluation.m3w_cost_budget_matched_eval import evaluate
            records, states = load_states(cfg, views, refs, identity)
            evaluate(cfg, data, views, predictions, controls, prior, identity, records, states, beat, args.verify)
        else:
            train(cfg, data, views, refs, identity, args, beat)


if __name__ == '__main__':
    main()
