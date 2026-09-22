"""Matched neural squared-fraction control on frozen source-only ramp forecasts."""
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
    raise RuntimeError('Native arm64 required before importing Torch')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_forest_cost as parent
from scripts.run_m3w_temporal_intervention import training_arrays
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write, array_hash
from src.world_model.m3w_fraction_square_head import fit, OBJECTIVE
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch

CONFIG = 'configs/m3w_fraction_square_v1.json'
CODE = ('scripts/run_m3w_fraction_square.py', 'src/world_model/m3w_fraction_square_head.py',
        'src/evaluation/m3w_fraction_square_eval.py', 'tests/test_m3w_fraction_square_head.py')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    fc, old, data, views, predictions, refs, neural, forest_report, pid = parent.load()
    if (cfg['sites'] != old['sites'] or cfg['seeds'] != old['seeds'] or cfg['training'] != old['training']
            or cfg['arm'] != 'ramp' or any(cfg[k] for k in ('threshold_search', 'model_selection',
                'closed_role_readout', 'risk_calibration', 'independent_confirmation', 'deployment',
                'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Frozen source-only matched training settings required')
    # parent.load returns the preceding temporal report; load the completed forests explicitly.
    forest_report = json.loads((ROOT/fc['reports']/'analysis.json').read_text())
    assert forest_report['identity'] == pid
    ranking_root = ROOT/'outputs/publication_readiness_2026_09/risk_ranking_v1'
    ranking = json.loads((ranking_root/'analysis.json').read_text()); assert_current(ranking['identity'])
    assert ranking['identity']['parent_analysis_sha256'] == file_digest(ROOT/fc['reports']/'analysis.json')
    bindings = dict(ranking['identity']['source_bindings'])
    def bind(path, expected=None):
        actual = file_digest(ROOT/path)
        if (expected is not None and actual != expected) or (path in bindings and bindings[path] != actual):
            raise ValueError('Changed source artifact: '+path)
        bindings[path] = actual
    for name in ('analysis.json', 'replay.json', 'separate_verification.json'):
        bind(str((ranking_root/name).relative_to(ROOT)))
    for name in ('replay.json', 'separate_verification.json'):
        r = json.loads((ranking_root/name).read_text())
        assert r['all_checks_passed'] and r['analysis_sha256'] == file_digest(ranking_root/'analysis.json')
    for row in ranking['archives']: bind(row['path'], row['sha256'])
    bind('data/stage_cvpr2027_experiments/risk_ranking_v1/decisions_complete.json', ranking['decision_manifest_sha256'])
    for path in (CONFIG, cfg['registration'], *CODE): bind(path)
    identity = dict(source_bindings=bindings, config=cfg, architecture=platform.machine(), torch=torch.__version__,
        runtime=dict(torch_threads=4, interop_threads=1, num_workers=0),
        all_sites_development_exposed=True, closed_role_readout=False)
    assert_current(identity)
    return cfg, old, data, views, predictions, refs, neural, forest_report, ranking, identity


def train(cfg, old, data, views, refs, neural, identity, args, beat):
    for key, meta in views.items():
        if args.view and args.view != key: continue
        ids, x, y, d, pr, bits, w, q = training_arrays(meta, data, refs, key, old, 'ramp')
        reference = neural[key, 'ramp']
        for field in ('mean', 'std', 'known', 'weights', 'constant'):
            np.testing.assert_array_equal(reference['preprocess'][field], pr[field])
        np.testing.assert_array_equal(reference['loss_weights'], w)
        ti = dict(identity=identity, view=key, inputs_sha256=array_hash(ids, x, d, q),
            labels_sha256=array_hash(y), weights_sha256=array_hash(w),
            reference_draws_sha256=array_hash(reference['draws']), training_sites=pr['training_sites'],
            producers=meta['training_producers'])
        folder = ROOT/cfg['output']/'trials'/key; receipt = folder/'complete.json'
        if receipt.exists():
            r = json.loads(receipt.read_text())
            assert r['identity'] == ti and r['fit']['complete']
            assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
            beat(state='cached_verified_complete', view=key); continue
        _, result = fit(x, y, d, data['sites'][ids], pr, w, reference['draws'], seed=meta['seed'],
            settings=cfg['training'], identity=ti, directory=folder, resume=args.resume,
            stop_at=args.stop_at, heartbeat=lambda **v: beat(view=key, **v))
        if not result['complete']:
            beat(state='pilot_complete_not_matrix', view=key, step=result['step']); return
        path = folder/'checkpoint.pt'; cp = torch.load(path, map_location='cpu', weights_only=False)
        assert cp['step'] == 12000 and cp['objective'] == OBJECTIVE
        np.testing.assert_array_equal(cp['draws'], reference['draws'])
        assert_current(identity)
        immutable_json(receipt, dict(identity=ti, view=key, fit=result, rows=len(ids),
            supported_rows=int(pr['known'].sum()), checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=file_digest(path)))
    beat(state='training_call_complete')


def load_states(cfg, views, identity):
    records, states = {}, {}
    for key in views:
        r = json.loads((ROOT/cfg['output']/'trials'/key/'complete.json').read_text())
        assert r['identity']['identity'] == identity and r['fit']['complete']
        assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
        cp = torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=False)
        assert cp['identity'] == r['identity'] and cp['step'] == 12000 and cp['objective'] == OBJECTIVE
        records[key], states[key] = r, cp
    return records, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-only', 'evaluate', 'verify', 'resume'): parser.add_argument('--'+name, action='store_true')
    parser.add_argument('--view'); parser.add_argument('--stop-at', type=int); args = parser.parse_args()
    modes = sum((args.audit_only, args.evaluate, args.verify))
    if modes > 1 or (modes and (args.view or args.stop_at is not None or args.resume)):
        raise ValueError('Disjoint training and complete readout modes required')
    if args.stop_at and not args.view: raise ValueError('Pilot must name one fixed view')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg = json.loads((ROOT/CONFIG).read_text()); root = ROOT/cfg['output']; root.mkdir(parents=True, exist_ok=True)
    def beat(**row):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **row)
        json_write(root/'heartbeat.json', event)
        with (root/'events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, old, data, views, predictions, refs, neural, forest, ranking, identity = load()
        if args.view and args.view not in views: raise ValueError('Unknown view')
        immutable_json(root/'identity.json', identity)
        if args.audit_only:
            beat(state='preflight_pass', source_bindings=len(identity['source_bindings']), planned_new_fits=12)
        elif args.evaluate or args.verify:
            from src.evaluation.m3w_fraction_square_eval import evaluate
            records, states = load_states(cfg, views, identity)
            evaluate(cfg, data, views, predictions, states, records, forest, ranking, identity, beat, args.verify)
        else:
            train(cfg, old, data, views, refs, neural, identity, args, beat)


if __name__ == '__main__': main()
