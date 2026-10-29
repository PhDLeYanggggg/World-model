"""Matched continuous action-cost learning on verified SOURCE TRAIN packets."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts import run_m3w_reference_coverage as previous
from scripts import run_m3w_train_feasibility as old
from scripts.report_m3w_reference_coverage import bank
from scripts.train_m3w_temporal_auxiliary_portable import unpack
from src.evaluation.m3w_train_packet_transport import verified_packet_bytes
from src.evaluation.m3w_forecast_support import aligned_predictions
from src.evaluation.m3w_native_metrics import native_errors
from src.evaluation.m3w_support_action_diagnostic import real_intervention_mask, causal_bank_envelope
from src.world_model import m3w_multiaction_cost as api
from src.world_model.m3w_european_source_forecast import baseline_numpy, BASELINES
from src.world_model.m3w_floor_relative import matched_features
from src.world_model.m3w_fixed_floor_probe import targets
from src.world_model.m3w_source_checkpoint import source_partition
from src.world_model.m3w_source_forest import fingerprint

NAME = 'european_multiaction_cost_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_multiaction_cost_v1.json'
REG = PUBLIC/'registration.json'
FILES = (CONFIG, Path(__file__), ROOT/'src/world_model/m3w_multiaction_cost.py',
         ROOT/'src/evaluation/m3w_multiaction_cost_readout.py',
         ROOT/'tests/test_m3w_multiaction_cost.py', ROOT/'tests/test_m3w_multiaction_cost_readout.py',
         PUBLIC/'protocol.md')
sha, once, json_write = previous.digest, previous.immutable_json, previous.json_write


def committed(path):
    rel = str(Path(path).relative_to(ROOT))
    if subprocess.check_output(['git', 'show', 'HEAD:'+rel], cwd=ROOT) != Path(path).read_bytes():
        raise ValueError('Commit frozen experiment asset first: '+rel)


def load():
    cfg = json.loads(CONFIG.read_text())
    if (cfg['arms'] != list(api.ARMS) or cfg['seeds'] != [17, 29, 43]
            or cfg['risk_budget'] != .02 or cfg['forecaster_seed'] != 43):
        raise ValueError('Fixed comparison, seeds and risk budget required')
    _, _, data, jobs, identity = previous.load()
    reg, _, ancestor = old.prior.admission()
    if len(reg['contexts']) != cfg['contexts']:
        raise ValueError('All six original contexts required')
    support = previous.PUBLIC/'multiaction_support.json'
    doc = json.loads(support.read_text())
    if (doc['required_views'] != 148 or doc['source_forecaster_seed'] != 43
            or doc['summary']['physical_plus_original']['positive_feasible'] != 148):
        raise ValueError('Verified preceding full-roster support required')
    registration = dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        source_bank_probe=dict(path=str(support.relative_to(ROOT)), sha256=sha(support)),
        source_contexts=reg['contexts'], contexts=sorted(reg['contexts']), seeds=cfg['seeds'],
        arms=cfg['arms'], fits=54, steps_per_fit=cfg['training']['steps'],
        forecaster_seed=43, training_partition='frozen_SOURCE_TRAIN_whole_record_fit',
        validation_partition='previously_exposed_SOURCE_TRAIN_inner_held_exploratory',
        feature_schema='380_existing_causal_geometry_rollout_plus_7_action_identity',
        normalizer_uses_physical_fitting_actions_only=True, matched_initialization_draws=True,
        risk_budget=.02, independent_roles_opened=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)
    once(REG, registration)
    return cfg, reg, ancestor, data, jobs, identity


def admission():
    values = load()
    frozen = json.loads(REG.read_text())
    for rel, digest in frozen['bindings'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Registered code changed: '+rel)
        committed(ROOT/rel)
    committed(REG)
    return values


def runtime(cfg):
    torch.set_num_threads(cfg['cpu_threads']); torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume atomic checkpoint')))

    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Preserve 10GiB disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['checkpoint_cap_bytes']:
            raise OSError('Registered private checkpoint cap exceeded')
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume the same fixed roster')

    def beat(**event):
        guard()
        event = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        json_write(PRIVATE/'heartbeat.json', event)
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)

    guard()
    return start, beat, guard


def original_row_ids(data, item, args):
    identity = item['packet']['identity']; source = identity['source']
    ids = np.flatnonzero(data['sites'] == source)
    tr, _, partition = source_partition(data['recordings'][ids], data['frames'][ids], source)
    tid = ids[tr]
    if partition != identity['partition'] or fingerprint(tid) != identity['training_ids_sha256']:
        raise ValueError('Original SOURCE TRAIN identity changed')
    for a, b in ((data['sites'][tid], args[4]), (data['recordings'][tid].astype(str), args[5].astype(str)),
                 (data['frames'][tid], args[6])):
        np.testing.assert_array_equal(a, b)
    if len(tid) != len(args[0]):
        raise ValueError('One exact original trajectory row per packet row required')
    return tid


def action_packet(data, ids, args, neural, design):
    physical = [baseline_numpy(data['history'][ids], k)-data['origin'][ids, None] for k in range(6)]
    switch = args[0][:, -1]
    if not np.isin(switch, [0, 1]).all():
        raise ValueError('Original causal floor switch must be Boolean')
    floor = np.where(switch[:, None, None].astype(bool), physical[3], physical[1])
    rollouts = [*physical, neural]
    features, envelopes, labels, ades, fdes = [], [], [], [], []

    def errors(p):
        return native_errors(p.astype(float)+data['origin'][ids, None], data['target_eval'][ids],
                             data['valid'][ids], np.ones(len(ids)))

    floor_ade, floor_fde = errors(floor)
    eligible = old.base.shared.eligible(args)
    for candidate in rollouts:
        x, envelope = matched_features(data['geometry'][ids], physical[1], floor, candidate, switch.astype(bool))
        np.testing.assert_array_equal(envelope, causal_bank_envelope(physical[1], floor, candidate))
        ade, fde = errors(candidate)
        features.append(x); envelopes.append(envelope); ades.append(ade); fdes.append(fde)
        labels.append(targets(data['baseline_ade'][ids, 1], floor_ade, ade, design['easy_cut']))
    np.testing.assert_array_equal(features[6], args[0])
    np.testing.assert_array_equal(envelopes[6], args[1])
    np.testing.assert_array_equal(labels[6], args[2])
    allowed = np.stack([eligible & real_intervention_mask(p, floor) for p in rollouts], 1)
    return dict(x=np.stack(features, 1), envelope=np.stack(envelopes, 1), y=np.stack(labels, 1),
        ade=np.stack(ades, 1), fde=np.stack(fdes, 1), floor_ade=floor_ade, floor_fde=floor_fde,
        cv_ade=data['baseline_ade'][ids, 1], hard=data['baseline_ade'][ids, 1] >= design['hard_cut'],
        allowed=allowed, rollouts=np.stack(rollouts, 1), floor_rollout=floor,
        sites=np.asarray(args[4]), recordings=np.asarray(args[5]).astype(str), frames=np.asarray(args[6]),
        ids=ids, scales=np.full(len(ids), args[-1]['scale']), args=args,
        source_packet_features_labels_replayed=True)


def prepare_context(context, items, data, jobs, identity, beat, *, include_held):
    job = next(j for j in jobs if j['key'] == context.split('_controller')[0])
    if job['old_identity']['seed'] != 43:
        raise ValueError('Frozen forecaster seed43 only')
    bank_ids = job['design']['held_ids']
    ref = identity['controls'][job['key']]['ADE_only_prediction']
    original = bank(ref, bank_ids)
    scopes = {'fit': {}}
    if include_held:
        scopes['inner_held'] = {}
    for item in items:
        ref = item['packet']; source = ref['identity']['source']
        content, status = verified_packet_bytes(ref, previous.PRIVATE/'verified_TRAIN_packets',
            lambda: (_ for _ in ()).throw(RuntimeError('No remote fetch in this experiment')), allow_fetch=False)
        raw = unpack(content); del content
        ids = original_row_ids(data, item, raw)
        fit, held = old.base.shared.prepare(raw, item['partition'])
        for scope, args in (('fit', fit), ('inner_held', held)):
            if scope not in scopes:
                continue
            take = np.isin(raw[5].astype(str), item['partition']['fit_recordings' if scope == 'fit' else 'held_recordings'])
            row_ids = ids[take]
            neural = aligned_predictions(row_ids, bank_ids, original)
            scopes[scope][source] = action_packet(data, row_ids, args, neural, job['design'])
        beat(state='context_source_prepared', context=context, source=source,
             packet_transport=status, fitting_rows=len(fit[0]), held_scored=False)
    return scopes


def combine(packets):
    names = sorted(packets)
    fields = ('x', 'envelope', 'y', 'scales', 'sites', 'recordings', 'frames', 'allowed', 'ids')
    out = {key:np.concatenate([packets[name][key] for name in names]) for key in fields}
    pr = api.preprocessing(out['x'], out['envelope'], out['y'], out['scales'], out['sites'])
    known = np.isfinite(out['y']).all((1, 2))
    groups = [np.flatnonzero(known & out['allowed'].any(1) & (out['sites'] == name)) for name in names]
    if any(not len(group) for group in groups):
        raise ValueError('Every fitting source needs supervised causal action rows')
    return dict(preprocess=pr, z=torch.from_numpy(api.standardized(out['x'], pr)),
        envelope=torch.from_numpy((out['envelope']/out['scales'][:, None]).astype(np.float32)),
        truth=torch.from_numpy(np.where(known[:, None, None], out['y']/out['scales'][:, None, None], 0).astype(np.float32)),
        allowed=torch.from_numpy(out['allowed']), groups=groups,
        input_hashes={key:fingerprint(value) for key, value in out.items()},
        known_training_rows=int(known.sum()), unknown_training_rows=int((~known).sum()))


def checkpoint(context, seed, arm, pilot=False, direct=False):
    return PRIVATE/('pilot' if pilot else 'heads')/f'{context}_seed{seed}_{arm}{"_direct" if direct else ""}.pt.gz'


def train(pilot=False, resume=False):
    cfg, reg, _, data, jobs, identity = admission()
    finish = PUBLIC/('pilot.json' if pilot else 'training_freeze.json')
    if finish.exists():
        raise ValueError('Completed phase immutable; do not duplicate training')
    if not pilot and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real resume/matched pilot required')
    started, beat, guard = runtime(cfg); receipts = []; measured = []
    contexts = sorted(reg['contexts'].items())[:1] if pilot else sorted(reg['contexts'].items())
    for context, items in contexts:
        packets = prepare_context(context, items, data, jobs, identity, beat, include_held=False)['fit']
        tensors = combine(packets); del packets
        for seed in cfg['seeds'][:1] if pilot else cfg['seeds']:
            states = []
            for arm in api.ARMS:
                path = checkpoint(context, seed, arm, pilot)
                kw = dict(data=tensors, arm=arm, settings=cfg['training'], seed=seed,
                    identity=dict(context=context, registration_sha256=sha(REG)), path=path,
                    heartbeat=beat, guard=guard)
                if pilot and not path.exists():
                    api.fit(**kw, stop_at=cfg['pilot_resume_at'])
                state = api.fit(**kw, stop_at=cfg['pilot_steps'] if pilot else None,
                                resume=path.exists() and (pilot or resume))
                if not any(not torch.equal(v, state['initial_model'][k]) for k, v in state['model'].items()):
                    raise ValueError('No actual neural cost-head parameter update')
                states.append(state)
                if pilot:
                    direct = api.fit(**{**kw, 'path':checkpoint(context, seed, arm, True, True)},
                        stop_at=cfg['pilot_steps'], resume=resume and checkpoint(context, seed, arm, True, True).exists())
                    for key in state:
                        if key != 'seconds':
                            api.exact(direct[key], state[key])
                    measured.append(state['seconds'])
                else:
                    receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
                    once(receipt, dict(context=context, seed=seed, arm=arm, step=state['step'],
                        checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                        seconds=state['seconds'], parameters=state['parameters'], draw_hash=state['draw_hash'],
                        first_trace=state['trace'][0], final_trace=state['trace'][-1],
                        unknown_rows_sampled=state['unknown_rows_sampled'],
                        normalizer_uses_neural_fields=False, inner_held_used=False,
                        registration_sha256=sha(REG), result_source='fresh_native_torch_action_cost_training'))
                    receipts.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
                    beat(state='fit_complete', context=context, seed=seed, arm=arm, fits=len(receipts), total=54)
            for other in states[1:]:
                for key in ('initial_model', 'preprocess', 'draw_hash', 'row_draws', 'sampler_rng', 'torch_rng', 'parameters'):
                    api.exact(states[0][key], other[key])
        del tensors
    elapsed = time.monotonic()-started
    if pilot:
        once(finish, dict(pid=os.getpid(), seconds=elapsed, actual_updates=120,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            projected_primary_training_seconds=sum(measured)*cfg['training']['steps']/cfg['pilot_steps']*18,
            exact_interrupted_resume_all_arms=True, matched_initialization_draws_all_arms=True,
            num_workers=0, cpu_threads=4, engineering_pass=True, held_scored=False,
            registration_sha256=sha(REG)))
    else:
        if len(receipts) != 54:
            raise ValueError('All54 fitted endpoints required')
        once(finish, dict(pid=os.getpid(), seconds=elapsed, count=54, optimizer_updates=108000, fits=receipts,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            registration_sha256=sha(REG), inner_held_scored=False, independent_roles_opened=False))
    beat(state='pilot_complete' if pilot else 'training_complete', seconds=elapsed)


def models():
    committed(PUBLIC/'training_freeze.json')
    freeze = json.loads((PUBLIC/'training_freeze.json').read_text())
    if freeze['count'] != 54 or freeze['registration_sha256'] != sha(REG) or freeze['inner_held_scored']:
        raise ValueError('Full committed54-fit freeze required before readout')
    out = {}
    for ref in freeze['fits']:
        path = ROOT/ref['path']; receipt = json.loads(path.read_text())
        checkpoint_ref = receipt['checkpoint']
        if sha(path) != ref['sha256'] or sha(ROOT/checkpoint_ref['path']) != checkpoint_ref['sha256']:
            raise ValueError('Frozen neural-cost endpoint changed')
        key = receipt['context'], receipt['seed'], receipt['arm']
        if key in out or receipt['step'] != 2000 or receipt['unknown_rows_sampled']:
            raise ValueError('Unique complete endpoints and known-only regression required')
        out[key] = receipt
    if len(out) != 54:
        raise ValueError('Complete54 endpoint roster required')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase == 'register':
            load()
        elif args.phase in ('pilot', 'train'):
            train(args.phase == 'pilot', args.resume)
        else:
            from src.evaluation.m3w_multiaction_cost_readout import evaluate
            evaluate(sys.modules[__name__], verify=args.phase == 'verify', resume=args.resume)


if __name__ == '__main__':
    main()
