"""Published logistic deferral-loss adapter with matched frozen forecast banks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_multiaction_cost as parent
from src.world_model import m3w_regression_deferral as api
from src.evaluation import m3w_multiaction_cost_readout as readout

NAME = 'european_regression_deferral_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_regression_deferral_v1.json'; REG = PUBLIC/'registration.json'
sha, once, json_write, fingerprint = parent.sha, parent.once, parent.json_write, parent.fingerprint
FILES = (CONFIG, Path(__file__), ROOT/'src/world_model/m3w_regression_deferral.py',
         ROOT/'tests/test_m3w_regression_deferral.py', PUBLIC/'protocol.md')


def load():
    cfg = json.loads(CONFIG.read_text()); pc, reg, ancestor, data, jobs, identity = parent.admission()
    learned = parent.models()
    if cfg['arms'] != list(api.ARMS) or cfg['seeds'] != pc['seeds'] or cfg['risk_budget'] != .02:
        raise ValueError('Matched banks, seeds and unchanged2%evaluation required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_registration_sha256=sha(parent.REG), parent_training_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        contexts=sorted(reg['contexts']), seeds=cfg['seeds'], arms=cfg['arms'], fits=54,
        steps_per_fit=2000, method='Mao_Mohri_Zhong_ICML2024_Eq3_logistic_surrogate_adapter',
        native_partial_ADE_application_not_original_bounded_loss_theorem_verification=True,
        no_one_hot_oracle_label=True, source_forecaster_seed=43, independent_roles_opened=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    return cfg, pc, reg, data, jobs, identity, learned


def admission():
    values = load()
    for rel, digest in json.loads(REG.read_text())['bindings'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Frozen deferral adapter changed')
        parent.committed(ROOT/rel)
    parent.committed(REG)
    return values


def runtime(cfg):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume same checkpoint')))

    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Preserve10GiB disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['checkpoint_cap_bytes']:
            raise OSError('Registered checkpoint cap exceeded')
        if time.monotonic()-started > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume same roster')

    def beat(**event):
        guard(); event = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        json_write(PRIVATE/'heartbeat.json', event)
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)

    guard(); return started, beat, guard


def with_floor_features(scopes, data):
    for packets in scopes.values():
        for p in packets.values():
            floor_x, _ = parent.matched_features(data['geometry'][p['ids']], p['rollouts'][:, 1],
                p['floor_rollout'], p['floor_rollout'], p['args'][0][:, -1].astype(bool))
            p['deferral_x'] = np.concatenate((floor_x[:, None], p['x']), 1)
    return scopes


def training_data(packets, pr):
    names = sorted(packets)
    x = np.concatenate([packets[s]['deferral_x'] for s in names])
    cost = np.concatenate([np.column_stack((packets[s]['floor_ade'], packets[s]['ade'])) for s in names])
    scales = np.concatenate([packets[s]['scales'] for s in names])
    sites = np.concatenate([packets[s]['sites'] for s in names])
    allowed = np.concatenate([packets[s]['allowed'] for s in names])
    known = np.isfinite(cost).all(1)
    if not (known | np.isnan(cost).all(1)).all():
        raise ValueError('Identical expert outcome support required')
    groups = [np.flatnonzero(known & allowed.any(1) & (sites == s)) for s in names]
    z = np.clip((x-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
    return dict(z=torch.from_numpy(z), cost=torch.from_numpy(np.where(known[:, None], cost/scales[:, None], 0).astype(np.float32)),
        preprocess=pr, groups=groups, input_hashes={k:fingerprint(v) for k,v in
            dict(features=x, labels=cost, sites=sites, scales=scales, allowed=allowed).items()})


def checkpoint(context, seed, arm, pilot=False, direct=False):
    return PRIVATE/('pilot' if pilot else 'heads')/f'{context}_seed{seed}_{arm}{"_direct" if direct else ""}.pt.gz'


def train(pilot=False, resume=False):
    cfg, _, reg, data, jobs, identity, learned = admission()
    finish = PUBLIC/('pilot.json' if pilot else 'training_freeze.json')
    if finish.exists():
        raise ValueError('Completed phase immutable')
    if not pilot and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real matched-loss pilot required')
    started, beat, guard = runtime(cfg); receipts, measured = [], []
    contexts = sorted(reg['contexts'].items())[:1] if pilot else sorted(reg['contexts'].items())
    for context, items in contexts:
        packets = with_floor_features(parent.prepare_context(context, items, data, jobs, identity, beat, include_held=False), data)['fit']
        reference = api.read_checkpoint(ROOT/learned[context, 17, 'physical_only']['checkpoint']['path'])
        tensors = training_data(packets, reference['preprocess']); del packets
        for seed in cfg['seeds'][:1] if pilot else cfg['seeds']:
            states = []
            for arm in api.ARMS:
                path = checkpoint(context, seed, arm, pilot)
                kw = dict(data=tensors, arm=arm, settings=cfg['training'], seed=seed,
                    identity=dict(context=context, registration_sha256=sha(REG)), path=path, heartbeat=beat, guard=guard)
                if pilot and not path.exists():
                    api.fit(**kw, stop_at=cfg['pilot_resume_at'])
                result = api.fit(**kw, stop_at=cfg['pilot_steps'] if pilot else None,
                                 resume=path.exists() and (pilot or resume))
                if not any(not torch.equal(v, result['initial_model'][k]) for k,v in result['model'].items()):
                    raise ValueError('No actual deferral-head parameter update')
                states.append(result)
                if pilot:
                    direct_path = checkpoint(context, seed, arm, True, True)
                    direct = api.fit(**{**kw, 'path':direct_path}, stop_at=cfg['pilot_steps'],
                        resume=resume and direct_path.exists())
                    for key in result:
                        if key != 'seconds':
                            api.exact(direct[key], result[key])
                    measured.append(result['seconds'])
                else:
                    if result['draw_hash'] != learned[context, seed, arm]['draw_hash']:
                        raise ValueError('Source row draws must match preceding moment-head training exactly')
                    receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
                    once(receipt, dict(context=context, seed=seed, arm=arm, step=result['step'],
                        checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                        seconds=result['seconds'], parameters=result['parameters'], draw_hash=result['draw_hash'],
                        first_trace=result['trace'][0], final_trace=result['trace'][-1], unknown_rows_sampled=0,
                        preceding_moment_row_draws_exact=True, inner_held_used=False,
                        registration_sha256=sha(REG), result_source='fresh_native_torch_published_loss_adapter'))
                    receipts.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
                    beat(state='fit_complete', context=context, seed=seed, arm=arm, fits=len(receipts), total=54)
            for other in states[1:]:
                for key in ('initial_model', 'preprocess', 'draw_hash', 'sampler_rng', 'torch_rng', 'parameters'):
                    api.exact(states[0][key], other[key])
        del tensors
    if pilot:
        once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started, actual_updates=120,
            projected_primary_optimizer_seconds=sum(measured)*100*18,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            engineering_pass=True, exact_resume_all_arms=True, matched_initialization_and_draws=True,
            registration_sha256=sha(REG), held_scored=False))
    else:
        if len(receipts) != 54:
            raise ValueError('All54 final endpoints required')
        once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started, count=54, optimizer_updates=108000,
            fits=receipts, registration_sha256=sha(REG), inner_held_scored=False,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, independent_roles_opened=False))
    beat(state='pilot_complete' if pilot else 'training_complete', seconds=time.monotonic()-started)


def models():
    parent.committed(PUBLIC/'training_freeze.json')
    freeze = json.loads((PUBLIC/'training_freeze.json').read_text()); result = {}
    if freeze['count'] != 54 or freeze['registration_sha256'] != sha(REG) or freeze['inner_held_scored']:
        raise ValueError('Committed complete54-fit freeze required')
    for ref in freeze['fits']:
        path = ROOT/ref['path']; row = json.loads(path.read_text())
        if sha(path) != ref['sha256'] or sha(ROOT/row['checkpoint']['path']) != row['checkpoint']['sha256']:
            raise ValueError('Frozen deferral endpoint changed')
        key = row['context'], row['seed'], row['arm']
        if key in result or row['step'] != 2000:
            raise ValueError('Unique complete endpoint required')
        result[key] = row
    if len(result) != 54:
        raise ValueError('All54 models required')
    return result


def evaluate(verify=False, resume=False):
    cfg, pc, reg, data, jobs, identity, parents = admission(); learned = models()
    finish = PUBLIC/'complete.json'
    if finish.exists() and not verify:
        raise ValueError('Completed readout immutable')
    started, beat, guard = runtime(cfg); rows, refs = [], []
    _, origin_refs = parent.old.direct.ORIGINS.models(reg)
    for context, items in sorted(reg['contexts'].items()):
        scopes = with_floor_features(parent.prepare_context(context, items, data, jobs, identity, beat, include_held=True), data)
        for seed in cfg['seeds']:
            states = {a:api.read_checkpoint(ROOT/learned[context, seed, a]['checkpoint']['path']) for a in api.ARMS}
            cost_states = {a:api.read_checkpoint(ROOT/parents[context, seed, a]['checkpoint']['path']) for a in api.ARMS}
            origin_ref = origin_refs[context, seed, 'self']['checkpoint']
            origin = parent.old.base.core.read_checkpoint(ROOT/origin_ref['path'])
            for scope, packets in scopes.items():
                original = parent.old.direct.origin_predictions(origin, {s:p['args'] for s,p in packets.items()})
                for source, packet in sorted(packets.items()):
                    path = PUBLIC/'groups'/f'{context}_{source}_seed{seed}_{scope}.json'
                    if path.exists() and not (verify or resume):
                        raise ValueError('Explicit partial readout resume required')
                    old_path = parent.PUBLIC/'groups'/path.name; old = json.loads(old_path.read_text())
                    choices, ranks, predicted = {}, {}, {}
                    for arm in api.ARMS:
                        p = parent.api.predict(cost_states[arm], packet['x'], packet['envelope'], packet['scales'])
                        predicted['frozen_moment_'+arm] = p
                        old_choice, old_rank = parent.api.select_actions(p, packet['allowed'], arm=arm)
                        if fingerprint(old_choice) != old['action_sha256'][arm]:
                            raise ValueError('Frozen moment-policy action mismatch')
                        choices[arm+'_moment_parent'], ranks[arm+'_moment_parent'] = old_choice, old_rank
                        scores = api.predict(states[arm], packet['deferral_x'])
                        choices[arm], ranks[arm] = api.choices(scores, packet['allowed'], arm=arm)
                        safe = ((p[..., 0] > p[..., 1]) & (p[..., 2] > 0) & (p[..., 3] > 0)
                            & (p[..., 1] <= .02*p[..., 2]) & (p[..., 4] <= .02*p[..., 3]))
                        choices[arm+'_cost_gate'], ranks[arm+'_cost_gate'] = api.choices(scores, packet['allowed'], arm=arm, risk_allowed=safe)
                    op = original[source]; take = parent.old.base.actions(op, parent.old.base.shared.eligible(packet['args'])) & packet['allowed'][:, 6]
                    choices['frozen_origin_binary'] = np.where(take, 6, -1); ranks['frozen_origin_binary'] = op[:, 0]-op[:, 1]
                    choices['floor'] = np.full(len(take), -1, np.int64); ranks['floor'] = np.zeros(len(take))
                    result = readout.group_readout(packet, choices, ranks, predicted)
                    for arm in api.ARMS:
                        if result['policies'][arm+'_moment_parent'] != old['result']['policies'][arm]:
                            raise ValueError('Preceding model metrics differ')
                    value = dict(context=context, source=source, seed=seed, scope=scope, result=result,
                        action_sha256={a:fingerprint(c) for a,c in choices.items()},
                        parent_group_sha256=sha(old_path), registration_sha256=sha(REG),
                        training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                        result_source='fresh_source_internal_published_loss_adapter', independent_roles_opened=False)
                    if verify:
                        if value != json.loads(path.read_text()):
                            raise ValueError('Exact deferral readout replay differs')
                    else:
                        once(path, value)
                    rows.append(value); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
            beat(state='readout_progress', context=context, views=len(rows), total=144)
        del scopes
    summary = readout.summarize(rows, pc)
    summary.update(method='published_regression_deferral_logistic_surrogate_adapter',
        calibration_fields_are_frozen_moment_controls_not_classifier_confidence=True,
        original_paper_experiments_reproduced=False, bounded_loss_theorem_assumptions_verified=False)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()) or json.loads(finish.read_text())['groups'] != refs:
            raise ValueError('Exact deferral summary replay differs')
        once(PUBLIC/'verification.json', dict(exact_replay=True, groups=len(refs),
            summary_sha256=sha(PUBLIC/'summary.json'), registration_sha256=sha(REG)))
    else:
        once(PUBLIC/'summary.json', summary)
        once(finish, dict(groups=refs, summary_sha256=sha(PUBLIC/'summary.json'),
            registration_sha256=sha(REG), pid=os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, independent_roles_opened=False))
    json_write(PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=os.getpid(), seconds=time.monotonic()-started, exact_replay=verify))
    beat(state='readout_verified' if verify else 'readout_complete',
         full_budget_supported=summary['primary_full_roster_budget_supported'])


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
            evaluate(args.phase == 'verify', args.resume)


if __name__ == '__main__':
    main()
