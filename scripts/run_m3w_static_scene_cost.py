"""Matched causal static-scene ablation, immutable endpoints and exact controls."""
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
from src.world_model import m3w_scene_context_cost as api
from src.world_model.m3w_static_scene_context import source_polygons, scene_features
from src.evaluation import m3w_multiaction_cost_readout as readout

NAME = 'european_static_scene_cost_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_static_scene_cost_v1.json'; REG = PUBLIC/'registration.json'
AUDIT = ROOT/'outputs/publication_readiness_2026_09/european_static_scene_audit_v1'
ARCHIVE = ROOT/'external_data/EuropeanSquares_source_audit/02.Resources.zip'
PACKED = ROOT/'data/stage_cvpr2027_experiments/european_source_forecast_v1/packed/receipt.json'
sha, once, json_write, fingerprint = parent.sha, parent.once, parent.json_write, parent.fingerprint
FILES = (CONFIG, Path(__file__), ROOT/'src/world_model/m3w_scene_context_cost.py',
    ROOT/'tests/test_m3w_scene_context_cost.py', ROOT/'src/world_model/m3w_static_scene_context.py',
    PUBLIC/'protocol.md', AUDIT/'audit.json', AUDIT/'verification.json')


def load():
    cfg = json.loads(CONFIG.read_text()); pc, reg, _, source, jobs, identity = parent.admission()
    audit = json.loads((AUDIT/'audit.json').read_text()); check = json.loads((AUDIT/'verification.json').read_text())
    if (not check['exact_replay'] or check['audit_sha256'] != sha(AUDIT/'audit.json')
            or audit['unique_recordings'] != 74 or audit['outside_expanded_bbox_over_contexts'] != 0
            or cfg['training'] != pc['training'] or cfg['seeds'] != pc['seeds'] or cfg['risk_budget'] != .02):
        raise ValueError('Verified TRAIN mapping and unchanged matched training required')
    packed = json.loads(PACKED.read_text()); polygons = source_polygons(ARCHIVE, packed['records'])
    once(REG, dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        static_resources_sha256=sha(ARCHIVE), packed_receipt_sha256=sha(PACKED),
        parent_registration_sha256=sha(parent.REG), parent_training_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        contexts=sorted(reg['contexts']), arms=cfg['arms'], seeds=cfg['seeds'], fits=54, steps_per_fit=2000,
        motion_fields=380, scene_fields=18, parameters=26309, additional_zero_initialized_weights=1152,
        legacy_weights_and_row_draws_matched=True, normalization='physical_fitting_only', risk_budget=.02,
        independent_roles_opened=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    return cfg, pc, reg, source, jobs, identity, parent.models(), packed, polygons


def admission():
    values = load()
    for rel, digest in json.loads(REG.read_text())['bindings'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Registered scene experiment changed')
        parent.committed(ROOT/rel)
    parent.committed(REG); return values


def runtime(cfg):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume same atomic checkpoint')))
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Preserve10GiB reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['checkpoint_cap_bytes']:
            raise OSError('Private checkpoint cap exceeded')
        if time.monotonic()-started > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h healthy-runtime boundary; resume fixed roster')
    def beat(**event):
        guard(); value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        json_write(PRIVATE/'heartbeat.json', value)
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
    guard(); return started, beat, guard


def scene_packets(scopes, source, packed, polygons):
    for packets in scopes.values():
        for packet in packets.values():
            n = len(packet['ids']); scene = np.empty((n, 7, 18), np.float32)
            for ri in np.unique(source['recordings'][packet['ids']]):
                at = np.flatnonzero(source['recordings'][packet['ids']] == ri); ids = packet['ids'][at]
                polygon = polygons[packed['records'][int(ri)]]['vertices']
                for action in range(7):
                    scene[at, action] = scene_features(history=source['history'][ids], polygon=polygon,
                        rollout=packet['rollouts'][at, action]+source['origin'][ids, None])
            packet['scene'] = scene
    return scopes


def training_data(packets):
    data = parent.combine(packets); names = sorted(packets)
    scene = np.concatenate([packets[s]['scene'] for s in names])
    envelope, labels, scales, sites = [np.concatenate([packets[s][key] for s in names])
        for key in ('envelope', 'y', 'scales', 'sites')]
    pr = api.preprocessing(data['preprocess'], scene, envelope, labels, scales, sites)
    normalized = np.clip((scene-pr['mean'][-18:])/pr['std'][-18:], -pr['clip'], pr['clip']).astype(np.float32)
    data['z'] = api.SceneCausalFeatures(data['z'], torch.from_numpy(normalized))
    data['preprocess'] = pr; data['input_hashes']['scene_causal_features'] = fingerprint(scene)
    return data


def checkpoint(context, seed, arm, pilot=False, direct=False):
    return PRIVATE/('pilot' if pilot else 'heads')/f'{context}_seed{seed}_{arm}{"_direct" if direct else ""}.pt.gz'


def neutral(resume=False):
    cfg, _, reg, source, jobs, identity, parents, packed, polygons = admission()
    if (PUBLIC/'neutral_replay.json').exists():
        raise ValueError('Completed neutral immutable')
    started, beat, guard = runtime(cfg); context, items = sorted(reg['contexts'].items())[0]
    packets = scene_packets(parent.prepare_context(context, items, source, jobs, identity, beat, include_held=False),
        source, packed, polygons)['fit']
    data = training_data(packets); ref = parents[context, 17, 'physical_neural']['checkpoint']
    old = parent.api.read_checkpoint(ROOT/ref['path'])
    with api.installed(False):
        state = parent.api.fit(data, arm='physical_neural', settings=cfg['training'], seed=17,
            identity=dict(registration_sha256=sha(REG)), path=PRIVATE/'neutral'/'checkpoint.pt.gz',
            heartbeat=beat, guard=guard, resume=resume)
    api.assert_neutral_state(old, state)
    once(PUBLIC/'neutral_replay.json', dict(pid=os.getpid(), seconds=time.monotonic()-started,
        updates=2000, legacy_weights_Adam_draws_trace_exact=True, scene_weights_zero=True,
        reference=ref, registration_sha256=sha(REG)))
    beat(state='neutral2000_exact')


def train(pilot=False, resume=False):
    cfg, _, reg, source, jobs, identity, parents, packed, polygons = admission()
    finish = PUBLIC/('pilot.json' if pilot else 'training_freeze.json')
    if finish.exists():
        raise ValueError('Completed phase immutable')
    if not pilot and (not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']
            or not json.loads((PUBLIC/'neutral_replay.json').read_text())['legacy_weights_Adam_draws_trace_exact']):
        raise ValueError('Real recovery and full disabled neutral required')
    started, beat, guard = runtime(cfg); refs, times = [], []
    contexts = sorted(reg['contexts'].items())[:1] if pilot else sorted(reg['contexts'].items())
    for context, items in contexts:
        packets = scene_packets(parent.prepare_context(context, items, source, jobs, identity, beat, include_held=False),
            source, packed, polygons)['fit']
        data = training_data(packets); del packets
        with api.installed():
            for seed in cfg['seeds'][:1] if pilot else cfg['seeds']:
                states = []
                for arm in parent.api.ARMS:
                    path = checkpoint(context, seed, arm, pilot)
                    kw = dict(data=data, arm=arm, settings=cfg['training'], seed=seed,
                        identity=dict(context=context, registration_sha256=sha(REG)), path=path,
                        heartbeat=beat, guard=guard)
                    if pilot and not path.exists():
                        parent.api.fit(**kw, stop_at=cfg['pilot_resume_at'])
                    state = parent.api.fit(**kw, stop_at=cfg['pilot_steps'] if pilot else None,
                        resume=path.exists() and (pilot or resume)); states.append(state)
                    if not torch.count_nonzero(state['model']['scene_weight']):
                        raise ValueError('No actual scene-weight update')
                    if pilot:
                        direct_path = checkpoint(context, seed, arm, True, True)
                        direct = parent.api.fit(**{**kw, 'path':direct_path}, stop_at=cfg['pilot_steps'],
                            resume=resume and direct_path.exists())
                        for key in state:
                            if key != 'seconds':
                                parent.api.exact(state[key], direct[key])
                        times.append(state['seconds'])
                    else:
                        prior = parent.api.read_checkpoint(ROOT/parents[context, seed, arm]['checkpoint']['path'])
                        for key in ('draw_hash', 'sampler_rng', 'torch_rng'):
                            parent.api.exact(state[key], prior[key])
                        for key in prior['initial_model']:
                            parent.api.exact(state['initial_model'][key], prior['initial_model'][key])
                        for key in prior['preprocess']:
                            value = state['preprocess'][key]
                            parent.api.exact(value[:380] if key in ('mean', 'std') else value, prior['preprocess'][key])
                        receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
                        once(receipt, dict(context=context, seed=seed, arm=arm, step=state['step'],
                            checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                            parameters=state['parameters'], seconds=state['seconds'], draw_hash=state['draw_hash'],
                            first_trace=state['trace'][0], final_trace=state['trace'][-1],
                            original_motion_initialization_preprocess_and_draws_exact=True,
                            registration_sha256=sha(REG), result_source='fresh_native_torch_static_scene_cost'))
                        refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
                        beat(state='fit_complete', fits=len(refs), total=54, context=context, seed=seed, arm=arm)
                for other in states[1:]:
                    for key in ('initial_model', 'preprocess', 'draw_hash', 'sampler_rng', 'torch_rng', 'parameters'):
                        parent.api.exact(states[0][key], other[key])
        del data
    if pilot:
        once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started, actual_updates=120,
            engineering_pass=True, exact_resume_all_arms=True, matched_initialization_and_draws=True,
            projected_primary_optimizer_seconds=sum(times)*100*18,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, held_scored=False))
    else:
        if len(refs) != 54:
            raise ValueError('All54endpoints required')
        once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started, fits=refs, count=54,
            optimizer_updates=108000, registration_sha256=sha(REG), inner_held_scored=False,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, independent_roles_opened=False))
    beat(state='pilot_complete' if pilot else 'training_complete')


def models():
    parent.committed(PUBLIC/'training_freeze.json'); frozen = json.loads((PUBLIC/'training_freeze.json').read_text())
    if frozen['count'] != 54 or frozen['inner_held_scored'] or frozen['registration_sha256'] != sha(REG):
        raise ValueError('Complete54fitfreeze required')
    result = {}
    for ref in frozen['fits']:
        row = json.loads((ROOT/ref['path']).read_text()); key = row['context'], row['seed'], row['arm']
        if sha(ROOT/ref['path']) != ref['sha256'] or sha(ROOT/row['checkpoint']['path']) != row['checkpoint']['sha256']:
            raise ValueError('Frozen endpoint changed')
        if key in result or row['step'] != 2000:
            raise ValueError('Unique final endpoints required')
        result[key] = row
    return result


def evaluate(verify=False, resume=False):
    cfg, pc, reg, source, jobs, identity, parents, packed, polygons = admission(); learned = models()
    finish = PUBLIC/'complete.json'
    if finish.exists() and not verify:
        raise ValueError('Completed readout immutable')
    started, beat, guard = runtime(cfg); rows, refs = [], []
    _, origin_refs = parent.old.direct.ORIGINS.models(reg)
    for context, items in sorted(reg['contexts'].items()):
        scopes = scene_packets(parent.prepare_context(context, items, source, jobs, identity, beat, include_held=True),
            source, packed, polygons)
        for seed in cfg['seeds']:
            states = {a:parent.api.read_checkpoint(ROOT/learned[context, seed, a]['checkpoint']['path']) for a in parent.api.ARMS}
            old_states = {a:parent.api.read_checkpoint(ROOT/parents[context, seed, a]['checkpoint']['path']) for a in parent.api.ARMS}
            origin = parent.old.base.core.read_checkpoint(ROOT/origin_refs[context, seed, 'self']['checkpoint']['path'])
            for scope, packets in scopes.items():
                original = parent.old.direct.origin_predictions(origin, {s:p['args'] for s,p in packets.items()})
                for name, packet in sorted(packets.items()):
                    path = PUBLIC/'groups'/f'{context}_{name}_seed{seed}_{scope}.json'
                    if path.exists() and not (verify or resume):
                        raise ValueError('Explicit partial readout resume required')
                    old_path = parent.PUBLIC/'groups'/path.name; old = json.loads(old_path.read_text())
                    choices, ranks, predicted = {}, {}, {}
                    for arm in parent.api.ARMS:
                        p = api.predict(states[arm], packet['x'], packet['scene'], packet['envelope'], packet['scales'])
                        predicted[arm] = p
                        for multiplier in (1., *pc['sensitivity_harm_multipliers']):
                            key = arm if multiplier == 1 else arm+'_harm'+str(int(multiplier))
                            choices[key], ranks[key] = parent.api.select_actions(p, packet['allowed'], arm=arm, harm_multiplier=multiplier)
                        prior = parent.api.predict(old_states[arm], packet['x'], packet['envelope'], packet['scales'])
                        key = arm+'_no_scene_parent'
                        choices[key], ranks[key] = parent.api.select_actions(prior, packet['allowed'], arm=arm)
                        if fingerprint(choices[key]) != old['action_sha256'][arm]:
                            raise ValueError('Original no-scene choices differ')
                    op = original[name]; take = parent.old.base.actions(op, parent.old.base.shared.eligible(packet['args'])) & packet['allowed'][:, 6]
                    choices['frozen_origin_binary'] = np.where(take, 6, -1); ranks['frozen_origin_binary'] = op[:, 0]-op[:, 1]
                    choices['floor'] = np.full(len(take), -1, np.int64); ranks['floor'] = np.zeros(len(take))
                    result = readout.group_readout(packet, choices, ranks, predicted)
                    for arm in parent.api.ARMS:
                        if result['policies'][arm+'_no_scene_parent'] != old['result']['policies'][arm]:
                            raise ValueError('Original no-scene metric replay differs')
                    left, right = choices['physical_neural_no_scene_parent'], choices['physical_neural']
                    aa, bb = readout.matched_choices(left, right, ranks['physical_neural_no_scene_parent'], ranks['physical_neural'],
                        packet['recordings'], packet['frames'], packet['ids'])
                    result['scene_minus_no_scene'] = dict(full=readout.paired_bounds(packet, left, right),
                        matched=readout.paired_bounds(packet, aa, bb), matched_count=int((aa >= 0).sum()),
                        matched_role='utility_only_not_safety')
                    value = dict(context=context, source=name, seed=seed, scope=scope, result=result,
                        action_sha256={a:fingerprint(c) for a,c in choices.items()}, parent_group_sha256=sha(old_path),
                        registration_sha256=sha(REG), training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                        result_source='fresh_source_internal_static_scene_readout', independent_roles_opened=False)
                    if verify:
                        if value != json.loads(path.read_text()):
                            raise ValueError('Static-scene readout replay differs')
                    else:
                        once(path, value)
                    rows.append(value); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
            beat(state='readout_progress', context=context, views=len(rows), total=144)
        del scopes
    summary = readout.summarize(rows, pc)
    for scope in summary['scopes']:
        selected = [r for r in rows if r['scope'] == scope]
        summary['scopes'][scope]['scene_minus_no_scene'] = {mode+'_'+bound:readout.interval(selected,
            lambda r,m=mode,b=bound:r['scene_minus_no_scene'][m][b], pc)
            for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')}
    summary.update(static_scene_context=True, matched_legacy_weights_and_draws=True, additional_weights=1152)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()) or json.loads(finish.read_text())['groups'] != refs:
            raise ValueError('Exact static-scene summary differs')
        once(PUBLIC/'verification.json', dict(exact_replay=True, groups=144, summary_sha256=sha(PUBLIC/'summary.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, groups=refs,
            summary_sha256=sha(PUBLIC/'summary.json'), registration_sha256=sha(REG), independent_roles_opened=False))
    json_write(PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=os.getpid(), seconds=time.monotonic()-started))
    beat(state='readout_verified' if verify else 'readout_complete', full_budget=summary['primary_full_roster_budget_supported'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'neutral', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase == 'register':
            load()
        elif args.phase == 'neutral':
            neutral(args.resume)
        elif args.phase in ('pilot', 'train'):
            train(args.phase == 'pilot', args.resume)
        else:
            evaluate(args.phase == 'verify', args.resume)


if __name__ == '__main__':
    main()
