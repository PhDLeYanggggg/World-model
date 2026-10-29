"""Matched shared-reference cost ablation over the frozen causal forecast bank."""
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
    raise RuntimeError('Use native arm64 .venv-pytorch before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_multiaction_cost as parent
from src.world_model import m3w_shared_reference_cost as api
from src.evaluation import m3w_multiaction_cost_readout as readout

NAME = 'european_shared_reference_cost_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_shared_reference_cost_v1.json'; REG = PUBLIC/'registration.json'
AUDIT = ROOT/'outputs/publication_readiness_2026_09/european_record_feasibility_audit_v1'
sha, once, json_write, fingerprint = parent.sha, parent.once, parent.json_write, parent.fingerprint
FILES = (CONFIG, Path(__file__), ROOT/'src/world_model/m3w_shared_reference_cost.py',
    ROOT/'tests/test_m3w_shared_reference_cost.py', PUBLIC/'protocol.md',
    AUDIT/'diagnosis.json', AUDIT/'verification.json')


def load():
    cfg = json.loads(CONFIG.read_text()); pc, reg, _, data, jobs, identity = parent.admission()
    learned = parent.models()
    check = json.loads((AUDIT/'verification.json').read_text())
    if (cfg['training'] != pc['training'] or cfg['risk_budget'] != .02
            or cfg['arms'] != list(parent.api.ARMS) or cfg['seeds'] != pc['seeds']
            or not check['exact_replay'] or check['diagnosis_sha256'] != sha(AUDIT/'diagnosis.json')):
        raise ValueError('Verified diagnosis and unchanged optimizer/loss/banks/budgets required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_registration_sha256=sha(parent.REG), parent_training_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        contexts=sorted(reg['contexts']), seeds=cfg['seeds'], arms=cfg['arms'], fits=54,
        steps_per_fit=2000, change='single_causal_floor_token_for_candidate_invariant_R_ER',
        parameters_matched_to_parent=True, no_new_neural_features_in_physical_only=True,
        floor_normalizer='frozen_physical_fitting_only_parent', forecaster_seed=43,
        same_source_draws_and_initial_weights_required=True, risk_budget=.02,
        independent_roles_opened=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    return cfg, pc, reg, data, jobs, identity, learned


def admission():
    values = load()
    for rel, digest in json.loads(REG.read_text())['bindings'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Registered shared-reference experiment changed')
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
            raise OSError('Checkpoint cap exceeded')
        if time.monotonic()-started > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h healthy-runtime boundary; resume same roster')
    def beat(**event):
        guard(); value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        json_write(PRIVATE/'heartbeat.json', value)
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
    guard(); return started, beat, guard


def floor_tokens(scopes, source):
    for packets in scopes.values():
        for packet in packets.values():
            floor, _ = parent.matched_features(source['geometry'][packet['ids']], packet['rollouts'][:, 1],
                packet['floor_rollout'], packet['floor_rollout'], packet['args'][0][:, -1].astype(bool))
            packet['floor_x'] = floor
    return scopes


def training_data(packets, *, shared):
    data = parent.combine(packets)
    if shared:
        floor = np.concatenate([packets[s]['floor_x'] for s in sorted(packets)])
        data['z'] = api.PairedCausalFeatures(data['z'],
            torch.from_numpy(parent.api.standardized(floor, data['preprocess'])))
        data['input_hashes']['floor_causal_features'] = fingerprint(floor)
    return data


def checkpoint(context, seed, arm, pilot=False, direct=False):
    return PRIVATE/('pilot' if pilot else 'heads')/f'{context}_seed{seed}_{arm}{"_direct" if direct else ""}.pt.gz'


def neutral(resume=False):
    cfg, _, reg, source, jobs, identity, parents = admission()
    if (PUBLIC/'neutral_replay.json').exists():
        raise ValueError('Completed neutral receipt immutable')
    started, beat, guard = runtime(cfg)
    context, items = sorted(reg['contexts'].items())[0]; seed, arm = 17, 'physical_neural'
    packets = parent.prepare_context(context, items, source, jobs, identity, beat, include_held=False)['fit']
    data = training_data(packets, shared=False); ref = parents[context, seed, arm]['checkpoint']
    old = parent.api.read_checkpoint(ROOT/ref['path']); path = PRIVATE/'neutral'/'checkpoint.pt.gz'
    with api.installed(False):
        result = parent.api.fit(data, arm=arm, settings=cfg['training'], seed=seed,
            identity=dict(context=context, registration_sha256=sha(REG)), path=path,
            heartbeat=beat, guard=guard, resume=resume)
    for key in old:
        if key not in ('identity', 'seconds'):
            parent.api.exact(old[key], result[key])
    once(PUBLIC/'neutral_replay.json', dict(pid=os.getpid(), seconds=time.monotonic()-started,
        exact_scientific_state_except_identity_runtime=True, updates=2000, reference=ref,
        checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path)), registration_sha256=sha(REG)))
    beat(state='full2000_neutral_exact')


def train(pilot=False, resume=False):
    cfg, _, reg, source, jobs, identity, parents = admission()
    finish = PUBLIC/('pilot.json' if pilot else 'training_freeze.json')
    if finish.exists():
        raise ValueError('Completed training phase immutable')
    if not pilot and (not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']
            or not json.loads((PUBLIC/'neutral_replay.json').read_text())['exact_scientific_state_except_identity_runtime']):
        raise ValueError('Real recovery pilot and full2000neutral required')
    started, beat, guard = runtime(cfg); receipts, measured = [], []
    contexts = sorted(reg['contexts'].items())[:1] if pilot else sorted(reg['contexts'].items())
    for context, items in contexts:
        packets = floor_tokens(parent.prepare_context(context, items, source, jobs, identity, beat, include_held=False), source)['fit']
        data = training_data(packets, shared=True); del packets
        with api.installed():
            for seed in cfg['seeds'][:1] if pilot else cfg['seeds']:
                states = []
                for arm in parent.api.ARMS:
                    path = checkpoint(context, seed, arm, pilot)
                    kw = dict(data=data, arm=arm, settings=cfg['training'], seed=seed,
                        identity=dict(context=context, registration_sha256=sha(REG)), path=path, heartbeat=beat, guard=guard)
                    if pilot and not path.exists():
                        parent.api.fit(**kw, stop_at=cfg['pilot_resume_at'])
                    result = parent.api.fit(**kw, stop_at=cfg['pilot_steps'] if pilot else None,
                        resume=path.exists() and (pilot or resume)); states.append(result)
                    if not any(not torch.equal(v, result['initial_model'][k]) for k,v in result['model'].items()):
                        raise ValueError('No actual parameter update')
                    if pilot:
                        direct_path = checkpoint(context, seed, arm, True, True)
                        direct = parent.api.fit(**{**kw, 'path':direct_path}, stop_at=cfg['pilot_steps'],
                            resume=resume and direct_path.exists())
                        for key in result:
                            if key != 'seconds':
                                parent.api.exact(direct[key], result[key])
                        measured.append(result['seconds'])
                    else:
                        prior = parent.api.read_checkpoint(ROOT/parents[context, seed, arm]['checkpoint']['path'])
                        for key in ('initial_model', 'preprocess', 'draw_hash', 'sampler_rng', 'torch_rng', 'parameters'):
                            parent.api.exact(result[key], prior[key])
                        receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
                        once(receipt, dict(context=context, seed=seed, arm=arm, step=result['step'],
                            checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                            parameters=result['parameters'], seconds=result['seconds'], draw_hash=result['draw_hash'],
                            first_trace=result['trace'][0], final_trace=result['trace'][-1],
                            parent_initial_draw_and_preprocess_exact=True, unknown_rows_sampled=0,
                            registration_sha256=sha(REG), result_source='fresh_native_torch_shared_reference_cost'))
                        receipts.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
                        beat(state='fit_complete', context=context, seed=seed, arm=arm, fits=len(receipts), total=54)
                for state in states[1:]:
                    for key in ('initial_model', 'preprocess', 'draw_hash', 'torch_rng', 'sampler_rng', 'parameters'):
                        parent.api.exact(states[0][key], state[key])
        del data
    if pilot:
        once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started, actual_updates=120,
            engineering_pass=True, exact_resume_all_arms=True, matched_initialization_and_draws=True,
            projected_primary_optimizer_seconds=sum(measured)*100*18,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, held_scored=False))
    else:
        if len(receipts) != 54:
            raise ValueError('All54 endpoints required')
        once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started, count=54, fits=receipts,
            optimizer_updates=108000, registration_sha256=sha(REG), inner_held_scored=False,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, independent_roles_opened=False))
    beat(state='pilot_complete' if pilot else 'training_complete')


def models():
    parent.committed(PUBLIC/'training_freeze.json')
    frozen = json.loads((PUBLIC/'training_freeze.json').read_text()); result = {}
    if frozen['count'] != 54 or frozen['inner_held_scored'] or frozen['registration_sha256'] != sha(REG):
        raise ValueError('Committed complete54-fit freeze required')
    for ref in frozen['fits']:
        path = ROOT/ref['path']; row = json.loads(path.read_text())
        if sha(path) != ref['sha256'] or sha(ROOT/row['checkpoint']['path']) != row['checkpoint']['sha256']:
            raise ValueError('Frozen shared-reference endpoint differs')
        key = row['context'], row['seed'], row['arm']
        if key in result or row['step'] != 2000:
            raise ValueError('Unique completed endpoint required')
        result[key] = row
    return result


def evaluate(verify=False, resume=False):
    cfg, pc, reg, source, jobs, identity, parents = admission(); learned = models()
    finish = PUBLIC/'complete.json'
    if finish.exists() and not verify:
        raise ValueError('Completed readout immutable')
    started, beat, guard = runtime(cfg); rows, refs = [], []
    _, origin_refs = parent.old.direct.ORIGINS.models(reg)
    for context, items in sorted(reg['contexts'].items()):
        scopes = floor_tokens(parent.prepare_context(context, items, source, jobs, identity, beat, include_held=True), source)
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
                        p = api.predict(states[arm], packet['x'], packet['floor_x'], packet['envelope'], packet['scales'])
                        predicted[arm] = p
                        for multiplier in (1., *pc['sensitivity_harm_multipliers']):
                            key = arm if multiplier == 1 else arm+'_harm'+str(int(multiplier))
                            choices[key], ranks[key] = parent.api.select_actions(p, packet['allowed'], arm=arm, harm_multiplier=multiplier)
                        prior = parent.api.predict(old_states[arm], packet['x'], packet['envelope'], packet['scales'])
                        key = arm+'_unshared_parent'
                        choices[key], ranks[key] = parent.api.select_actions(prior, packet['allowed'], arm=arm)
                        if fingerprint(choices[key]) != old['action_sha256'][arm]:
                            raise ValueError('Original comparator changed')
                    op = original[name]; take = parent.old.base.actions(op, parent.old.base.shared.eligible(packet['args'])) & packet['allowed'][:, 6]
                    choices['frozen_origin_binary'] = np.where(take, 6, -1); ranks['frozen_origin_binary'] = op[:, 0]-op[:, 1]
                    choices['floor'] = np.full(len(take), -1, np.int64); ranks['floor'] = np.zeros(len(take))
                    result = readout.group_readout(packet, choices, ranks, predicted)
                    for arm in parent.api.ARMS:
                        if result['policies'][arm+'_unshared_parent'] != old['result']['policies'][arm]:
                            raise ValueError('Parent metric replay differs')
                    left, right = choices['physical_neural_unshared_parent'], choices['physical_neural']
                    aa, bb = readout.matched_choices(left, right, ranks['physical_neural_unshared_parent'], ranks['physical_neural'],
                        packet['recordings'], packet['frames'], packet['ids'])
                    result['shared_minus_unshared'] = dict(full=readout.paired_bounds(packet, left, right),
                        matched=readout.paired_bounds(packet, aa, bb), matched_count=int((aa >= 0).sum()),
                        matched_role='utility_only_not_safety')
                    value = dict(context=context, source=name, seed=seed, scope=scope, result=result,
                        action_sha256={a:fingerprint(c) for a,c in choices.items()}, parent_group_sha256=sha(old_path),
                        shared_R_ER_candidate_invariant=True, registration_sha256=sha(REG),
                        training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                        result_source='fresh_source_internal_shared_reference_readout', independent_roles_opened=False)
                    if verify:
                        if value != json.loads(path.read_text()):
                            raise ValueError('Shared-reference readout replay differs')
                    else:
                        once(path, value)
                    rows.append(value); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
            beat(state='readout_progress', context=context, views=len(rows), total=144)
        del scopes
    summary = readout.summarize(rows, pc)
    for scope in summary['scopes']:
        selected = [r for r in rows if r['scope'] == scope]
        summary['scopes'][scope]['shared_minus_unshared'] = {mode+'_'+bound:readout.interval(selected,
            lambda r,m=mode,b=bound:r['shared_minus_unshared'][m][b], pc)
            for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')}
    summary.update(reference_shared=True, original_parameters_initialization_and_draws_matched=True)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()) or json.loads(finish.read_text())['groups'] != refs:
            raise ValueError('Exact shared-reference summary differs')
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
