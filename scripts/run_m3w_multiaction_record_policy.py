"""Bounded exact-record decision repair; frozen forecasts and cost heads."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time
import resource
import shutil
import signal

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from scripts import run_m3w_multiaction_cost as parent
from src.world_model import m3w_multiaction_record_policy as api
from src.world_model.m3w_causal_descriptor_head import descriptors
from src.evaluation import m3w_multiaction_cost_readout as readout

NAME = 'european_multiaction_record_policy_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_multiaction_record_policy_v1.json'
REG = PUBLIC/'registration.json'
sha, once, json_write, fingerprint = parent.sha, parent.once, parent.json_write, parent.fingerprint
FILES = (CONFIG, Path(__file__), ROOT/'src/world_model/m3w_multiaction_record_policy.py',
         ROOT/'tests/test_m3w_multiaction_record_policy.py', PUBLIC/'protocol.md')


def load():
    cfg = json.loads(CONFIG.read_text())
    if cfg['risk_budget'] != .02 or cfg['arms'] != list(api.ARMS) or cfg['seeds'] != [17, 29, 43]:
        raise ValueError('Unchanged banks, seeds and2%risk required')
    pc, reg, ancestor, data, jobs, identity = parent.admission()
    learned = parent.models()
    check = json.loads((parent.PUBLIC/'verification.json').read_text())
    if not check['exact_replay'] or check['summary_sha256'] != sha(parent.PUBLIC/'summary.json'):
        raise ValueError('Exact preceding negative readout required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_registration_sha256=sha(parent.REG), parent_training_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        parent_summary_sha256=sha(parent.PUBLIC/'summary.json'), parent_verification_sha256=sha(parent.PUBLIC/'verification.json'),
        contexts=sorted(reg['contexts']), seeds=cfg['seeds'], arms=cfg['arms'], fits=54,
        steps_per_fit=cfg['training']['steps'], gradient_estimator='exact_full_record_chunked_not_sampled',
        cost_heads_and_forecaster_frozen=True, reference_event_roster_and_eligibility_unchanged=True,
        risk_budget=.02, positive_real_easy_support_training_constraint=True,
        independent_roles_opened=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    return cfg, pc, reg, ancestor, data, jobs, identity, learned


def admission():
    values = load()
    doc = json.loads(REG.read_text())
    for rel, digest in doc['bindings'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Frozen decision experiment code changed')
        parent.committed(ROOT/rel)
    parent.committed(REG)
    return values


def runtime(cfg):
    torch.set_num_threads(cfg['cpu_threads']); torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume atomic checkpoint')))

    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Preserve10GiB disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['checkpoint_cap_bytes']:
            raise OSError('Registered decision checkpoint cap exceeded')
        if time.monotonic()-started > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume same roster')

    def beat(**event):
        guard()
        value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        json_write(PRIVATE/'heartbeat.json', value)
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)

    guard()
    return started, beat, guard


def context_packets(context, items, data, jobs, identity, beat, include_held=False):
    scopes = parent.prepare_context(context, items, data, jobs, identity, beat, include_held=include_held)
    for packets in scopes.values():
        for packet in packets.values():
            packet['descriptors'] = np.stack([descriptors(data['geometry'][packet['ids']],
                packet['floor_rollout'], packet['rollouts'][:, action], packet['x'][:, action], packet['args'][-1])
                for action in range(7)], 1)
    return scopes


def causal_inputs(packet, cost_state, arm):
    p = parent.api.predict(cost_state, packet['x'], packet['envelope'], packet['scales'])
    choice, rank = parent.api.select_actions(p, packet['allowed'], arm=arm)
    anchor = choice[:, None] == np.arange(7)[None]
    features = np.concatenate((p/packet['scales'][:, None, None], packet['descriptors'],
                               (packet['envelope']/packet['scales'][:, None])[..., None]), -1)
    if features.shape != (len(choice), 7, 12) or not np.isfinite(features).all():
        raise ValueError('Finite past-only descriptor/moment schema required')
    return features, anchor, choice, rank, p


def prepare(context, seed, arm, packets, parent_receipt):
    state = api.read_checkpoint(ROOT/parent_receipt['checkpoint']['path'])
    fields = {key:[] for key in ('features', 'anchor', 'allowed', 'y', 'envelope', 'scales', 'sites', 'recordings')}
    audit = []
    for source, packet in sorted(packets.items()):
        features, anchor, choice, _, p = causal_inputs(packet, state, arm)
        old_group = parent.PUBLIC/'groups'/f'{context}_{source}_seed{seed}_fit.json'
        old = json.loads(old_group.read_text())
        if (fingerprint(choice) != old['action_sha256'][arm]
                or fingerprint(p) != old['prediction_sha256'][arm]):
            raise ValueError('Preceding fitting decisions/moments do not reproduce')
        score = old['result']['policies'][arm]['policy']
        known = np.isfinite(packet['y']).all((1, 2)); yes = (choice >= 0) & known
        indices = np.where(choice >= 0, choice, 0)
        predicted = p[np.arange(len(choice)), indices][yes].sum(0)
        observed = packet['y'][np.arange(len(choice)), indices][yes].sum(0)
        audit.append(dict(source=source, parent_action_sha256=fingerprint(choice), parent_group_sha256=sha(old_group),
            known_selected=int(yes.sum()), selected_unknown=score['selected_unknown'],
            predicted_known_selected_moment_sum=predicted.tolist(), observed_known_selected_moment_sum=observed.tolist(),
            actual_known_all_positive_risk=score['known_positive_risk'],
            actual_known_easy_positive_risk=score['known_easy_positive_risk']))
        for key in fields:
            fields[key].append(features if key == 'features' else anchor if key == 'anchor' else packet[key])
    arrays = {key:np.concatenate(values) for key, values in fields.items()}
    binding = dict(context=context, seed=seed, arm=arm, registration_sha256=sha(REG),
        parent_cost_checkpoint_sha256=parent_receipt['checkpoint']['sha256'],
        input_hashes={key:fingerprint(value) for key,value in arrays.items()})
    data = api.prepare(arrays['features'], arrays['anchor'], arrays['allowed'], arrays['y'], arrays['envelope'],
                       arrays['scales'], arrays['sites'], arrays['recordings'], arm=arm, identity=binding)
    once(PUBLIC/'parent_fitting_audit'/f'{context}_seed{seed}_{arm}.json',
        dict(identity=binding, sources=audit, fitting_only=True, no_optimizer_update=True,
             preceding_action_and_cost_replay_exact=True))
    return data


def checkpoint(context, seed, arm, pilot=False, direct=False):
    return PRIVATE/('pilot' if pilot else 'heads')/f'{context}_seed{seed}_{arm}{"_direct" if direct else ""}.pt.gz'


def train(pilot=False, resume=False):
    cfg, _, reg, _, source, jobs, identity, learned = admission()
    finish = PUBLIC/('pilot.json' if pilot else 'training_freeze.json')
    if finish.exists():
        raise ValueError('Completed phase immutable')
    if not pilot and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real exact-record pilot required')
    started, beat, guard = runtime(cfg); receipts, measured = [], []
    contexts = sorted(reg['contexts'].items())[:1] if pilot else sorted(reg['contexts'].items())
    for context, items in contexts:
        packets = context_packets(context, items, source, jobs, identity, beat)['fit']
        for seed in cfg['seeds'][:1] if pilot else cfg['seeds']:
            states = []
            for arm in api.ARMS:
                data = prepare(context, seed, arm, packets, learned[context, seed, arm])
                path = checkpoint(context, seed, arm, pilot)
                kwargs = dict(data=data, arm=arm, settings=cfg['training'], seed=seed, path=path,
                              heartbeat=beat, guard=guard)
                if pilot and not path.exists():
                    api.fit(**kwargs, stop_at=cfg['pilot_resume_at'])
                result = api.fit(**kwargs, stop_at=cfg['pilot_steps'] if pilot else None,
                                 resume=path.exists() and (pilot or resume))
                if not any(not torch.equal(v, result['initial_model'][k]) for k,v in result['model'].items()):
                    raise ValueError('No actual decision-head parameter update')
                states.append(result)
                if pilot:
                    direct_path = checkpoint(context, seed, arm, True, True)
                    direct = api.fit(**{**kwargs, 'path':direct_path}, stop_at=cfg['pilot_steps'],
                        resume=resume and direct_path.exists())
                    for key in result:
                        if key != 'seconds':
                            api.exact(direct[key], result[key])
                    measured.append(result['seconds'])
                else:
                    receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
                    once(receipt, dict(context=context, seed=seed, arm=arm, step=result['step'],
                        checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                        seconds=result['seconds'], parameters=result['parameters'], full_gradient_rows=result['full_gradient_rows'],
                        first_trace=result['trace'][0], final_trace=result['trace'][-1],
                        registration_sha256=sha(REG), exact_whole_record_gradient=True, inner_held_used=False,
                        result_source='fresh_native_torch_record_aware_decision_training'))
                    receipts.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
                    beat(state='fit_complete', context=context, seed=seed, arm=arm, fits=len(receipts), total=54)
                del data
            for other in states[1:]:
                for key in ('initial_model', 'record_keys', 'record_sizes', 'record_risk_scales',
                            'minimum_real_easy_support', 'torch_rng', 'parameters', 'full_gradient_rows'):
                    # Different action banks can have a different minimum real easy row.
                    if key == 'minimum_real_easy_support':
                        continue
                    api.exact(states[0][key], other[key])
        del packets
    if pilot:
        once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started, actual_updates=60,
            projected_primary_optimizer_seconds=sum(measured)*cfg['training']['steps']/cfg['pilot_steps']*18,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            exact_resume_all_arms=True, matched_initialization_full_row_population=True,
            engineering_pass=True, held_scored=False, registration_sha256=sha(REG)))
    else:
        if len(receipts) != 54:
            raise ValueError('Full54-fit roster required')
        once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started, count=54, optimizer_updates=27000,
            fits=receipts, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            registration_sha256=sha(REG), inner_held_scored=False, independent_roles_opened=False))
    beat(state='pilot_complete' if pilot else 'training_complete', seconds=time.monotonic()-started)


def models():
    parent.committed(PUBLIC/'training_freeze.json')
    freeze = json.loads((PUBLIC/'training_freeze.json').read_text()); result = {}
    if freeze['count'] != 54 or freeze['registration_sha256'] != sha(REG) or freeze['inner_held_scored']:
        raise ValueError('Complete committed54-fit freeze required before readout')
    for ref in freeze['fits']:
        path = ROOT/ref['path']; row = json.loads(path.read_text())
        if sha(path) != ref['sha256'] or sha(ROOT/row['checkpoint']['path']) != row['checkpoint']['sha256']:
            raise ValueError('Frozen decision-head endpoint changed')
        key = row['context'], row['seed'], row['arm']
        if key in result or row['step'] != 500:
            raise ValueError('Unique complete endpoints required')
        result[key] = row
    if len(result) != 54:
        raise ValueError('Complete model roster required')
    return result


def comparison(packet, choices, ranks, left, right):
    aa, bb = readout.matched_choices(choices[left], choices[right], ranks[left], ranks[right],
        packet['recordings'], packet['frames'], packet['ids'])
    return dict(full=readout.paired_bounds(packet, choices[left], choices[right]),
        matched=readout.paired_bounds(packet, aa, bb), matched_count=int((aa >= 0).sum()),
        matched_empty=not np.any(aa >= 0), matched_role='utility_only_not_safety')


def evaluate(verify=False, resume=False):
    cfg, pc, reg, _, data, jobs, identity, parents = admission(); learned = models()
    finish = PUBLIC/'complete.json'
    if finish.exists() and not verify:
        raise ValueError('Completed readout immutable; use verify')
    started, beat, guard = runtime(cfg); rows, refs = [], []
    _, origin_refs = parent.old.direct.ORIGINS.models(reg)
    for context, items in sorted(reg['contexts'].items()):
        scopes = context_packets(context, items, data, jobs, identity, beat, include_held=True)
        for seed in cfg['seeds']:
            states = {arm:api.read_checkpoint(ROOT/learned[context, seed, arm]['checkpoint']['path']) for arm in api.ARMS}
            cost_states = {arm:api.read_checkpoint(ROOT/parents[context, seed, arm]['checkpoint']['path']) for arm in api.ARMS}
            origin_ref = origin_refs[context, seed, 'self']['checkpoint']
            origin = parent.old.base.core.read_checkpoint(ROOT/origin_ref['path'])
            for scope, packets in scopes.items():
                originals = parent.old.direct.origin_predictions(origin, {source:p['args'] for source,p in packets.items()})
                for source, packet in sorted(packets.items()):
                    path = PUBLIC/'groups'/f'{context}_{source}_seed{seed}_{scope}.json'
                    if path.exists() and not (verify or resume):
                        raise ValueError('Explicit readout resume required')
                    choices, ranks, p = {}, {}, {}
                    parent_group = parent.PUBLIC/'groups'/path.name
                    old = json.loads(parent_group.read_text())
                    for arm in api.ARMS:
                        features, anchor, choice, rank, p[arm] = causal_inputs(packet, cost_states[arm], arm)
                        if (fingerprint(choice) != old['action_sha256'][arm]
                                or fingerprint(p[arm]) != old['prediction_sha256'][arm]):
                            raise ValueError('Frozen moment-policy comparator does not reproduce')
                        choices[arm+'_moment_parent'], ranks[arm+'_moment_parent'] = choice, rank
                        choices[arm], ranks[arm] = api.predict(states[arm], features, anchor, packet['allowed'], arm=arm)
                    original_p = originals[source]
                    take = parent.old.base.actions(original_p, parent.old.base.shared.eligible(packet['args'])) & packet['allowed'][:, 6]
                    choices['frozen_origin_binary'] = np.where(take, 6, -1)
                    ranks['frozen_origin_binary'] = original_p[:, 0]-original_p[:, 1]
                    choices['floor'] = np.full(len(take), -1, np.int64); ranks['floor'] = np.zeros(len(take))
                    result = readout.group_readout(packet, choices, ranks, p)
                    for arm in api.ARMS:
                        result['comparisons'][arm+'_minus_moment_parent'] = comparison(packet, choices, ranks, arm+'_moment_parent', arm)
                        if result['policies'][arm+'_moment_parent'] != old['result']['policies'][arm]:
                            raise ValueError('Parent metric replay differs')
                    value = dict(context=context, source=source, seed=seed, scope=scope, result=result,
                        action_sha256={key:fingerprint(v) for key,v in choices.items()},
                        parent_group_sha256=sha(parent_group), registration_sha256=sha(REG),
                        training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                        result_source='fresh_run_fixed_decision_final_source_internal', independent_roles_opened=False)
                    if verify:
                        if value != json.loads(path.read_text()):
                            raise ValueError('Exact decision-policy readout replay differs')
                    else:
                        once(path, value)
                    rows.append(value); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
            beat(state='readout_progress', context=context, views=len(rows), total=144)
        del scopes
    summary = readout.summarize(rows, pc)
    for scope in summary['scopes']:
        selected = [r for r in rows if r['scope'] == scope]
        summary['scopes'][scope]['decision_minus_moment_parent'] = {
            mode+'_'+bound:readout.interval(selected,
                lambda r, m=mode,b=bound:r['comparisons']['physical_neural_minus_moment_parent'][m][b], pc)
            for mode in ('full', 'matched') for bound in ('lower_pp', 'upper_pp', 'known_pp')}
    summary.update(exact_record_training=True, parent_cost_heads_frozen=True,
                   scene_joint_inference_implemented=False)
    if verify:
        doc = json.loads(finish.read_text())
        if doc['groups'] != refs or summary != json.loads((PUBLIC/'summary.json').read_text()):
            raise ValueError('Exact aggregate decision readout differs')
        once(PUBLIC/'verification.json', dict(exact_replay=True, groups=len(refs),
            summary_sha256=sha(PUBLIC/'summary.json'), registration_sha256=sha(REG)))
    else:
        once(PUBLIC/'summary.json', summary)
        once(finish, dict(groups=refs, summary_sha256=sha(PUBLIC/'summary.json'),
            registration_sha256=sha(REG), training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
            pid=os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, independent_roles_opened=False))
    json_write(PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=os.getpid(), seconds=time.monotonic()-started,
             peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, exact_replay=verify))
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
