"""Policy-region moment learning with fixed causal cost models and readout."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_static_scene_cost as old
from scripts import report_m3w_feature_cost_evidence as aggregates
from src.world_model import m3w_policy_region_cost as api
import numpy as np

parent = old.parent; torch = old.torch; readout = old.readout
NAME = 'european_policy_region_cost_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_policy_region_cost_v1.json'; REG = PUBLIC/'registration.json'
AUDIT = ROOT/'outputs/publication_readiness_2026_09/european_selected_cost_audit_v1'
sha, once, json_write, fingerprint = parent.sha, parent.once, parent.json_write, parent.fingerprint
FILES = (CONFIG, Path(__file__), ROOT/'src/world_model/m3w_policy_region_cost.py',
    ROOT/'tests/test_m3w_policy_region_cost.py', PUBLIC/'protocol.md',
    AUDIT/'diagnosis.json.gz', AUDIT/'diagnosis_archive.json', AUDIT/'summary.json', AUDIT/'verification.json')


def load():
    cfg = json.loads(CONFIG.read_text()); oc, pc, reg, source, jobs, identity, _, packed, polygons = old.admission()
    manifest = json.loads((AUDIT/'diagnosis_archive.json').read_text())
    check = json.loads((AUDIT/'verification.json').read_text())
    content = (AUDIT/'diagnosis.json.gz').read_bytes()
    if (sha(AUDIT/'diagnosis.json.gz') != manifest['archive_sha256']
            or hashlib.sha256(gzip.decompress(content)).hexdigest() != check['diagnosis_sha256']
            or not check['exact_replay'] or cfg['training'] != oc['training']
            or cfg['region_weight'] != 1 or cfg['region_moments'] != list(api.MOMENTS)
            or cfg['risk_budget'] != .02 or cfg['seeds'] != oc['seeds']):
        raise ValueError('Verified diagnosis and fixed matched objective required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_registration_sha256=sha(old.REG), parent_training_freeze_sha256=sha(old.PUBLIC/'training_freeze.json'),
        contexts=sorted(reg['contexts']), arms=cfg['arms'], seeds=cfg['seeds'], fits=54, steps_per_fit=2000,
        parameters=26309, change='fixed_teacher_selected_record_region_moment_training',
        region_weight=1., original_row_draws_and_initialization_matched=True, risk_budget=.02,
        independent_roles_opened=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    return cfg, pc, reg, source, jobs, identity, old.models(), packed, polygons


def admission():
    values = load()
    for rel, digest in json.loads(REG.read_text())['bindings'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Registered region experiment changed')
        parent.committed(ROOT/rel)
    parent.committed(REG); return values


def runtime(cfg):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    def guard():
        if old.shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Preserve10GiB reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['checkpoint_cap_bytes']:
            raise OSError('Private checkpoint cap exceeded')
        if time.monotonic()-started > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h healthy-runtime boundary; resume same roster')
    def beat(**event):
        guard(); row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        json_write(PRIVATE/'heartbeat.json', row)
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    old.signal.signal(old.signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume same checkpoint')))
    guard(); return started, beat, guard


def region_data(packets, checkpoint, arm):
    state = parent.api.read_checkpoint(ROOT/checkpoint['path']); choices = []
    for source, p in sorted(packets.items()):
        predicted = old.api.predict(state, p['x'], p['scene'], p['envelope'], p['scales'])
        choice, _ = parent.api.select_actions(predicted, p['allowed'], arm=arm); choices.append(choice)
    values = [np.concatenate([packets[s][k] for s in sorted(packets)]) for k in ('y', 'scales', 'sites', 'recordings')]
    return api.prepare_regions(*values, np.concatenate(choices), arm=arm)


def checkpoint(context, seed, arm, pilot=False, direct=False):
    return PRIVATE/('pilot' if pilot else 'heads')/f'{context}_seed{seed}_{arm}{"_direct" if direct else ""}.pt.gz'


def neutral(resume=False):
    cfg, _, reg, source, jobs, identity, controls, packed, polygons = admission()
    if (PUBLIC/'neutral_replay.json').exists():
        raise ValueError('Completed neutral immutable')
    started, beat, guard = runtime(cfg); context, items = sorted(reg['contexts'].items())[0]
    packets = old.scene_packets(parent.prepare_context(context, items, source, jobs, identity, beat, include_held=False),
        source, packed, polygons)['fit']; data = old.training_data(packets)
    ref = controls[context, 17, 'physical_neural']['checkpoint']; regions = region_data(packets, ref, 'physical_neural')
    result = api.fit(data, arm='physical_neural', settings=cfg['training'], seed=17,
        identity=dict(registration_sha256=sha(REG)), path=PRIVATE/'neutral'/'checkpoint.pt.gz',
        regions=regions, region_weight=0, heartbeat=beat, guard=guard, resume=resume)
    api.assert_neutral(parent.api.read_checkpoint(ROOT/ref['path']), result)
    once(PUBLIC/'neutral_replay.json', dict(pid=os.getpid(), seconds=time.monotonic()-started,
        updates=2000, all_legacy_scientific_state_except_identity_runtime_exact=True,
        region_weight=0, reference=ref, registration_sha256=sha(REG)))
    beat(state='neutral2000_exact')


def train(pilot=False, resume=False):
    cfg, _, reg, source, jobs, identity, controls, packed, polygons = admission()
    finish = PUBLIC/('pilot.json' if pilot else 'training_freeze.json')
    if finish.exists():
        raise ValueError('Completed training phase immutable')
    if not pilot and (not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']
            or not json.loads((PUBLIC/'neutral_replay.json').read_text())['all_legacy_scientific_state_except_identity_runtime_exact']):
        raise ValueError('Real recovery pilot and full neutral required')
    started, beat, guard = runtime(cfg); refs, times = [], []
    contexts = sorted(reg['contexts'].items())[:1] if pilot else sorted(reg['contexts'].items())
    for context, items in contexts:
        packets = old.scene_packets(parent.prepare_context(context, items, source, jobs, identity, beat, include_held=False),
            source, packed, polygons)['fit']; data = old.training_data(packets)
        for seed in cfg['seeds'][:1] if pilot else cfg['seeds']:
            states = []
            for arm in parent.api.ARMS:
                teacher = controls[context, seed, arm]['checkpoint']; regions = region_data(packets, teacher, arm)
                path = checkpoint(context, seed, arm, pilot)
                kw = dict(data=data, arm=arm, settings=cfg['training'], seed=seed,
                    identity=dict(context=context, teacher=teacher, registration_sha256=sha(REG)),
                    path=path, regions=regions, region_weight=1., heartbeat=beat, guard=guard)
                if pilot and not path.exists():
                    api.fit(**kw, stop_at=cfg['pilot_resume_at'])
                state = api.fit(**kw, stop_at=cfg['pilot_steps'] if pilot else None,
                    resume=path.exists() and (pilot or resume)); states.append(state)
                if pilot:
                    direct_path = checkpoint(context, seed, arm, True, True)
                    direct = api.fit(**{**kw, 'path':direct_path}, stop_at=cfg['pilot_steps'], resume=resume and direct_path.exists())
                    for key in state:
                        if key != 'seconds':
                            parent.api.exact(state[key], direct[key])
                    times.append(state['seconds'])
                else:
                    prior = parent.api.read_checkpoint(ROOT/teacher['path'])
                    for key in ('initial_model', 'preprocess', 'draw_hash', 'sampler_rng', 'torch_rng', 'parameters', 'input_hashes'):
                        parent.api.exact(state[key], prior[key])
                    if all(torch.equal(v, prior['model'][k]) for k,v in state['model'].items()):
                        raise ValueError('Region objective did not change any weight')
                    receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
                    once(receipt, dict(context=context, seed=seed, arm=arm, step=state['step'],
                        checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                        parameters=state['parameters'], seconds=state['seconds'], draw_hash=state['draw_hash'],
                        first_trace=state['trace'][0], final_trace=state['trace'][-1],
                        teacher=teacher, region_keys=regions['keys'], selected_known_region_rows=regions['selected_known_rows'],
                        unknown_teacher_selected_rows_not_labeled=regions['unknown_selected_rows'],
                        region_arrays=state['region_arrays'], original_initialization_preprocess_and_draws_exact=True,
                        registration_sha256=sha(REG), result_source='fresh_native_torch_policy_region_cost'))
                    refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
                    beat(state='fit_complete', fits=len(refs), total=54, context=context, seed=seed, arm=arm)
            for other in states[1:]:
                for key in ('initial_model', 'preprocess', 'draw_hash', 'sampler_rng', 'torch_rng', 'parameters'):
                    parent.api.exact(states[0][key], other[key])
        del data, packets
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
        raise ValueError('Committed complete54fitfreeze required')
    result = {}
    for ref in frozen['fits']:
        row = json.loads((ROOT/ref['path']).read_text()); key = row['context'], row['seed'], row['arm']
        if sha(ROOT/ref['path']) != ref['sha256'] or sha(ROOT/row['checkpoint']['path']) != row['checkpoint']['sha256']:
            raise ValueError('Frozen region endpoint differs')
        if key in result or row['step'] != 2000:
            raise ValueError('Unique completed endpoint required')
        result[key] = row
    return result


def evaluate(verify=False, resume=False):
    cfg, pc, reg, source, jobs, identity, controls, packed, polygons = admission(); learned = models()
    finish = PUBLIC/'complete.json'
    if finish.exists() and not verify:
        raise ValueError('Completed readout immutable')
    started, beat, guard = runtime(cfg); rows, refs = [], []
    _, origin_refs = parent.old.direct.ORIGINS.models(reg)
    expected = aggregates.aggregate_groups(old.NAME)
    for context, items in sorted(reg['contexts'].items()):
        scopes = old.scene_packets(parent.prepare_context(context, items, source, jobs, identity, beat, include_held=True), source, packed, polygons)
        for seed in cfg['seeds']:
            states = {a:parent.api.read_checkpoint(ROOT/learned[context, seed, a]['checkpoint']['path']) for a in parent.api.ARMS}
            prior = {a:parent.api.read_checkpoint(ROOT/controls[context, seed, a]['checkpoint']['path']) for a in parent.api.ARMS}
            origin = parent.old.base.core.read_checkpoint(ROOT/origin_refs[context, seed, 'self']['checkpoint']['path'])
            for scope, packets in scopes.items():
                original = parent.old.direct.origin_predictions(origin, {s:p['args'] for s,p in packets.items()})
                for name, packet in sorted(packets.items()):
                    path = PUBLIC/'groups'/f'{context}_{name}_seed{seed}_{scope}.json'
                    if path.exists() and not (verify or resume):
                        raise ValueError('Explicit partial readout resume required')
                    old_group = expected[path.name]; choices, ranks, predicted = {}, {}, {}
                    for arm in parent.api.ARMS:
                        p = old.api.predict(states[arm], packet['x'], packet['scene'], packet['envelope'], packet['scales'])
                        predicted[arm] = p
                        for multiplier in (1., *pc['sensitivity_harm_multipliers']):
                            key = arm if multiplier == 1 else arm+'_harm'+str(int(multiplier))
                            choices[key], ranks[key] = parent.api.select_actions(p, packet['allowed'], arm=arm, harm_multiplier=multiplier)
                        p = old.api.predict(prior[arm], packet['x'], packet['scene'], packet['envelope'], packet['scales'])
                        key = arm+'_MSE_parent'; choices[key], ranks[key] = parent.api.select_actions(p, packet['allowed'], arm=arm)
                        if fingerprint(choices[key]) != old_group['action_sha256'][arm]:
                            raise ValueError('Frozen MSE comparator differs')
                    op = original[name]; take = parent.old.base.actions(op, parent.old.base.shared.eligible(packet['args'])) & packet['allowed'][:, 6]
                    choices['frozen_origin_binary'] = np.where(take, 6, -1); ranks['frozen_origin_binary'] = op[:, 0]-op[:, 1]
                    choices['floor'] = np.full(len(take), -1, np.int64); ranks['floor'] = np.zeros(len(take))
                    result = readout.group_readout(packet, choices, ranks, predicted)
                    for arm in parent.api.ARMS:
                        if result['policies'][arm+'_MSE_parent'] != old_group['result']['policies'][arm]:
                            raise ValueError('Frozen comparator metric replay differs')
                    left, right = choices['physical_neural_MSE_parent'], choices['physical_neural']
                    aa, bb = readout.matched_choices(left, right, ranks['physical_neural_MSE_parent'], ranks['physical_neural'], packet['recordings'], packet['frames'], packet['ids'])
                    result['region_minus_parent'] = dict(full=readout.paired_bounds(packet, left, right),
                        matched=readout.paired_bounds(packet, aa, bb), matched_count=int((aa>=0).sum()), matched_role='utility_only_not_safety')
                    value = dict(context=context, source=name, seed=seed, scope=scope, result=result,
                        action_sha256={a:fingerprint(c) for a,c in choices.items()}, registration_sha256=sha(REG),
                        training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                        result_source='fresh_source_internal_policy_region_readout', independent_roles_opened=False)
                    if verify:
                        if value != json.loads(path.read_text()):
                            raise ValueError('Exact region readout differs')
                    else:
                        once(path, value)
                    rows.append(value); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
            beat(state='readout_progress', context=context, views=len(rows), total=144)
        del scopes
    summary = readout.summarize(rows, pc)
    for scope in summary['scopes']:
        selected = [r for r in rows if r['scope']==scope]
        summary['scopes'][scope]['region_minus_parent'] = {m+'_'+b:readout.interval(selected,
            lambda r,m=m,b=b:r['region_minus_parent'][m][b], pc)
            for m in ('full','matched') for b in ('known_pp','lower_pp','upper_pp')}
    summary.update(region_moment_loss=True, original_parameters_initialization_and_draws_matched=True)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()) or json.loads(finish.read_text())['groups'] != refs:
            raise ValueError('Exact region summary differs')
        once(PUBLIC/'verification.json', dict(exact_replay=True, groups=144, summary_sha256=sha(PUBLIC/'summary.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, groups=refs,
            summary_sha256=sha(PUBLIC/'summary.json'), registration_sha256=sha(REG), independent_roles_opened=False))
    json_write(PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'), dict(pid=os.getpid(), seconds=time.monotonic()-started))
    beat(state='readout_verified' if verify else 'readout_complete', full_budget=summary['primary_full_roster_budget_supported'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register','pilot','neutral','train','evaluate','verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase == 'register': load()
        elif args.phase == 'neutral': neutral(args.resume)
        elif args.phase in ('pilot','train'): train(args.phase=='pilot', args.resume)
        else: evaluate(args.phase=='verify', args.resume)
