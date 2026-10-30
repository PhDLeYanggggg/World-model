"""Registered matched benefit/risk fits and full original SOURCE policy evaluation."""
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
    raise RuntimeError('Native arm64 required before importing Torch')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_query_unit_critic as unit
from scripts import run_m3w_position_anchor as anchor
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.world_model import m3w_coupled_benefit_risk as api
from src.world_model import m3w_coupled_benefit_allocation as decoder
from src.evaluation import m3w_coupled_benefit_risk as evaluation
from src.evaluation.m3w_metric_bundle import pack

np = api.np; source = unit.parent
NAME = 'european_coupled_benefit_risk_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_coupled_benefit_risk_v1.json'; REG = PUBLIC/'registration.json'
sha, once, write = unit.sha, unit.once, unit.write
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_coupled_benefit_risk.py',
    ROOT/'src/world_model/m3w_coupled_benefit_allocation.py', ROOT/'src/evaluation/m3w_coupled_benefit_risk.py',
    ROOT/'tests/test_m3w_coupled_benefit_risk.py', ROOT/'tests/test_m3w_coupled_benefit_allocation.py', PUBLIC/'protocol.md')


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); old_cfg, values, store, banks, models, _ = unit.admission()
    checked = json.loads((unit.PUBLIC/'verification.json').read_text())
    parent = json.loads((anchor.PUBLIC/'verification.json').read_text())
    if (cfg['arms'] != list(api.ARMS) or cfg['training'] != old_cfg['training'] or cfg['seeds'] != [17, 29, 43]
            or cfg['risk_budget'] != .02 or cfg['parameters_per_head'] != 30857
            or not checked['all72groups_exact'] or checked['summary_sha256'] != sha(unit.PUBLIC/'summary.json')
            or not parent['all24groups_exact'] or parent['summary_sha256'] != sha(anchor.PUBLIC/'summary.json')
            or any(cfg[k] for k in ('held_scored', 'independent_roles_opened', 'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Verified prior controls and unchanged task/budget required')
    controls = {}
    for ref in json.loads((unit.PUBLIC/'training_freeze.json').read_text())['fits']:
        if sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('Original unit fit receipt changed')
        r = json.loads((ROOT/ref['path']).read_text()); cp = r['checkpoint']
        if sha(ROOT/cp['path']) != cp['sha256']: raise ValueError('Original unit checkpoint changed')
        if r['arm'] == 'selected_set': controls[r['context'], r['seed']] = cp
    if len(controls) != 18: raise ValueError('All18 original selected-set controls required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        source_registration_sha256=sha(unit.REG), source_training_freeze_sha256=sha(unit.PUBLIC/'training_freeze.json'),
        source_verification_sha256=sha(unit.PUBLIC/'verification.json'), prior_anchor_summary_sha256=sha(anchor.PUBLIC/'summary.json'),
        prior_anchor_verification_sha256=sha(anchor.PUBLIC/'verification.json'), contexts=json.loads(unit.REG.read_text())['contexts'],
        changed_factor='release_benefit_constraint_only_same_reference_moments_and_risk_objective',
        own_training_or_policy_outcomes_read=False, fresh_fits=36, optimizer_updates=72000, full_refit_updates=4000,
        primary='released_minus_protected_full_joint_completion_lower', references_frozen=True,
        risk_budget=.02, role='exposed_SOURCE_fitting_resubstitution_only', independent_roles_opened=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel) != digest: raise ValueError('Registered coupled experiment changed: '+rel)
            source.frozen.BASE.engine.committed(ROOT/rel)
        source.frozen.BASE.engine.committed(REG)
    return cfg, values, store, banks, models, controls


def runtime(cfg):
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1); start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume coupled-cost experiment')))
    def guard():
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12h boundary')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('10GiB reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['private_checkpoint_cap_bytes']: raise OSError('1GiB checkpoint cap')
    def beat(**event):
        guard(); row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE/'heartbeat.json', row)
        with (PRIVATE/'events.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    beat(state='runtime_ready', architecture=platform.machine(), device='cpu', threads=4, interop_threads=1, workers=0, resource_probing=False)
    return start, beat, guard


def checkpoint(context, seed, mode, phase='heads'):
    return PRIVATE/phase/f'{context}_seed{seed}_{mode}.pt.gz'


def fit(data, cfg, context, seed, mode, phase, beat, guard, **kw):
    return api.fit(data, mode=mode, seed=seed, settings=cfg['training'],
        identity=dict(registration_sha256=sha(REG), context=context, cost_seed=seed),
        path=checkpoint(context, seed, mode, phase), heartbeat=lambda **event: beat(context=context, seed=seed, **event), guard=guard, **kw)


def exact_fit(left, right):
    for k in left:
        if k != 'seconds': api.parent.exact(left[k], right[k])


def context_packets(context, items, values, store, banks, beat):
    return source.frozen.BASE.prepared_context(context, items, values, store, banks, beat)[1::2]


def pilot(resume=False):
    cfg, values, store, banks, models, _ = admission(); start, beat, guard = runtime(cfg)
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Completed pilot immutable')
    old, main = aggregate_groups(source.frozen.source.parent.NAME), aggregate_groups(source.frozen.NAME)
    context, items = sorted(values[2]['contexts'].items())[0]
    packets, changed = context_packets(context, items, values, store, banks, beat)
    data, _, _ = source.prepare(context, 17, packets, changed, store, models, old, main, beat)
    states = []; cells = []
    for mode in api.ARMS:
        path = checkpoint(context, 17, mode, 'pilot')
        if not path.exists(): fit(data, cfg, context, 17, mode, 'pilot', beat, guard, stop_at=50)
        recovered = fit(data, cfg, context, 17, mode, 'pilot', beat, guard, stop_at=100, resume=True)
        dp = checkpoint(context, 17, mode, 'pilot_direct')
        direct = fit(data, cfg, context, 17, mode, 'pilot_direct', beat, guard, stop_at=100, resume=resume and dp.exists())
        exact_fit(recovered, direct)
        if mode == 'protected_benefit':
            api.assert_old_control(recovered, api.parent.read_checkpoint(unit.checkpoint(context, 17, 'selected_set', 'pilot')))
        if recovered['parameters'] != 30857: raise ValueError('Matched parameter count changed')
        cells.append(dict(mode=mode, exact_interrupted_recovery=True, parameters=recovered['parameters'],
            first_trace=recovered['trace'][0], final_trace=recovered['trace'][-1])); states.append(recovered)
    api.assert_matched(*states)
    once(PUBLIC/'pilot.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, actual_optimizer_updates=400,
        cells=cells, protected_old100update_control_exact=True, engineering_pass=True,
        own_policy_outcomes_read=False, registration_sha256=sha(REG)))
    beat(state='real400update_pilot_complete')


def train(resume=False):
    cfg, values, store, banks, models, controls = admission(); start, beat, guard = runtime(cfg)
    if (PUBLIC/'training_freeze.json').exists(): raise ValueError('Completed training immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']: raise ValueError('Real pilot required')
    old, main = aggregate_groups(source.frozen.source.parent.NAME), aggregate_groups(source.frozen.NAME)
    first = sorted(values[2]['contexts'])[0]; refs = []
    for context, items in sorted(values[2]['contexts'].items()):
        packets, changed = context_packets(context, items, values, store, banks, beat)
        for seed in cfg['seeds']:
            data, _, _ = source.prepare(context, seed, packets, changed, store, models, old, main, beat); states = []
            for mode in api.ARMS:
                path = checkpoint(context, seed, mode)
                state = fit(data, cfg, context, seed, mode, 'heads', beat, guard, resume=resume and path.exists())
                if mode == 'protected_benefit': api.assert_old_control(state, api.parent.read_checkpoint(ROOT/controls[context, seed]['path']))
                if state['step'] != 2000 or state['parameters'] != 30857 or state['unknown_rows_with_loss']:
                    raise ValueError('Full matched training required')
                if context == first and seed == 17:
                    rp = checkpoint(context, seed, mode, 'full_refit')
                    exact_fit(state, fit(data, cfg, context, seed, mode, 'full_refit', beat, guard, resume=resume and rp.exists()))
                    once(PUBLIC/('first_full_refit_'+mode+'.json'), dict(scientific_state_exact=True, updates=2000,
                        checkpoint_sha256=sha(path), refit_checkpoint_sha256=sha(rp)))
                receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{mode}.json'
                once(receipt, dict(context=context, seed=seed, mode=mode, step=state['step'], parameters=state['parameters'],
                    checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                    input_hashes=state['input_hashes'], draw_hash=state['draw_hash'], query_normalizer=state['query_normalizer'],
                    first_trace=state['trace'][0], final_trace=state['trace'][-1], seconds=state['seconds'],
                    unknown_rows_drawn=state['unknown_rows_drawn'], unknown_rows_with_loss=0,
                    original_references_frozen=True, registration_sha256=sha(REG)))
                refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt))); states.append(state)
                beat(state='head_complete', heads=len(refs), total=36, context=context, seed=seed, mode=mode)
            api.assert_matched(*states); del data, states
        del changed, packets
    once(PUBLIC/'training_freeze.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, fits=refs,
        fresh_optimizer_updates=72000, exact_refit_updates=4000, protected_old18controls_exact=True,
        registration_sha256=sha(REG), independent_roles_opened=False))
    beat(state='all36_coupled_heads_frozen')


def evaluate(verify=False, resume=False):
    cfg, values, store, banks, models, controls = admission(); start, beat, guard = runtime(cfg)
    if (PUBLIC/'complete.json').exists() and not verify: raise ValueError('Completed readout immutable')
    freeze = json.loads((PUBLIC/'training_freeze.json').read_text()); fitted = {}
    for ref in freeze['fits']:
        if sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('Fit receipt changed')
        r = json.loads((ROOT/ref['path']).read_text()); cp = r['checkpoint']
        if sha(ROOT/cp['path']) != cp['sha256']: raise ValueError('Checkpoint changed')
        fitted[r['context'], r['seed'], r['mode']] = cp
    if len(fitted) != 36: raise ValueError('All36 frozen fits required')
    old, main = aggregate_groups(source.frozen.source.parent.NAME), aggregate_groups(source.frozen.NAME)
    unit_results = aggregate_groups(unit.NAME); previous = aggregate_groups(NAME) if verify else {}
    rows, refs, all_ids, records = [], [], set(), set()
    for context, items in sorted(values[2]['contexts'].items()):
        packets, changed = context_packets(context, items, values, store, banks, beat)
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            data, ranges, _ = source.prepare(context, seed, packets, changed, store, models, old, main, beat)
            states = {a: api.parent.read_checkpoint(ROOT/fitted[context, seed, a]['path']) for a in api.ARMS}
            api.assert_matched(*states.values()); api.assert_old_control(states[api.ARMS[0]], api.parent.read_checkpoint(ROOT/controls[context, seed]['path']))
            predictions = {a: api.predict(s, data['causal']) for a, s in states.items()}
            for name, packet in sorted(packets.items()):
                key = f'{context}_{name}_seed{seed}_fit.json'; sl = ranges[name]; path = PUBLIC/'groups'/key
                p = {a: v[sl] for a, v in predictions.items()}; oldrow = unit_results[key]
                if (api.parent.digest(packet['ids']) != oldrow['ids_sha256']
                        or api.parent.digest(p[api.ARMS[0]]) != oldrow['prediction_sha256']['selected_set']):
                    raise ValueError('Old full prediction/row control changed')
                if path.exists() and not verify:
                    if not resume: raise ValueError('Explicit partial readout resume required')
                    row = json.loads(path.read_text())
                    if row['training_freeze_sha256'] != sha(PUBLIC/'training_freeze.json'): raise ValueError('Partial model freeze differs')
                else:
                    control = api.parent.read_checkpoint(unit.PRIVATE/'progress'/(key+'.pt.gz'))['choices']
                    if api.parent.digest(control['set_joint']) != oldrow['action_sha256']['set_joint']: raise ValueError('Old choices changed')
                    progress = PRIVATE/'progress'/(key+'.pt.gz')
                    choices, ranks, info = decoder.allocate(p, packet['allowed'], packet['recordings'], packet['frames'], packet['ids'], control,
                        seconds=cfg['solver_seconds_per_query'], checkpoint=(lambda _: None) if verify else lambda v: api.parent.save_checkpoint(progress, v),
                        heartbeat=lambda **kw: beat(state='coupled_allocation', context=context, source=name, seed=seed, **kw),
                        resume_state=api.parent.read_checkpoint(progress) if resume and progress.exists() and not verify else None)
                    result = evaluation.readout(packet, choices, ranks, info, p, job['design']['easy_cut'], values[3]['origin'][packet['ids']], data['rms'])
                    if result['policies']['protected_joint'] != oldrow['result']['policies']['set_joint']:
                        raise ValueError('Original complete policy score changed')
                    row = dict(context=context, source=name, seed=seed, result=result, old_control_exact=True,
                        paired_training_verified=True, ids_sha256=oldrow['ids_sha256'],
                        prediction_sha256={a: api.parent.digest(v) for a, v in p.items()},
                        action_sha256={a: api.parent.digest(c) for a, c in choices.items()},
                        training_freeze_sha256=sha(PUBLIC/'training_freeze.json'), registration_sha256=sha(REG),
                        independent_roles_opened=False, held_scored=False)
                    if verify:
                        if row != previous[key]: raise ValueError('Full inference/allocation/scoring replay differs')
                    else: once(path, row)
                all_ids.update(map(int, packet['ids'])); records.update((name, str(r)) for r in packet['recordings'])
                rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                beat(state='coupled_group_complete', groups=len(rows), total=72, context=context, source=name, seed=seed)
            del data, states, predictions
        del changed, packets
    if len(all_ids) != 149510 or len(records) != 74: raise ValueError('Complete original target cohort required')
    summary = evaluation.summarize(rows, json.loads(REG.read_text())['contexts'], cfg)
    summary.update(unique_target_rows=len(all_ids), unique_recordings=len(records))
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()): raise ValueError('Canonical summary differs')
        once(PUBLIC/'verification.json', dict(all72groups_exact=True, all72_original_controls_exact=True,
            summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(PUBLIC/'complete.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, groups=refs,
            summary_sha256=sha(PUBLIC/'summary.json'), training_freeze_sha256=sha(PUBLIC/'training_freeze.json')))
        pack(ROOT, PUBLIC)
    write(PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'), dict(pid=os.getpid(),
        seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all72_verified' if verify else 'all72_complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'pilot', 'train', 'eval', 'verify'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'register': admission(True)
        elif args.phase == 'pilot': pilot(args.resume)
        elif args.phase == 'train': train(args.resume)
        else: evaluate(args.phase == 'verify', args.resume)
