"""Fresh matched selected-query risk fits, then fixed complete-policy readouts."""
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
    raise RuntimeError('Use native arm64 .venv-pytorch before importing Torch')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_shared_count_bank as prerequisite
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.world_model import m3w_selected_query_critic as api
from src.world_model import m3w_critic_allocation as decoder
from src.evaluation import m3w_selected_query_critic_readout as evaluation

frozen = prerequisite.parent
NAME = 'european_selected_query_critic_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_selected_query_critic_v1.json'; REG = PUBLIC/'registration.json'
sha, once, write = frozen.sha, frozen.once, frozen.write
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_selected_query_critic.py',
    ROOT/'src/world_model/m3w_critic_allocation.py', ROOT/'src/evaluation/m3w_selected_query_critic_readout.py',
    ROOT/'src/world_model/m3w_rational_harm.py', ROOT/'src/world_model/m3w_query_excess.py',
    ROOT/'tests/test_m3w_selected_query_critic.py', ROOT/'tests/test_m3w_critic_allocation.py',
    ROOT/'tests/test_m3w_selected_query_critic_readout.py', PUBLIC/'protocol.md')


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); _, values, store, banks, models = prerequisite.admission()
    check = json.loads((prerequisite.PUBLIC/'verification.json').read_text())
    if (cfg['arms'] != list(api.ARMS) or cfg['seeds'] != [17, 29, 43] or cfg['risk_budget'] != .02
            or cfg['training']['steps'] != 2000 or cfg['input_fields'] != 462 or cfg['parameters_per_head'] != 30726
            or not check['all72groups_exact'] or check['summary_sha256'] != sha(prerequisite.PUBLIC/'summary.json')
            or cfg['row_risk_loss_weight'] != .5 or cfg['selected_query_score_loss_weight'] != .5
            or cfg['rms_minimum'] != 1e-6
            or any(cfg[k] for k in ('held_scored', 'independent_roles_opened', 'deployment_changed',
                                   'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Verified complete control, unchanged2% budget and fitting-only design required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        prerequisite_complete_sha256=sha(prerequisite.PUBLIC/'complete.json'),
        prerequisite_summary_sha256=sha(prerequisite.PUBLIC/'summary.json'),
        prerequisite_verification_sha256=sha(prerequisite.PUBLIC/'verification.json'),
        parent_frame_complete_sha256=sha(frozen.PUBLIC/'complete.json'),
        parent_frame_summary_sha256=sha(frozen.PUBLIC/'summary.json'),
        parent_registration_sha256=sha(frozen.REG), contexts=json.loads(frozen.REG.read_text())['contexts'],
        fresh_fits=36, primary_optimizer_updates=72000, first_full_refit_updates=4000,
        real_pilot_updates=400, cost_head_seeds=cfg['seeds'], forecaster_seed=43,
        selected_context='original_joint_matched_initial_selected_actions',
        primary='selected_set_joint_minus_row_local_joint_full_completion_lower_utility',
        role='exposed_SOURCE_fitting_resubstitution_only', new_trajectory_forecaster_fits=0,
        risk_budget=.02, known_outcome_mask_is_not_an_input=True, held_scored=False,
        independent_roles_opened=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel) != digest: raise ValueError('Registered critic dependency changed: '+rel)
            frozen.BASE.engine.committed(ROOT/rel)
        frozen.BASE.engine.committed(REG)
    return cfg, values, store, banks, models


def runtime(cfg):
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume original critic checkpoints')))
    def guard():
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12h boundary; explicit resume')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('10GiB free disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['private_checkpoint_cap_bytes']:
            raise OSError('Private256MiB checkpoint cap')
    def beat(**event):
        guard(); row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE/'heartbeat.json', row)
        with (PRIVATE/'events.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    beat(state='runtime_ready', architecture=platform.machine(), threads=4, interop_threads=1, workers=0, device='cpu')
    return start, beat, guard


def prepare(context, seed, packets, changed, store, models, old, main, beat):
    model_ref = models[context, seed, 'admitted_score']['checkpoint']
    if sha(ROOT/model_ref['path']) != model_ref['sha256']: raise ValueError('Original cost head changed')
    state = api.read_checkpoint(ROOT/model_ref['path'])
    api.exact(state['preprocess'], changed['preprocess'])
    fields = {k: [] for k in ('prediction', 'envelope', 'scales', 'allowed', 'initial', 'sites', 'recordings', 'frames', 'ids', 'targets')}
    independent, ranges = {}, {}; offset = 0
    for name, packet in sorted(packets.items()):
        key = f'{context}_{name}_seed{seed}_fit.json'
        p, ind, _ = frozen.infer(packet, state, store, old[key])
        cp = api.read_checkpoint(frozen.PRIVATE/'progress'/(key+'.pt.gz'))
        initial = cp['choices']['joint_matched']
        if (cp['completed_queries'] != len(decoder.base.query_groups(packet['recordings'], packet['frames'], packet['ids']))
                or frozen.BASE.engine.fingerprint(initial) != main[key]['action_sha256']['joint_matched']):
            raise ValueError('Original joint selected set changed')
        for arm, choice in (('independent', ind), ('joint_matched', initial)):
            if evaluation.bank.policy_metrics(packet, choice) != main[key]['result']['policies'][arm]:
                raise ValueError('Original full policy score differs')
        values = dict(prediction=p, envelope=packet['envelope'], scales=packet['scales'], allowed=packet['allowed'],
            initial=initial, sites=np.repeat(name, len(initial)), recordings=packet['recordings'],
            frames=packet['frames'], ids=packet['ids'], targets=packet['y'])
        for k in fields: fields[k].append(values[k])
        ranges[name] = slice(offset, offset+len(initial)); offset += len(initial); independent[name] = ind
    arrays = {k: np.concatenate(v) for k, v in fields.items()}
    if changed['z'].shape != (offset, 7, 462): raise ValueError('Complete frozen462-field ordering required')
    np.testing.assert_array_equal(changed['allowed'].numpy(), arrays['allowed'])
    known = np.isfinite(arrays['targets']).all((1, 2))
    np.testing.assert_array_equal(changed['known'].numpy(), known)
    np.testing.assert_allclose(changed['truth'].numpy()[known],
        arrays['targets'][known]/arrays['scales'][known, None, None], rtol=3e-6, atol=1e-7)
    c = api.causal_inputs(changed['z'], *(arrays[k] for k in
        ('prediction', 'envelope', 'scales', 'allowed', 'initial', 'sites', 'recordings', 'frames', 'ids')))
    data = api.training_data(c, arrays['targets'])
    beat(state='full_causal_critic_context_prepared', context=context, seed=seed, rows=offset,
        queries=len(c['keys']), original_selected_rows=int((c['initial'] >= 0).sum()),
        unknown_rows=data['unknown_rows'], source_count=len(c['source_groups']), labels_only_for_loss=True)
    return data, ranges, independent


def checkpoint(context, seed, arm, phase='heads'):
    return PRIVATE/phase/f'{context}_seed{seed}_{arm}.pt.gz'


def fit(data, cfg, context, seed, arm, phase, beat, guard, **kwargs):
    return api.fit(data, arm=arm, seed=seed, settings=cfg['training'], path=checkpoint(context, seed, arm, phase),
        identity=dict(registration_sha256=sha(REG), context=context, cost_seed=seed),
        heartbeat=lambda **kw: beat(context=context, seed=seed, **kw), guard=guard, **kwargs)


def scientific_equal(a, b):
    for k in a:
        if k != 'seconds': api.exact(a[k], b[k])


def pilot(resume=False):
    cfg, values, store, banks, models = admission()
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Completed pilot immutable')
    start, beat, guard = runtime(cfg)
    old, main = aggregate_groups(frozen.source.parent.NAME), aggregate_groups(frozen.NAME)
    context, items = sorted(values[2]['contexts'].items())[0]
    _, packets, _, changed = frozen.BASE.prepared_context(context, items, values, store, banks, beat)
    data, _, _ = prepare(context, 17, packets, changed, store, models, old, main, beat)
    states, cells = [], []
    for arm in cfg['arms']:
        path = checkpoint(context, 17, arm, 'pilot')
        if not path.exists():
            fit(data, cfg, context, 17, arm, 'pilot', beat, guard, stop_at=cfg['pilot_split_updates'])
        state = fit(data, cfg, context, 17, arm, 'pilot', beat, guard, resume=True, stop_at=cfg['pilot_final_updates'])
        direct = fit(data, cfg, context, 17, arm, 'pilot_direct', beat, guard,
            resume=resume and checkpoint(context, 17, arm, 'pilot_direct').exists(), stop_at=cfg['pilot_final_updates'])
        scientific_equal(state, direct)
        if (state['parameters'] != cfg['parameters_per_head'] or state['unknown_rows_with_loss']
                or not state['unknown_rows_drawn'] or not any(not api.torch.equal(v, state['initial_model'][k])
                    for k, v in state['model'].items())):
            raise ValueError('Real training/activation, same size and exact recovery required')
        cells.append(dict(arm=arm, updates=state['step'], seconds=state['seconds'], exact_recovery=True,
            first_monitor=state['trace'][0], final_monitor=state['trace'][-1],
            unknown_rows_drawn=state['unknown_rows_drawn'], unknown_rows_with_loss=0))
        states.append(state)
    api.assert_matched(*states)
    once(PUBLIC/'pilot.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        actual_optimizer_updates=400, paired_design_exact=True, cells=cells, engineering_pass=True,
        rough_full_training_seconds=max(v['seconds'] for v in cells)*20*36,
        projection_excludes_preparation_refit_readout=True, outcome_readout=False,
        registration_sha256=sha(REG)))
    beat(state='real400update_matched_critic_pilot_complete')


def train(resume=False):
    cfg, values, store, banks, models = admission()
    if (PUBLIC/'training_freeze.json').exists(): raise ValueError('Completed36-head freeze immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']: raise ValueError('Real pilot required')
    start, beat, guard = runtime(cfg)
    old, main = aggregate_groups(frozen.source.parent.NAME), aggregate_groups(frozen.NAME); refs = []
    first = sorted(values[2]['contexts'])[0]
    for context, items in sorted(values[2]['contexts'].items()):
        _, packets, _, changed = frozen.BASE.prepared_context(context, items, values, store, banks, beat)
        for seed in cfg['seeds']:
            data, _, _ = prepare(context, seed, packets, changed, store, models, old, main, beat); states = []
            for arm in cfg['arms']:
                path = checkpoint(context, seed, arm); receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
                state = fit(data, cfg, context, seed, arm, 'heads', beat, guard, resume=resume and path.exists())
                if state['step'] != 2000 or state['parameters'] != 30726 or state['unknown_rows_with_loss']:
                    raise ValueError('Complete registered real training required')
                if context == first and seed == 17:
                    rp = checkpoint(context, seed, arm, 'full_refit')
                    refit = fit(data, cfg, context, seed, arm, 'full_refit', beat, guard, resume=resume and rp.exists())
                    scientific_equal(state, refit)
                    once(PUBLIC/('first_full_refit_'+arm+'.json'), dict(scientific_state_exact=True,
                        updates=2000, checkpoint_sha256=sha(path), refit_checkpoint_sha256=sha(rp)))
                row = dict(context=context, seed=seed, arm=arm, step=state['step'], parameters=state['parameters'],
                    checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                    first_trace=state['trace'][0], final_trace=state['trace'][-1], draw_hash=state['draw_hash'],
                    input_hashes=state['input_hashes'], rms=state['rms'].tolist(), unknown_rows_drawn=state['unknown_rows_drawn'],
                    unknown_rows_with_loss=0, seconds=state['seconds'], registration_sha256=sha(REG),
                    result_source='fresh_native_arm64_Torch_supervised_critic_fit')
                once(receipt, row); refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
                states.append(state); beat(state='critic_head_complete', context=context, seed=seed, arm=arm, heads=len(refs))
            api.assert_matched(*states)
            del states, data
        del changed, packets
    once(PUBLIC/'training_freeze.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, fits=refs,
        optimizer_updates=72000, full_refit_updates=4000, held_scored=False,
        registration_sha256=sha(REG), independent_roles_opened=False))
    beat(state='all36critic_heads_frozen')


def evaluate(verify=False, resume=False):
    cfg, values, store, banks, models = admission()
    if (PUBLIC/'complete.json').exists() and not verify: raise ValueError('Completed readout immutable')
    freeze = json.loads((PUBLIC/'training_freeze.json').read_text()); trained = {}
    for ref in freeze['fits']:
        if sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('Training receipt changed')
        receipt = json.loads((ROOT/ref['path']).read_text()); cp = receipt['checkpoint']
        if sha(ROOT/cp['path']) != cp['sha256']: raise ValueError('Trained critic changed')
        trained[receipt['context'], receipt['seed'], receipt['arm']] = cp
    if len(trained) != 36 or freeze['registration_sha256'] != sha(REG): raise ValueError('Complete fixed36 fits required')
    start, beat, guard = runtime(cfg)
    old, main = aggregate_groups(frozen.source.parent.NAME), aggregate_groups(frozen.NAME)
    previous = aggregate_groups(NAME) if verify else {}
    prior_refs = {Path(v['path']).name: v for v in json.loads((PUBLIC/'complete.json').read_text())['groups']} if verify else {}
    rows, refs = [], []
    for context, items in sorted(values[2]['contexts'].items()):
        _, packets, _, changed = frozen.BASE.prepared_context(context, items, values, store, banks, beat)
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            data, ranges, independent = prepare(context, seed, packets, changed, store, models, old, main, beat)
            states = {arm: api.read_checkpoint(ROOT/trained[context, seed, arm]['path']) for arm in cfg['arms']}
            api.assert_matched(states['row_local'], states['selected_set'])
            for state in states.values():
                api.exact(state['input_hashes'], data['input_hashes'])
            prediction = {arm: api.predict(state, data['causal']) for arm, state in states.items()}
            for name, packet in sorted(packets.items()):
                key = f'{context}_{name}_seed{seed}_fit.json'; path = PUBLIC/'groups'/key; sl = ranges[name]
                progress = PRIVATE/'progress'/(key+'.pt.gz'); progress.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and not verify:
                    if not resume: raise ValueError('Explicit partial evaluation resume required')
                    row = json.loads(path.read_text())
                    if row['registration_sha256'] != sha(REG): raise ValueError('Partial registration changed')
                else:
                    if progress.exists() and not (resume or verify): raise ValueError('Explicit allocation resume required')
                    p = {a: v[sl] for a, v in prediction.items()}
                    choices, ranks, info = decoder.allocate(p, packet['allowed'], packet['recordings'], packet['frames'], packet['ids'],
                        seconds=cfg['solver_seconds_per_query'], checkpoint=(lambda _: None) if verify else lambda v: api.save_checkpoint(progress, v),
                        heartbeat=lambda **kw: beat(state='critic_query_allocation', context=context, source=name, seed=seed, **kw),
                        resume_state=api.read_checkpoint(progress) if resume and progress.exists() and not verify else None)
                    result = evaluation.readout(packet, choices, ranks, info, p, data['causal']['prediction'][sl],
                        data['causal']['initial'][sl], independent[name], job['design']['easy_cut'],
                        values[3]['origin'][packet['ids']], data['rms'])
                    row = dict(context=context, source=name, seed=seed, result=result,
                        ids_sha256=main[key]['ids_sha256'], parent_controls_exact=True, paired_training_verified=True,
                        prediction_sha256={a: frozen.BASE.engine.fingerprint(v) for a, v in p.items()},
                        action_sha256={a: frozen.BASE.engine.fingerprint(v) for a, v in choices.items()},
                        training_freeze_sha256=sha(PUBLIC/'training_freeze.json'), registration_sha256=sha(REG),
                        held_scored=False, independent_roles_opened=False,
                        result_source='fresh_critic_inference_allocation_and_scoring_on_cached_verified_SOURCE_fitting_assets')
                    if verify:
                        if row != previous[key]: raise ValueError('Separate complete critic replay differs')
                    else: once(path, row)
                rows.append(row); refs.append(prior_refs[key] if verify else dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                beat(state='critic_group_complete', groups=len(rows), total=72, context=context, source=name, seed=seed)
            del data, prediction, states
        del packets, changed
    summary = evaluation.summarize(rows, json.loads(REG.read_text())['contexts'], cfg)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()) or refs != json.loads((PUBLIC/'complete.json').read_text())['groups']:
            raise ValueError('Complete critic summary differs')
        once(PUBLIC/'verification.json', dict(all72groups_exact=True, all144parent_full_policy_controls_exact=True,
            summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(PUBLIC/'complete.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, groups=refs,
            summary_sha256=sha(PUBLIC/'summary.json'), training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
            independent_roles_opened=False, deployment_changed=False))
    write(PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=os.getpid(), seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all72critic_groups_verified' if verify else 'all72critic_groups_complete',
        fitting_advancement_gate=summary['fitting_advancement_gate'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'pilot', 'train', 'eval', 'verify')); p.add_argument('--resume', action='store_true')
    args = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'register': admission(register=True)
        elif args.phase == 'pilot': pilot(args.resume)
        elif args.phase == 'train': train(args.resume)
        else: evaluate(args.phase == 'verify', args.resume)
