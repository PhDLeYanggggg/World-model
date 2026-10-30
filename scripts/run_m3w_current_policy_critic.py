"""Matched current-decoder versus frozen-original query supervision."""
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

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_query_unit_critic as prior
from scripts import probe_m3w_selected_scores as diagnosis
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.world_model import m3w_current_policy_critic as api

parent = prior.parent; np = prior.np
NAME = 'european_current_policy_critic_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_current_policy_critic_v1.json'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_current_policy_critic.py',
    ROOT/'tests/test_m3w_current_policy_critic.py', PUBLIC/'protocol.md')
sha, once, write = prior.sha, prior.once, prior.write


def roster(path):
    freeze = json.loads(path.read_text()); out = {}
    for ref in freeze['fits']:
        if sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('Frozen fit receipt changed')
        row = json.loads((ROOT/ref['path']).read_text()); cp = row['checkpoint']
        if sha(ROOT/cp['path']) != cp['sha256']: raise ValueError('Frozen checkpoint changed')
        out[row['context'], row['seed'], row['arm']] = cp
    if len(out) != 36: raise ValueError('Complete36 matched heads required')
    return out


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); old_cfg, values, store, banks, models, _ = prior.admission()
    check = json.loads((diagnosis.PUBLIC/'verification.json').read_text())
    if (cfg['training'] != old_cfg['training'] or cfg['arms'] != old_cfg['arms'] or cfg['seeds'] != old_cfg['seeds']
            or cfg['query_target'] != 'current_decoded' or cfg['risk_budget'] != .02
            or cfg['training_solver_seconds_per_query'] != 2. or cfg['solver_seconds_per_query'] != 2.
            or not check['all72groups_exact'] or check['summary_sha256'] != sha(diagnosis.PUBLIC/'summary.json')
            or any(cfg[k] for k in ('held_scored', 'independent_roles_opened', 'deployment_changed',
                                   'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Verified score diagnosis and otherwise unchanged source-fitting design required')
    complete = json.loads((prior.PUBLIC/'complete.json').read_text())
    if len(complete['groups']) != 72 or complete['summary_sha256'] != sha(prior.PUBLIC/'summary.json'):
        raise ValueError('Complete fixed-query policy control required')
    controls = roster(prior.PUBLIC/'training_freeze.json')
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        prior_registration_sha256=sha(prior.REG), prior_training_freeze_sha256=sha(prior.PUBLIC/'training_freeze.json'),
        prior_complete_sha256=sha(prior.PUBLIC/'complete.json'), prior_summary_sha256=sha(prior.PUBLIC/'summary.json'),
        score_audit_summary_sha256=sha(diagnosis.PUBLIC/'summary.json'), score_audit_verification_sha256=sha(diagnosis.PUBLIC/'verification.json'),
        contexts=json.loads(prior.REG.read_text())['contexts'], cached_verified_fixed_query_heads=36,
        fresh_current_query_heads=36, primary_optimizer_updates=72000, full_refit_updates=4000,
        real_pilot_optimizer_updates=600, changed_factor='query_supervision_action_set_only_current_decoder_stop_gradient',
        pointwise_loss_architecture_and_query_RMS_unchanged=True, frozen_original_context_inputs=True,
        fixed_training_weights=[.5, .5], training_solver_seconds=2., training_incomplete_solver_refuses=True,
        primary='current_query_set_joint_minus_fixed_query_set_joint_full_completion_lower_utility',
        role='exposed_SOURCE_fitting_resubstitution_only', new_trajectory_forecaster_fits=0,
        independent_roles_opened=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel) != digest: raise ValueError('Registered current-query definition changed: '+rel)
            parent.frozen.BASE.engine.committed(ROOT/rel)
        parent.frozen.BASE.engine.committed(REG)
    return cfg, values, store, banks, models, controls


def runtime(cfg):
    parent.api.torch.set_num_threads(4); parent.api.torch.set_num_interop_threads(1); start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume frozen current-query roster')))
    def guard():
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12h healthy-run boundary')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('10GiB reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['private_checkpoint_cap_bytes']:
            raise OSError('256MiB private checkpoint cap')
    def beat(**event):
        guard(); row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE/'heartbeat.json', row)
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    beat(state='runtime_ready', architecture=platform.machine(), device='cpu', threads=4, interop_threads=1, workers=0)
    return start, beat, guard


def checkpoint(context, seed, arm, phase='heads'):
    return PRIVATE/phase/f'{context}_seed{seed}_{arm}.pt.gz'


def fit(data, cfg, context, seed, arm, phase, beat, guard, *, target='current_decoded', **kwargs):
    prior.checked_normalizer(data, context, seed)
    return api.fit(data, target=target, arm=arm, seed=seed, settings=cfg['training'],
        seconds=cfg['training_solver_seconds_per_query'], path=checkpoint(context, seed, arm, phase),
        identity=dict(registration_sha256=sha(REG), context=context, cost_seed=seed),
        heartbeat=lambda **event: beat(context=context, seed=seed, **event), guard=guard, **kwargs)


def pilot(resume=False):
    cfg, values, store, banks, models, _ = admission()
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Completed pilot immutable')
    start, beat, guard = runtime(cfg)
    old, main = aggregate_groups(parent.frozen.source.parent.NAME), aggregate_groups(parent.frozen.NAME)
    context, items = sorted(values[2]['contexts'].items())[0]
    _, packets, _, changed = parent.frozen.BASE.prepared_context(context, items, values, store, banks, beat)
    data, _, _ = parent.prepare(context, 17, packets, changed, store, models, old, main, beat)
    states = []; cells = []
    for arm in cfg['arms']:
        control = fit(data, cfg, context, 17, arm, 'pilot_fixed_control', beat, guard, target='frozen_original',
            stop_at=100, resume=resume and checkpoint(context, 17, arm, 'pilot_fixed_control').exists())
        prior.compare_scientific(control, parent.api.read_checkpoint(prior.checkpoint(context, 17, arm, 'pilot')), identity=False)
        if not checkpoint(context, 17, arm, 'pilot').exists():
            fit(data, cfg, context, 17, arm, 'pilot', beat, guard, stop_at=50)
        recovered = fit(data, cfg, context, 17, arm, 'pilot', beat, guard, resume=True, stop_at=100)
        direct = fit(data, cfg, context, 17, arm, 'pilot_direct', beat, guard, stop_at=100,
            resume=resume and checkpoint(context, 17, arm, 'pilot_direct').exists())
        prior.compare_scientific(recovered, direct); parent.api.assert_matched(control, recovered)
        if recovered['unknown_rows_with_loss'] or recovered['parameters'] != 30726: raise ValueError('Protected matched training required')
        cells.append(dict(arm=arm, actual_optimizer_updates=300, fixed_control_exact=True,
            interrupted_recovery_exact=True, first_trace=recovered['trace'][0], final_trace=recovered['trace'][-1]))
        states.append(recovered)
    parent.api.assert_matched(*states)
    elapsed = time.monotonic()-start
    once(PUBLIC/'pilot.json', dict(pid=os.getpid(), seconds=elapsed,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, actual_optimizer_updates=600,
        cells=cells, engineering_pass=True, single_context_runtime_extrapolation_seconds=elapsed*76000/400,
        extrapolation_not_measured_full_runtime=True, new_policy_outcomes_read=False, registration_sha256=sha(REG)))
    beat(state='real600update_current_query_pilot_complete')


def train(resume=False):
    cfg, values, store, banks, models, controls = admission()
    if (PUBLIC/'training_freeze.json').exists(): raise ValueError('Completed36 current-query fits immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']: raise ValueError('Real recovery pilot required')
    start, beat, guard = runtime(cfg); refs = []
    old, main = aggregate_groups(parent.frozen.source.parent.NAME), aggregate_groups(parent.frozen.NAME)
    first = sorted(values[2]['contexts'])[0]
    for context, items in sorted(values[2]['contexts'].items()):
        _, packets, _, changed = parent.frozen.BASE.prepared_context(context, items, values, store, banks, beat)
        for seed in cfg['seeds']:
            data, _, _ = parent.prepare(context, seed, packets, changed, store, models, old, main, beat); states = []
            for arm in cfg['arms']:
                path = checkpoint(context, seed, arm)
                state = fit(data, cfg, context, seed, arm, 'heads', beat, guard, resume=resume and path.exists())
                parent.api.assert_matched(state, parent.api.read_checkpoint(ROOT/controls[context, seed, arm]['path']))
                if state['step'] != 2000 or state['parameters'] != 30726 or state['unknown_rows_with_loss']:
                    raise ValueError('Full original-budget paired training required')
                if context == first and seed == 17:
                    rp = checkpoint(context, seed, arm, 'full_refit')
                    refit = fit(data, cfg, context, seed, arm, 'full_refit', beat, guard, resume=resume and rp.exists())
                    prior.compare_scientific(state, refit)
                    once(PUBLIC/('first_full_refit_'+arm+'.json'), dict(scientific_state_exact=True, updates=2000,
                        checkpoint_sha256=sha(path), refit_checkpoint_sha256=sha(rp)))
                receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
                once(receipt, dict(context=context, seed=seed, arm=arm, step=state['step'], parameters=state['parameters'],
                    checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                    input_hashes=state['input_hashes'], draw_hash=state['draw_hash'], query_normalizer=state['identity']['query_unit_normalizer'],
                    first_trace=state['trace'][0], final_trace=state['trace'][-1], seconds=state['seconds'],
                    unknown_rows_drawn=state['unknown_rows_drawn'], unknown_rows_with_loss=0,
                    cached_fixed_control=controls[context, seed, arm], paired_original_design_exact=True, registration_sha256=sha(REG)))
                refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt))); states.append(state)
                beat(state='current_query_head_complete', heads=len(refs), context=context, seed=seed, arm=arm)
            parent.api.assert_matched(*states); del data, states
        del packets, changed
    once(PUBLIC/'training_freeze.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, fits=refs,
        fresh_optimizer_updates=72000, exact_refit_updates=4000, cached_verified_fixed_query_heads=36,
        registration_sha256=sha(REG), independent_roles_opened=False))
    beat(state='all36_current_query_critics_frozen')


def summarize(rows, cfg):
    reg = json.loads(REG.read_text()); evaluation = parent.evaluation
    summary = evaluation.summarize(rows, reg['contexts'], cfg)
    interval = lambda get: evaluation.bank.legacy.interval(rows, get, cfg)
    contrasts = {arm: {mode+'_'+bound: interval(lambda r, a=arm, m=mode, b=bound: r['current_query_contrasts'][a][m][b])
        for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')} for arm in ('row', 'set')}
    secondary = summary['fitting_gates']
    gates = dict(complete36fresh_fits72groups=True, all144_fixed_query_controls_exact=all(r['fixed_query_controls_exact'] for r in rows),
        paired_training=secondary['paired_feature_initialization_and_training_draws'], same_count_context_controls=secondary['same_count_controls_every_query'],
        current_minus_fixed_full_lower_CI_positive=contrasts['set']['full_lower_pp']['CI95'] is not None and contrasts['set']['full_lower_pp']['CI95'][0] > 0,
        positive_floor_lower_utility=secondary['context_vs_floor_completion_lower_CI_positive'],
        complete_recording_risk=secondary['complete_context_recording_risk'], main_optima_certified=secondary['all_main_query_optima_certified'])
    summary.update(primary='current_query_set_joint_minus_fixed_original_query_set_joint',
        current_query_contrasts=contrasts, context_secondary_gates=secondary, fitting_gates=gates, fitting_advancement_gate=all(gates.values()),
        fixed_query_accuracy={a: interval(lambda r, a=a: r['policies'][a]['known_ADE_gain_floor_percent'])
            for a in ('fixed_query_row_joint', 'fixed_query_set_joint')},
        fixed_query_safety={a: evaluation.bank.safety(rows, a) for a in ('fixed_query_row_joint', 'fixed_query_set_joint')},
        changed_factor='query_supervision_selection_only', cached_verified_fixed_query_heads=36,
        result_source='fresh_current_query_critic_fit_inference_allocation_exposed_SOURCE_fitting_only')
    return summary


def evaluate(verify=False, resume=False):
    cfg, values, store, banks, models, controls = admission()
    if not json.loads((prior.PUBLIC/'verification.json').read_text())['all72groups_exact']:
        raise ValueError('Separately verified complete fixed-query readout required before new policy scoring')
    if (PUBLIC/'complete.json').exists() and not verify: raise ValueError('Completed current-query readout immutable')
    fitted = roster(PUBLIC/'training_freeze.json'); start, beat, guard = runtime(cfg)
    old, main = aggregate_groups(parent.frozen.source.parent.NAME), aggregate_groups(parent.frozen.NAME)
    fixed_results = aggregate_groups(prior.NAME); previous = aggregate_groups(NAME) if verify else {}; rows = []; refs = []
    previous_refs = {Path(r['path']).name: r for r in json.loads((PUBLIC/'complete.json').read_text())['groups']} if verify else {}
    for context, items in sorted(values[2]['contexts'].items()):
        _, packets, _, changed = parent.frozen.BASE.prepared_context(context, items, values, store, banks, beat)
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            data, ranges, independent = parent.prepare(context, seed, packets, changed, store, models, old, main, beat)
            states = {a: parent.api.read_checkpoint(ROOT/fitted[context, seed, a]['path']) for a in cfg['arms']}
            fixed = {a: parent.api.read_checkpoint(ROOT/controls[context, seed, a]['path']) for a in cfg['arms']}
            for a in cfg['arms']:
                parent.api.assert_matched(states[a], fixed[a]); parent.api.exact(states[a]['input_hashes'], data['input_hashes'])
            p = {a: parent.api.predict(s, data['causal']) for a, s in states.items()}
            p0 = {a: parent.api.predict(s, data['causal']) for a, s in fixed.items()}
            for source, packet in sorted(packets.items()):
                key = f'{context}_{source}_seed{seed}_fit.json'; path = PUBLIC/'groups'/key; sl = ranges[source]
                progress = PRIVATE/'progress'/(key+'.pt.gz'); progress.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and not verify:
                    if not resume: raise ValueError('Explicit partial resume required')
                    row = json.loads(path.read_text())
                else:
                    if progress.exists() and not (verify or resume): raise ValueError('Explicit allocation resume required')
                    new = {a: v[sl] for a, v in p.items()}
                    choices, ranks, info = parent.decoder.allocate(new, packet['allowed'], packet['recordings'], packet['frames'], packet['ids'],
                        seconds=cfg['solver_seconds_per_query'], checkpoint=(lambda _: None) if verify else lambda v: parent.api.save_checkpoint(progress, v),
                        heartbeat=lambda **kw: beat(state='current_query_allocation', context=context, source=source, seed=seed, **kw),
                        resume_state=parent.api.read_checkpoint(progress) if resume and progress.exists() and not verify else None)
                    fixed_choices = parent.api.read_checkpoint(prior.PRIVATE/'progress'/(key+'.pt.gz'))['choices']
                    for prefix, a in (('row', 'row_local'), ('set', 'selected_set')):
                        c = fixed_choices[prefix+'_joint']; pred = p0[a][sl]
                        if (parent.frozen.BASE.engine.fingerprint(pred) != fixed_results[key]['prediction_sha256'][a]
                                or parent.frozen.BASE.engine.fingerprint(c) != fixed_results[key]['action_sha256'][prefix+'_joint']):
                            raise ValueError('Fixed-query critic/control choice changed')
                        label = 'fixed_query_'+prefix+'_joint'; choices[label] = c; rank = np.zeros(len(c)); at = np.flatnonzero(c >= 0)
                        rank[at] = pred[at, c[at], 0]-pred[at, c[at], 1]; ranks[label] = rank
                    result = parent.evaluation.readout(packet, choices, ranks, info, new, data['causal']['prediction'][sl],
                        data['causal']['initial'][sl], independent[source], job['design']['easy_cut'],
                        values[3]['origin'][packet['ids']], data['rms'])
                    contrasts = {}
                    for prefix in ('row', 'set'):
                        left, right = 'fixed_query_'+prefix+'_joint', prefix+'_joint'
                        if result['policies'][left] != fixed_results[key]['result']['policies'][right]:
                            raise ValueError('Fixed full-policy control score changed')
                        contrasts[prefix] = parent.evaluation.bank.pair(packet, choices[left], ranks[left], packet, choices[right], ranks[right])
                    result['current_query_contrasts'] = contrasts
                    row = dict(context=context, source=source, seed=seed, result=result, parent_controls_exact=True,
                        fixed_query_controls_exact=True, paired_training_verified=True, ids_sha256=fixed_results[key]['ids_sha256'],
                        prediction_sha256={a: parent.frozen.BASE.engine.fingerprint(v) for a, v in new.items()},
                        action_sha256={a: parent.frozen.BASE.engine.fingerprint(v) for a, v in choices.items()},
                        training_freeze_sha256=sha(PUBLIC/'training_freeze.json'), registration_sha256=sha(REG), independent_roles_opened=False)
                    if verify:
                        if row != previous[key]: raise ValueError('Separate current-query replay differs')
                    else: once(path, row)
                rows.append(row); refs.append(previous_refs[key] if verify else dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                beat(state='current_query_group_complete', groups=len(rows), total=72, context=context, source=source, seed=seed)
            del data, states, fixed, p, p0
        del packets, changed
    summary = summarize(rows, cfg)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()): raise ValueError('Current-query canonical summary differs')
        once(PUBLIC/'verification.json', dict(all72groups_exact=True, all144_fixed_query_controls_exact=True,
            summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(PUBLIC/'complete.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, groups=refs,
            summary_sha256=sha(PUBLIC/'summary.json'), training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
            independent_roles_opened=False, deployment_changed=False))
    write(PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=os.getpid(), seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all72_current_query_groups_verified' if verify else 'all72_current_query_groups_complete', fitting_advancement_gate=summary['fitting_advancement_gate'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'pilot', 'train', 'eval', 'verify'))
    p.add_argument('--resume', action='store_true'); a = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.phase == 'register': admission(register=True)
        elif a.phase == 'pilot': pilot(a.resume)
        elif a.phase == 'train': train(a.resume)
        else: evaluate(a.phase == 'verify', a.resume)
