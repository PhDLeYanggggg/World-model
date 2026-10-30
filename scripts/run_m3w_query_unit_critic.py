"""One-factor query-unit repair with36 cached and36 freshly fitted critics."""
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
from scripts import run_m3w_selected_query_critic as parent
from scripts import probe_m3w_query_critic_capacity as diagnosis
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.world_model import m3w_query_unit_critic as api

np = parent.np
NAME = 'european_query_unit_critic_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_query_unit_critic_v1.json'; REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_query_unit_critic.py',
    ROOT/'tests/test_m3w_query_unit_critic.py', PUBLIC/'protocol.md')
sha, once, write = parent.sha, parent.once, parent.write


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); old_cfg, values, store, banks, models = parent.admission()
    check = json.loads((diagnosis.PUBLIC/'verification.json').read_text())
    freeze = json.loads((parent.PUBLIC/'training_freeze.json').read_text())
    if (cfg['training'] != old_cfg['training'] or cfg['seeds'] != old_cfg['seeds']
            or cfg['arms'] != old_cfg['arms'] or cfg['normalization'] != 'selected_query'
            or cfg['risk_budget'] != .02 or cfg['query_rms_floor'] != 1e-6
            or cfg['row_risk_loss_weight'] != .5 or cfg['selected_query_score_loss_weight'] != .5
            or not check['all72groups_exact'] or check['summary_sha256'] != sha(diagnosis.PUBLIC/'summary.json')
            or len(freeze['fits']) != 36
            or any(cfg[k] for k in ('held_scored', 'independent_roles_opened', 'deployment_changed',
                                   'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Verified training-unit diagnosis and otherwise unchanged matched experiment required')
    controls = {}
    for ref in freeze['fits']:
        if sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('Original fitting receipt changed')
        r = json.loads((ROOT/ref['path']).read_text()); cp = r['checkpoint']
        if sha(ROOT/cp['path']) != cp['sha256']: raise ValueError('Original control checkpoint changed')
        controls[r['context'], r['seed'], r['arm']] = cp
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parent_registration_sha256=sha(parent.REG), parent_training_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        capacity_summary_sha256=sha(diagnosis.PUBLIC/'summary.json'), capacity_verification_sha256=sha(diagnosis.PUBLIC/'verification.json'),
        contexts=json.loads(parent.REG.read_text())['contexts'], cached_verified_critic_controls=36,
        fresh_fits=36, primary_optimizer_updates=72000, full_refit_updates=4000, real_pilot_updates=600,
        changed_factor='query_score_normalizer_only_original_selected_fitting_target_RMS',
        unchanged_pointwise_objective=True, unchanged_architecture_sampler_and_decoder=True,
        primary='new_query_units_set_joint_minus_old_units_set_joint_full_completion_lower_utility',
        parent_policy_readout_complete_at_registration=(parent.PUBLIC/'complete.json').exists(),
        design_from_training_and_capacity_not_parent_policy_scores=True,
        role='exposed_SOURCE_fitting_resubstitution_only', risk_budget=.02,
        independent_roles_opened=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel) != digest: raise ValueError('Registered query-unit implementation changed: '+rel)
            parent.frozen.BASE.engine.committed(ROOT/rel)
        parent.frozen.BASE.engine.committed(REG)
    return cfg, values, store, banks, models, controls


def runtime(cfg):
    api.parent.torch.set_num_threads(4); api.parent.torch.set_num_interop_threads(1)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume frozen query-unit roster')))
    def guard():
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12h boundary')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('10GiB reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['private_checkpoint_cap_bytes']:
            raise OSError('256MiB checkpoint cap')
    def beat(**event):
        guard(); row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE/'heartbeat.json', row)
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    beat(state='runtime_ready', architecture=platform.machine(), device='cpu', threads=4, workers=0)
    return start, beat, guard


def checked_normalizer(data, context, seed):
    normalizer = api.query_normalizer(data)
    source = json.loads((diagnosis.PUBLIC/'summary.json').read_text())
    row = next(r for r in source['heads'] if (r['context'], r['seed']) == (context, seed))
    np.testing.assert_allclose(normalizer['rms'], np.maximum(row['label_query_score_RMS_source_balanced'], 1e-6), rtol=1e-12, atol=1e-14)
    return normalizer


def checkpoint(context, seed, arm, phase='heads'):
    return PRIVATE/phase/f'{context}_seed{seed}_{arm}.pt.gz'


def fit(data, cfg, context, seed, arm, phase, beat, guard, *, normalization='selected_query', **kw):
    checked_normalizer(data, context, seed)
    return api.fit(data, normalization=normalization, arm=arm, seed=seed, settings=cfg['training'],
        identity=dict(registration_sha256=sha(REG), context=context, cost_seed=seed),
        path=checkpoint(context, seed, arm, phase),
        heartbeat=lambda **event: beat(context=context, seed=seed, **event), guard=guard, **kw)


def compare_scientific(left, right, identity=True):
    for key in left:
        if key != 'seconds' and (identity or key != 'identity'): api.parent.exact(left[key], right[key])


def pilot(resume=False):
    cfg, values, store, banks, models, _ = admission()
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Completed pilot immutable')
    start, beat, guard = runtime(cfg)
    old, main = aggregate_groups(parent.frozen.source.parent.NAME), aggregate_groups(parent.frozen.NAME)
    context, items = sorted(values[2]['contexts'].items())[0]
    _, packets, _, changed = parent.frozen.BASE.prepared_context(context, items, values, store, banks, beat)
    data, _, _ = parent.prepare(context, 17, packets, changed, store, models, old, main, beat)
    states, cells = [], []
    for arm in cfg['arms']:
        control = fit(data, cfg, context, 17, arm, 'pilot_old_control', beat, guard, normalization='all_action',
            stop_at=100, resume=resume and checkpoint(context, 17, arm, 'pilot_old_control').exists())
        cached = api.parent.read_checkpoint(parent.checkpoint(context, 17, arm, 'pilot'))
        compare_scientific(control, cached, identity=False)
        path = checkpoint(context, 17, arm, 'pilot')
        if not path.exists(): fit(data, cfg, context, 17, arm, 'pilot', beat, guard, stop_at=50)
        recovered = fit(data, cfg, context, 17, arm, 'pilot', beat, guard, resume=True, stop_at=100)
        direct = fit(data, cfg, context, 17, arm, 'pilot_direct', beat, guard, stop_at=100,
            resume=resume and checkpoint(context, 17, arm, 'pilot_direct').exists())
        compare_scientific(recovered, direct)
        api.parent.assert_matched(control, recovered)
        if recovered['unknown_rows_with_loss'] or recovered['parameters'] != 30726: raise ValueError('Real protected training required')
        states.append(recovered); cells.append(dict(arm=arm, real_updates=300, exact_cached_old_control=True,
            exact_interrupted_recovery=True, first_trace=recovered['trace'][0], final_trace=recovered['trace'][-1]))
    api.parent.assert_matched(*states)
    once(PUBLIC/'pilot.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, actual_optimizer_updates=600,
        cells=cells, engineering_pass=True, current_policy_outcomes_read=False, registration_sha256=sha(REG)))
    beat(state='real600update_query_unit_pilot_complete')


def train(resume=False):
    cfg, values, store, banks, models, controls = admission()
    if (PUBLIC/'training_freeze.json').exists(): raise ValueError('Completed36 new fits immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']: raise ValueError('Real pilot required')
    start, beat, guard = runtime(cfg)
    old, main = aggregate_groups(parent.frozen.source.parent.NAME), aggregate_groups(parent.frozen.NAME)
    first = sorted(values[2]['contexts'])[0]; refs = []
    for context, items in sorted(values[2]['contexts'].items()):
        _, packets, _, changed = parent.frozen.BASE.prepared_context(context, items, values, store, banks, beat)
        for seed in cfg['seeds']:
            data, _, _ = parent.prepare(context, seed, packets, changed, store, models, old, main, beat); states = []
            for arm in cfg['arms']:
                path = checkpoint(context, seed, arm)
                state = fit(data, cfg, context, seed, arm, 'heads', beat, guard, resume=resume and path.exists())
                api.parent.assert_matched(state, api.parent.read_checkpoint(ROOT/controls[context, seed, arm]['path']))
                if state['step'] != 2000 or state['parameters'] != 30726 or state['unknown_rows_with_loss']:
                    raise ValueError('Full original-budget matched training required')
                if context == first and seed == 17:
                    rp = checkpoint(context, seed, arm, 'full_refit')
                    repeated = fit(data, cfg, context, seed, arm, 'full_refit', beat, guard, resume=resume and rp.exists())
                    compare_scientific(state, repeated)
                    once(PUBLIC/('first_full_refit_'+arm+'.json'), dict(scientific_state_exact=True, updates=2000,
                        checkpoint_sha256=sha(path), refit_checkpoint_sha256=sha(rp)))
                receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
                once(receipt, dict(context=context, seed=seed, arm=arm, step=state['step'], parameters=state['parameters'],
                    checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                    input_hashes=state['input_hashes'], draw_hash=state['draw_hash'], query_normalizer=state['identity']['query_unit_normalizer'],
                    first_trace=state['trace'][0], final_trace=state['trace'][-1], seconds=state['seconds'],
                    unknown_rows_drawn=state['unknown_rows_drawn'], unknown_rows_with_loss=0,
                    cached_control=controls[context, seed, arm], paired_old_design_exact=True, registration_sha256=sha(REG)))
                refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt))); states.append(state)
                beat(state='unit_critic_head_complete', heads=len(refs), context=context, seed=seed, arm=arm)
            api.parent.assert_matched(*states); del data, states
        del changed, packets
    once(PUBLIC/'training_freeze.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, fits=refs,
        fresh_optimizer_updates=72000, exact_refit_updates=4000, cached_verified_control_heads=36,
        registration_sha256=sha(REG), independent_roles_opened=False))
    beat(state='all36_query_unit_critics_frozen')


def evaluate(verify=False, resume=False):
    cfg, values, store, banks, models, controls = admission()
    original_verification = json.loads((parent.PUBLIC/'verification.json').read_text())
    if not original_verification['all72groups_exact']: raise ValueError('Complete separately verified original critic readout required')
    if (PUBLIC/'complete.json').exists() and not verify: raise ValueError('Completed new readout immutable')
    freeze = json.loads((PUBLIC/'training_freeze.json').read_text()); fitted = {}
    for ref in freeze['fits']:
        if sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('New fit receipt changed')
        r = json.loads((ROOT/ref['path']).read_text()); cp = r['checkpoint']
        if sha(ROOT/cp['path']) != cp['sha256']: raise ValueError('New trained checkpoint changed')
        fitted[r['context'], r['seed'], r['arm']] = cp
    if len(fitted) != 36: raise ValueError('Complete36 frozen new heads required')
    start, beat, guard = runtime(cfg)
    old, main = aggregate_groups(parent.frozen.source.parent.NAME), aggregate_groups(parent.frozen.NAME)
    old_results = aggregate_groups(parent.NAME); previous = aggregate_groups(NAME) if verify else {}
    oldrefs = {Path(r['path']).name: r for r in json.loads((PUBLIC/'complete.json').read_text())['groups']} if verify else {}
    rows, refs = [], []
    for context, items in sorted(values[2]['contexts'].items()):
        _, packets, _, changed = parent.frozen.BASE.prepared_context(context, items, values, store, banks, beat)
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            data, ranges, independent = parent.prepare(context, seed, packets, changed, store, models, old, main, beat)
            states = {a: api.parent.read_checkpoint(ROOT/fitted[context, seed, a]['path']) for a in cfg['arms']}
            old_states = {a: api.parent.read_checkpoint(ROOT/controls[context, seed, a]['path']) for a in cfg['arms']}
            for arm in cfg['arms']:
                api.parent.assert_matched(states[arm], old_states[arm])
                api.parent.exact(states[arm]['input_hashes'], data['input_hashes'])
            p = {a: api.parent.predict(s, data['causal']) for a, s in states.items()}
            old_p = {a: api.parent.predict(s, data['causal']) for a, s in old_states.items()}
            for name, packet in sorted(packets.items()):
                key = f'{context}_{name}_seed{seed}_fit.json'; sl = ranges[name]; path = PUBLIC/'groups'/key
                progress = PRIVATE/'progress'/(key+'.pt.gz'); progress.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and not verify:
                    if not resume: raise ValueError('Explicit partial resume required')
                    row = json.loads(path.read_text())
                else:
                    if progress.exists() and not (verify or resume): raise ValueError('Explicit action resume required')
                    new = {a: v[sl] for a, v in p.items()}
                    choices, ranks, info = parent.decoder.allocate(new, packet['allowed'], packet['recordings'], packet['frames'], packet['ids'],
                        seconds=cfg['solver_seconds_per_query'], checkpoint=(lambda _: None) if verify else lambda v: api.parent.save_checkpoint(progress, v),
                        heartbeat=lambda **kw: beat(state='unit_critic_allocation', context=context, source=name, seed=seed, **kw),
                        resume_state=api.parent.read_checkpoint(progress) if resume and progress.exists() and not verify else None)
                    old_choices = api.parent.read_checkpoint(parent.PRIVATE/'progress'/(key+'.pt.gz'))['choices']
                    for prefix, arm in (('row', 'row_local'), ('set', 'selected_set')):
                        if parent.frozen.BASE.engine.fingerprint(old_p[arm][sl]) != old_results[key]['prediction_sha256'][arm]:
                            raise ValueError('Original critic inference changed')
                        c = old_choices[prefix+'_joint']
                        if parent.frozen.BASE.engine.fingerprint(c) != old_results[key]['action_sha256'][prefix+'_joint']:
                            raise ValueError('Original critic choice changed')
                        label = 'old_units_'+prefix+'_joint'; choices[label] = c
                        at = np.flatnonzero(c >= 0); rank = np.zeros(len(c)); pred = old_p[arm][sl]
                        rank[at] = pred[at, c[at], 0]-pred[at, c[at], 1]; ranks[label] = rank
                    result = parent.evaluation.readout(packet, choices, ranks, info, new, data['causal']['prediction'][sl],
                        data['causal']['initial'][sl], independent[name], job['design']['easy_cut'],
                        values[3]['origin'][packet['ids']], data['rms'])
                    comparisons = {}
                    for prefix in ('row', 'set'):
                        old_name, new_name = 'old_units_'+prefix+'_joint', prefix+'_joint'
                        if result['policies'][old_name] != old_results[key]['result']['policies'][new_name]:
                            raise ValueError('Original full critic policy score changed')
                        comparisons[prefix] = parent.evaluation.bank.pair(packet, choices[old_name], ranks[old_name],
                            packet, choices[new_name], ranks[new_name])
                    result['query_unit_contrasts'] = comparisons
                    row = dict(context=context, source=name, seed=seed, result=result, parent_controls_exact=True,
                        old_critic_controls_exact=True, paired_training_verified=True, ids_sha256=old_results[key]['ids_sha256'],
                        prediction_sha256={a: parent.frozen.BASE.engine.fingerprint(v) for a, v in new.items()},
                        action_sha256={a: parent.frozen.BASE.engine.fingerprint(c) for a, c in choices.items()},
                        training_freeze_sha256=sha(PUBLIC/'training_freeze.json'), registration_sha256=sha(REG),
                        independent_roles_opened=False, result_source='fresh_query_unit_critic_inference_allocation_SOURCE_fitting_only')
                    if verify:
                        if row != previous[key]: raise ValueError('Separate query-unit replay differs')
                    else: once(path, row)
                rows.append(row); refs.append(oldrefs[key] if verify else dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                beat(state='unit_group_complete', groups=len(rows), total=72, context=context, source=name, seed=seed)
            del data, states, old_states, p, old_p
        del changed, packets
    summary = parent.evaluation.summarize(rows, json.loads(REG.read_text())['contexts'], cfg)
    interval = lambda get: parent.evaluation.bank.legacy.interval(rows, get, cfg)
    contrasts = {arm: {mode+'_'+bound: interval(lambda r, a=arm, m=mode, b=bound: r['query_unit_contrasts'][a][m][b])
        for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')} for arm in ('row', 'set')}
    context_gates = summary['fitting_gates']
    gates = dict(complete36fresh_fits72groups=True, all144old_critic_controls_exact=all(r['old_critic_controls_exact'] for r in rows),
        original_matched_training=context_gates['paired_feature_initialization_and_training_draws'],
        same_count_context_controls=context_gates['same_count_controls_every_query'],
        new_units_set_minus_old_units_lower_CI_positive=contrasts['set']['full_lower_pp']['CI95'] is not None and contrasts['set']['full_lower_pp']['CI95'][0] > 0,
        positive_floor_lower_utility=context_gates['context_vs_floor_completion_lower_CI_positive'],
        complete_recording_risk=context_gates['complete_context_recording_risk'],
        main_optima_certified=context_gates['all_main_query_optima_certified'])
    summary.update(primary='query_unit_repair_set_joint_minus_old_units_set_joint',
        cached_verified_critic_controls=36, query_unit_contrasts=contrasts, context_secondary_gates=context_gates,
        fitting_gates=gates, fitting_advancement_gate=all(gates.values()),
        old_units_accuracy={a: interval(lambda r, a=a: r['policies'][a]['known_ADE_gain_floor_percent'])
            for a in ('old_units_row_joint', 'old_units_set_joint')},
        old_units_safety={a: parent.evaluation.bank.safety(rows, a) for a in ('old_units_row_joint', 'old_units_set_joint')})
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()): raise ValueError('Query-unit summary differs')
        once(PUBLIC/'verification.json', dict(all72groups_exact=True, all144old_critic_controls_exact=True,
            summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(PUBLIC/'complete.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, groups=refs,
            summary_sha256=sha(PUBLIC/'summary.json'), training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
            independent_roles_opened=False, deployment_changed=False))
    write(PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=os.getpid(), seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all72_query_unit_groups_verified' if verify else 'all72_query_unit_groups_complete',
        fitting_advancement_gate=summary['fitting_advancement_gate'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'pilot', 'train', 'eval', 'verify'))
    p.add_argument('--resume', action='store_true'); a = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.phase == 'register': admission(register=True)
        elif a.phase == 'pilot': pilot(a.resume)
        elif a.phase == 'train': train(a.resume)
        else: evaluate(a.phase == 'verify', a.resume)
