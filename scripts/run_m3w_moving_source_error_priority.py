"""Matched in-source versus source-held error discovery; original rules unchanged."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_admission_priority as parent
from src.world_model import m3w_source_error_priority as api

source, old, direct = parent.source, parent.old, parent.direct
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_source_error_priority_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_source_error_priority_v1'
CONFIG = ROOT/'configs/m3w_moving_source_error_priority_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/world_model/m3w_source_error_priority.py',
    ROOT/'tests/test_m3w_source_error_priority.py')


def register(beat):
    cfg = source.read(CONFIG); values = parent.register(beat)
    jobs, parts = values[3], values[5]
    if (cfg['arms'] != list(api.ARMS) or cfg['teacher_heads'] != 8 or cfg['student_heads'] != 6
            or cfg['training'] != source.read(parent.CONFIG)['training'] or cfg['priority_multiplier'] != 10
            or cfg['risk_budget'] != .02 or cfg['easy_net_budget'] != .02
            or any(cfg[k] for k in ('new_H_readout', 'threshold_search', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed new model comparison and unchanged rules required')
    if source.read(parent.PUBLIC/'verification.json')['analysis_sha256'] != source.sha(parent.PUBLIC/'analysis.json'):
        raise ValueError('Prior negative must remain exactly verified')
    source_counts = [len(parts[j['key']][0]['names']) for j in jobs]
    if source_counts != [2, 3, 3]:
        raise ValueError('Same three fixed B contexts required')
    source.once(REG, source.stable(dict(configuration=cfg,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        previous_registration=source.artifact(parent.REG), previous_result=source.artifact(parent.PUBLIC/'analysis.json'),
        warm_freeze=source.artifact(parent.parent.PUBLIC/'training_freeze.json'),
        jobs=jobs, partitions={j['key']: parts[j['key']] for j in jobs}, B_sources_per_context=source_counts,
        original_rules_unchanged=True, no_new_H=True, independent_confirmation=False,
        original24group_primary='not_estimable_not_passed', submission_ready=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)))
    return cfg, values


def subpart(data, job, part, names):
    if not names or not set(names) < set(part['names']):
        raise ValueError('Strict nonempty subset of B required')
    ids = source.np.flatnonzero(source.np.isin(data['sites'], names))
    routing = dict(part['routing_job'], evaluation_sites=sorted(names),
        critic_sites=sorted(set(job['critic_sites'])-set(names)),
        evaluation_scope=dict(rows=len(ids), ids_sha256=source.array_hash(ids)))
    return dict(part=part['part'], names=sorted(names), routing_job=routing)


def teacher_packet(values, job, omitted):
    _, _, data, _, refs, parts, _, routers, *_ = values
    part = parts[job['key']][0]
    p = old.packet(data, job, refs, subpart(data, job, part, set(part['names'])-{omitted}), training=True)
    router = api.base.read_checkpoint(source.checked(routers[job['key'], 1]['checkpoint']))
    direct.parent.api.assert_excluded(router, p['causal'], p['sites'])
    logits = direct.parent.api.router.predict(router, p['causal'])
    d, _ = direct.parent.api.condition(p['learning'], logits)
    return api.teacher_data(d)


def path_for(job, name, kind):
    return PRIVATE/kind/(job['key']+'_'+name+'.pt.gz')


def teacher_fit(d, job, omitted, cfg, beat, guard, *, kind='teachers', resume=False, stop_at=None):
    warm = api.scratch(cfg['training'], job['seed'])
    return api.parent.fit(d, warm, arm='continued_mse', settings=cfg['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job['key'], omitted_source=omitted,
            role='scratch_inner_B_error_discovery', initialization='random_no_full_B_weights'),
        path=path_for(job, omitted, kind), resume=resume, stop_at=stop_at, guard=guard,
        heartbeat=lambda **v: beat(job=job['key'], omitted_source=omitted, kind=kind, **v))


def student_data(values, job):
    _, _, data, _, refs, parts, points, routers, _, fitted, _, warm_refs = values
    return parent.fitting(data, job, refs, parts, points, routers, fitted, warm_refs)


def student_fit(d, warm, job, arm, cfg, mask_receipt, beat, guard, *, kind='students', resume=False):
    return api.parent.fit(d, warm, arm='false_safe_priority', settings=cfg['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job['key'], discovery_arm=arm,
            discovery=mask_receipt, warm_freeze=source.artifact(parent.parent.PUBLIC/'training_freeze.json')),
        path=path_for(job, arm, kind), resume=resume, guard=guard,
        heartbeat=lambda **v: beat(job=job['key'], discovery_arm=arm, kind=kind, **v))


def pilot(cfg, values, beat, guard, resume):
    job = values[3][0]; omitted = values[5][job['key']][0]['names'][0]
    d = teacher_packet(values, job, omitted); path = path_for(job, omitted, 'pilot_resume')
    step = api.base.read_checkpoint(path)['step'] if path.exists() else 0
    if step < 50:
        teacher_fit(d, job, omitted, cfg, beat, guard, kind='pilot_resume', resume=resume and path.exists(), stop_at=50)
    a = teacher_fit(d, job, omitted, cfg, beat, guard, kind='pilot_resume', resume=True, stop_at=100)
    b = teacher_fit(d, job, omitted, cfg, beat, guard, kind='pilot_direct',
        resume=resume and path_for(job, omitted, 'pilot_direct').exists(), stop_at=100)
    api.kin.parent.parent.router.exact_state(a, b)
    source.np.testing.assert_array_equal(api.predict(a, d['causal']), api.predict(b, d['causal']))
    if a['warm_step'] != 0 or omitted in a['source_sites']:
        raise ValueError('Real scratch teacher must exclude its held source')
    source.once(PUBLIC/'pilot.json', dict(exact_resume=True, updates_including_repeat=200,
        parameters=b['parameters'], fitting_sources=b['source_sites'], omitted_source=omitted,
        observed_rows=int(d['observed_future'].sum()), rows=len(d['causal']['ids']),
        seconds=b['seconds'], validation_readout=False))


def teachers(cfg, values, beat, guard, resume):
    if not source.read(PUBLIC/'pilot.json')['exact_resume']:
        raise ValueError('Actual resume pilot required')
    receipts, masks = [], []
    for job in values[3]:
        names = values[5][job['key']][0]['names']; predictions, provenance, refs = {}, {}, []
        full = old.packet(values[2], job, values[4], values[5][job['key']][0], training=False)
        for omitted in names:
            d = teacher_packet(values, job, omitted)
            state = teacher_fit(d, job, omitted, cfg, beat, guard,
                resume=resume and path_for(job, omitted, 'teachers').exists())
            held = old.packet(values[2], job, values[4],
                subpart(values[2], job, values[5][job['key']][0], {omitted}), training=False)
            api.assert_excluded(state, held['causal'], [omitted])
            if state['step'] != 2000 or state['parameters'] != 32130 or state['warm_step'] != 0:
                raise ValueError('Complete real scratch teacher required')
            if not receipts:
                repeat = teacher_fit(d, job, omitted, cfg, beat, guard, kind='teacher_full_refit',
                    resume=resume and path_for(job, omitted, 'teacher_full_refit').exists())
                api.kin.parent.parent.router.exact_state(state, repeat)
                source.once(PUBLIC/'teacher_full_refit.json', dict(exact=True,
                    checkpoint=source.artifact(path_for(job, omitted, 'teacher_full_refit'))))
                del repeat
            predictions[omitted] = api.predict(state, full['causal']); provenance[omitted] = state['source_sites']
            out = PUBLIC/'teachers'/(job['key']+'_'+omitted+'.json')
            source.once(out, source.stable(dict(job=job['key'], omitted_source=omitted,
                checkpoint=source.artifact(path_for(job, omitted, 'teachers')), steps=state['step'],
                parameters=state['parameters'], seconds=state['seconds'], trace=state['trace'],
                source_sites=state['source_sites'], fit_ids_hash=source.array_hash(state['fit_ids']),
                fit_rows=len(state['fit_ids']), preprocessing_sources=state['preprocess']['source_names'],
                extra_preprocessing_sources=state['extra_preprocess']['source_names'],
                prediction_hash=source.array_hash(predictions[omitted]), warm_step=0,
                held_source_excluded_from_weights_and_preprocessing=True, validation_readout=False)))
            refs.append(source.artifact(out)); receipts.append(source.artifact(out))
            beat(state='scratch_error_teacher_complete', complete=len(receipts), total=8)
            del d, held, state; source.gc.collect(); guard()
        discovered, mapping = api.discovery_pair(full['sites'], predictions, provenance)
        cache = PRIVATE/'discovery'/(job['key']+'.npz'); cache.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(ids=full['ids'], **discovered)
        if cache.exists():
            with source.np.load(cache, allow_pickle=False) as saved:
                for k, v in payload.items():
                    source.np.testing.assert_array_equal(v, saved[k])
        else:
            source.np.savez_compressed(cache, **payload)
        d, warm = student_data(values, job)
        out = PUBLIC/'discovery'/(job['key']+'.json')
        source.once(out, source.stable(dict(job=job['key'], mapping=mapping, teachers=refs,
            cache=source.artifact(cache), ids_hash=source.array_hash(full['ids']),
            prediction_hashes={a: source.array_hash(v) for a, v in discovered.items()},
            priority_channels={a: api.prioritize(d, v)['priority_mask'].sum((0, 1)).tolist() for a, v in discovered.items()},
            unknown_priority_channels=0, B_only=True, V_used=False)))
        masks.append(source.artifact(out))
        del full, d, warm, predictions, discovered; source.gc.collect(); guard()
    source.once(PUBLIC/'teacher_freeze.json', dict(teachers=receipts, discovery=masks,
        complete_heads=8, primary_updates=16000, additional_refit_updates=2000, all_before_V=True))


def students(cfg, values, beat, guard, resume):
    freeze = source.read(PUBLIC/'teacher_freeze.json')
    if freeze['complete_heads'] != 8 or not freeze['all_before_V']:
        raise ValueError('All discovery heads and masks must be frozen first')
    receipts = []
    for job in values[3]:
        d, warm = student_data(values, job); states = []
        mr = next(ref for ref in freeze['discovery'] if source.read(source.checked(ref))['job'] == job['key'])
        discovery = source.read(source.checked(mr))
        with source.np.load(source.checked(discovery['cache']), allow_pickle=False) as saved:
            source.np.testing.assert_array_equal(saved['ids'], d['causal']['ids'])
            for arm in api.ARMS:
                chosen = api.prioritize(d, saved[arm])
                s = student_fit(chosen, warm, job, arm, cfg, mr, beat, guard,
                    resume=resume and path_for(job, arm, 'students').exists())
                if s['step'] != 2000 or s['parameters'] != 32130:
                    raise ValueError('All 2000 real neural updates required')
                if not receipts:
                    repeat = student_fit(chosen, warm, job, arm, cfg, mr, beat, guard, kind='student_full_refit',
                        resume=resume and path_for(job, arm, 'student_full_refit').exists())
                    api.kin.parent.parent.router.exact_state(s, repeat)
                    source.once(PUBLIC/'student_full_refit.json', dict(exact=True,
                        checkpoint=source.artifact(path_for(job, arm, 'student_full_refit'))))
                    del repeat
                out = PUBLIC/'students'/(job['key']+'_'+arm+'.json')
                source.once(out, source.stable(dict(job=job['key'], arm=arm,
                    checkpoint=source.artifact(path_for(job, arm, 'students')), discovery=mr,
                    steps=s['step'], parameters=s['parameters'], seconds=s['seconds'], trace=s['trace'],
                    priority_mask_hash=s['priority_mask_hash'], prediction_hash=s['warm_prediction_hash'],
                    priority_channels=chosen['priority_mask'].sum((0, 1)).tolist(),
                    source_sites=s['source_sites'], input_hash=s['input_hash'], target_hash=s['target_hash'],
                    validation_readout=False)))
                receipts.append(source.artifact(out)); states.append(s)
                beat(state='error_priority_student_complete', complete=len(receipts), total=6)
        api.assert_student_pair(*states)
        del d, warm, states; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(students=receipts, teacher_freeze=source.artifact(PUBLIC/'teacher_freeze.json'),
        complete_heads=6, primary_updates=12000, additional_refit_updates=2000, all_before_V=True))


def evaluate(cfg, values, beat, guard, verify=False):
    freeze = source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_heads'] != 6 or not freeze['all_before_V']:
        raise ValueError('All six students frozen before V readout')
    trained = {}
    for ref in freeze['students']:
        r = source.read(source.checked(ref)); trained[r['job'], r['arm']] = r
    _, _, data, jobs, refs, parts, points, routers, critics, fitted, *_ = values
    all_rows, receipts = [], []
    for job in jobs:
        part = parts[job['key']][2]; p = old.packet(data, job, refs, part, training=False)
        c, ids = p['causal'], p['ids']; key = job['key']+'_val2_propose1'
        previous = source.read(parent.PUBLIC/'groups'/(key+'.json'))
        _, original_predictions, _ = direct.decisions(p, job, part, points, routers, critics, fitted, True)
        with source.np.load(source.checked(previous['cache']), allow_pickle=False) as saved:
            source.np.testing.assert_array_equal(saved['ids'], ids)
            choices = {name: saved[name].copy() for name in previous['choice_hashes']}
            for name, choice in choices.items():
                if source.array_hash(choice) != previous['choice_hashes'][name]:
                    raise ValueError('Frozen old action hashes must match')
        predictions = {}
        for arm in api.ARMS:
            state = api.base.read_checkpoint(source.checked(trained[job['key'], arm]['checkpoint']))
            api.assert_excluded(state, c, part['names'])
            predictions[arm] = api.predict(state, c)
            choices[arm+'_veto'], choices[arm+'_matched'] = api.kin.parent.choose(
                original_predictions['router'], original_predictions['all_action'], predictions[arm], c)
        paths = {name: old.rollout(p['paths'], p['floor'], choice) for name, choice in choices.items()}
        target = data['target_eval'][ids]-data['origin'][ids, None]; valid = data['valid'][ids]
        rows = []
        for site in sorted(set(p['sites'])):
            at = p['sites'] == site
            policies = {name: source.readout.policy(v[at], p['floor'][at], p['cv'][at], target[at], valid[at],
                p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for name, v in paths.items()}
            old_row = next(r for r in previous['rows'] if r['site'] == site)
            for name, policy in old_row['policies'].items():
                api.base.exact(policies[name], policy)
            contrasts = {name: source.readout.paired(paths[left][at], paths[right][at], p['floor'][at], target[at], valid[at])
                for name, left, right in (('held_minus_in_source', 'in_source_errors_veto', 'held_source_errors_veto'),
                    ('held_minus_warm', 'kinematic_veto', 'held_source_errors_veto'),
                    ('held_minus_affine', 'readout_veto', 'held_source_errors_veto'))}
            groups = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
            matched = {a: source.matched_utility(paths[a+'_matched'][at], paths[a+'_veto'][at], p['floor'][at],
                target[at], valid[at], choices[a+'_matched'][at], choices[a+'_veto'][at], groups) for a in api.ARMS}
            rows.append(dict(job=job['key'], site=str(site), policies=policies, contrasts=contrasts, matched=matched))
        payload = dict(ids=ids, **{a+'_prediction': v for a, v in predictions.items()}, **choices)
        cache = PRIVATE/'evaluation'/(key+'.npz'); cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            with source.np.load(cache, allow_pickle=False) as saved:
                for k, v in payload.items():
                    source.np.testing.assert_array_equal(saved[k], v)
        elif verify:
            raise ValueError('Original predictions required for replay')
        else:
            source.np.savez_compressed(cache, **payload)
        out = PUBLIC/'groups'/(key+'.json')
        source.once(out, source.stable(dict(rows=rows, cache=source.artifact(cache),
            prediction_hashes={a: source.array_hash(v) for a, v in predictions.items()},
            choice_hashes={a: source.array_hash(v) for a, v in choices.items()},
            old_controls='cached_verified_actions_fresh_metric_recomputation_exact')))
        all_rows.extend(rows); receipts.append(source.artifact(out))
        beat(state='held_source_error_readout', complete=len(receipts), total=3)
        del p, state, predictions, paths; source.gc.collect(); guard()
    macro = parent.parent.macro
    policies = {name: dict(known=macro(all_rows, lambda r, n=name: r['policies'][n]['known_ADE_gain_floor_percent']),
        hard=macro(all_rows, lambda r, n=name: r['policies'][n]['subsets']['hard']['gain_percent']),
        switch_percent=macro(all_rows, lambda r, n=name: 100*r['policies'][n]['switch_rate']),
        safety=direct.safety(all_rows, name)) for name in all_rows[0]['policies']}
    contrasts = {name: {k: macro(all_rows, lambda r, n=name, k=k: r['contrasts'][n][k])
        for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')} for name in all_rows[0]['contrasts']}
    contexts = []
    for job in jobs:
        rs = [r for r in all_rows if r['job'] == job['key']]
        a, b = (direct.safety(rs, name+'_veto') for name in api.ARMS)
        lower = macro(rs, lambda r: r['contrasts']['held_minus_in_source']['completion_lower_pp'])['mean']
        active = sum(r['policies']['held_source_errors_veto']['switch_rate'] for r in rs) > 0
        ok = lower is not None and lower > 0 and active and all(b[k] <= a[k] for k in
            ('known_all_violations', 'known_easy_violations', 'undefined_selected_easy', 'undefined_full_easy_net'))
        contexts.append(dict(job=job['key'], completion_lower_pp=lower, control_safety=a,
            held_error_safety=b, expansion_condition=bool(ok)))
    lower = contrasts['held_minus_affine']['completion_lower_pp']['mean']
    result = dict(status='fresh_run', held_policy=policies, contrasts=contrasts, per_context=contexts,
        sources=len({r['site'] for r in all_rows}), group_receipts=receipts,
        expansion_supported=all(v['expansion_condition'] for v in contexts) and lower is not None and lower > 0,
        unavailable_matched_source_views={a: sum(r['matched'][a]['contrast']['known_pp'] is None for r in all_rows) for a in api.ARMS},
        original24group_primary='not_estimable_not_passed', independent_confirmation=False,
        no_new_H=True, original_rules_unchanged=True, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'analysis.json', source.stable(result))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all3_predictions_actions_metrics_exact=True,
            old_cached_control_metrics_exact=True, analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'teachers', 'students', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        old.previous.teacher.lineage.PRIVATE = PRIVATE
        started, guard, beat = old.previous.teacher.lineage.runtime({}, source.read(CONFIG), args.phase)
        cfg, values = register(beat)
        if args.phase == 'register':
            return
        for p in (*FILES, REG):
            content = source.subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT,
                capture_output=True, check=True).stdout
            if source.hashlib.sha256(content).hexdigest() != source.sha(p):
                raise ValueError('Commit the exact prospective comparison before fitting')
        receipt = PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phases remain immutable')
        if args.phase in ('pilot', 'teachers', 'students'):
            globals()[args.phase](cfg, values, beat, guard, args.resume)
        else:
            evaluate(cfg, values, beat, guard, args.phase == 'verify')
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_torch=True, no_new_H=True, new_encoder_training=args.phase in ('pilot', 'teachers', 'students')))
        beat(state='complete')


if __name__ == '__main__':
    main()
