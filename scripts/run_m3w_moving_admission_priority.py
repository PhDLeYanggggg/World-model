"""Fixed warm-error priority versus matched continued neural risk training."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_kinematic_risk as parent
from src.world_model import m3w_admission_priority as api

source, old, direct = parent.source, parent.old, parent.direct
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_admission_risk_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_admission_risk_v1'
CONFIG = ROOT/'configs/m3w_moving_admission_priority_v1.json'
REG = PUBLIC/'registration_v2.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/world_model/m3w_admission_priority.py',
    ROOT/'tests/test_m3w_admission_priority.py')
POLICIES = parent.POLICIES+tuple(a+s for a in api.ARMS for s in ('_veto', '_matched'))


def register(beat):
    cfg = source.read(CONFIG)
    _, original, data, jobs, refs, parts, points, routers, critics, fitted, refits = parent.register(beat)
    if (cfg['arms'] != list(api.ARMS) or cfg['priority_multiplier'] != api.MULTIPLIER or cfg['new_heads'] != 6
            or cfg['primary_updates'] != 12000 or cfg['training'] != source.read(parent.CONFIG)['training']
            or cfg['risk_budget'] != .02 or cfg['easy_net_budget'] != .02
            or any(cfg[k] for k in ('new_H_readout', 'threshold_search', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed objective comparison and unchanged original rules required')
    if source.read(parent.PUBLIC/'verification.json')['analysis_sha256'] != source.sha(parent.PUBLIC/'analysis.json'):
        raise ValueError('Previous negative result must remain verified')
    warm = {}
    for ref in source.read(parent.PUBLIC/'training_freeze.json')['fits']:
        r = source.read(source.checked(ref)); warm[r['job'], r['arm']] = r
    source.once(REG, source.stable(dict(configuration=cfg, jobs=jobs,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        parent_registration=source.artifact(parent.REG), warm_freeze=source.artifact(parent.PUBLIC/'training_freeze.json'),
        previous_negative=source.artifact(parent.PUBLIC/'analysis.json'),
        fitting_only_diagnosis=source.artifact(PUBLIC/'fitting_loss_decomposition.json'),
        proposer_part=1, fitting_part=0, validation_part=2, exposed_SOURCE_only=True,
        original_rules_unchanged=True, original24group_primary='not_estimable_not_passed',
        independent_confirmation=False, no_new_H=True, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)))
    return cfg, original, data, jobs, refs, parts, points, routers, critics, fitted, refits, warm


def fitting(data, job, refs, parts, points, routers, fitted, warm_refs):
    d, _ = parent.fitting(data, job, refs, parts, points, routers, fitted)
    warm = api.base.read_checkpoint(source.checked(warm_refs[job['key'], 'kinematic']['checkpoint']))
    if warm['step'] != 2000 or warm['parameters'] != 32130:
        raise ValueError('Completed frozen kinematic neural warm model required')
    return api.prepare(d, warm), warm


def path_for(job, arm, kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.pt.gz')


def one(d, warm, job, arm, cfg, warm_refs, beat, guard, kind='fits', resume=False, stop_at=None):
    return api.fit(d, warm, arm=arm, settings=cfg['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job,
            warm_checkpoint=warm_refs[job['key'], 'kinematic']['checkpoint']),
        path=path_for(job, arm, kind), resume=resume, stop_at=stop_at, guard=guard,
        heartbeat=lambda **v: beat(job=job['key'], kind=kind, **v))


def pilot(cfg, original, data, jobs, refs, parts, points, routers, critics, fitted, refits, warm_refs, beat, guard, resume):
    job = jobs[0]; d, warm = fitting(data, job, refs, parts, points, routers, fitted, warm_refs)
    model = api.initialize(warm); api.base.exact(model.state_dict(), warm['model'])
    rows = []
    for arm in api.ARMS:
        path = path_for(job, arm, 'pilot_resume'); step = api.base.read_checkpoint(path)['step'] if path.exists() else 0
        if step < 50:
            one(d, warm, job, arm, cfg, warm_refs, beat, guard, 'pilot_resume', resume and path.exists(), 50)
        a = one(d, warm, job, arm, cfg, warm_refs, beat, guard, 'pilot_resume', True, 100)
        b = one(d, warm, job, arm, cfg, warm_refs, beat, guard, 'pilot_direct',
            resume and path_for(job, arm, 'pilot_direct').exists(), 100)
        api.parent.parent.parent.router.exact_state(a, b)
        source.np.testing.assert_array_equal(api.predict(a, d['causal']), api.predict(b, d['causal']))
        rows.append(dict(arm=arm, steps=100, exact_resume=True, parameters=b['parameters']))
    source.once(PUBLIC/'pilot.json', dict(rows=rows, both_resume_predictions_exact=True,
        initial_warm_weights_exact=True, actual_updates_including_repeats=400, no_validation_used=True))


def train(cfg, original, data, jobs, refs, parts, points, routers, critics, fitted, refits, warm_refs, beat, guard, resume):
    if not source.read(PUBLIC/'pilot.json')['both_resume_predictions_exact']:
        raise ValueError('Real resume pilot first')
    receipts = []
    for job in jobs:
        d, warm = fitting(data, job, refs, parts, points, routers, fitted, warm_refs); states = []
        for arm in api.ARMS:
            s = one(d, warm, job, arm, cfg, warm_refs, beat, guard,
                resume=resume and path_for(job, arm).exists())
            if s['step'] != 2000 or s['parameters'] != 32130 or not all(int(v['step']) == 2000 for v in s['optimizer']['state'].values()):
                raise ValueError('Complete actual neural optimizer updates required')
            if api.torch.equal(s['model']['encoder.weight'], warm['model']['encoder.weight']):
                raise ValueError('Common neural encoder must actually train')
            for key in ('sampler', 'draw_hash', 'preprocess', 'fit_ids', 'source_sites'):
                api.base.exact(s[key], warm[key])
            if not receipts:
                repeat = one(d, warm, job, arm, cfg, warm_refs, beat, guard, 'full_refit',
                    resume and path_for(job, arm, 'full_refit').exists())
                api.parent.parent.parent.router.exact_state(s, repeat)
                source.once(PUBLIC/'first_full_refit.json', dict(exact=True,
                    checkpoint=source.artifact(path_for(job, arm, 'full_refit'))))
            out = PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            source.once(out, source.stable(dict(job=job['key'], arm=arm,
                checkpoint=source.artifact(path_for(job, arm)), warm_checkpoint=warm_refs[job['key'], 'kinematic']['checkpoint'],
                steps=s['step'], parameters=s['parameters'], seconds=s['seconds'], trace=s['trace'],
                initial_model_hash=source.array_hash(s['initial_model']['encoder.weight'].numpy()),
                input_hash=s['input_hash'], target_hash=s['target_hash'], proposal_hash=s['proposal_hash'],
                priority_mask_hash=s['priority_mask_hash'], warm_prediction_hash=s['warm_prediction_hash'],
                priority_channels=d['priority_mask'].sum((0, 1)).tolist(),
                source_sites=s['source_sites'], neural_encoder_updated=True, validation_readout=False)))
            receipts.append(source.artifact(out)); states.append(s)
            beat(state='priority_fit_frozen', complete=len(receipts), total=6)
        api.assert_matched(*states)
        del d, warm, states; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(fits=receipts, complete_heads=6,
        primary_updates=12000, additional_full_refit_updates=2000, all_before_validation=True))


def evaluate(cfg, original, data, jobs, refs, parts, points, routers, critics, fitted, refits, warm_refs, beat, guard, *, verify):
    freeze = source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_heads'] != 6 or not freeze['all_before_validation']:
        raise ValueError('Freeze all six actual fits before any held-source evaluation')
    trained = {}
    for ref in freeze['fits']:
        r = source.read(source.checked(ref)); trained[r['job'], r['arm']] = r
    held, fitting_rows, receipts = [], [], []
    for job in jobs:
        for part_id in (0, 2):
            part = parts[job['key']][part_id]
            p = old.packet(data, job, refs, part, training=False); c, ids = p['causal'], p['ids']
            choices, predictions, hashes = direct.decisions(p, job, part, points, routers, critics, fitted, part_id == 2)
            states = {'readout': (parent.parent.api, refits[job['key']]['checkpoint'])}
            states.update({a: (api.parent, warm_refs[job['key'], a]['checkpoint']) for a in api.parent.ARMS})
            states.update({a: (api, trained[job['key'], a]['checkpoint']) for a in api.ARMS})
            for arm, (model_api, ref) in states.items():
                s = api.base.read_checkpoint(source.checked(ref))
                if part_id == 2:
                    if arm == 'readout':
                        api.parent.parent.parent.assert_excluded(s, c, part['names'])
                    else:
                        parent.assert_excluded(s, c, part['names'])
                predictions[arm] = model_api.predict(s, c); hashes[arm] = source.array_hash(predictions[arm])
                choices[arm+'_veto'], choices[arm+'_matched'] = api.parent.parent.choose(
                    predictions['router'], predictions['all_action'], predictions[arm], c)
            target = data['target_eval'][ids]-data['origin'][ids, None]; valid = data['valid'][ids]
            truth = old.amplitude.targets(p, target, valid, job['easy_cut'])
            paths = {name: old.rollout(p['paths'], p['floor'], choice) for name, choice in choices.items()}
            rows = []
            for site in sorted(set(p['sites'])):
                at = p['sites'] == site
                policies = {name: source.readout.policy(v[at], p['floor'][at], p['cv'][at], target[at], valid[at],
                    p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for name, v in paths.items()}
                local_truth = dict(moments=truth['moments'][at], observed_future=truth['observed_future'][at])
                contrasts = {name: source.readout.paired(paths[left][at], paths[right][at], p['floor'][at], target[at], valid[at])
                    for name, left, right in (('priority_minus_continued', 'continued_mse_veto', 'false_safe_priority_veto'),
                        ('priority_minus_warm', 'kinematic_veto', 'false_safe_priority_veto'),
                        ('priority_minus_affine', 'readout_veto', 'false_safe_priority_veto'))}
                q = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
                matched = {arm: source.matched_utility(paths[arm+'_matched'][at], paths[arm+'_veto'][at], p['floor'][at],
                    target[at], valid[at], choices[arm+'_matched'][at], choices[arm+'_veto'][at], q) for arm in api.ARMS}
                reliability = {arm: direct.reliability(predictions[arm][at], local_truth,
                    dict(proposals=choices['public_router'][at], admitted=choices[arm+'_veto'][at])) for arm in ('kinematic', *api.ARMS)}
                flow = {arm: parent.parent.flow(local_truth, choices['public_router'][at], choices['kinematic_veto'][at],
                    choices[arm+'_veto'][at]) for arm in api.ARMS}
                rows.append(dict(job=job['key'], site=str(site), policies=policies, contrasts=contrasts,
                    matched=matched, reliability=reliability, action_flow=flow))
            if part_id == 0:
                fitting_rows.extend(rows)
            else:
                key = job['key']+'_val2_propose1'; previous = source.read(parent.PUBLIC/'groups'/(key+'.json'))
                for name, h in previous['prediction_hashes'].items():
                    api.base.exact(h, hashes[name])
                for name, h in previous['choice_hashes'].items():
                    api.base.exact(h, source.array_hash(choices[name]))
                for r in rows:
                    prev = next(v for v in previous['rows'] if v['site'] == r['site'])
                    for name in parent.POLICIES:
                        api.base.exact(r['policies'][name], prev['policies'][name])
                cache = PRIVATE/'choices'/(key+'.npz'); cache.parent.mkdir(parents=True, exist_ok=True)
                payload = dict(ids=ids, **{a+'_prediction': predictions[a] for a in api.ARMS}, **choices)
                if cache.exists():
                    with source.np.load(cache, allow_pickle=False) as saved:
                        for name, value in payload.items():
                            source.np.testing.assert_array_equal(saved[name], value)
                elif verify:
                    raise ValueError('Saved original inference required for verification')
                else:
                    source.np.savez_compressed(cache, **payload)
                out = PUBLIC/'groups'/(key+'.json')
                source.once(out, source.stable(dict(rows=rows, prediction_hashes=hashes,
                    choice_hashes={k: source.array_hash(v) for k, v in choices.items()},
                    cache=source.artifact(cache), old_controls_exact=True)))
                receipts.append(source.artifact(out)); held.extend(rows)
                beat(state='fixed_priority_readout', complete=len(receipts), total=3)
            del p, predictions, paths, truth, states; source.gc.collect(); guard()
    source.once(PUBLIC/'fitting_readout.json', source.stable(fitting_rows))
    macro = parent.macro
    policy = lambda rs: {name: dict(known=macro(rs, lambda r, n=name: r['policies'][n]['known_ADE_gain_floor_percent']),
        hard=macro(rs, lambda r, n=name: r['policies'][n]['subsets']['hard']['gain_percent']),
        switch_percent=macro(rs, lambda r, n=name: 100*r['policies'][n]['switch_rate']), safety=direct.safety(rs, name)) for name in POLICIES}
    contrasts = {scope: {name: {k: macro(rs, lambda r, n=name, k=k: r['contrasts'][n][k])
        for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')} for name in rs[0]['contrasts']}
        for scope, rs in (('fit', fitting_rows), ('held', held))}
    contexts = []
    for job in jobs:
        rs = [r for r in held if r['job'] == job['key']]
        control, priority = (direct.safety(rs, a+'_veto') for a in api.ARMS)
        lower = macro(rs, lambda r: r['contrasts']['priority_minus_continued']['completion_lower_pp'])['mean']
        active = sum(r['policies']['false_safe_priority_veto']['switch_rate'] for r in rs) > 0
        condition = lower is not None and lower > 0 and active and all(priority[k] <= control[k] for k in
            ('known_all_violations', 'known_easy_violations', 'undefined_selected_easy', 'undefined_full_easy_net'))
        contexts.append(dict(job=job['key'], completion_lower_pp=lower, control_safety=control,
            priority_safety=priority, expansion_condition=bool(condition)))
    gain = contrasts['held']['priority_minus_affine']['completion_lower_pp']['mean']
    result = dict(held_policy=policy(held), fitting_policy=policy(fitting_rows), contrasts=contrasts,
        per_context=contexts, group_receipts=receipts, sources=len({r['site'] for r in held}), source_views=len(held),
        expansion_supported=all(c['expansion_condition'] for c in contexts) and gain is not None and gain > 0,
        unavailable_matched_source_views={a: sum(r['matched'][a]['contrast']['known_pp'] is None for r in held) for a in api.ARMS},
        original24group_primary='not_estimable_not_passed', no_new_H=True, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'analysis.json', source.stable(result))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all3_predictions_choices_metrics_exact=True,
            all_parent_controls_exact=True, analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        old.previous.teacher.lineage.PRIVATE = PRIVATE
        started, guard, beat = old.previous.teacher.lineage.runtime({}, source.read(CONFIG), args.phase)
        values = register(beat)
        if args.phase == 'register':
            return
        for p in (*FILES, REG):
            content = source.subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT,
                capture_output=True, check=True).stdout
            if source.hashlib.sha256(content).hexdigest() != source.sha(p):
                raise ValueError('Commit exact objective comparison before fitting')
        receipt = PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phases remain immutable')
        if args.phase in ('pilot', 'train'):
            globals()[args.phase](*values, beat, guard, args.resume)
        else:
            evaluate(*values, beat, guard, verify=args.phase == 'verify')
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_torch=True, no_new_H=True, new_encoder_training=args.phase in ('pilot', 'train')))
        beat(state='complete')


if __name__ == '__main__':
    main()
