"""Real proposal-conditioned risk learning without changing the evaluation rules."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before importing Torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_public_deferral as parent
from src.world_model import m3w_proposal_conditioned_risk as api

old, source = parent.old, parent.source
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/moving_proposal_risk_v1'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/moving_proposal_risk_v1'
CONFIG = ROOT / 'configs/m3w_moving_proposal_risk_v1.json'
REG = PUBLIC / 'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC / 'protocol.md',
    ROOT/'src/world_model/m3w_proposal_conditioned_risk.py',
    ROOT/'tests/test_m3w_proposal_conditioned_risk.py', ROOT/'tests/test_m3w_proposal_risk_runner.py')
POLICIES = ('physical_floor', 'public_router', 'all_action_veto', 'all_action_matched',
    'conditioned_veto', 'conditioned_matched')


def register(beat):
    spec = source.read(CONFIG)
    if (spec['training'] != source.read(parent.CONFIG)['training'] or spec['new_heads'] != 108
            or spec['primary_updates'] != 216000 or spec['parameters'] != 28229
            or spec['risk_budget'] != .02 or spec['easy_net_budget'] != .02
            or any(spec[k] for k in ('new_H_readout', 'threshold_search', 'identity_certification',
                'deployment_changed', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Full source-separated budget and original rules required')
    _, cfg, data, jobs, refs, parts, points = parent.register(beat)
    if source.read(parent.PUBLIC/'verification.json')['analysis_sha256'] != source.sha(parent.PUBLIC/'analysis.json'):
        raise ValueError('Parent must remain verified and immutable')
    routers = {}
    for ref in source.read(parent.PUBLIC/'training_freeze.json')['fits']:
        row = source.read(source.checked(ref)); routers[row['job'], row['part']] = row
    if len(routers) != 54:
        raise ValueError('All frozen proposers required')
    source.once(REG, source.stable(dict(configuration=spec,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        parent_registration=source.artifact(parent.REG),
        parent_freeze=source.artifact(parent.PUBLIC/'training_freeze.json'),
        point_freeze=source.artifact(old.PUBLIC/'training_freeze.json'),
        jobs=jobs, partitions=parts, forecasts=refs,
        original_rules_unchanged=True, original24group_primary='not_estimable_not_passed',
        exposed_SOURCE_only=True, new_H_readout=False, independent_confirmation=False,
        identity_certification=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)))
    return spec, cfg, data, jobs, refs, parts, points, routers


def checkpoint(job, proposer, critic, kind='fits'):
    return PRIVATE/kind/(job['key']+f'_router{proposer}_critic{critic}.pt.gz')


def training_packet(data, job, refs, part, proposer, points, routers):
    p = old.packet(data, job, refs, part, training=False)
    a = api.base.read_checkpoint(source.checked(routers[job['key'], proposer]['checkpoint']))
    api.assert_excluded(a, p['causal'], part['names'])
    logits = api.router.predict(a, p['causal'])
    # Proposals and their loss-support mask are fixed before B supervision is read.
    choice, _ = api.router.choose(logits, p['causal']['allowed'])
    control = api.base.read_checkpoint(source.checked(points[job['key'], part['part']]['checkpoint']))
    if set(control['source_sites']) != set(part['names']):
        raise ValueError('Only own-B control preprocessing may be reused')
    source.np.testing.assert_array_equal(control['fit_ids'], p['ids'])
    target = data['target_eval'][p['ids']]-data['origin'][p['ids'], None]
    learning = old.amplitude.training_data(p, target, data['valid'][p['ids']], job['easy_cut'], control['preprocess'])
    conditioned, second = api.condition(learning, logits)
    source.np.testing.assert_array_equal(choice, second)
    take = choice >= 0
    coverage = dict(rows=len(take), proposed=int(take.sum()),
        proposed_known=int((take & learning['observed_future']).sum()),
        proposed_unknown=int((take & ~learning['observed_future']).sum()),
        queries=len(p['causal']['groups']),
        queries_with_proposals=sum(bool(take[g].any()) for g in p['causal']['groups']),
        action_counts=source.np.bincount(choice+1, minlength=8).tolist())
    if not coverage['proposed']:
        raise ValueError('No proposed training actions: honest hard blocker, not zero-loss success')
    p.update(learning=conditioned, coverage=coverage, proposal_hash=source.array_hash(choice),
        router_checkpoint=routers[job['key'], proposer]['checkpoint'])
    return p, control


def one(p, job, a, b, spec, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return api.fit(p['learning'], seed=job['seed'], settings=spec['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job, proposer_part=a, critic_part=b,
            proposal_hash=p['proposal_hash'], router_checkpoint=p['router_checkpoint'], no_new_H=True),
        path=checkpoint(job, a, b, kind), resume=resume, stop_at=stop_at,
        heartbeat=lambda **v: beat(job=job['key'], proposer=a, critic=b, **v), guard=guard)


def pilot(spec, cfg, data, jobs, refs, parts, points, routers, beat, guard, resume):
    rows = []
    for fold in range(3):
        job = next(j for j in jobs if j['critic_fold'] == fold)
        p, control = training_packet(data, job, refs, parts[job['key']][0], 1, points, routers)
        path = checkpoint(job, 1, 0, 'pilot_resume')
        step = api.base.read_checkpoint(path)['step'] if path.exists() else 0
        if path.exists() and not resume:
            raise ValueError('Explicit resume required')
        if step < 50:
            one(p, job, 1, 0, spec, beat, guard, kind='pilot_resume', resume=path.exists(), stop_at=50)
        a = one(p, job, 1, 0, spec, beat, guard, kind='pilot_resume', resume=True, stop_at=100)
        path = checkpoint(job, 1, 0, 'pilot_direct')
        b = one(p, job, 1, 0, spec, beat, guard, kind='pilot_direct', resume=resume and path.exists(), stop_at=100)
        api.router.exact_state(a, b)
        source.np.testing.assert_array_equal(api.predict(a, p['causal']), api.predict(b, p['causal']))
        if not any(not api.router.torch.equal(b['model'][k], b['initial_model'][k]) for k in b['model']):
            raise ValueError('Real optimizer updates required')
        rows.append(dict(job=job['key'], steps=100, seconds=b['seconds'], parameters=b['parameters'],
            exact_resume=True, budget=b['budget'], coverage=p['coverage']))
        del p, control, a, b; source.gc.collect(); guard()
    source.once(PUBLIC/'pilot.json', source.stable(dict(cells=rows, all3_resume_exact=True, actual_updates=600,
        projected_primary_seconds_not_completion=36*sum(r['seconds']/100*r['budget'] for r in rows),
        validation_readout=False)))


def train(spec, cfg, data, jobs, refs, parts, points, routers, beat, guard, resume):
    if not source.read(PUBLIC/'pilot.json')['all3_resume_exact']:
        raise ValueError('Real resume pilot first')
    receipts = []
    for job in jobs:
        for part in parts[job['key']]:
            b = part['part']
            for a in sorted(set(range(3))-{b}):
                p, control = training_packet(data, job, refs, part, a, points, routers)
                path = checkpoint(job, a, b)
                state = one(p, job, a, b, spec, beat, guard, resume=resume and path.exists())
                if state['step'] != 2000 or state['budget'] != 2000 or state['parameters'] != spec['parameters']:
                    raise ValueError('Complete registered budget required')
                for key in ('initial_model', 'preprocess', 'sampler', 'draw_hash', 'fit_ids', 'source_sites'):
                    api.base.exact(state[key], control[key])
                if not receipts:
                    repeat_path = checkpoint(job, a, b, 'full_refit')
                    repeat = one(p, job, a, b, spec, beat, guard, kind='full_refit', resume=resume and repeat_path.exists())
                    api.router.exact_state(state, repeat)
                    source.once(PUBLIC/'first_full_refit.json', dict(exact=True, checkpoint=source.artifact(repeat_path)))
                    del repeat
                rowpath = PUBLIC/'fits'/(path.name.removesuffix('.pt.gz')+'.json')
                source.once(rowpath, source.stable(dict(job=job['key'], proposer=a, critic=b, seed=job['seed'],
                    checkpoint=source.artifact(path), steps=state['step'], budget=state['budget'],
                    parameters=state['parameters'], seconds=state['seconds'], trace=state['trace'],
                    observed_supervised_draws=state['observed_supervised_draws'],
                    unknown_bound_supervised_draws=state['unknown_bound_supervised_draws'],
                    fit_sources=state['source_sites'], fit_ids_sha256=source.array_hash(state['fit_ids']),
                    proposal_hash=p['proposal_hash'], coverage=p['coverage'],
                    cached_control=points[job['key'], b]['checkpoint'], parent_init_preprocess_sampler_exact=True,
                    validation_readout=False)))
                receipts.append(source.artifact(rowpath))
                beat(state='proposal_conditioned_fit_frozen', job=job['key'], proposer=a, critic=b,
                    complete=len(receipts), total=108)
                del p, control, state; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(fits=receipts, complete_heads=108,
        primary_updates=216000, all_before_validation=True, parent_init_preprocess_sampler_exact=True))


def evaluate(spec, cfg, data, jobs, refs, parts, points, routers, beat, guard, *, verify):
    freeze = source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_heads'] != 108 or not freeze['all_before_validation']:
        raise ValueError('Freeze every registered model before readout')
    fitted = {}
    for ref in freeze['fits']:
        r = source.read(source.checked(ref)); fitted[r['job'], r['proposer'], r['critic']] = r
    records, receipts = [], []
    for job in jobs:
        for part in parts[job['key']]:
            v = part['part']; others = sorted(set(range(3))-{v})
            p = old.packet(data, job, refs, part, training=False); c, ids = p['causal'], p['ids']
            pending = []
            for a in others:
                b = next(k for k in others if k != a)
                predictions, hashes = {}, {}
                for name, ref, predictor in (
                        ('router', routers[job['key'], a], api.router.predict),
                        ('all_action', points[job['key'], b], api.predict),
                        ('conditioned', fitted[job['key'], a, b], api.predict)):
                    state = api.base.read_checkpoint(source.checked(ref['checkpoint']))
                    api.assert_excluded(state, c, part['names'])
                    predictions[name] = predictor(state, c)
                    hashes[name] = source.array_hash(predictions[name])
                chosen = api.causal_choices(predictions['router'], predictions['all_action'], predictions['conditioned'], c)
                key = job['key']+f'_val{v}_propose{a}'
                parent_group = source.read(parent.PUBLIC/'groups'/(key+'.json'))
                for ours, previous in (('public_router', 'public_router'), ('all_action_veto', 'public_veto'),
                        ('all_action_matched', 'public_matched')):
                    if source.array_hash(chosen[ours]) != parent_group['choice_hashes'][previous]:
                        raise ValueError('Cached parent control must replay exactly')
                if (hashes['router'] != parent_group['prediction_hashes'][f'public{a}']
                        or hashes['all_action'] != parent_group['prediction_hashes'][f'point{b}']):
                    raise ValueError('Frozen model predictions changed')
                paths = {name: old.rollout(p['paths'], p['floor'], x) for name, x in chosen.items()}
                pending.append((a, b, key, chosen, paths, predictions, hashes))
            # Validation outcomes are accessed only after both orientation decisions.
            target, valid = data['target_eval'][ids]-data['origin'][ids, None], data['valid'][ids]
            truth = old.amplitude.targets(p, target, valid, job['easy_cut'])
            for a, b, key, chosen, paths, predictions, hashes in pending:
                cache = PRIVATE/'choices'/(key+'.npz'); cache.parent.mkdir(exist_ok=True)
                cached = dict(chosen, ids=ids)
                if cache.exists():
                    with source.np.load(cache, allow_pickle=False) as previous:
                        for name, value in cached.items():
                            source.np.testing.assert_array_equal(previous[name], value)
                elif verify:
                    raise ValueError('Original causal cache missing')
                else:
                    source.np.savez_compressed(cache, **cached)
                rows = []
                for site in sorted(set(p['sites'])):
                    at = p['sites'] == site
                    q = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
                    policies = {name: source.readout.policy(x[at], p['floor'][at], p['cv'][at], target[at], valid[at],
                        p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for name, x in paths.items()}
                    matched, full, costs = {}, {}, {}
                    for name in ('all_action', 'conditioned'):
                        ma, ve = name+'_matched', name+'_veto'
                        matched[name] = source.matched_utility(paths[ma][at], paths[ve][at], p['floor'][at],
                            target[at], valid[at], chosen[ma][at], chosen[ve][at], q)
                        full[name] = source.readout.paired(paths[ma][at], paths[ve][at], p['floor'][at], target[at], valid[at])
                        costs[name] = {subset: old.selected_costs(predictions[name][at], truth['moments'][at],
                            truth['observed_future'][at], chosen[choice_name][at], q)
                            for subset, choice_name in (('same_proposals', 'public_router'), ('admitted', ve))}
                    direct = source.readout.paired(paths['all_action_veto'][at], paths['conditioned_veto'][at],
                        p['floor'][at], target[at], valid[at])
                    rows.append(dict(site=str(site), job=job['key'], seed=job['seed'], validation_part=v,
                        proposer_part=a, critic_part=b, policies=policies, nonempty_matched=matched,
                        secondary_full_strategy=full, conditioned_minus_all_action=direct, cost_reliability=costs))
                path = PUBLIC/'groups'/(key+'.json')
                source.once(path, source.stable(dict(rows=rows, prediction_hashes=hashes,
                    choice_hashes={name: source.array_hash(x) for name, x in chosen.items()},
                    decision_cache=source.artifact(cache), parent_control_replay_exact=True, no_new_H=True)))
                receipts.append(source.artifact(path)); records.extend(rows)
                beat(state='proposal_risk_replay' if verify else 'proposal_risk_validation', complete=len(receipts), total=108)
            del p, pending, truth; source.gc.collect(); guard()
    if len(records) != 276 or len(receipts) != 108:
        raise ValueError('Every required original view retained')
    sites = sorted({r['site'] for r in records})
    ci = lambda getter: source.group_interval(records, getter, sites, cfg)
    measures = ('known_pp', 'completion_lower_pp', 'completion_upper_pp')
    matched = {a: {m: ci(lambda r, a=a, m=m: r['nonempty_matched'][a]['contrast'][m]) for m in measures}
        for a in ('all_action', 'conditioned')}
    full = {a: {m: ci(lambda r, a=a, m=m: r['secondary_full_strategy'][a][m]) for m in measures}
        for a in ('all_action', 'conditioned')}
    direct = {m: ci(lambda r, m=m: r['conditioned_minus_all_action'][m]) for m in measures}
    safety = {}
    for arm in POLICIES:
        views = [view for r in records for view in r['policies'][arm]['recordings'].values()]
        if len(views) != 696:
            raise ValueError('Every original recording view retained')
        safety[arm] = dict(required=696, supported=sum(x['supported'] for x in views),
            known_all_violations=sum(x['known_all_risk'] is not None and x['known_all_risk'] > .02 for x in views),
            known_easy_violations=sum(x['known_easy_risk'] is not None and x['known_easy_risk'] > .02 for x in views),
            undefined_selected_easy=sum(x['completion_easy_risk_upper'] is None for x in views),
            full_original_risk_pass=all(x['supported'] for x in views))
    analysis = source.stable(dict(nonempty_matched=matched, secondary_full_strategy=full,
        conditioned_minus_all_action=direct, safety=safety,
        switch_percent={a: ci(lambda r, a=a: 100*r['policies'][a]['switch_rate']) for a in POLICIES},
        unavailable_primary_views={a: sum(r['nonempty_matched'][a]['contrast']['known_pp'] is None for r in records)
            for a in ('all_action', 'conditioned')}, group_receipts=receipts, source_units=len(sites),
        new_H_readout=False, independent_confirmation=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False))
    source.once(PUBLIC/'analysis.json', analysis)
    primary = matched['conditioned']['completion_lower_pp']
    source.once(PUBLIC/'decision.json', dict(development_criteria_pass=bool(primary['CI95'] is not None
        and primary['CI95'][0] > 0 and safety['conditioned_veto']['full_original_risk_pass']),
        deployment_changed=False, submission_ready=False, original24group_primary='not_estimable_not_passed',
        identity_certification=False, new_H_readout=False, stage5c_executed=False, smc_enabled=False))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all108_policy_blocks_and_prediction_hashes_exact=True,
            parent_control_replay_exact=True, analysis_sha256=source.sha(PUBLIC/'analysis.json'), no_new_H=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        old.previous.teacher.lineage.PRIVATE = PRIVATE
        start, guard, beat = old.previous.teacher.lineage.runtime({}, source.read(CONFIG), args.phase)
        values = register(beat)
        if args.phase == 'register':
            return
        for path in (*FILES, REG):
            body = source.subprocess.run(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT,
                capture_output=True, check=True).stdout
            if source.hashlib.sha256(body).hexdigest() != source.sha(path):
                raise ValueError('Commit scientific definition before real fitting')
        receipt = PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phase is immutable')
        if args.phase in ('pilot', 'train'):
            globals()[args.phase](*values, beat, guard, args.resume)
        else:
            evaluate(*values, beat, guard, verify=args.phase == 'verify')
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_torch=True, new_H_readout=False, new_trajectory_training=False))
        beat(state='complete')


if __name__ == '__main__':
    main()
