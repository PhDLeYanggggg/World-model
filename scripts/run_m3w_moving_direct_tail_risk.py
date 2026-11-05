"""Six-head SOURCE pilot of directly predicted mean versus upper-tail risk."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_proposal_risk as parent
from src.world_model import m3w_direct_tail_risk as api

old, source = parent.old, parent.source
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_direct_tail_risk_pilot_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_direct_tail_risk_pilot_v1'
CONFIG = ROOT/'configs/m3w_moving_direct_tail_risk_pilot_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md',
    ROOT/'src/world_model/m3w_direct_tail_risk.py', ROOT/'tests/test_m3w_direct_tail_risk.py')
POLICIES = parent.POLICIES + ('mean_veto', 'mean_matched', 'quantile90_veto', 'quantile90_matched')


def register(beat):
    spec = source.read(CONFIG)
    _, cfg, data, jobs, refs, parts, points, routers = parent.register(beat)
    if (spec['training'] != source.read(parent.CONFIG)['training'] or spec['new_heads'] != 6
            or spec['primary_updates'] != 12000 or spec['parameters'] != 28034
            or spec['arms'] != list(api.ARMS) or spec['quantile'] != .9
            or spec['risk_budget'] != .02 or spec['easy_net_budget'] != .02
            or any(spec[k] for k in ('new_H_readout', 'threshold_search', 'identity_certification',
                'deployment_changed', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Registered pilot only; no automatic rule or budget change')
    selected = [next(j for j in jobs if j['critic_fold'] == f) for f in range(3)]
    critics = {}
    for ref in source.read(parent.PUBLIC/'training_freeze.json')['fits']:
        r = source.read(source.checked(ref)); critics[r['job'], r['proposer'], r['critic']] = r
    if source.read(parent.PUBLIC/'verification.json')['analysis_sha256'] != source.sha(parent.PUBLIC/'analysis.json'):
        raise ValueError('Verified frozen parent required')
    source.once(REG, source.stable(dict(configuration=spec, jobs=selected,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        parent_registration=source.artifact(parent.REG), parent_training=source.artifact(parent.PUBLIC/'training_freeze.json'),
        original_rules_unchanged=True, proposer_part=1, critic_part=0, validation_part=2,
        exposed_SOURCE_only=True, new_H_readout=False, pilot_not_full=True,
        identity_certification=False, original24group_primary='not_estimable_not_passed',
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)))
    return spec, cfg, data, selected, refs, parts, points, routers, critics


def checkpoint(job, arm, kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.pt.gz')


def one(p, job, arm, spec, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return api.fit(p['learning'], arm=arm, seed=job['seed'], settings=spec['training'],
        identity=dict(registration_sha256=source.sha(REG), job=job, proposer_part=1, critic_part=0,
            proposal_hash=p['proposal_hash'], router_checkpoint=p['router_checkpoint']),
        path=checkpoint(job, arm, kind), resume=resume, stop_at=stop_at, guard=guard,
        heartbeat=lambda **v: beat(job=job['key'], **v))


def pilot(spec, cfg, data, jobs, refs, parts, points, routers, critics, beat, guard, resume):
    job = jobs[0]
    p, _ = parent.training_packet(data, job, refs, parts[job['key']][0], 1, points, routers)
    rows = []
    for arm in api.ARMS:
        path = checkpoint(job, arm, 'smoke_resume')
        step = api.base.read_checkpoint(path)['step'] if path.exists() else 0
        if path.exists() and not resume:
            raise ValueError('Explicit resume required')
        if step < 50:
            one(p, job, arm, spec, beat, guard, kind='smoke_resume', resume=path.exists(), stop_at=50)
        a = one(p, job, arm, spec, beat, guard, kind='smoke_resume', resume=True, stop_at=100)
        path = checkpoint(job, arm, 'smoke_direct')
        b = one(p, job, arm, spec, beat, guard, kind='smoke_direct', resume=resume and path.exists(), stop_at=100)
        api.parent.router.exact_state(a, b)
        source.np.testing.assert_array_equal(api.predict(a, p['causal']), api.predict(b, p['causal']))
        rows.append(dict(arm=arm, exact_resume=True, seconds=b['seconds'], parameters=b['parameters']))
    source.once(PUBLIC/'pilot.json', dict(actual_smoke_updates=400, rows=rows,
        all_resume_exact=True, validation_readout=False))


def train(spec, cfg, data, jobs, refs, parts, points, routers, critics, beat, guard, resume):
    if not source.read(PUBLIC/'pilot.json')['all_resume_exact']:
        raise ValueError('Real resume smoke test required')
    receipts = []
    for job in jobs:
        p, control = parent.training_packet(data, job, refs, parts[job['key']][0], 1, points, routers)
        states = []
        for arm in api.ARMS:
            path = checkpoint(job, arm)
            state = one(p, job, arm, spec, beat, guard, resume=resume and path.exists())
            if state['step'] != 2000 or state['budget'] != 2000 or state['parameters'] != 28034:
                raise ValueError('Full registered pilot update budget required')
            for key in ('preprocess', 'sampler', 'draw_hash', 'fit_ids', 'source_sites'):
                api.base.exact(state[key], control[key])
            if not receipts:
                path2 = checkpoint(job, arm, 'full_refit')
                repeat = one(p, job, arm, spec, beat, guard, kind='full_refit', resume=resume and path2.exists())
                api.parent.router.exact_state(state, repeat)
                source.once(PUBLIC/'first_full_refit.json', dict(exact=True, checkpoint=source.artifact(path2)))
            for opt in state['optimizer']['state'].values():
                if int(opt['step']) != 2000:
                    raise ValueError('Actual optimizer updates required')
            if not any(not api.torch.equal(v, state['initial_model'][k]) for k, v in state['model'].items()):
                raise ValueError('Real weight changes required')
            rowpath = PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            source.once(rowpath, source.stable(dict(job=job['key'], arm=arm, checkpoint=source.artifact(path),
                steps=state['step'], parameters=state['parameters'], seconds=state['seconds'], trace=state['trace'],
                observed_draws=state['observed_supervised_draws'], bound_draws=state['unknown_bound_supervised_draws'],
                coverage=p['coverage'], cached_control=points[job['key'], 0]['checkpoint'],
                parent_preprocess_sampler_exact=True, validation_readout=False)))
            receipts.append(source.artifact(rowpath)); states.append(state)
            beat(state='direct_tail_fit_frozen', completed=len(receipts), total=6)
        api.assert_matched(*states)
        del p, control, states; source.gc.collect(); guard()
    source.once(PUBLIC/'training_freeze.json', dict(fits=receipts, complete_heads=6,
        primary_updates=12000, all_before_validation=True, matched_loss_only=True))


def reliability(excess, truth, choices):
    result = {}
    for name, chosen in choices.items():
        rows = source.np.flatnonzero((chosen >= 0) & truth['observed_future'])
        p = excess[rows, chosen[rows]]
        y = api.signed(truth['moments'][rows, chosen[rows]])
        result[name] = dict(known_rows=len(rows), unknown_rows=int(((chosen >= 0) & ~truth['observed_future']).sum()),
            predicted_excess_sum=p.sum(0).tolist(), actual_excess_sum=y.sum(0).tolist(),
            underprediction_fraction=(y > p).mean(0).tolist() if len(rows) else None,
            positive_actual_excess_fraction=(y > 0).mean(0).tolist() if len(rows) else None,
            mse=((y-p)**2).mean(0).tolist() if len(rows) else None,
            descriptive_available_labels_only=True)
    return result


def decisions(p, job, part, points, routers, critics, fitted, excluded):
    c = p['causal']; states, predictions, hashes = {}, {}, {}
    for name, ref, predict in (
            ('router', routers[job['key'], 1], api.parent.router.predict),
            ('all_action', points[job['key'], 0], api.parent.predict),
            ('conditioned', critics[job['key'], 1, 0], api.parent.predict),
            *[(arm, fitted[job['key'], arm], api.predict) for arm in api.ARMS]):
        state = api.base.read_checkpoint(source.checked(ref['checkpoint']))
        if excluded or name == 'router':
            api.parent.assert_excluded(state, c, part['names'])
        states[name] = state
        predictions[name] = predict(state, c); hashes[name] = source.array_hash(predictions[name])
    chosen = api.parent.causal_choices(predictions['router'], predictions['all_action'], predictions['conditioned'], c)
    for arm in api.ARMS:
        chosen[arm+'_veto'], chosen[arm+'_matched'] = api.choose(predictions['router'], predictions['all_action'], predictions[arm], c)
    return chosen, predictions, hashes


def safety(rows, name):
    views = [v for r in rows for v in r['policies'][name]['recordings'].values()]
    easy = [100*v['known_easy_net_degradation'] for v in views if v['known_easy_net_degradation'] is not None]
    return dict(required=len(views), supported=sum(v['supported'] for v in views),
        known_all_violations=sum(v['known_all_risk'] is not None and v['known_all_risk'] > .02 for v in views),
        known_easy_violations=sum(v['known_easy_risk'] is not None and v['known_easy_risk'] > .02 for v in views),
        undefined_selected_easy=sum(v['completion_easy_risk_upper'] is None for v in views),
        undefined_full_easy_net=sum(v['known_easy_net_degradation'] is None for v in views),
        worst_defined_signed_easy_net_percent=max(easy) if easy else None,
        full_original_risk_pass=all(v['supported'] for v in views))


def evaluate(spec, cfg, data, jobs, refs, parts, points, routers, critics, beat, guard, *, verify):
    freeze = source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_heads'] != 6 or not freeze['all_before_validation']:
        raise ValueError('All six models must be frozen before validation')
    fitted = {}
    for ref in freeze['fits']:
        r = source.read(source.checked(ref)); fitted[r['job'], r['arm']] = r
    groups, records, fit_diagnostics = [], [], []
    for job in jobs:
        for part_id in (0, 2):
            part = parts[job['key']][part_id]
            p = old.packet(data, job, refs, part, training=False); ids = p['ids']
            chosen, predictions, hashes = decisions(p, job, part, points, routers, critics, fitted, part_id == 2)
            # No outcomes or their validity masks enter any model or policy decision.
            target = data['target_eval'][ids]-data['origin'][ids, None]
            valid = data['valid'][ids]
            truth = old.amplitude.targets(p, target, valid, job['easy_cut'])
            diag = {arm: reliability(predictions[arm], truth,
                dict(proposals=chosen['public_router'], admitted=chosen[arm+'_veto'])) for arm in api.ARMS}
            if part_id == 0:
                fit_diagnostics.append(dict(job=job['key'], scope='fitting_not_generalization', reliability=diag))
                continue
            key = job['key']+'_val2_propose1'
            previous = source.read(parent.PUBLIC/'groups'/(key+'.json'))
            for name in parent.POLICIES:
                if source.array_hash(chosen[name]) != previous['choice_hashes'][name]:
                    raise ValueError('Every frozen parent policy must reproduce')
            for name in ('router', 'all_action', 'conditioned'):
                if hashes[name] != previous['prediction_hashes'][name]:
                    raise ValueError('Frozen prediction hash changed')
            cache = PRIVATE/'choices'/(key+'.npz'); cache.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(chosen, ids=ids, mean_prediction=predictions['mean'], quantile_prediction=predictions['quantile90'])
            if cache.exists():
                with source.np.load(cache, allow_pickle=False) as saved:
                    for name, value in payload.items():
                        source.np.testing.assert_array_equal(saved[name], value)
            elif verify:
                raise ValueError('Original decisions missing')
            else:
                source.np.savez_compressed(cache, **payload)
            paths = {name: old.rollout(p['paths'], p['floor'], choice) for name, choice in chosen.items()}
            rows = []
            for site in sorted(set(p['sites'])):
                at = p['sites'] == site
                q = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
                policies = {name: source.readout.policy(x[at], p['floor'][at], p['cv'][at], target[at], valid[at],
                    p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for name, x in paths.items()}
                prev = next(r for r in previous['rows'] if r['site'] == site)
                for name in parent.POLICIES:
                    if policies[name] != prev['policies'][name]:
                        raise ValueError('Frozen control metrics changed')
                paired = {name: source.readout.paired(paths[left][at], paths[right][at], p['floor'][at], target[at], valid[at])
                    for name, left, right in (
                        ('quantile_minus_mean', 'mean_veto', 'quantile90_veto'),
                        ('mean_minus_conditioned', 'conditioned_veto', 'mean_veto'),
                        ('quantile_minus_conditioned', 'conditioned_veto', 'quantile90_veto'))}
                matched = {arm: source.matched_utility(paths[arm+'_matched'][at], paths[arm+'_veto'][at], p['floor'][at],
                    target[at], valid[at], chosen[arm+'_matched'][at], chosen[arm+'_veto'][at], q) for arm in api.ARMS}
                rows.append(dict(job=job['key'], site=str(site), policies=policies, paired=paired, matched=matched))
            path = PUBLIC/'groups'/(key+'.json')
            source.once(path, source.stable(dict(rows=rows, reliability=diag, prediction_hashes=hashes,
                choice_hashes={k: source.array_hash(v) for k, v in chosen.items()}, cache=source.artifact(cache),
                parent_controls_exact=True, scope='exposed_SOURCE_pilot')))
            groups.append(source.artifact(path)); records.extend(rows)
            beat(state='direct_tail_readout', complete=len(groups), total=3)
            del p, predictions, paths, truth; source.gc.collect(); guard()
    source.once(PUBLIC/'fitting_diagnostics.json', source.stable(fit_diagnostics))
    sites = sorted({r['site'] for r in records})

    def macro(rows, getter):
        by_site = {}
        for site in sorted({r['site'] for r in rows}):
            vals = [getter(r) for r in rows if r['site'] == site]
            by_site[site] = float(source.np.mean(vals)) if all(v is not None for v in vals) else None
        return dict(by_source=by_site, mean=float(source.np.mean(list(by_site.values())))
            if by_site and all(v is not None for v in by_site.values()) else None, CI95=None, pilot_descriptive=True)

    policy = {name: dict(known=macro(records, lambda r, n=name: r['policies'][n]['known_ADE_gain_floor_percent']),
        hard=macro(records, lambda r, n=name: r['policies'][n]['subsets']['hard']['gain_percent']),
        switch_percent=macro(records, lambda r, n=name: 100*r['policies'][n]['switch_rate']), safety=safety(records, name))
        for name in POLICIES}
    paired = {name: {m: macro(records, lambda r, n=name, m=m: r['paired'][n][m])
        for m in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')}
        for name in records[0]['paired']}
    per_context = []
    for job in jobs:
        rows = [r for r in records if r['job'] == job['key']]
        mean, quant = safety(rows, 'mean_veto'), safety(rows, 'quantile90_veto')
        lower = macro(rows, lambda r: r['paired']['quantile_minus_mean']['completion_lower_pp'])['mean']
        active = sum(r['policies']['quantile90_veto']['switch_rate'] for r in rows) > 0
        expand = lower is not None and lower > 0 and active and all(quant[k] <= mean[k]
            for k in ('known_all_violations', 'known_easy_violations'))
        per_context.append(dict(job=job['key'], direct_lower_pp=lower, mean_safety=mean,
            quantile_safety=quant, quantile_any_intervention=active, expansion_condition=bool(expand)))
    analysis = dict(policy=policy, paired=paired, per_context=per_context, group_receipts=groups,
        sources=len(sites), source_views=len(records), fitting_diagnostics=source.artifact(PUBLIC/'fitting_diagnostics.json'),
        matched_primary_unavailable={arm: sum(r['matched'][arm]['contrast']['known_pp'] is None for r in records) for arm in api.ARMS},
        pilot_expansion_supported=all(v['expansion_condition'] for v in per_context),
        original24group_primary='not_estimable_not_passed', no_new_H=True, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'analysis.json', source.stable(analysis))
    if verify:
        source.once(PUBLIC/'verification.json', dict(all3_policy_prediction_blocks_exact=True,
            all_parent_controls_exact=True, analysis_sha256=source.sha(PUBLIC/'analysis.json')))


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
            saved = source.subprocess.run(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT,
                capture_output=True, check=True).stdout
            if source.hashlib.sha256(saved).hexdigest() != source.sha(path):
                raise ValueError('Commit registered definition before fitting')
        receipt = PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phase is immutable')
        if args.phase in ('pilot', 'train'):
            globals()[args.phase](*values, beat, guard, args.resume)
        else:
            evaluate(*values, beat, guard, verify=args.phase == 'verify')
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_torch=True, new_H_readout=False, primary_scope='pilot_not_full'))
        beat(state='complete')


if __name__ == '__main__':
    main()
