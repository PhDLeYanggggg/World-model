"""Train genuinely source-separated C neural critics, keeping original risk rules."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_adaptive_record_cost as previous
from src.world_model import m3w_amplitude_cost as amplitude
from src.world_model.m3w_source_separated_critics import ARMS, partition_jobs, choices
from src.world_model.m3w_cross_critic_veto import rollout
from src.evaluation.m3w_selected_cost_audit import selected_costs

source = previous.source
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/moving_source_separated_critics_v1'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/moving_source_separated_critics_v1'
CONFIG = ROOT / 'configs/m3w_moving_source_separated_critics_v1.json'
REG = PUBLIC / 'registration.json'
INNER = ROOT / 'outputs/publication_readiness_2026_09/moving_hurdle_reference_v1/registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC / 'protocol.md',
    ROOT / 'src/world_model/m3w_source_separated_critics.py',
    ROOT / 'tests/test_m3w_source_separated_critics.py',
    ROOT / 'tests/test_m3w_source_separated_runner.py')


def register(beat):
    spec = source.read(CONFIG)
    if (spec['training'] != source.read(previous.CONFIG)['training']
            or spec['arms'] != list(ARMS) or spec['parts'] != [0, 1, 2] or spec['replicas'] != [0, 1]
            or spec['replica_seed_offset'] != 1009 or spec['new_neural_cost_heads'] != 108
            or spec['primary_optimizer_updates'] != 216000 or spec['positive_harm_budget'] != .02
            or spec['easy_net_degradation_budget'] != .02
            or spec['normalization'] != 'own_inner_part_physical_fitting_only'
            or any(spec[k] for k in ('threshold_search', 'new_official_data_roles', 'new_H_prediction_or_policy_scoring',
                'new_trajectory_training', 'analytic_identity_certification_added', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed source-separated neural development study and original risk rules required')
    _, cfg, data, jobs, refs, _, _ = previous.register(beat)
    mapping = source.read(INNER)['inner_source_folds']
    partitions = {j['key']: partition_jobs(data['sites'], j, mapping[j['key']]) for j in jobs}
    if len(jobs) != 18 or sum(len(p) for p in partitions.values()) != 54:
        raise ValueError('All original contexts and inner parts required')
    source.once(REG, source.stable(dict(bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        configuration=spec, parent_registration=source.artifact(previous.REG),
        old_inner_source_registration=source.artifact(INNER), jobs=jobs, forecasts=refs, partitions=partitions,
        new_neural_training=True, new_H_prediction=False, independent_confirmation=False,
        original24group_primary='not_estimable_not_passed', analytic_identity_certification_added=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)))
    return spec, cfg, data, jobs, refs, partitions


def packet(data, job, refs, part, *, training):
    raw = source.view(data, part['routing_job'], refs, training=False)
    ids = raw['ids']
    if (set(raw['sites']) != set(part['names']) or set(raw['sites']) & set(job['producer_sites'])
            or set(raw['sites']) & set(job['evaluation_sites']) or 'learning' in raw):
        raise ValueError('Causal routing must not read any target or fit old full-C preprocessing')
    result = amplitude.causal_bank(old_causal=raw['causal'], geometry=data['geometry'][ids],
        shared_context=data['shared_context'][ids], paths=raw['paths'], floor_index=job['floor_index'])
    out = dict(result, ids=ids, sites=raw['sites'], recordings=raw['recordings'], frames=raw['frames'])
    if training:
        target, valid = data['target_eval'][ids]-data['origin'][ids, None], data['valid'][ids]
        physical = source.api.base.training_data(raw['causal'], raw['paths'], target, valid,
            job['floor_index'], job['easy_cut'])
        pr = physical['preprocess']
        if set(pr['source_names']) != set(part['names']):
            raise ValueError('Own single-part normalization, target RMS and bias only')
        out['learning'] = amplitude.training_data(result, target, valid, job['easy_cut'], pr)
    return out


def state_path(job, part, replica, kind='fits'):
    return PRIVATE / kind / (job['key'] + f'_part{part}_replica{replica}.pt.gz')


def one(p, job, part, replica, spec, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return amplitude.fit(p['learning'], arm='amplitude_bank',
        seed=job['seed']+replica*spec['replica_seed_offset'], settings=spec['training'],
        identity=dict(registration_sha256=source.sha(REG), outer_job=job, inner_part=part, replica=replica,
            fitting_role='single_inner_C_source_part', no_new_H=True),
        path=state_path(job, part, replica, kind), resume=resume, stop_at=stop_at,
        heartbeat=lambda **v: beat(job=job['key'], part=part, replica=replica, **v), guard=guard)


def pilot(spec, cfg, data, jobs, refs, parts, beat, guard, resume):
    cells = []
    for fold in range(3):
        job = next(j for j in jobs if j['critic_fold'] == fold)
        part = parts[job['key']][0]
        p = packet(data, job, refs, part, training=True)
        for replica in spec['replicas']:
            path = state_path(job, 0, replica, 'pilot_resume')
            if path.exists() and not resume: raise ValueError('Explicit pilot resume required')
            step = amplitude.base.base.read_checkpoint(path)['step'] if path.exists() else 0
            if step < spec['pilot_stop']:
                one(p, job, 0, replica, spec, beat, guard, kind='pilot_resume', resume=path.exists(), stop_at=spec['pilot_stop'])
            a = one(p, job, 0, replica, spec, beat, guard, kind='pilot_resume', resume=True, stop_at=spec['pilot_target'])
            path = state_path(job, 0, replica, 'pilot_direct')
            b = one(p, job, 0, replica, spec, beat, guard, kind='pilot_direct', resume=resume and path.exists(), stop_at=spec['pilot_target'])
            source.recovered(a, b)
            cells.append(dict(job=job['key'], replica=replica, step=b['step'], budget=b['budget'],
                seconds=b['seconds'], parameters=b['parameters'], fit_rows=len(b['fit_ids']),
                source_names=b['source_sites'], exact_resume=True))
        del p, a, b
        source.gc.collect(); guard()
    projection = 18*sum(r['seconds']/r['step']*r['budget'] for r in cells)
    source.once(PUBLIC / 'pilot.json', source.stable(dict(cells=cells, all6_resume_cells_exact=True,
        actual_pilot_optimizer_updates=1200, projected_primary_seconds_not_completed=projection,
        validation_readout=False, new_H_prediction=False)))


def train(spec, cfg, data, jobs, refs, parts, beat, guard, resume):
    if not source.read(PUBLIC / 'pilot.json')['all6_resume_cells_exact']:
        raise ValueError('Real own-source training and exact resume pilot required')
    receipts = []
    for job in jobs:
        for part in parts[job['key']]:
            p = packet(data, job, refs, part, training=True)
            for replica in spec['replicas']:
                k = part['part']; path = state_path(job, k, replica)
                state = one(p, job, k, replica, spec, beat, guard, resume=resume and path.exists())
                if (state['step'] != 2000 or state['budget'] != 2000 or state['parameters'] != 28229
                        or set(state['source_sites']) != set(part['names'])
                        or not source.np.array_equal(state['fit_ids'], p['ids'])):
                    raise ValueError('Complete own-part budget and normalization required')
                if not receipts:
                    repeat_path = state_path(job, k, replica, 'full_refit')
                    repeat = one(p, job, k, replica, spec, beat, guard, kind='full_refit', resume=resume and repeat_path.exists())
                    source.recovered(state, repeat)
                    source.once(PUBLIC / 'first_full_refit.json', dict(exact=True, checkpoint=source.artifact(repeat_path)))
                    del repeat
                receipt = PUBLIC / 'fits' / (path.stem.replace('.pt', '') + '.json')
                source.once(receipt, source.stable(dict(job=job['key'], part=k, replica=replica,
                    cost_seed=state['seed'], fit_sources=state['source_sites'], fit_ids_sha256=source.array_hash(state['fit_ids']),
                    preprocess_array_hashes={name: source.array_hash(state['preprocess'][name]) for name in ('mean', 'std', 'rms', 'bias')},
                    checkpoint=source.artifact(path), steps=state['step'], budget=state['budget'], seconds=state['seconds'],
                    parameters=state['parameters'], trace=state['trace'], minimum_query_visits=int(state['sampler']['counts'].min()),
                    own_part_preprocessing=True, new_neural_training=True, validation_readout=False)))
                receipts.append(source.artifact(receipt))
                beat(state='source_separated_fit_frozen', job=job['key'], part=k, replica=replica, complete=len(receipts), total=108)
                del state
            del p
            source.gc.collect(); guard()
    source.once(PUBLIC / 'training_freeze.json', dict(fits=receipts, complete_new_heads=108,
        primary_optimizer_updates=216000, all_before_nested_validation=True, new_H_prediction=False))


def evaluate(spec, cfg, data, jobs, refs, parts, beat, guard, *, verify):
    freeze = source.read(PUBLIC / 'training_freeze.json')
    if freeze['complete_new_heads'] != 108 or not freeze['all_before_nested_validation']:
        raise ValueError('Freeze all108 full-budget models before any validation label scores')
    models = {}
    for ref in freeze['fits']:
        r = source.read(source.checked(ref)); models[r['job'], r['part'], r['replica']] = r
    records, receipts = [], []
    for job in jobs:
        all_names = {r['part']: set(r['names']) for r in parts[job['key']]}
        for part in parts[job['key']]:
            v = part['part']; others = sorted(set(range(3))-{v})
            p = packet(data, job, refs, part, training=False)
            c, ids = p['causal'], p['ids']
            predictions, hashes = {}, {}
            for fitpart in others:
                for replica in (0, 1):
                    r = models[job['key'], fitpart, replica]
                    state = amplitude.base.base.read_checkpoint(source.checked(r['checkpoint']))
                    if (set(state['source_sites']) != all_names[fitpart] or set(state['fit_ids']) & set(ids)
                            or set(state['preprocess']['source_names']) & all_names[v]):
                        raise ValueError('Every normalization/label fit must exclude validation sources')
                    predictions[fitpart, replica] = amplitude.predict(state, c)
                    hashes[f'part{fitpart}_replica{replica}'] = source.array_hash(predictions[fitpart, replica])
                    del state
            orientations = []
            for proposer in others:
                critic = next(k for k in others if k != proposer)
                chosen = choices(predictions[proposer, 0], predictions[proposer, 1], predictions[critic, 0], c['allowed'], c['groups'], ids)
                paths = {arm: rollout(p['paths'], p['floor'], value) for arm, value in chosen.items()}
                orientations.append((proposer, critic, chosen, paths))
            # Validation labels enter scoring only after every causal prediction/action/path above.
            target, valid = data['target_eval'][ids]-data['origin'][ids, None], data['valid'][ids]
            truth = amplitude.targets(p, target, valid, job['easy_cut'])
            for proposer, critic, chosen, paths in orientations:
                key = job['key'] + f'_val{v}_propose{proposer}'
                cache = PRIVATE / 'choices' / (key+'.npz'); cache.parent.mkdir(exist_ok=True)
                cached = dict(chosen, ids=ids)
                if cache.exists():
                    with source.np.load(cache, allow_pickle=False) as old:
                        for name, value in cached.items(): source.np.testing.assert_array_equal(old[name], value)
                elif verify: raise ValueError('Every decision cache must predate replay')
                else: source.np.savez_compressed(cache, **cached)
                pred = dict(proposer=predictions[proposer, 0], same_data=predictions[proposer, 1], disjoint_data=predictions[critic, 0])
                cells = [dict(model=m, region=a, scores=selected_costs(x, truth['moments'], truth['observed_future'], choice, c['groups']))
                    for m, x in pred.items() for a, choice in chosen.items()]
                rows = []
                for site in sorted(set(p['sites'])):
                    at = p['sites'] == site
                    q = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
                    policies = {a: source.readout.policy(path[at], p['floor'][at], p['cv'][at], target[at], valid[at],
                        p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for a, path in paths.items()}
                    contrasts, full, harm = {}, {}, {}
                    y, known = truth['moments'][at], truth['observed_future'][at]
                    for kind in ('same_data', 'disjoint_data'):
                        first, second = kind+'_matched_self', kind+'_veto'
                        contrasts[kind] = source.matched_utility(paths[first][at], paths[second][at], p['floor'][at], target[at], valid[at],
                            chosen[first][at], chosen[second][at], q)
                        contrasts[kind]['comparison'] = second+'_minus_query_count_and_alpha_matched_self'
                        full[kind] = source.readout.paired(paths[first][at], paths[second][at], p['floor'][at], target[at], valid[at])
                        def mass(arm):
                            a = chosen[arm][at]; ix = source.np.flatnonzero((a >= 0) & known)
                            return y[ix, a[ix]].sum(0)
                        a, b = mass(first), mass(second)
                        den = float(y[:, 0, 2].sum())
                        harm[kind] = float(100*(a[1]-b[1])/den) if den > 0 else None
                    rows.append(dict(site=str(site), job=job['key'], seed=job['seed'], validation_part=v,
                        proposer_part=proposer, critic_part=critic, policies=policies, nonempty_matched=contrasts,
                        secondary_full_strategy=full, matched_known_harm_reduction_pp=harm))
                for (k, r), value in predictions.items():
                    if source.array_hash(value) != hashes[f'part{k}_replica{r}']:
                        raise ValueError('Scoring must not mutate predictions')
                output = PUBLIC / 'groups' / (key+'.json')
                source.once(output, source.stable(dict(job=job['key'], validation_part=v, proposer_part=proposer, critic_part=critic,
                    rows=rows, prediction_hashes=hashes, choice_hashes={a: source.array_hash(x) for a, x in chosen.items()},
                    cost_cells=cells, decision_cache=source.artifact(cache), validation_ids_sha256=source.array_hash(ids),
                    own_preprocessing_excludes_validation=True, risk_and_proposer_fit_sources_disjoint=True, new_H_prediction=False)))
                receipts.append(source.artifact(output)); records.extend(rows)
                beat(state='nested_C_replayed' if verify else 'nested_C_scored', job=job['key'], validation_part=v,
                    proposer_part=proposer, complete=len(receipts), total=108)
            del p, c, predictions, orientations, truth
            source.gc.collect(); guard()
    sites = sorted({r['site'] for r in records})
    if len(records) != 276 or len(sites) != 23 or len(receipts) != 108:
        raise ValueError('All source/orientation views required, never report a selected subset')
    ci = lambda getter: source.group_interval(records, getter, sites, cfg)
    contrasts, full, harm = {}, {}, {}
    for kind in ('same_data', 'disjoint_data'):
        contrasts[kind] = {m: ci(lambda r, m=m, k=kind: r['nonempty_matched'][k]['contrast'][m])
            for m in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')}
        full[kind] = {m: ci(lambda r, m=m, k=kind: r['secondary_full_strategy'][k][m])
            for m in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')}
        harm[kind] = ci(lambda r, k=kind: r['matched_known_harm_reduction_pp'][k])
    safety = {}
    for arm in ARMS:
        views = [v for r in records for v in r['policies'][arm]['recordings'].values()]
        if len(views) != 696: raise ValueError('Retain all recording/orientation views')
        easy = [v['known_easy_net_degradation'] for v in views if v['known_easy_net_degradation'] is not None]
        safety[arm] = dict(required_record_views=696, supported_record_views=sum(v['supported'] for v in views),
            known_all_risk_violations=sum(v['known_all_risk'] is not None and v['known_all_risk'] > .02 for v in views),
            known_easy_risk_violations=sum(v['known_easy_risk'] is not None and v['known_easy_risk'] > .02 for v in views),
            undefined_easy_views=sum(v['completion_easy_risk_upper'] is None for v in views),
            worst_defined_signed_easy_net_percent=100*max(easy) if easy else None,
            full_original_risk_pass=all(v['supported'] for v in views))
    summary = source.stable(dict(nonempty_matched=contrasts, secondary_full_strategy=full,
        matched_known_harm_reduction_pp=harm, safety=safety,
        known_utility={a: ci(lambda r, a=a: r['policies'][a]['known_ADE_gain_floor_percent']) for a in ARMS},
        switch_percent={a: ci(lambda r, a=a: 100*r['policies'][a]['switch_rate']) for a in ARMS},
        nonempty_query_views={k: sum(r['nonempty_matched'][k]['nonempty_queries'] for r in records) for k in contrasts},
        empty_query_views={k: sum(r['nonempty_matched'][k]['empty_queries'] for r in records) for k in contrasts},
        unavailable_source_views={k: sum(r['nonempty_matched'][k]['contrast']['known_pp'] is None for r in records) for k in contrasts},
        group_receipts=receipts, new_neural_training=True, independent_confirmation=False,
        nested_source_excluded_C_development=True, new_H_prediction=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False))
    source.once(PUBLIC / 'analysis.json', summary)
    positive = lambda row: row['CI95'] is not None and row['CI95'][0] > 0
    passing = bool(positive(contrasts['disjoint_data']['completion_lower_pp']) and positive(harm['disjoint_data'])
        and safety['disjoint_data_veto']['full_original_risk_pass'])
    source.once(PUBLIC / 'decision.json', dict(source_excluded_mechanism_and_original_risk_pass=passing,
        new_deployment=False, independent_confirmation=False, new_H_prediction=False,
        analytic_identity_certification_added=False, original24group_primary='not_estimable_not_passed',
        submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if verify:
        source.once(PUBLIC / 'verification.json', dict(all216_unique_nested_C_predictions_and_all108policy_blocks_exact=True,
            all1620_cost_cells_exact=True, all_query_count_and_alpha_histograms_exact=True,
            analysis_sha256=source.sha(PUBLIC / 'analysis.json'), no_new_H_prediction=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        previous.teacher.lineage.PRIVATE = PRIVATE
        spec = source.read(CONFIG)
        start, guard, beat = previous.teacher.lineage.runtime({}, spec, args.phase)
        spec, cfg, data, jobs, refs, parts = register(beat)
        if args.phase == 'register': return
        for path in (*FILES, REG):
            body = source.subprocess.run(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
            if source.hashlib.sha256(body).hexdigest() != source.sha(path):
                raise ValueError('Commit unchanged full study before any pilot fitting')
        receipt = PUBLIC / (args.phase+'_execution.json')
        if receipt.exists(): raise ValueError('Completed phase is immutable')
        if args.phase == 'pilot': pilot(spec, cfg, data, jobs, refs, parts, beat, guard, args.resume)
        elif args.phase == 'train': train(spec, cfg, data, jobs, refs, parts, beat, guard, args.resume)
        else: evaluate(spec, cfg, data, jobs, refs, parts, beat, guard, verify=args.phase == 'verify')
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            new_H_prediction=False, new_trajectory_training=False, real_torch=True))
        beat(state='complete')


if __name__ == '__main__': main()
