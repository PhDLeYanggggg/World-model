"""Fixed C-only cross-critic veto; no training, new Hreadout or deployment."""
import argparse
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_adaptive_record_cost as run
from src.world_model.m3w_cross_critic_veto import ARMS, choices, rollout
from src.evaluation.m3w_selected_cost_audit import selected_costs

source = run.source
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/moving_cross_critic_veto_v1'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/moving_cross_critic_veto_v1'
CONFIG = ROOT / 'configs/m3w_moving_cross_critic_veto_v1.json'
REG = PUBLIC / 'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC / 'protocol.md', ROOT / 'src/world_model/m3w_cross_critic_veto.py',
    ROOT / 'tests/test_m3w_cross_critic_veto.py')


def registered():
    spec = source.read(CONFIG)
    if (spec['arms'] != list(ARMS) or spec['budget'] != .02 or spec['expected_jobs'] != 18
            or spec['expected_source_views'] != 138 or spec['expected_record_views'] != 348
            or spec['expected_fresh_C_cost_arrays_each_phase'] != 36
            or spec['role'] != 'C_fitting_diagnostic_not_validation_or_calibration'
            or any(spec[k] for k in ('new_training', 'new_H_prediction_or_policy_scoring', 'threshold_search',
                'new_data_roles', 'independent_confirmation', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Only the fixed supervised-C mechanism diagnostic is admitted')
    banks = {}
    for name, base, count in (('adaptive', run.PUBLIC, 'complete_new_adaptive_cost_heads'),
                               ('frozen', run.teacher.PUBLIC, 'complete_new_neural_cost_heads')):
        freeze = source.read(base / 'training_freeze.json')
        if freeze[count] != 18:
            raise ValueError('Both completed frozen neural banks required')
        banks[name] = {source.read(source.checked(r))['job']['key']: source.read(source.checked(r)) for r in freeze['fits']}
    original = source.read(run.REG)
    if set(banks['adaptive']) != set(banks['frozen']) or len(original['jobs']) != 18:
        raise ValueError('Do not subset models, seeds or source roles')
    source.once(REG, source.stable(dict(bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        configuration=spec, original_training_registration=source.artifact(run.REG),
        adaptive_freeze=source.artifact(run.PUBLIC / 'training_freeze.json'),
        frozen_freeze=source.artifact(run.teacher.PUBLIC / 'training_freeze.json'),
        prior_C_diagnostic=source.artifact(run.PUBLIC / 'fitting_cost_diagnostics.json'),
        jobs=original['jobs'], forecasts=original['forecasts'],
        prior_H_results_already_exposed=True, definition_frozen_before_this_C_prototype_scoring=True,
        independent_data_critics=False, validation_or_calibration=False, new_data_roles=False,
        new_H_prediction=False, new_training=False, new_deployment=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)))
    return spec, banks


def diagnostic_decision(summary):
    positive = lambda r: r['CI95'] is not None and source.np.isfinite(r['CI95']).all() and r['CI95'][0] > 0
    utility = positive(summary['primary_nonempty_matched']['completion_lower_pp'])
    harm = positive(summary['matched_known_harm_reduction_pp'])
    return dict(C_nonempty_count_matched_lower_positive=bool(utility), C_matched_known_harm_reduction_positive=bool(harm),
        worth_source_excluded_followup=bool(utility and harm),
        fitting_only_not_validation_or_calibration=True, independent_confirmation=False,
        new_deployment=False, submission_ready=False, original24group_primary='not_estimable_not_passed',
        stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    spec, banks = registered()
    if args.phase == 'register':
        return
    for path in (*FILES, REG):
        content = source.subprocess.run(['git', 'show', 'HEAD:' + str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
        if source.hashlib.sha256(content).hexdigest() != source.sha(path):
            raise ValueError('Commit unchanged diagnostic before Cprototype scoring')
    if (PUBLIC / (args.phase + '_execution.json')).exists():
        raise ValueError('Completed phase remains immutable')
    if args.phase == 'verify' and not (PUBLIC / 'analysis.json').exists():
        raise ValueError('Complete C-only run before exact replay')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        run.teacher.lineage.PRIVATE = PRIVATE
        cfg = source.read(run.teacher.PARENT.CONFIG)
        start, guard, beat = run.teacher.lineage.runtime(cfg, spec, args.phase)
        _, cfg, data, jobs, refs, _, _ = run.register(beat)
        run.api.base.base.exact(source.stable(jobs), source.read(REG)['jobs'])
        records, receipts = [], []
        for job in jobs:
            output = PUBLIC / 'groups' / (job['key'] + '.json')
            if args.phase == 'run' and output.exists() and not args.resume:
                raise ValueError('Explicit same-binding resume required for partial C run')
            p = run.teacher.PARENT.view(data, job, refs, training=True)
            p.pop('learning')
            c, ids = p['causal'], p['ids']
            if (set(p['sites']) != set(job['critic_sites']) or set(p['sites']) & set(job['producer_sites'])
                    or set(p['sites']) & set(job['evaluation_sites'])):
                raise ValueError('Only original C rows, never P or H rows')
            prediction = {}
            for name in ('adaptive', 'frozen'):
                fit = banks[name][job['key']]
                state = run.api.base.base.read_checkpoint(source.checked(fit['checkpoint']))
                prediction[name] = run.api.parent.predict(state, c)
                if source.array_hash(prediction[name]) != fit['fitting_predictions_sha256']:
                    raise ValueError('Fresh C prediction must equal its original completed fitting receipt')
                del state
            chosen = choices(prediction['adaptive'], prediction['frozen'], c['allowed'], c['groups'], ids)
            ph = {k: source.array_hash(v) for k, v in prediction.items()}
            ch = {k: source.array_hash(v) for k, v in chosen.items()}
            paths = {k: rollout(p['paths'], p['floor'], choice) for k, choice in chosen.items()}
            cache = PRIVATE / 'choices' / (job['key'] + '.npz')
            cache.parent.mkdir(exist_ok=True)
            if cache.exists():
                with source.np.load(cache, allow_pickle=False) as old:
                    for key, value in dict(chosen, ids=ids).items():
                        source.np.testing.assert_array_equal(old[key], value)
            elif args.phase == 'verify':
                raise ValueError('Every causal decision cache must predate replay')
            else:
                source.np.savez_compressed(cache, **chosen, ids=ids)
            # Targets were loaded with data, but do not enter either predictor or choice.
            target = data['target_eval'][ids] - data['origin'][ids, None]
            valid = data['valid'][ids]
            truth = run.api.parent.targets(p, target, valid, job['easy_cut'])
            cells = [dict(model=model, region=arm, scores=selected_costs(pred, truth['moments'], truth['observed_future'],
                choice, c['groups'])) for model, pred in prediction.items() for arm, choice in chosen.items()]
            old_cells = banks['adaptive'][job['key']]['C_region_cells']
            for model, arm, old_name in (('adaptive', 'adaptive_self', 'adaptive_selection'),
                                         ('frozen', 'frozen_self', 'frozen_teacher')):
                actual = next(r['scores'] for r in cells if r['model'] == model and r['region'] == arm)
                previous = next(r['scores'] for r in old_cells if r['model'] == old_name and r['region'] == old_name)
                run.api.base.base.exact(source.stable(actual), source.stable(previous))
            parts = []
            for site in sorted(set(p['sites'])):
                at = p['sites'] == site
                q = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
                policies = {arm: source.readout.policy(path[at], p['floor'][at], p['cv'][at], target[at], valid[at],
                    p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for arm, path in paths.items()}
                paired = source.readout.paired(paths['self_count_matched'][at], paths['cross_veto'][at], p['floor'][at], target[at], valid[at])
                matched = source.matched_utility(paths['self_count_matched'][at], paths['cross_veto'][at], p['floor'][at],
                    target[at], valid[at], chosen['self_count_matched'][at], chosen['cross_veto'][at], q)
                matched['comparison'] = 'cross_veto_minus_causal_count_matched_self'
                y, known = truth['moments'][at], truth['observed_future'][at]
                def mass(arm):
                    choice = chosen[arm][at]
                    ix = source.np.flatnonzero((choice >= 0) & known)
                    return y[ix, choice[ix]].sum(0)
                self_mass, veto_mass = mass('self_count_matched'), mass('cross_veto')
                denominator = float(y[:, 0, 2].sum())
                harm_reduction = float(100*(self_mass[1]-veto_mass[1])/denominator) if denominator > 0 else None
                parts.append(dict(site=str(site), job=job['key'], seed=job['seed'], producer_fold=job['producer_fold'],
                    critic_fold=job['critic_fold'], policies=policies, secondary_full_strategy_paired=paired,
                    primary_nonempty_matched=matched, matched_known_harm_reduction_pp=harm_reduction))
            if ph != {k: source.array_hash(v) for k, v in prediction.items()} or ch != {k: source.array_hash(v) for k, v in chosen.items()}:
                raise ValueError('Labels mutated causal predictions or actions')
            source.once(output, source.stable(dict(job=job, rows=parts, prediction_hashes=ph, choice_hashes=ch,
                cost_cells=cells, C_ids_sha256=source.array_hash(ids), decision_cache=source.artifact(cache),
                original_C_diagonal_cost_cells_exact=True, targets_used_for_scoring_after_choices=True,
                new_H_prediction=False, shared_C_training_not_independent_critics=True)))
            receipts.append(source.artifact(output))
            records.extend(parts)
            beat(state='C_cross_critic_replayed' if args.phase == 'verify' else 'C_cross_critic_scored',
                 job=job['key'], complete=len(receipts), total=18)
            del p, c, prediction, chosen, paths, truth, cells, parts, target, valid, y
            source.gc.collect()
            guard()
        sites = sorted({r['site'] for r in records})
        if len(records) != 138 or len(sites) != 23:
            raise ValueError('All138original Csource views required')
        ci = lambda getter: source.group_interval(records, getter, sites, cfg)
        primary = {m: ci(lambda r, m=m: r['primary_nonempty_matched']['contrast'][m])
            for m in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')}
        secondary = {m: ci(lambda r, m=m: r['secondary_full_strategy_paired'][m])
            for m in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')}
        safety = {}
        for arm in ARMS:
            views = [v for r in records for v in r['policies'][arm]['recordings'].values()]
            if len(views) != 348:
                raise ValueError('Every C record view remains in each strategy safety denominator')
            easy = [v['known_easy_net_degradation'] for v in views if v['known_easy_net_degradation'] is not None]
            safety[arm] = dict(required_record_views=len(views), supported_record_views=sum(v['supported'] for v in views),
                known_all_risk_violations=sum(v['known_all_risk'] is not None and v['known_all_risk'] > .02 for v in views),
                known_easy_risk_violations=sum(v['known_easy_risk'] is not None and v['known_easy_risk'] > .02 for v in views),
                undefined_easy_views=sum(v['completion_easy_risk_upper'] is None for v in views),
                worst_defined_signed_easy_net_percent=100*max(easy) if easy else None,
                full_original_risk_pass=all(v['supported'] for v in views))
        summary = source.stable(dict(primary_nonempty_matched=primary, secondary_full_strategy_paired=secondary,
            matched_known_harm_reduction_pp=ci(lambda r: r['matched_known_harm_reduction_pp']), safety=safety,
            known_utility={a: ci(lambda r, a=a: r['policies'][a]['known_ADE_gain_floor_percent']) for a in ARMS},
            switch_percent={a: ci(lambda r, a=a: 100*r['policies'][a]['switch_rate']) for a in ARMS},
            nonempty_query_views=sum(r['primary_nonempty_matched']['nonempty_queries'] for r in records),
            empty_query_views=sum(r['primary_nonempty_matched']['empty_queries'] for r in records),
            empty_source_views=sum(r['primary_nonempty_matched']['contrast']['known_pp'] is None for r in records),
            group_receipts=receipts, C_fitting_only=True, new_H_prediction=False, new_training=False,
            independent_confirmation=False, deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
        source.once(PUBLIC / 'analysis.json', summary)
        source.once(PUBLIC / 'decision.json', source.stable(diagnostic_decision(summary)))
        source.once(PUBLIC / (args.phase + '_execution.json'), dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            fresh_C_cost_arrays=36, original_diagonal_cells_exact=36, new_H_prediction=False, new_training=False))
        if args.phase == 'verify':
            source.once(PUBLIC / 'verification.json', dict(all36C_cost_arrays_and_all144cost_cells_exact=True,
                all_four_policy_and_count_matched_blocks_exact=True, analysis_sha256=source.sha(PUBLIC / 'analysis.json'),
                new_H_prediction=False))
        beat(state='complete')


if __name__ == '__main__':
    main()
