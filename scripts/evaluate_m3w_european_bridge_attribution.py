"""Frozen paired scoring and neural-trajectory attribution on development scenes."""
import json
import numpy as np
from scripts import build_m3w_european_selection_data as adapter
from src.world_model.m3w_bridge_attribution import POLICIES, query_groups, decisions
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics


def seed_summary(rows, cfg):
    out = {}
    for producer in range(3):
        for controller in range(3):
            if producer == controller: continue
            selected = [g for g in rows if (g['producer'], g['controller']) == (producer, controller)]
            assert sorted(g['seed'] for g in selected) == [17, 29, 43]
            result = {}
            for contrast in selected[0]['contrasts']:
                for subset in selected[0]['contrasts'][contrast]:
                    key = contrast+'__'+subset
                    localities = sorted(selected[0]['contrasts'][contrast][subset]['by_scene'])
                    a = np.array([[g['contrasts'][contrast][subset]['by_scene'][s]['gain_percent']
                                   for s in localities] for g in selected], float)
                    if not np.isfinite(a).all():
                        result[key] = dict(status='not_estimable', reason='Missing locality denominator'); continue
                    local = a.mean(0)
                    boot = np.random.default_rng(cfg['bootstrap_seed']).choice(local,
                        size=(cfg['bootstrap_resamples'], len(local))).mean(1)
                    result[key] = dict(gain_percent=float(local.mean()), CI=np.quantile(boot, [.025, .975]).tolist(),
                        by_locality=dict(zip(localities, local.tolist())), seed_points=a.mean(1).tolist(),
                        seed_std=float(a.mean(1).std(ddof=1)), independent_localities=len(local), seeds=3,
                        groups_not_independent=True)
            out[f'producer{producer}_controller{controller}'] = result
    return out


def evaluate(run, cfg, bank, pid, identity):
    freeze = run.PUBLIC/'decision_freeze.json'; run.require_committed(freeze)
    frozen = json.loads(freeze.read_text()); assert frozen['identity'] == identity
    ref = frozen['manifest']; assert run.artifact(run.ROOT/ref['path']) == ref
    done = json.loads((run.ROOT/ref['path']).read_text())
    assert done['all_passed'] and done['identity'] == identity
    training = run.checked_training(identity)
    for ref in done['groups']:
        assert run.artifact(run.ROOT/ref['path']) == ref
        r = json.loads((run.ROOT/ref['path']).read_text()); a = r['artifact']
        assert run.artifact(run.ROOT/a['path']) == a
    data = adapter.load(run.parent, pid); labels = adapter.load(run.parent, pid, labels=True)
    sites = data['sites']; n = len(sites); roster = sorted(set(sites)); checks = dict(coordinates=0, metrics=0)
    queries = query_groups(sites, data['recordings'], data['frames']); ids = np.arange(n)
    moving = np.linalg.norm(data['history'][:, -1]-data['history'][:, -2], axis=1) > 0
    def errors(pred):
        a, f = native_errors(pred, labels['target_eval'], labels['valid'], np.ones(n))
        other = run.base.cross.independent.coordinate_errors(pred, labels['target_eval'], labels['valid'])
        for u, v in zip((a, f), other):
            run.base.cross.independent.close(u, v); checks['coordinates'] += 1
        return a, f
    classical = [errors(baseline_numpy(data['history'], k)) for k in range(6)]
    cv = classical[1][0]; all_rows = np.ones(n, bool)
    def metric(m, r, mask):
        value = paired_scene_metrics(m[mask], r[mask], sites[mask], expected_scenes=roster,
            dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
            bootstrap_resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'])
        run.base.cross.independent.check_metric(m, r, sites, mask, value, cfg)
        checks['metrics'] += 1
        return value
    rows, refs = [], []
    for g in training['groups']:
        name = g['group']; seed = int(name.split('_seed')[1].split('_')[0]); path = run.PUBLIC/'groups'/(name+'.json')
        if path.exists():
            value = json.loads(path.read_text()); assert value['identity'] == identity and value['verified']
            rows.append(value); refs.append(run.artifact(path)); continue
        run.beat('evaluate_group', group=name)
        pairs, meta = run.selection(data, bank, g); predictions, decisions_by_pair, cost_scores = {}, {}, {}
        counts_by_pair, envs = {}, {}
        for pair, (x, env, rp, pp) in pairs.items():
            rr = json.loads((run.PRIVATE/'decisions'/(name+'_'+pair+'.json')).read_text())
            assert run.array_hash(x) == rr['input_sha256']
            values = run.scores(pair, name); assert run.array_hash(*values) == rr['score_sha256']
            out, common, counts = decisions(*values, moving, env, queries, ids)
            with np.load(run.PRIVATE/'decisions'/(name+'_'+pair+'.npz'), allow_pickle=False) as z:
                for k in POLICIES: np.testing.assert_array_equal(out[k], z[k])
                np.testing.assert_array_equal(env, z['envelope']); np.testing.assert_array_equal(counts, z['counts'])
                np.testing.assert_array_equal(common, z['common'])
            ra, rf = errors(rp.astype(float)+data['origin'][:, None])
            pa, pf = errors(pp.astype(float)+data['origin'][:, None])
            for policy in POLICIES:
                predictions[pair+'__'+policy] = (np.where(out[policy], pa, ra), np.where(out[policy], pf, rf))
            decisions_by_pair[pair] = out; cost_scores[pair] = values; counts_by_pair[pair] = counts; envs[pair] = env
        full = predictions['full__neural']
        masks = dict(all=all_rows, easy=(cv > 0)&(cv <= g['easy_cut']), hard=cv >= g['hard_cut'], complete=labels['valid'].all(1))
        views = {}
        for pair in cfg['pairs']:
            out = decisions_by_pair[pair]; ra = predictions[pair+'__reference'][0]
            nu, nr, ru, rr = cost_scores[pair]
            for policy in POLICIES:
                key = pair+'__'+policy; ade, fde = predictions[key]; use = out[policy]
                per_site = {}
                for site in roster:
                    pop = sites == site; known = pop & np.isfinite(ade); dist = ade[known]
                    event_risk = {}
                    for event in ('all', 'easy', 'hard'):
                        ok = known & masks[event]; den = float(ra[ok].sum())
                        harm = float(np.maximum(ade[ok]-ra[ok], 0).sum())
                        event_risk[event] = dict(rows=int(ok.sum()), reference_error_sum=den, positive_harm_sum=harm,
                            realized_ratio=harm/den if den > 0 else None)
                    predicted = {}
                    for family, score in (('neural', nr), ('ridge', rr)):
                        den = float(score[pop, 0].sum()); harm = float(score[pop & use, 1].sum())
                        predicted[family] = dict(reference_moment_sum=den, intervention_harm_sum=harm,
                            predicted_ratio=harm/den if den > 0 else None)
                    per_site[site] = dict(p95=float(np.quantile(dist, .95)), p99=float(np.quantile(dist, .99)),
                        positive_harm=event_risk, predicted=predicted, switch_rate=float((use & (envs[pair] > 0))[pop].mean()))
                cvmetrics = {s: metric(ade, cv, mask) for s, mask in masks.items()}
                worst = cvmetrics['easy']['worst_scene_gain_percent']
                views[key] = dict(ADE_vs_full_neural={s: metric(ade, full[0], mask) for s, mask in masks.items()},
                    ADE_vs_CV=cvmetrics, ADE_vs_training_selected=metric(ade, classical[meta['easy']['baseline_index']][0], all_rows),
                    FDE_vs_full_neural=metric(fde, full[1], all_rows), per_locality=per_site,
                    switch_rate=float((use & (envs[pair] > 0)).mean()), unknown_label_actions=int((use & ~np.isfinite(ade)).sum()),
                    zero_CV_harmed_rows=int(((cv == 0)&(ade > 0)).sum()),
                    worst_easy_degradation=None if worst is None else max(0., -worst),
                    easy_pass=worst is not None and worst >= -2 and not ((cv == 0)&(ade > 0)).any())
        comparisons = {}
        for pair in cfg['pairs']:
            for m, r in (('neural', 'ridge'), ('neural_matched', 'ridge_matched'),
                         ('neural_matched', 'hash_matched'), ('neural', 'neural_utility_ridge_risk'),
                         ('neural', 'ridge_utility_neural_risk')):
                comparisons[pair+'__'+m+'_vs_'+r] = (pair+'__'+m, pair+'__'+r)
        comparisons['full_neural_vs_motion_neural'] = ('full__neural', 'motion_only__neural')
        contrasts = {k: {s: metric(predictions[m][0], predictions[r][0], mask) for s, mask in masks.items()}
                     for k, (m, r) in comparisons.items()}
        coverage = {}
        for pair in cfg['pairs']:
            out = decisions_by_pair[pair]; counts = counts_by_pair[pair]
            different = out['neural_matched'] != out['ridge_matched']
            coverage[pair] = dict(queries=len(queries), zero_count_queries=int((counts == 0).sum()),
                matched_actions=int(counts.sum()), matched_disagree_rows=int(different.sum()),
                matched_disagree_queries=sum(bool(different[q].any()) for q in queries),
                effective_intervention_counts_match=True, unknown_labels_included_in_all_choices=True)
        value = dict(identity=identity, group=name, producer=g['producer'], controller=g['controller'], seed=seed,
            views=views, contrasts=contrasts, coverage=coverage, result_source='fresh_run_fixed_readout',
            role='reused_opened_model_selection', verified=True)
        run.immutable_json(path, value); rows.append(value); refs.append(run.artifact(path))
    run.immutable_json(run.PUBLIC/'seed_averaged_metrics.json', seed_summary(rows, cfg))
    summaries = {g['group']: {k: dict(gain_vs_full_neural=v['ADE_vs_full_neural']['all']['equal_scene_gain_percent'],
        gain_vs_training_selected=v['ADE_vs_training_selected']['equal_scene_gain_percent'],
        easy_degradation=v['worst_easy_degradation'], easy_pass=v['easy_pass'], switch_rate=v['switch_rate'])
        for k, v in g['views'].items()} for g in rows}
    run.immutable_json(run.PUBLIC/'summary_metrics.json', dict(result_source='fresh_run_fixed_readout',
        role='reused_opened_model_selection', groups=summaries, independent_localities=roster, rows=n,
        supported_ADE=int(np.isfinite(cv).sum()), unknown_future=int((~np.isfinite(cv)).sum()),
        new_heads=36, new_updates=72000, cached_full_heads=36, groups_are_dependent=True,
        candidate_selected=False, deployment_changed=False, calibration_opened=False, confirmation_opened=False,
        stage5c_executed=False, smc_enabled=False))
    run.immutable_json(run.PUBLIC/'completion_checks.json', dict(identity=identity, all_passed=True,
        independent_checks=checks, evaluation_receipts=refs, training=run.artifact(run.PUBLIC/'training_receipt.json'),
        decisions=run.artifact(freeze), summary=run.artifact(run.PUBLIC/'summary_metrics.json'),
        engineering_completion_not_scientific_success=True))
    run.beat('readout_complete', groups=len(rows), **checks)
