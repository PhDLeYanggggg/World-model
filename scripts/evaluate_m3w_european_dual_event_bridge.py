"""Complete fixed-family evaluation on reused model-selection localities."""
import json
import numpy as np
from scripts import build_m3w_european_selection_data as adapter
from src.world_model.m3w_dual_event_bridge import POLICIES, scalar_replay
from src.world_model.m3w_european_source_forecast import baseline_numpy, BASELINES
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics


def seed_summary(groups, policies, roster, count, seed):
    """Average seeds within each locality before resampling six localities."""
    out = {}
    for producer in range(3):
        for controller in range(3):
            if producer == controller: continue
            selected = [g for g in groups if g['producer'] == producer and g['controller'] == controller]
            assert sorted(g['seed'] for g in selected) == [17, 29, 43]
            row = {}
            for policy in policies:
                a = np.array([[g['views'][policy]['ADE_vs_easy_add']['all']['by_scene'][s]['gain_percent']
                               for s in roster] for g in selected])
                local = a.mean(0)
                boot = np.random.default_rng(seed).choice(local, size=(count, len(roster))).mean(1)
                row[policy] = dict(equal_locality_gain=float(local.mean()),
                    CI=np.quantile(boot, [.025, .975]).tolist(), by_locality=dict(zip(roster, local.tolist())),
                    seed_point_estimates=a.mean(1).tolist(), seed_std=float(a.mean(1).std(ddof=1)),
                    independent_localities=len(roster), seeds=3, groups_not_independent=True)
            out[f'producer{producer}_controller{controller}'] = row
    return out


def evaluate(run, cfg, bank, pid, identity):
    freeze = run.PUBLIC/'decision_freeze.json'; run.require_committed(freeze)
    frozen = json.loads(freeze.read_text())
    assert frozen['identity'] == identity
    ref = frozen['manifest']; assert run.artifact(run.ROOT/ref['path']) == ref
    done = json.loads((run.ROOT/ref['path']).read_text())
    assert done['all_passed'] and done['identity'] == identity
    trained = run.checked_training(identity)
    for ref in done['groups']:
        assert run.artifact(run.ROOT/ref['path']) == ref
        r = json.loads((run.ROOT/ref['path']).read_text()); a = r['artifact']
        assert run.artifact(run.ROOT/a['path']) == a
    data = adapter.load(run.parent, pid)
    labels = adapter.load(run.parent, pid, labels=True)
    sites = data['sites']; n = len(sites); ones = np.ones(n); roster = sorted(set(sites))
    moving = np.linalg.norm(data['history'][:, -1]-data['history'][:, -2], axis=1) > 0
    checks = dict(coordinates=0, metrics=0)
    def errors(pred):
        a, f = native_errors(pred, labels['target_eval'], labels['valid'], ones)
        other = run.base.cross.independent.coordinate_errors(pred, labels['target_eval'], labels['valid'])
        for u, v in zip((a, f), other):
            run.base.cross.independent.close(u, v); checks['coordinates'] += 1
        return a, f
    classical = [errors(baseline_numpy(data['history'], k)) for k in range(6)]
    cv = classical[1][0]
    def metric(m, r, mask):
        value = paired_scene_metrics(m[mask], r[mask], sites[mask], expected_scenes=roster,
            dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
            bootstrap_resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'])
        run.base.cross.independent.check_metric(m, r, sites, mask, value, cfg)
        checks['metrics'] += 1
        return value
    rows = np.ones(n, bool)
    run.immutable_json(run.PUBLIC/'classical_baselines.json', dict(result_source='fresh_run',
        role='reused_opened_model_selection', baselines={k: metric(v[0], cv, rows) for k, v in zip(BASELINES, classical)},
        winner_not_selected_from_readout=True))
    summaries, all_groups, refs = {}, [], []
    for g in trained['groups']:
        name = g['group']; seed = int(name.split('_seed')[1].split('_')[0])
        path = run.PUBLIC/'groups'/(name+'.json')
        if path.exists():
            value = json.loads(path.read_text()); assert value['identity'] == identity and value['verified']
            summaries[name] = value['summary']; all_groups.append(value); refs.append(run.artifact(path)); continue
        run.beat('evaluate_group', group=name)
        x, env, pair, old, meta = run.selection_pair(data, bank, g['producer'], seed, g['controller'])
        receipt = json.loads((run.PRIVATE/'decisions'/(name+'.json')).read_text())
        assert run.array_hash(x) == receipt['input_sha256']
        with np.load(run.PRIVATE/'decisions'/(name+'.npz'), allow_pickle=False) as z:
            out = {k: z[k].copy() for k in z.files}
        np.testing.assert_array_equal(env, out['envelope'])
        ra, rf = errors(pair['easy']+data['origin'][:, None])
        pa, pf = errors(pair['all']+data['origin'][:, None])
        neural = np.load(run.parent.PRIVATE/'forecasts'/f"fold{g['producer']}_seed{seed}.npy", mmap_mode='r')
        na, nf = errors(neural.astype(float)+data['origin'][:, None])
        cv_p = baseline_numpy(data['history'], 1)-data['origin'][:, None]
        damp_p = baseline_numpy(data['history'], 3)-data['origin'][:, None]
        add = {}
        for event in ('easy', 'all'):
            floor = np.where(old[event]['floor_bit'][:, None, None], damp_p, cv_p)
            prediction = np.where(old[event]['add_only'][:, None, None], neural, floor)
            add[event] = errors(prediction+data['origin'][:, None])
        masks = dict(all=rows, easy=(cv > 0)&(cv <= g['easy_cut']), hard=cv >= g['hard_cut'], complete=labels['valid'].all(1))
        views = {}; summary = {}
        for policy in (*POLICIES, 'old_easy_add', 'old_all_add', 'raw_neural', 'training_selected'):
            if policy in POLICIES:
                family = 'ridge' if policy == 'ridge_dual' else 'heads'
                scores = [out[family+'__'+t] for t in run.TASKS]
                use = out[policy]
                np.testing.assert_array_equal(use, scalar_replay(*scores, moving, env, policy))
                ade, fde = np.where(use, pa, ra), np.where(use, pf, rf)
                switch_rate = float((use & (env > 0)).mean())
            elif policy == 'raw_neural': ade, fde = na, nf; switch_rate = None
            elif policy == 'training_selected':
                ade, fde = classical[meta['easy']['baseline_index']]; switch_rate = None
            else:
                ade, fde = add['easy' if policy == 'old_easy_add' else 'all']; switch_rate = None
            view = dict(ADE_vs_easy_add={s: metric(ade, add['easy'][0], m) for s, m in masks.items()},
                ADE_vs_reference={s: metric(ade, ra, m) for s, m in masks.items()},
                ADE_vs_CV={s: metric(ade, cv, m) for s, m in masks.items()},
                ADE_vs_training_selected=metric(ade, classical[meta['easy']['baseline_index']][0], rows),
                FDE_vs_easy_add=metric(fde, add['easy'][1], rows),
                switch_rate=switch_rate, zero_CV_harmed_rows=int(((cv == 0)&(ade > 0)).sum()))
            if policy in POLICIES:
                risk = {}
                for event in ('all', 'easy'):
                    moments = out[family+'__'+event+'_risk']; by_site = {}
                    for site in roster:
                        pop = sites == site; known = pop & masks[event] & np.isfinite(ade)
                        den = float(ra[known].sum()); harm = float(np.maximum(ade[known]-ra[known], 0).sum())
                        pd = float(moments[pop, 0].sum()); ph = float(moments[pop & use, 1].sum())
                        by_site[site] = dict(event_rows=int(known.sum()), reference_ADE_sum=den,
                            positive_harm_sum=harm, realized_ratio=harm/den if den > 0 else None,
                            predicted_ratio=ph/pd if pd > 0 else None)
                    risk[event] = by_site
                view['risk_reliability'] = risk
            views[policy] = view
            easy = view['ADE_vs_CV']['easy']['worst_scene_gain_percent']
            summary[policy] = dict(gain_vs_easy_add=view['ADE_vs_easy_add']['all']['equal_scene_gain_percent'],
                CI=view['ADE_vs_easy_add']['all']['scene_bootstrap_ci95'],
                gain_vs_reference=view['ADE_vs_reference']['all']['equal_scene_gain_percent'],
                hard_gain_vs_easy_add=view['ADE_vs_easy_add']['hard']['equal_scene_gain_percent'],
                gain_vs_training_selected=view['ADE_vs_training_selected']['equal_scene_gain_percent'],
                worst_easy_degradation=None if easy is None else max(0., -easy),
                easy_pass=easy is not None and easy >= -2 and view['zero_CV_harmed_rows'] == 0,
                switch_rate=switch_rate, zero_CV_harmed_rows=view['zero_CV_harmed_rows'])
        value = dict(identity=identity, producer=g['producer'], controller=g['controller'], seed=seed,
            group=name, views=views, summary=summary, verified=True, result_source='fresh_run',
            role='reused_opened_model_selection', calibration_opened=False, confirmation_opened=False)
        run.immutable_json(path, value); summaries[name] = summary; all_groups.append(value); refs.append(run.artifact(path))
    run.immutable_json(run.PUBLIC/'seed_averaged_metrics.json', seed_summary(all_groups, POLICIES, roster,
        cfg['bootstrap_resamples'], cfg['bootstrap_seed']))
    run.immutable_json(run.PUBLIC/'summary_metrics.json', dict(result_source='fresh_run_training_and_readout',
        data_role='reused_opened_model_selection', groups=summaries, groups_are_dependent=True,
        rows=n, localities=roster, supported_ADE=int(np.isfinite(cv).sum()), unknown_future=int((~np.isfinite(cv)).sum()),
        calibration_opened=False, confirmation_opened=False, candidate_selected=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False))
    run.immutable_json(run.PUBLIC/'completion_checks.json', dict(identity=identity, all_passed=True,
        independent_checks=checks, evaluation_receipts=refs, training=run.artifact(run.PUBLIC/'training_receipt.json'),
        decisions=run.artifact(freeze), summary=run.artifact(run.PUBLIC/'summary_metrics.json'),
        engineering_completion_not_scientific_success=True))
    run.beat('readout_complete', groups=len(summaries), **checks)
