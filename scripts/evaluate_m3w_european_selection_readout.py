"""Frozen family readout, with paired locality-level uncertainty and no refits."""
import json
import subprocess
import numpy as np
from scripts import build_m3w_european_selection_data as adapter
from src.world_model.m3w_frozen_selection import POLICIES
from src.world_model.m3w_european_source_forecast import baseline_numpy, BASELINES
from src.world_model.m3w_incumbent_relative import replay
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics


def evaluate(run, cfg, bank, identity):
    freeze = run.PUBLIC/'decision_freeze.json'
    rel = str(freeze.relative_to(run.ROOT))
    if subprocess.check_output(['git', 'show', 'HEAD:'+rel], cwd=run.ROOT) != freeze.read_bytes():
        raise ValueError('All decisions must be committed before outcome readout')
    public = json.loads(freeze.read_text()); ref = public['manifest']
    if run.artifact(run.ROOT/ref['path']) != ref: raise ValueError('Decision manifest changed')
    manifest = json.loads((run.ROOT/ref['path']).read_text())
    if manifest['identity'] != identity or not manifest['all_passed'] or len(manifest['groups']) != 36:
        raise ValueError('Incomplete frozen inference family')
    for ref in manifest['groups']:
        if run.artifact(run.ROOT/ref['path']) != ref: raise ValueError('Decision receipt changed')
        rr = json.loads((run.ROOT/ref['path']).read_text())
        for a in rr['artifacts'].values():
            if run.artifact(run.ROOT/a['path']) != a: raise ValueError('Frozen prediction changed')
    data = adapter.load(run, identity)
    labels = adapter.load(run, identity, labels=True)
    n = len(data['sites']); ones = np.ones(n); roster = cfg['localities']; sites = data['sites']
    coordinate_checks = 0
    def error(p):
        nonlocal coordinate_checks
        a, f = native_errors(p, labels['target_eval'], labels['valid'], ones)
        other = run.inc.base.cross.independent.coordinate_errors(p, labels['target_eval'], labels['valid'])
        for u, v in zip((a, f), other): run.inc.base.cross.independent.close(u, v); coordinate_checks += 1
        return a, f
    baselines = [error(baseline_numpy(data['history'], i)) for i in range(6)]
    cv = baselines[1][0]
    def metric(m, r, mask):
        v = paired_scene_metrics(m[mask], r[mask], sites[mask], expected_scenes=roster,
            dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
            bootstrap_resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'])
        run.inc.base.cross.independent.check_metric(m, r, sites, mask, v, cfg)
        return v
    all_rows = np.ones(n, bool)
    baseline_table = {key: metric(a, cv, all_rows) for key, (a, _) in zip(BASELINES, baselines)}
    run.immutable_json(run.PUBLIC/'classical_baselines.json', dict(role='model_selection',
        result_source='fresh_run', rows=n, ADE_vs_CV=baseline_table, diagnostic_not_fitted_on_selection=True))
    summaries, refs = {}, []
    for name, group in bank['groups'].items():
        path = run.PUBLIC/'groups'/(name+'.json')
        if path.exists():
            value = json.loads(path.read_text())
            if value['identity'] != identity or not value['verified']: raise ValueError('Resume result identity mismatch')
            summaries[name] = value['summary']; refs.append(run.artifact(path)); continue
        run.beat('evaluate_group', group=name)
        receipt = json.loads((run.PRIVATE/'decisions'/(name+'.json')).read_text())
        with np.load(run.PRIVATE/'decisions'/(name+'.npz'), allow_pickle=False) as z:
            out = {k: z[k].copy() for k in z.files}
        neural = np.load(run.ROOT/receipt['artifacts']['neural']['path'], allow_pickle=False, mmap_mode='r')
        cv_p = baseline_numpy(data['history'], 1)-data['origin'][:, None]
        damp_p = baseline_numpy(data['history'], 3)-data['origin'][:, None]
        floor = np.where(out['floor_bit'][:, None, None], damp_p, cv_p)
        da, df = error(floor+data['origin'][:, None]); na, nf = error(neural.astype(float)+data['origin'][:, None])
        selected = {p: np.where(out[p], na, da) for p in POLICIES}
        masks = dict(all=all_rows, easy=(cv > 0)&(cv <= group['easy_cut']),
            hard=cv >= group['hard_cut'], complete=labels['valid'].all(1))
        moving = np.linalg.norm(data['history'][:, -1]-data['history'][:, -2], axis=1) > 0
        for p in ('floor_reference', 'incumbent_reference', 'ridge_incumbent', 'previous_matched', 'add_only', 'remove_only'):
            arm = 'incumbent_reference' if p in ('add_only', 'remove_only') else p
            role = 'floor_reference' if arm in ('floor_reference', 'previous_matched') else 'incumbent_reference'
            direction = {'add_only': 'add', 'remove_only': 'remove'}.get(p, 'both')
            np.testing.assert_array_equal(out[p], replay(out[arm+'__utility'], out[arm+'__risk'], moving,
                out['old_stop'], arm=role, direction=direction))
        views = {}
        for p in POLICIES:
            err = selected[p]; use = out[p]
            view = dict(ADE_vs_incumbent={s: metric(err, selected['old_stop'], m) for s, m in masks.items()},
                ADE_vs_floor={s: metric(err, da, m) for s, m in masks.items()},
                ADE_vs_CV={s: metric(err, cv, m) for s, m in masks.items()},
                ADE_vs_training_selected=metric(err, baselines[group['baseline_index']][0], all_rows),
                FDE_vs_incumbent=metric(np.where(use, nf, df), np.where(out['old_stop'], nf, df), all_rows),
                switch_rate=float(use.mean()), switched_rows=int(use.sum()),
                unknown_ADE_switches=int((use & ~np.isfinite(na)).sum()),
                zero_CV=dict(rows=int((cv == 0).sum()), harmed_rows=int(((cv == 0)&(err > 0)).sum())))
            event_mask = masks[group['event']]
            risk = {}
            arm = 'incumbent_reference' if p in ('add_only', 'remove_only') else p
            referr = selected['old_stop'] if arm in ('incumbent_reference', 'ridge_incumbent') else da
            for site in roster:
                population = sites == site; known = population & event_mask & np.isfinite(err)
                denom = float(referr[known].sum()); harm = float(np.maximum(err[known]-referr[known], 0).sum())
                r = dict(event_rows=int(known.sum()), denominator=denom, positive_harm=harm,
                    realized_event_harm_ratio=harm/denom if denom > 0 else None)
                key = arm+'__risk'
                if key in out:
                    action = (use != out['old_stop']) if arm in ('incumbent_reference', 'ridge_incumbent') else use
                    den = float(out[key][population, 0].sum()); num = float(out[key][population & action, 1].sum())
                    r['predicted_event_harm_ratio'] = num/den if den > 0 else None
                risk[site] = r
            view['risk_reliability'] = risk
            views[p] = view
        summary = {}
        for p, v in views.items():
            easy = v['ADE_vs_CV']['easy']['worst_scene_gain_percent']
            summary[p] = dict(gain_vs_incumbent=v['ADE_vs_incumbent']['all']['equal_scene_gain_percent'],
                CI=v['ADE_vs_incumbent']['all']['scene_bootstrap_ci95'],
                hard_gain_vs_incumbent=v['ADE_vs_incumbent']['hard']['equal_scene_gain_percent'],
                gain_vs_training_selected=v['ADE_vs_training_selected']['equal_scene_gain_percent'],
                worst_easy_degradation=None if easy is None else max(0., -easy),
                easy_pass=easy is not None and easy >= -2 and v['zero_CV']['harmed_rows'] == 0,
                zero_CV_harmed_rows=v['zero_CV']['harmed_rows'], switch_rate=v['switch_rate'])
        value = dict(identity=identity, group=name, metadata=group, views=views, summary=summary,
            protected_floor_ADE_vs_CV={s: metric(da, cv, m) for s, m in masks.items()},
            verified=True, decision_replays=6, result_source='fresh_run', no_fitting=True)
        run.immutable_json(path, value); summaries[name] = summary; refs.append(run.artifact(path))
    summary = dict(result_source='fresh_run', data_role='model_selection', groups=summaries,
        localities=roster, rows=n, supported_ADE=int(np.isfinite(cv).sum()), supported_endpoint=int(labels['valid'][:, -1].sum()),
        all_groups_reported=True, groups_are_dependent=True, candidate_selected=False,
        deployment_changed=False, no_new_training=True, calibration_opened=False, confirmation_opened=False,
        stage5c_executed=False, smc_enabled=False, coordinate_unit='image_pixel', time_unit='raw_annotation_steps')
    run.immutable_json(run.PUBLIC/'summary_metrics.json', summary)
    run.immutable_json(run.PUBLIC/'completion_checks.json', dict(identity=identity, all_passed=True,
        coordinate_checks_this_invocation=coordinate_checks, evaluation_receipts=refs,
        summary=run.artifact(run.PUBLIC/'summary_metrics.json'), data=run.artifact(run.PUBLIC/'data_audit.json'),
        decisions=run.artifact(freeze), engineering_completed_not_scientific_gate_pass=True))
    run.beat('readout_complete', groups=len(summaries), rows=n)
