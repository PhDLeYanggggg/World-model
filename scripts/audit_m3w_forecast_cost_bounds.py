"""Frozen-control missing-outcome bounds and training-only cost support audit."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_joint_controls import load, view_scores
from scripts.run_m3w_native_forecast import assert_current, immutable_json, array_hash
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_forecast_cost_bounds import disagreement, partial_gain_bounds, bounded_fractions, energy_concentration
import numpy as np
import torch


def summary(mask, bounds, selected):
    use = mask & selected
    possible = use & ~bounds['complete'] & bounds['exact_CV_still_possible'] & (bounds['full_disagreement'] > 0)
    return dict(rows=int(mask.sum()), selected=int(use.sum()),
        full_grid_absolute_gain_lower=float(np.where(use, bounds['lower'], 0)[mask].mean()),
        full_grid_absolute_gain_upper=float(np.where(use, bounds['upper'], 0)[mask].mean()),
        selected_unknown_radius=float(bounds['unknown_radius'][use].sum()),
        incomplete_selected=int((use & ~bounds['complete']).sum()),
        incomplete_possible_zero_CV_harm_rows=int(possible.sum()),
        incomplete_possible_zero_CV_harm_sum_upper=float(bounds['exact_CV_harm_upper'][possible].sum()),
        complete_observed_zero_CV_harm_rows=int((use & bounds['complete'] & (bounds['exact_CV_harm_upper'] > 0)).sum()))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    _, data, _, views, gain, _, identity = load()
    bindings = dict(identity['source_bindings'])
    for path in ('scripts/audit_m3w_forecast_cost_bounds.py', 'src/evaluation/m3w_forecast_cost_bounds.py', 'tests/test_m3w_forecast_cost_bounds.py'):
        bindings[path] = file_digest(ROOT/path)
    # Selection is fixed before reading any target/validity arrays in this audit.
    fixed = {key:view_scores(key, meta, gain, data) for key, meta in views.items()}
    n = len(data['sites'])
    y = np.empty((n, 12, 2), np.float32)
    valid = np.zeros((n, 12), bool)
    for rec in np.unique(data['recordings']):
        rows = data['recordings'] == rec
        root = ROOT/'data/stage_cvpr2027_experiments/sdd_auxiliary_v1/inputs'/rec
        y[rows] = np.load(root/'target.npy', allow_pickle=False)
        valid[rows] = np.load(root/'valid.npy', allow_pickle=False)
    training, outer = [], []
    for key, meta in views.items():
        for kind in ('inputs', 'targets'):
            path = meta[kind+'_path']
            assert file_digest(ROOT/path) == meta[kind+'_sha256']
            bindings[path] = meta[kind+'_sha256']
        with np.load(ROOT/meta['inputs_path'], allow_pickle=False) as z:
            ids, p = z['ids'].copy(), z['prediction'].copy()
        with np.load(ROOT/meta['targets_path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            cost = np.column_stack((z['benefit'], z['harm']))
        assert meta['outer_site'] not in data['sites'][ids]
        b = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        d = disagreement(p, b, data['scale'][ids]).mean(1)
        full = valid[ids].all(1)
        fraction = bounded_fractions(cost, d, full)
        direct = partial_gain_bounds(p, b, y[ids], valid[ids], data['scale'][ids])
        np.testing.assert_allclose(cost[full, 0]-cost[full, 1], direct['lower'][full], atol=1e-10, rtol=1e-10)
        training.append(dict(view=key, training_sites=sorted(set(data['sites'][ids])), rows=len(ids),
            complete_rows=int(full.sum()), query_ids_sha256=array_hash(ids),
            max_cost_bound_excess=float((cost[full].sum(1)-d[full]).max()),
            positive_disagreement_complete_rows=int((full & (d > 0)).sum()),
            native_harm=energy_concentration(cost[full, 1]), fraction_harm=energy_concentration(fraction[full, 1]),
            native_benefit=energy_concentration(cost[full, 0]), fraction_benefit=energy_concentration(fraction[full, 0]),
            fraction_target_quantiles={name:np.quantile(fraction[full, j], [0, .5, .9, .99, 1]).tolist() for j,name in enumerate(('benefit', 'harm'))}))
        ids, p, b, score, eligible, pr = fixed[key]
        bounds = partial_gain_bounds(p, b, y[ids], valid[ids], data['scale'][ids])
        d = bounds['full_disagreement']
        chosen = {'uncontrolled':np.ones(len(ids), bool), 'stop_mse_strict':eligible}
        exceed = score.sum(1)-d
        outer.append(dict(view=key, rows=len(ids), score_bound_violation_rows=int((exceed > 1e-8*(1+d)).sum()),
            selected_score_bound_violation_rows=int((eligible & (exceed > 1e-8*(1+d))).sum()),
            score_bound_max_excess=float(exceed.max()), controls={name:summary(np.ones(len(ids), bool), bounds, use) for name,use in chosen.items()},
            strict_selected_ids_sha256=array_hash(ids[eligible])))
        print(json.dumps(dict(view=key, training_geometry_checked=True, fixed_outcome_bounds_checked=True)), flush=True)
    result = dict(result_source='fresh_run_analytic_bounds_and_training_support_cached_verified_forecasts_and_scores',
        source_bindings=bindings, training=training, frozen_outer_diagnostics=outer,
        inference_future_labels=False, label_support_controls_intervention=False,
        selection_rule_changed=False, new_training=False, independent_confirmation=False,
        risk_calibrated=False, closed_role_readout=False, statistical_safety_claim=False,
        primary_metric_changed=False, absolute_interval_unit='annotation_pixel_full_12_step_ADE_gain_not_relative_percent',
        independent_scene_roles='not_assigned_and_no_new_roles_in_this_audit', stage5c_executed=False, smc_enabled=False)
    assert_current(dict(source_bindings=bindings))
    path = ROOT/'outputs/publication_readiness_2026_09/native_cost_bounds_v1/analysis.json'
    immutable_json(path, result)
    print(json.dumps(dict(output=str(path), sha256=file_digest(path))))


if __name__ == '__main__':
    main()
