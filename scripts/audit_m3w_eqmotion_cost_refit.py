"""Post-readout fixed-population diagnosis, never a policy or threshold search."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_m3w_eqmotion_cost_refit import load
from scripts.run_m3w_bounded_cost import read_arrays
from scripts.run_m3w_native_forecast import immutable_json, assert_current
from scripts.verify_m3w_native_joint_controls import distances
from src.evaluation.m3w_experiment_contract import file_digest


def groups(old, new):
    old, new = np.asarray(old), np.asarray(new)
    if old.dtype.kind != 'b' or new.dtype.kind != 'b' or old.shape != new.shape or old.ndim != 1:
        raise ValueError('Aligned frozen decision vectors required')
    return dict(frozen_only=old & ~new, refit_only=new & ~old, overlap=old & new,
                frozen_all=old, refit_all=new)


def summarize(mask, valid, cv, candidate, old, new, distance, speed):
    complete = mask & valid.all(1)
    result = dict(selected=int(mask.sum()), complete=int(complete.sum()),
                  unknown_ADE=int((mask & ~valid.any(1)).sum()), incomplete=int((mask & ~valid.all(1)).sum()))
    if not complete.any():
        return dict(result, costs=None)
    delta = cv[complete]-candidate[complete]
    costs = dict(realized_net_gain=float(delta.mean()), realized_benefit=float(np.maximum(delta, 0).mean()),
                 realized_harm=float(np.maximum(-delta, 0).mean()), baseline_ADE=float(cv[complete].mean()),
                 disagreement=float(distance[complete].mean()), past_speed=float(speed[complete].mean()))
    for name, scores in (('frozen', old), ('refit', new)):
        p = scores[complete]
        costs[name] = dict(predicted_benefit=float(p[:, 0].mean()), predicted_harm=float(p[:, 1].mean()),
            native_cost_MSE=float(((p-np.column_stack((np.maximum(delta,0),np.maximum(-delta,0))))**2).mean()),
            wrong_positive_gain_fraction=float(((p[:, 0] > p[:, 1]) & (delta < 0)).mean()))
    return dict(result, costs=costs)


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, _, predictions, _, identity = load()
    public = ROOT/cfg['reports']; a = json.loads((public/'analysis.json').read_text())
    receipt = json.loads((public/'independent_verification.json').read_text())
    assert receipt['all_checks_passed'] and receipt['analysis_sha256'] == file_digest(public/'analysis.json')
    rows = []
    for rec in a['archives']:
        key = rec['view']; assert file_digest(ROOT/rec['path']) == rec['sha256']
        with np.load(ROOT/rec['path'], allow_pickle=False) as z:
            ids = z['ids'].copy(); old = z['frozen_bounded_fraction'].copy(); new = z['refit_bounded_fraction'].copy()
            d = z['distance'].copy()
            masks = {policy:groups(z['frozen_bounded_fraction_'+policy], z['refit_bounded_fraction_'+policy])
                     for policy in ('strict_stop', 'matched_count')}
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); p = z['prediction'].copy()
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        speed = np.linalg.norm(past[:, -1].astype(float)-past[:, -2], axis=1)*data['scale'][ids]
        y, valid = read_arrays(data, ids, 'target'), read_arrays(data, ids, 'valid')
        b = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        cv, _ = distances(b, y, valid, data['scale'][ids]); pe, _ = distances(p, y, valid, data['scale'][ids])
        for policy, partition in masks.items():
            for label, mask in partition.items():
                rows.append(dict(view=key, policy=policy, group=label,
                    **summarize(mask, valid, cv, pe, old, new, d, speed)))
    out = dict(analysis_sha256=file_digest(public/'analysis.json'), script_sha256=file_digest(Path(__file__)),
        result_source='fresh_post_readout_diagnostic_cached_verified_frozen_decisions',
        rows=rows, same_population_for_both_score_comparisons=True,
        complete_outcomes_only_for_cost_means=True, unknown_as_zero=False,
        threshold_search=False, new_training=False, decision_changes=False,
        independent_confirmation=False, causal_explanation_proven=False)
    assert_current(identity); immutable_json(public/'same_population_forensics.json', out)
    print(json.dumps(dict(rows=len(rows), decision_changes=False, all_checks_passed=True)))


if __name__ == '__main__': main()
