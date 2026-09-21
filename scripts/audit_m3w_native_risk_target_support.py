"""Training-only support for a possible separate easy-harm target; no head fitting."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_matched_coverage import load
from scripts.run_m3w_native_forecast import immutable_json
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg, _, data, views, old, identity = load()
    results = []
    for key, meta in views.items():
        with np.load(ROOT/meta['inputs_path'], allow_pickle=False) as z:
            ids, pred = z['ids'].copy(), z['prediction'].copy()
        with np.load(ROOT/meta['targets_path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            cv, b, h = z['baseline_ade'].copy(), z['benefit'].copy(), z['harm'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] != meta['outer_site']))
        if np.any(data['sites'][ids] == meta['outer_site']):
            raise ValueError('Outer-site rows cannot enter target feasibility statistics')
        known, complete = np.isfinite(cv), data['valid'][ids].all(1)
        g = data['geometry'][ids]
        same = np.all(pred == g[:, 332:356].reshape(-1, 12, 2), axis=(1, 2))
        xy = g[:, :16].reshape(-1, 8, 2)
        stopped_now = np.all(xy[:, -1] == xy[:, -2], axis=1)
        stationary_history = np.all(xy == xy[:, :1], axis=(1, 2))
        rr = next(r for r in old['training'] if r['view'] == key and r['arm'] == 'ridge')
        pr = torch.load(ROOT/rr['checkpoint'], map_location='cpu', weights_only=False)['preprocess']
        groups = dict(all=np.ones(len(ids), bool), full_future=complete,
            zero_CV_complete=complete & (cv == 0),
            positive_easy_diagnostic=(cv > 0) & (cv <= pr['positive_easy_cut']),
            current_observed_step_zero=stopped_now, stationary_past=stationary_history)
        info = {}
        for name, mask in groups.items():
            use = mask & known
            positive = use & (h > 0)
            info[name] = dict(indexed_rows=int(mask.sum()), supported_rows=int(use.sum()),
                unknown_rows=int((mask & ~known).sum()), beneficial_rows=int((use & (b > 0)).sum()),
                harmful_rows=int(positive.sum()), changed_candidate_rows=int((mask & ~same).sum()),
                zero_CV_harmful_rows=int((positive & complete & (cv == 0)).sum()),
                mean_benefit=float(b[use].mean()) if use.any() else None,
                mean_harm=float(h[use].mean()) if use.any() else None,
                harm_q50_q90_q99_max=np.quantile(h[positive], [.5, .9, .99, 1]).tolist() if positive.any() else None,
                harmful_rows_by_training_scene={s:int((positive & (data['sites'][ids] == s)).sum()) for s in pr['training_sites']})
        results.append(dict(view=key, outer_site=meta['outer_site'], training_sites=pr['training_sites'],
            input_sha256=meta['inputs_sha256'], targets_sha256=meta['targets_sha256'],
            groups=info, supervised_targets_only_not_inference_inputs=True))
    result = dict(result_source='fresh_run_training_only_target_feasibility_on_cached_verified_nested_views',
        parent_analysis_sha256=file_digest(ROOT/reg['parent_analysis']),
        code_sha256=file_digest(Path(__file__)), source_bindings=identity['source_bindings'], views=results,
        outer_training_rows=0, new_training=False, threshold_search=False,
        independent_confirmation=False, rows_are_overlapping_and_repeated_across_seeds=True,
        scope='post_hoc_feasibility_not_new_target_validation_or_deployment')
    immutable_json(ROOT/reg['reports']/'risk_target_support.json', result)
    print(json.dumps(dict(views=len(results), outer_training_rows=0, new_training=False)))


if __name__ == '__main__':
    main()
