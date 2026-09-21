"""Separate label, selection and scalar-error verification of frozen risk heads."""
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_protected_risk import load
from scripts.run_m3w_native_forecast import array_hash, immutable_json
from scripts.verify_m3w_native_gain_harm import errors, check_mean
from scripts.verify_m3w_native_matched_coverage import partition_choice
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def labels(cv, harm, complete, protected):
    y = np.full((len(cv), 2), np.nan)
    idx = np.flatnonzero(complete)
    value = harm[idx].copy()
    if protected:
        value[cv[idx] != 0] = 0
    y[idx, 0], y[idx, 1] = (value > 0).astype(float), value
    return y


def check_event(p, y, complete, record):
    p, y = p[complete], y[complete]
    assert len(p) == record['rows'] and int(y.sum()) == record['positive_rows']
    np.testing.assert_allclose(np.mean((p-y)**2), record['brier'], rtol=1e-12, atol=1e-12)
    low = p <= .01
    assert int(low.sum()) == record['score_le_0p01_rows']
    assert int(y[low].sum()) == record['score_le_0p01_positive_rows']
    ece = 0.
    for k in range(10):
        use = (p >= k/10) & ((p < (k+1)/10) if k < 9 else (p <= 1))
        if use.any():
            ece += abs(p[use].sum()-y[use].sum())/len(p)
    # The producer averages float32 scores; this sum-first reduction promotes
    # after summation. Permit their dtype rounding, not a scientific tolerance.
    tolerance = 4*np.finfo(p.dtype).eps if np.issubdtype(p.dtype, np.floating) else 1e-12
    np.testing.assert_allclose(ece, record['ece'], rtol=0, atol=tolerance)
    return float(abs(ece-record['ece']))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg, parent, data, views, old, identity = load()
    public = ROOT/reg['reports']
    analysis = json.loads((public/'analysis.json').read_text())
    replay = json.loads((public/'verification.json').read_text())
    assert analysis['identity'] == identity and replay['all_checks_passed']
    assert replay['analysis_sha256'] == file_digest(public/'analysis.json')
    choice_checks, reductions, labels_checked, event_checks = 0, 0, 0, 0
    slices, seed_harm, seed_unknown, seed_incomplete = [], {}, {}, {}
    ece_rounding = []
    seed_max = {}
    for key, meta in views.items():
        site, seed = meta['outer_site'], meta['seed']
        rb = next(x for x in analysis['score_archives'] if x['view'] == key)
        assert file_digest(ROOT/rb['path']) == rb['sha256']
        with np.load(ROOT/rb['path'], allow_pickle=False) as z:
            ids = z['ids'].copy()
            risk = {a:z[a].copy() for a in reg['arms']}
        sb = next(x for x in old['score_archives'] if x['view'] == key)
        assert file_digest(ROOT/sb['path']) == sb['sha256']
        with np.load(ROOT/sb['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            m, a = z['mse'].copy(), z['underharm4'].copy()
        with np.load(ROOT/meta['outer_prediction']['prediction']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            prediction = z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == site))
        baseline = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        cv, cf = errors(baseline, data['target'][ids], data['valid'][ids], data['scale'][ids])
        ne, nf = errors(prediction, data['target'][ids], data['valid'][ids], data['scale'][ids])
        complete = data['valid'][ids].all(1)
        h = np.maximum(ne-cv, 0)
        same = np.all(prediction == baseline, axis=(1, 2))
        with np.load(ROOT/meta['targets_path'], allow_pickle=False) as z:
            train_ids = z['ids'].copy()
            tc, th = z['baseline_ade'].copy(), z['harm'].copy()
        np.testing.assert_array_equal(train_ids, np.flatnonzero(data['sites'] != site))
        train_full = data['valid'][train_ids].all(1)
        paired_draws = None
        for arm in reg['arms']:
            tr = next(x for x in analysis['training'] if x['view'] == key and x['arm'] == arm)
            assert file_digest(ROOT/tr['checkpoint']) == tr['checkpoint_sha256']
            cp = torch.load(ROOT/tr['checkpoint'], map_location='cpu', weights_only=False)
            truth = labels(tc, th, train_full, arm == 'zero_reference_harm')
            assert array_hash(truth) == cp['identity']['labels_sha256']
            assert array_hash(train_ids) == cp['identity']['ids_sha256']
            np.testing.assert_array_equal(cp['preprocess']['known'], train_full)
            assert cp['draws'][~train_full].sum() == 0
            if paired_draws is None:
                paired_draws = cp['draws']
            else:
                np.testing.assert_array_equal(cp['draws'], paired_draws)
            labels_checked += len(truth)
            y = labels(cv, h, complete, arm == 'zero_reference_harm')
            yz = labels(cv, h, complete, True)
            record = next(x for x in analysis['quality'] if x['view'] == key and x['arm'] == arm)
            ece_rounding.append(check_event(risk[arm][:, 0], y[:, 0], complete, record['own_event']))
            ece_rounding.append(check_event(risk[arm][:, 0], yz[:, 0], complete, record['zero_event_ranking']))
            np.testing.assert_allclose(np.mean((risk[arm][complete, 1]-y[complete, 1])**2),
                                       record['own_cost_mse'], rtol=1e-12, atol=1e-12)
            event_checks += 2
        pr_record = next(x for x in old['training'] if x['view'] == key and x['arm'] == 'ridge')
        pr = torch.load(ROOT/pr_record['checkpoint'], map_location='cpu', weights_only=False)['preprocess']
        masks = dict(complete=complete, zero_CV=complete & (cv == 0),
            hard=cv >= pr['hard_cut'], positive_easy=(cv > 0) & (cv <= pr['positive_easy_cut']))
        xy = data['geometry'][ids, :16].reshape(-1, 8, 2)
        stop = (xy[:, -1, 0] == xy[:, -2, 0]) & (xy[:, -1, 1] == xy[:, -2, 1])
        net = m[:, 0]-m[:, 1]
        eligible = (net > 0) & ~same
        pools = dict(net_only=eligible, stop_veto=eligible & ~stop,
            all_harm_guard=eligible & (risk['all_harm'][:, 0] <= .01),
            zero_harm_guard=eligible & (risk['zero_reference_harm'][:, 0] <= .01))
        anchor = (a[:, 0] > a[:, 1]) & (a[:, 1] <= .1*a[:, 0]) & ~same
        k = min(int(anchor.sum()), *(int(v.sum()) for v in pools.values()))
        cap = next(x for x in analysis['policy_capacities'] if x['view'] == key)
        assert k == cap['common_count'] and int(anchor.sum()) == cap['requested_count']
        assert {n:int(v.sum()) for n,v in pools.items()} == cap['capacity']
        policies = {n+'_online':v for n,v in pools.items()}
        policies.update({n+'_matched':partition_choice(net, v, ids, k) for n,v in pools.items()})
        policies.update(parent_asym_strict=anchor, native_without_guard=~same,
            parent_mse_strict=eligible & (m[:, 1] <= .1*m[:, 0]))
        for name, use in policies.items():
            assert array_hash(ids[use]) == cap['selected_ids_sha256'][name]
            choice_checks += 1
            actual, end = np.where(use, ne, cv), np.where(use, nf, cf)
            record = analysis['summary'][name]['seeds'][str(seed)]
            check_mean(actual, cv, np.ones(len(ids), bool), record['ADE']['by_scene'][site])
            check_mean(end, cf, np.ones(len(ids), bool), record['FDE']['by_scene'][site])
            reductions += 2
            for subgroup, mask in masks.items():
                check_mean(actual, cv, mask, record['subsets'][subgroup]['by_scene'][site])
                reductions += 1
            harmed = masks['zero_CV'] & (actual > 0)
            zmax = float(actual[masks['zero_CV']].max())
            index = name, seed
            seed_harm[index] = seed_harm.get(index, 0)+int(harmed.sum())
            seed_max[index] = max(seed_max.get(index, 0.), zmax)
            seed_unknown[index] = seed_unknown.get(index, 0)+int((use & ~np.isfinite(cv)).sum())
            seed_incomplete[index] = seed_incomplete.get(index, 0)+int((use & ~complete).sum())
            selected_known = use & complete
            q = risk['zero_reference_harm'][:, 0]
            slices.append(dict(view=key, policy=name, selected_rows=int(use.sum()),
                unknown_ADE_selected_rows=int((use & ~np.isfinite(cv)).sum()),
                incomplete_future_selected_rows=int((use & ~complete).sum()),
                zero_CV_harmed_rows=int(harmed.sum()), zero_CV_max_harm=zmax,
                harmed_current_stop_rows=int((harmed & stop).sum()),
                complete_selected_rows=int(selected_known.sum()),
                selected_zero_event_rate=float(harmed.sum()/selected_known.sum()) if selected_known.any() else None,
                selected_mean_zero_event_score=float(q[selected_known].mean()) if selected_known.any() else None,
                harmed_zero_event_score_min=float(q[harmed].min()) if harmed.any() else None,
                harmed_zero_event_score_max=float(q[harmed].max()) if harmed.any() else None))
        print(json.dumps(dict(view=key, independent_labels_choices_metrics='passed')), flush=True)
    rows, max_error_rounding = [], 0.
    for name, record in analysis['summary'].items():
        for seed, ss in record['seeds'].items():
            key = name, int(seed)
            assert seed_harm[key] == ss['zero_CV_harmed_rows']
            np.testing.assert_allclose(seed_max[key], ss['zero_CV_max_absolute_harm'], rtol=1e-12, atol=1e-12)
            max_error_rounding = max(max_error_rounding, abs(seed_max[key]-ss['zero_CV_max_absolute_harm']))
            assert seed_unknown[key] == ss['unknown_ADE_selected_rows']
            assert seed_incomplete[key] == ss['unknown_complete_risk_selected_rows']
        seeds = list(record['seeds'].values())
        rows.append(dict(policy=name, ADE_gain_percent=record['ADE']['equal_scene_gain_percent'],
            CI_low=record['ADE']['scene_bootstrap_ci95'][0], CI_high=record['ADE']['scene_bootstrap_ci95'][1],
            hard_gain_percent=record['subsets']['hard']['equal_scene_gain_percent'],
            positive_easy_degradation_percent=-record['subsets']['positive_easy']['equal_scene_gain_percent'],
            mean_switch_percent=100*float(np.mean([s['intervention_rate'] for s in seeds])),
            zero_harmed_query_seed_instances=sum(s['zero_CV_harmed_rows'] for s in seeds),
            worst_zero_absolute_harm=max(s['zero_CV_max_absolute_harm'] for s in seeds),
            unknown_ADE_selected_instances=sum(s['unknown_ADE_selected_rows'] for s in seeds),
            incomplete_risk_selected_instances=sum(s['unknown_complete_risk_selected_rows'] for s in seeds)))
    helper_paths = ('scripts/verify_m3w_native_protected_risk.py', 'scripts/verify_m3w_native_matched_coverage.py',
                    'scripts/verify_m3w_native_gain_harm.py', 'scripts/run_m3w_native_forecast.py')
    result = dict(result_source='fresh_run_independent_labels_partition_and_scalar_reductions',
        analysis_sha256=file_digest(public/'analysis.json'), replay_sha256=file_digest(public/'verification.json'),
        verifier_bindings={p:file_digest(ROOT/p) for p in helper_paths},
        policy_choices_checked=choice_checks, scalar_scene_reductions_checked=reductions,
        training_target_rows_checked=labels_checked, event_metric_checks=event_checks,
        max_ECE_sum_vs_mean_rounding_difference=max(ece_rounding),
        max_absolute_error_arithmetic_rounding_difference=max_error_rounding,
        ECE_check_tolerance='4_machine_eps_of_original_probability_dtype',
        all_checks_passed=True, independent_confirmation=False, new_training=False,
        threshold_search=False, deployment=False)
    immutable_json(public/'independent_verification.json', result)
    immutable_json(public/'failure_slices.json', dict(provenance=result, slices=slices))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    path = public/'results.csv'
    if path.exists():
        assert path.read_bytes() == out.getvalue().encode()
    else:
        path.write_bytes(out.getvalue().encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
