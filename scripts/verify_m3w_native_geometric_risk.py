"""Independent target/distance/selection reductions for the geometric factorial."""
import csv
import io
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_geometric_risk import load
from scripts.run_m3w_native_forecast import array_hash, immutable_json
from scripts.verify_m3w_native_gain_harm import errors, check_mean
from scripts.verify_m3w_native_matched_coverage import partition_choice
from scripts.verify_m3w_native_protected_risk import check_event
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg, parent, data, views, gain, previous, identity = load()
    public = ROOT/reg['reports']
    analysis = json.loads((public/'analysis.json').read_text())
    replay = json.loads((public/'verification.json').read_text())
    assert analysis['identity'] == identity and replay['all_checks_passed']
    assert replay['analysis_sha256'] == file_digest(public/'analysis.json')
    checks, reductions, target_rows, quality_checks = 0, 0, 0, 0
    slices, seed_counts, ece_rounding = [], {}, []
    for key, meta in views.items():
        site, seed = meta['outer_site'], meta['seed']
        sb = next(r for r in analysis['score_archives'] if r['view'] == key)
        assert file_digest(ROOT/sb['path']) == sb['sha256']
        with np.load(ROOT/sb['path'], allow_pickle=False) as z:
            ids, distance = z['ids'].copy(), z['distance'].copy()
            probabilities = {a:z[a].copy() for a in reg['arms']}
        with np.load(ROOT/meta['outer_prediction']['prediction']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); pred = z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == site))
        baseline = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        independent_d, _ = errors(pred, baseline, np.ones((len(ids), 12), bool), data['scale'][ids])
        np.testing.assert_allclose(distance, independent_d, rtol=1e-12, atol=1e-12)
        cv, cf = errors(baseline, data['target'][ids], data['valid'][ids], data['scale'][ids])
        ne, nf = errors(pred, data['target'][ids], data['valid'][ids], data['scale'][ids])
        full = data['valid'][ids].all(1)
        same = np.all(pred == baseline, axis=(1, 2))
        truth = np.full(len(ids), np.nan); truth[full] = (cv[full] == 0).astype(float)
        cost = np.where(cv == 0, np.maximum(ne-cv, 0), 0.)
        np.testing.assert_allclose(distance[full]*truth[full], cost[full], rtol=1e-10, atol=1e-10)
        with np.load(ROOT/meta['targets_path'], allow_pickle=False) as z:
            train_ids, train_cv = z['ids'].copy(), z['baseline_ade'].copy()
        np.testing.assert_array_equal(train_ids, np.flatnonzero(data['sites'] != site))
        complete_train = data['valid'][train_ids].all(1)
        train_y = np.full(len(train_ids), np.nan)
        train_y[complete_train] = (train_cv[complete_train] == 0).astype(float)
        first_draws = None
        for arm, p in probabilities.items():
            record = next(r for r in analysis['training'] if r['view'] == key and r['arm'] == arm)
            assert file_digest(ROOT/record['checkpoint']) == record['checkpoint_sha256']
            cp = torch.load(ROOT/record['checkpoint'], map_location='cpu', weights_only=False)
            assert cp['identity']['target_sha256'] == array_hash(train_y)
            assert cp['identity']['ids_sha256'] == array_hash(train_ids)
            np.testing.assert_array_equal(cp['preprocess']['known'], complete_train)
            assert cp['draws'][~complete_train].sum() == 0
            if first_draws is None:
                first_draws = cp['draws']
            else:
                np.testing.assert_array_equal(first_draws, cp['draws'])
            target_rows += len(train_y)
            q = next(r for r in analysis['quality'] if r['view'] == key and r['arm'] == arm)
            ece_rounding.append(check_event(p, truth, full, q['zero_reference_event']))
            ece_rounding.append(check_event(np.where(same, 0., p), np.where(same, 0., truth), full, q['protected_harm_event']))
            np.testing.assert_allclose(np.mean((p[full]*distance[full]-cost[full])**2), q['native_protected_cost_mse'], rtol=1e-12, atol=1e-12)
            quality_checks += 2
        gb = next(r for r in gain['score_archives'] if r['view'] == key)
        with np.load(ROOT/gb['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); m, a = z['mse'].copy(), z['underharm4'].copy()
        rb = next(r for r in previous['score_archives'] if r['view'] == key)
        with np.load(ROOT/rb['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); old_p = z['zero_reference_harm'][:, 0].copy()
        net = m[:, 0]-m[:, 1]; pos = (net > 0) & ~same
        strict = pos & (m[:, 1] <= .1*m[:, 0])
        xy = data['geometry'][ids, :16].reshape(-1, 8, 2); stop = np.all(xy[:, -1] == xy[:, -2], axis=1)
        policies = dict(net_only=pos, mse_strict=strict, stop_mse_strict=strict & ~stop,
            previous_protected_net=pos & (old_p <= .01), previous_protected_strict=strict & (old_p <= .01))
        pools = dict(mse=strict, stop=strict & ~stop, previous=strict & (old_p <= .01))
        for arm, p in probabilities.items():
            policies[arm+'_net'] = pos & (p <= .01)
            policies[arm+'_strict'] = strict & (p <= .01); pools[arm] = policies[arm+'_strict']
        anchor = (a[:, 0] > a[:, 1]) & (a[:, 1] <= .1*a[:, 0]) & ~same
        common = min(int(anchor.sum()), *(int(v.sum()) for v in pools.values()))
        cap = next(r for r in analysis['capacities'] if r['view'] == key)
        assert cap['common_count'] == common and cap['requested_count'] == int(anchor.sum())
        assert cap['capacity'] == {k:int(v.sum()) for k,v in pools.items()}
        policies.update({k+'_matched':partition_choice(net, v, ids, common) for k,v in pools.items()})
        prior = next(r for r in gain['training'] if r['view'] == key and r['arm'] == 'ridge')
        pr = torch.load(ROOT/prior['checkpoint'], map_location='cpu', weights_only=False)['preprocess']
        masks = dict(complete=full, zero_CV=full & (cv == 0), hard=cv >= pr['hard_cut'],
            positive_easy=(cv > 0) & (cv <= pr['positive_easy_cut']))
        for name, use in policies.items():
            assert array_hash(ids[use]) == cap['selected_ids_sha256'][name]; checks += 1
            ade, fde = np.where(use, ne, cv), np.where(use, nf, cf)
            r = analysis['summary'][name]['seeds'][str(seed)]
            check_mean(ade, cv, np.ones(len(ids), bool), r['ADE']['by_scene'][site])
            check_mean(fde, cf, np.ones(len(ids), bool), r['FDE']['by_scene'][site]); reductions += 2
            for k, mask in masks.items():
                check_mean(ade, cv, mask, r['subsets'][k]['by_scene'][site]); reductions += 1
            harmed = masks['zero_CV'] & (ade > 0)
            counts = dict(zero_harmed=int(harmed.sum()), unknown=int((use & ~np.isfinite(cv)).sum()),
                incomplete=int((use & ~full).sum()))
            acc = seed_counts.setdefault((name, seed), dict(zero_harmed=0, unknown=0, incomplete=0))
            for k, v in counts.items():
                acc[k] += v
            slices.append(dict(view=key, policy=name, selected_rows=int(use.sum()), **counts,
                zero_harmed_current_stop_rows=int((harmed & stop).sum()),
                zero_harmed_unique_tracks=len(np.unique(data['tracks'][ids][harmed])),
                max_zero_CV_harm=float(ade[masks['zero_CV']].max())))
        print(json.dumps(dict(view=key, independent_reductions='passed')), flush=True)
    table = []
    for name, r in analysis['summary'].items():
        for seed, s in r['seeds'].items():
            c = seed_counts[name, int(seed)]
            assert c['zero_harmed'] == s['zero_CV_harmed_rows']
            assert c['unknown'] == s['unknown_ADE_selected_rows']
            assert c['incomplete'] == s['incomplete_risk_selected_rows']
        seeds = list(r['seeds'].values())
        table.append(dict(policy=name, ADE_gain_percent=r['ADE']['equal_scene_gain_percent'],
            CI_low=r['ADE']['scene_bootstrap_ci95'][0], CI_high=r['ADE']['scene_bootstrap_ci95'][1],
            hard_gain_percent=r['subsets']['hard']['equal_scene_gain_percent'],
            positive_easy_degradation_percent=-r['subsets']['positive_easy']['equal_scene_gain_percent'],
            mean_switch_percent=100*float(np.mean([s['intervention_rate'] for s in seeds])),
            zero_harmed_query_seed_instances=sum(s['zero_CV_harmed_rows'] for s in seeds),
            max_zero_absolute_harm=max(s['zero_CV_max_absolute_harm'] for s in seeds),
            unknown_ADE_selected_instances=sum(s['unknown_ADE_selected_rows'] for s in seeds),
            incomplete_risk_selected_instances=sum(s['incomplete_risk_selected_rows'] for s in seeds)))
    helpers = ('scripts/verify_m3w_native_geometric_risk.py', 'scripts/verify_m3w_native_gain_harm.py',
               'scripts/verify_m3w_native_matched_coverage.py', 'scripts/verify_m3w_native_protected_risk.py')
    result = dict(result_source='fresh_run_independent_target_distance_selection_and_scalar_checks',
        analysis_sha256=file_digest(public/'analysis.json'), replay_sha256=file_digest(public/'verification.json'),
        verifier_bindings={p:file_digest(ROOT/p) for p in helpers}, policy_choices_checked=checks,
        scalar_scene_reductions_checked=reductions, training_target_rows_checked=target_rows,
        event_metric_checks=quality_checks, max_ECE_rounding_difference=max(ece_rounding),
        all_checks_passed=True, independent_confirmation=False, new_training=False, threshold_search=False)
    immutable_json(public/'independent_verification.json', result)
    immutable_json(public/'failure_slices.json', dict(provenance=result, slices=slices))
    content = io.StringIO(newline=''); writer = csv.DictWriter(content, fieldnames=list(table[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(table)
    path = public/'results.csv'
    if path.exists():
        assert path.read_bytes() == content.getvalue().encode()
    else:
        path.write_bytes(content.getvalue().encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
