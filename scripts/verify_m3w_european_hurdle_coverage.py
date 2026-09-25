"""Separate scalar sorting, coordinate-error and additive-path verification."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_hurdle_support_coverage as run
import numpy as np


def manual_choices(utility, risk, moving, sites, ids, budget):
    eligible = moving & (utility > 0) & (risk['product_mse'][:, 0] > 0) & (risk['hurdle'][:, 0] > 0)
    output = {}
    for label, source in (('product', 'product_mse'), ('hurdle', 'hurdle')):
        bank = risk[source]
        threshold = np.asarray(budget, dtype=bank.dtype) * bank[:, 0]
        output[label + '_original'] = np.array([
            bool(moving[i] and utility[i] > 0 and bank[i, 0] > 0 and bank[i, 1] <= threshold[i])
            for i in range(len(ids))])
    output['hurdle_at_product'] = np.zeros(len(ids), bool)
    output['product_at_hurdle'] = np.zeros(len(ids), bool)
    output['product_common'] = output['product_original'] & eligible
    output['hurdle_common'] = output['hurdle_original'] & eligible
    for site in sorted(set(sites)):
        rows = np.flatnonzero(sites == site)
        pool = [int(i) for i in rows if eligible[i]]
        for anchor, scorer in (('product', 'hurdle'), ('hurdle', 'product')):
            bank = risk['product_mse' if scorer == 'product' else 'hurdle']
            k = int(output[anchor + '_common'][rows].sum())
            ranked = sorted(pool, key=lambda i: (float(bank[i, 1]) / float(bank[i, 0]), int(ids[i])))
            output[scorer + '_at_' + anchor][ranked[:k]] = True
    return output


def coordinate_errors(prediction, target, valid):
    p = np.asarray(prediction, float)
    safe = np.where(valid[..., None], target, 0).astype(float)
    dist = np.sqrt((p[..., 0] - safe[..., 0]) ** 2 + (p[..., 1] - safe[..., 1]) ** 2)
    counts = valid.sum(1)
    ade = np.full(len(p), np.nan)
    np.divide(np.where(valid, dist, 0).sum(1), counts, out=ade, where=counts > 0)
    fde = np.full(len(p), np.nan)
    fde[valid[:, -1]] = dist[valid[:, -1], -1]
    return ade, fde


def close(a, b):
    np.testing.assert_allclose(a, b, atol=1e-8, rtol=1e-10)


def check_metric(cost, reference, sites, mask, row, cfg):
    good = mask & np.isfinite(cost) & np.isfinite(reference)
    assert row['indexed_rows'] == int(mask.sum())
    assert row['supported_rows'] == int(good.sum())
    assert row['unknown_rows'] == int((mask & ~good).sum())
    gains = []
    for site in sorted(set(sites)):
        use = good & (sites == site); expected = row['by_scene'][str(site)]
        assert expected['rows'] == int(use.sum())
        if not use.any():
            assert expected['status'] == 'no_supported_labels'
            continue
        mm, rr = float(cost[use].mean()), float(reference[use].mean())
        close([mm, rr, mm-rr], [expected['model_error'], expected['reference_error'], expected['absolute_harm']])
        close(np.quantile(cost[use], [.95, .99]), [expected['model_p95'], expected['model_p99']])
        if rr > 0:
            gain = 100 * (1 - mm/rr); gains.append(gain)
            close(gain, expected['gain_percent'])
        else:
            assert expected['gain_percent'] is None
    if len(gains) == len(set(sites)):
        close([np.mean(gains), min(gains)], [row['equal_scene_gain_percent'], row['worst_scene_gain_percent']])
        ix = np.random.default_rng(cfg['bootstrap_seed']).integers(0, len(gains), (cfg['bootstrap_resamples'], len(gains)))
        close(np.quantile(np.asarray(gains)[ix].mean(1), [.025, .975]), row['scene_bootstrap_ci95'])
    else:
        assert row['equal_scene_gain_percent'] is None and row['scene_bootstrap_ci95'] is None


def check_decomposition(errors, cv, sites, mask, row, cfg):
    by_component = {k: [] for k in ('total', 'ranking_at_product_count', 'coverage_with_hurdle_ranking',
                                  'coverage_with_product_ranking', 'ranking_at_hurdle_count',
                                  'full_total', 'support_difference')}
    for site in sorted(set(sites)):
        use = mask & (sites == site) & np.isfinite(cv)
        expected = row['by_locality'][str(site)]
        denominator = cv[use].sum()
        assert expected['rows'] == int(use.sum())
        close(denominator, expected['reference_sum'])
        if not use.any() or denominator == 0:
            assert expected['status'] != 'defined'
            continue
        p, h, hp, ph = (errors[k][use].sum() for k in (
            'product_common', 'hurdle_common', 'hurdle_at_product', 'product_at_hurdle'))
        full = errors['product_original'][use].sum() - errors['hurdle_original'][use].sum()
        parts = dict(total=(p-h), ranking_at_product_count=(p-hp), coverage_with_hurdle_ranking=(hp-h),
                     coverage_with_product_ranking=(p-ph), ranking_at_hurdle_count=(ph-h),
                     full_total=full, support_difference=full-(p-h))
        for name, value in parts.items():
            value = float(100 * value / denominator)
            close(value, expected[name]); by_component[name].append(value)
        close(parts['total'], parts['ranking_at_product_count'] + parts['coverage_with_hurdle_ranking'])
        close(parts['total'], parts['ranking_at_hurdle_count'] + parts['coverage_with_product_ranking'])
    for name, vals in by_component.items():
        if len(vals) != len(set(sites)):
            assert row[name] is None
            continue
        ix = np.random.default_rng(cfg['bootstrap_seed']).integers(0, len(vals), (cfg['bootstrap_resamples'], len(vals)))
        close(vals, row[name]['scene_differences_pp'])
        close(np.mean(vals), row[name]['mean_gain_difference_pp'])
        close(np.quantile(np.asarray(vals)[ix].mean(1), [.025, .975]), row[name]['ci95_pp'])


def main():
    run.PRIVATE.mkdir(parents=True, exist_ok=True)
    run.torch.set_num_threads(4); run.torch.set_num_interop_threads(1)
    ctx = run.load(); cfg, _, data, identity, _, _ = ctx
    analysis = json.loads((run.PUBLIC / 'analysis.json').read_text())
    replay = json.loads((run.PUBLIC / 'replay.json').read_text())
    assert replay['all_passed'] and replay['analysis_sha256'] == run.digest(run.PUBLIC / 'analysis.json')
    assert analysis['identity'] == identity
    assert analysis['decision_manifest_sha256'] == run.digest(run.PRIVATE / 'decisions_complete.json')
    archives = run.read_decisions(identity)
    checks = reductions = decompositions = 0
    run.beat('separate_verifier_start')
    for key, candidate, fold, seed, design, utility, moving, risks in run.jobs(ctx):
        ids = design['held_ids']; sites = data['sites'][ids]
        bits = manual_choices(utility, risks, moving, sites, ids, cfg['risk_budget'])
        with np.load(ROOT / archives[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            for name in run.ARMS:
                np.testing.assert_array_equal(bits[name], z[name]); checks += 1
        for site, counts in archives[key]['counts'].items():
            ix = sites == site
            for name in ('product', 'hurdle'):
                assert counts[name + '_count'] == int(bits[name + '_common'][ix].sum())
                assert counts[name + '_original_count'] == int(bits[name + '_original'][ix].sum())
                assert counts[name + '_outside_common'] == int((bits[name + '_original'] & ~bits[name + '_common'])[ix].sum())
        # Outcomes are inspected only after the independent causal choices match.
        ade, fde = coordinate_errors(run.prediction(data, identity, candidate, fold, seed, ids),
                                     data['target_eval'][ids], data['valid'][ids])
        cv, cvf = data['baseline_ade'][ids, 1], data['baseline_fde'][ids, 1]
        masks = dict(all=np.ones(len(ids), bool), easy=(cv > 0) & (cv <= design['easy_cut']), hard=cv >= design['hard_cut'])
        errors = {name: np.where(use, ade, cv) for name, use in bits.items()}
        for name, use in bits.items():
            expected = analysis['views'][key + '_' + name]
            assert expected['decision_sha256'] == run.array_hash(ids, use)
            assert expected['selected_rows'] == int(use.sum())
            assert expected['selected_unknown_ADE'] == int((use & ~np.isfinite(ade)).sum())
            assert expected['selected_unknown_FDE'] == int((use & ~np.isfinite(fde)).sum())
            zero = np.isfinite(cv) & (cv == 0)
            assert expected['zero_CV'] == dict(rows=int(zero.sum()), harmed_rows=int((errors[name][zero] > 0).sum()))
            scorer = risks['hurdle' if name.startswith('hurdle') else 'product_mse']
            violations = use & (scorer[:, 1] > cfg['risk_budget'] * scorer[:, 0])
            assert expected['predicted_risk_violations'] == int(violations.sum())
            for subset, mask in masks.items():
                check_metric(errors[name], cv, sites, mask, expected['ADE_vs_CV'][subset], cfg); reductions += 1
            check_metric(np.where(use, fde, cvf), cvf, sites, masks['all'], expected['FDE_vs_CV'], cfg); reductions += 1
            for site in sorted(set(sites)):
                ix = sites == site; e = expected['by_locality'][str(site)]
                chosen = ix & use & np.isfinite(ade)
                assert e['selected'] == int((ix & use).sum()) and e['selected_known_ADE'] == int(chosen.sum())
                assert e['predicted_risk_violations'] == int((ix & violations).sum())
                if chosen.any():
                    close(np.maximum(ade[chosen]-cv[chosen], 0).mean(), e['selected_positive_harm_mean'])
                    close((cv[chosen]-ade[chosen]).mean(), e['selected_net_gain_mean'])
                else:
                    assert e['selected_positive_harm_mean'] is None and e['selected_net_gain_mean'] is None
            easy = expected['ADE_vs_CV']['easy']['worst_scene_gain_percent']
            assert expected['observed_preservation'] == bool(easy is not None and easy >= -2 and not (errors[name][zero] > 0).any())
        for subset, mask in masks.items():
            check_decomposition(errors, cv, sites, mask, analysis['decomposition'][key][subset], cfg); decompositions += 1
        run.beat('separate_group_verified', group=key)
    run.assert_identity(identity)
    run.immutable_json(run.PUBLIC / 'separate_verification.json', dict(all_passed=True,
        analysis_sha256=run.digest(run.PUBLIC / 'analysis.json'), verifier_sha256=run.digest(Path(__file__)),
        scalar_sorting_choice_arrays=checks, coordinate_metric_reductions=reductions,
        additive_decompositions=decompositions, bootstrap_recomputed=True,
        limitations='shared_verified_forecast_banks_and_labels_not_independent_research_confirmation'))
    run.beat('phase_complete', separate_choices=checks, reductions=reductions, decompositions=decompositions)


if __name__ == '__main__':
    main()
