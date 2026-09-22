"""Diagnose fixed temporal actions and choices; no new policy or threshold fit."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def decomposition(ramp, uniform, uniform_at_ramp, ramp_at_uniform):
    a, b, c, d = [np.asarray(x, float) for x in
                  (ramp, uniform, uniform_at_ramp, ramp_at_uniform)]
    if a.ndim != 1 or not len(a) or any(x.shape != a.shape or not np.isfinite(x).all()
                                      for x in (a, b, c, d)):
        raise ValueError('Aligned finite scene gains required')
    out = dict(total=a-b, action_at_ramp_choices=a-c, choices_on_uniform=c-b,
               action_at_uniform_choices=d-b, choices_on_ramp=a-d)
    np.testing.assert_allclose(out['total'], out['action_at_ramp_choices']+out['choices_on_uniform'], atol=1e-12)
    np.testing.assert_allclose(out['total'], out['action_at_uniform_choices']+out['choices_on_ramp'], atol=1e-12)
    return {k: dict(by_scene_pp=v.tolist(), equal_scene_pp=float(v.mean())) for k, v in out.items()}


def selected_costs(reference, candidate, complete, selected, score):
    r, e, q = np.asarray(reference, float), np.asarray(candidate, float), np.asarray(score, float)
    c, s = np.asarray(complete), np.asarray(selected)
    if (r.ndim != 1 or e.shape != r.shape or q.shape != (len(r), 2)
            or c.shape != r.shape or s.shape != r.shape or c.dtype != bool or s.dtype != bool
            or np.isinf(r).any() or np.isinf(e).any() or not np.isfinite(q).all()
            or (q < 0).any() or np.any(r[np.isfinite(r)] < 0) or np.any(e[np.isfinite(e)] < 0)
            or not np.array_equal(np.isnan(r), np.isnan(e)) or np.any(c & np.isnan(r))):
        raise ValueError('Paired nonnegative costs, support and fixed decisions required')
    groups = dict(complete=c, incomplete_observed=~c & np.isfinite(r), unknown=~np.isfinite(r))
    out = {}
    for name, mask in groups.items():
        take = s & mask
        row = dict(rows=int(take.sum()),
                   predicted_benefit_mean=float(q[take, 0].mean()) if take.any() else None,
                   predicted_harm_mean=float(q[take, 1].mean()) if take.any() else None)
        if name != 'unknown' and take.any():
            delta = r[take]-e[take]
            row.update(beneficial=int((delta > 0).sum()), harmful=int((delta < 0).sum()),
                       tied=int((delta == 0).sum()), realized_benefit_mean=float(np.maximum(delta, 0).mean()),
                       realized_harm_mean=float(np.maximum(-delta, 0).mean()),
                       realized_harm_sum=float(np.maximum(-delta, 0).sum()))
            row['harm_underpredicted'] = row['predicted_harm_mean'] < row['realized_harm_mean']
        out[name] = row
    assert sum(v['rows'] for v in out.values()) == int(s.sum())
    return out


def verify_gates(report):
    p = report['summaries']['ramp_strict']
    expected = dict(
        positive_primary_ci=report['contrasts']['uniform_strict']['ci95_pp'][0] > 0,
        each_seed_positive_cv=all(v['ADE']['equal_scene_gain_percent'] > 0 for v in p['seeds'].values()),
        aggregate_easy=all(v['subsets']['positive_easy']['equal_scene_gain_percent'] >= -2
                           for v in p['seeds'].values()),
        each_scene_seed_easy=all(v['gain_percent'] >= -2 for s in p['seeds'].values()
                                 for v in s['subsets']['positive_easy']['by_scene'].values()),
        exact_zero=all(v['zero_CV_harmed'] == 0 for v in p['seeds'].values()),
        positive_old_scalar_ci=report['contrasts']['scalar_log_strict']['ci95_pp'][0] > 0)
    if expected != report['primary_gates'] or all(expected.values()) != report['primary_joint_empirical_pass']:
        raise ValueError('Recorded gate does not match fixed criteria')
    return expected


def main():
    from scripts.run_m3w_temporal_intervention import load, load_states
    from scripts.run_m3w_bounded_cost import read_arrays
    from scripts.run_m3w_native_forecast import immutable_json, assert_current, array_hash
    from scripts.verify_m3w_temporal_intervention import manual_candidates
    from scripts.verify_m3w_native_joint_controls import distances
    from src.evaluation.m3w_experiment_contract import file_digest
    import torch

    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, predictions, scalar, refs, identity = load()
    _, states = load_states(cfg, views, identity)
    public = ROOT/cfg['reports']; report = json.loads((public/'analysis.json').read_text())
    digest = file_digest(public/'analysis.json')
    assert report['identity'] == identity
    receipts = {}
    for name in ('replay.json', 'separate_verification.json'):
        r = json.loads((public/name).read_text())
        assert r['all_checks_passed'] and r['analysis_sha256'] == digest
        receipts[name] = file_digest(public/name)
    n = len(data['sites']); y, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    b = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, _ = distances(b, y, valid, data['scale'])
    rows, zero_cases = [], []
    for archive in report['archives']:
        key = archive['view']; meta = views[key]
        assert file_digest(ROOT/archive['path']) == archive['sha256']
        with np.load(ROOT/archive['path'], allow_pickle=False) as z:
            ids = z['ids'].copy()
            choices = {a: z[a+'_strict'].copy() for a in ('ramp', 'uniform')}
            scores = {a: z[a+'_score'].copy() for a in ('ramp', 'uniform')}
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); p = z['prediction'].copy()
        proposals, _ = manual_candidates(b[ids], p)
        for arm, q in proposals.items():
            error, _ = distances(q, y[ids], valid[ids], data['scale'][ids])
            full = valid[ids].all(1); take = choices[arm]; pr = states[key, arm]['preprocess']
            masks = dict(all=np.ones(len(ids), bool),
                         positive_easy=(cv[ids] > 0) & (cv[ids] <= pr['positive_easy_cut']),
                         hard=cv[ids] >= pr['hard_cut'])
            for subset, mask in masks.items():
                rows.append(dict(view=key, site=meta['outer_site'], seed=meta['seed'], arm=arm,
                                 subset=subset, costs=selected_costs(cv[ids], error, full, take & mask, scores[arm])))
            harmed = take & full & (cv[ids] == 0) & (error > 0)
            recorded = report['summaries'][arm+'_strict']['seeds'][str(meta['seed'])]['subsets']['zero_CV']['by_scene'][meta['outer_site']]
            np.testing.assert_allclose(error[harmed].sum(), recorded['absolute_harm']*recorded['rows'], rtol=1e-10, atol=1e-8)
            for j in np.flatnonzero(harmed):
                zero_cases.append(dict(view=key, arm=arm, row_identity_sha256=array_hash(ids[j:j+1]),
                                       reference_ade=0., selected_ade=float(error[j]),
                                       predicted_benefit=float(scores[arm][j, 0]),
                                       predicted_harm=float(scores[arm][j, 1]),
                                       complete_label=True, unit='annotation_pixel'))
    summaries = report['summaries']; sites = cfg['sites']
    def gains(name):
        return [summaries[name]['ADE']['by_scene'][s]['gain_percent'] for s in sites]
    parts = decomposition(*[gains(k) for k in ('ramp_strict', 'uniform_strict', 'uniform_at_ramp', 'ramp_at_uniform')])
    np.testing.assert_allclose(parts['total']['equal_scene_pp'], report['contrasts']['uniform_strict']['mean_gain_difference_pp'], atol=1e-12)
    counts = {}
    for arm in ('ramp', 'uniform'):
        r = [v['costs']['complete'] for v in rows if v['arm'] == arm and v['subset'] == 'all']
        counts[arm] = dict(complete_selected_views=len(r),
                          complete_selected_harm_underprediction_views=sum(v.get('harm_underpredicted', False) for v in r))
    out = dict(result_source='fresh_post_readout_diagnosis_from_verified_fixed_decisions',
               analysis_sha256=digest, verification_sources=receipts,
               code_sha256=file_digest(Path(__file__)), tests_sha256=file_digest(ROOT/'tests/test_m3w_temporal_diagnosis.py'),
               physical_sites=sites, seeds=cfg['seeds'], action_choice_decomposition=parts,
               conditional_costs=rows, harm_underprediction=counts, exact_zero_cases=zero_cases,
               independently_recomputed_gate_arithmetic=verify_gates(report),
               incomplete_costs_describe_observed_support_only=True, unknown_outcomes_not_counted_safe=True,
               overlapping_rows_and_repeated_seeds_not_independent=True,
               changed_decisions=False, new_training=False, threshold_search=False,
               independent_research_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity); assert file_digest(public/'analysis.json') == digest
    immutable_json(public/'action_choice_diagnosis.json', out)
    print(json.dumps(dict(decomposition=parts, underprediction=counts, exact_zero_cases=zero_cases), indent=2))


if __name__ == '__main__':
    main()
