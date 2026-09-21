"""Separate arithmetic and exhaustive real-query verification; no policy fitting."""
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_joint_controls import load, view_scores, ARMS
from scripts.run_m3w_native_forecast import assert_current, immutable_json
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np


def distances(p, y, mask, scale):
    """Explicit xy arithmetic without the experiment metric implementation."""
    delta = np.asarray(p, float)-np.asarray(y, float)
    d = np.sqrt(delta[:, :, 0]**2+delta[:, :, 1]**2)*scale[:, None]
    counts = mask.sum(1)
    ade = np.full(len(p), np.nan)
    has = counts > 0
    ade[has] = np.sum(np.where(mask, d, 0), axis=1)[has]/counts[has]
    return ade, np.where(mask[:, -1], d[:, -1], np.nan)


def check_reduction(model, reference, sites, roster, expected):
    values = []
    assert np.array_equal(np.isnan(model), np.isnan(reference))
    for site in roster:
        use = (sites == site) & np.isfinite(reference)
        r = expected['by_scene'][site]
        assert int(use.sum()) == r['rows']
        if not use.any():
            assert r['model_error'] is None
            continue
        mm, rr = float(np.mean(model[use])), float(np.mean(reference[use]))
        np.testing.assert_allclose([mm, rr], [r['model_error'], r['reference_error']], rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(np.percentile(model[use], [95, 99]), [r['model_p95'], r['model_p99']], rtol=1e-12, atol=1e-10)
        if rr == 0:
            assert r['gain_percent'] is None
        else:
            gain = 100*(1-mm/rr)
            np.testing.assert_allclose(gain, r['gain_percent'], rtol=1e-10, atol=1e-10)
            values.append(gain)
    if len(values) == len(roster):
        np.testing.assert_allclose(np.mean(values), expected['equal_scene_gain_percent'], atol=1e-10, rtol=1e-10)
        rng = np.random.default_rng(expected['bootstrap_seed'])
        indices = rng.integers(len(roster), size=(expected['bootstrap_resamples'], len(roster)))
        interval = np.quantile(np.asarray(values)[indices].mean(1), [.025, .975])
        np.testing.assert_allclose(interval, expected['scene_bootstrap_ci95'], atol=1e-10, rtol=1e-10)
    else:
        assert expected['equal_scene_gain_percent'] is None
    return len(roster)


def exhaustive_query(q, c, data, ids_index, prediction, baseline, cost, eligible, preprocess):
    rows = np.flatnonzero(c['context_frame_ids'] == q['frame'])
    tid = c['context_target_rows'][rows]
    target = tid >= 0
    loc = ids_index[tid[target]]
    n = len(rows)
    b = c['context_cv_rollout'][rows].copy()
    cand = b.copy()
    for normalized, output in ((baseline[loc], b), (prediction[loc], cand)):
        ids = tid[target]
        # Inverse of row-vector local rotation, independently from restore().
        output[target] = np.matmul(normalized*data['scale'][ids, None, None],
                                  np.swapaxes(data['rotation'][ids], 1, 2))+data['origin'][ids, None]
    valid = c['context_cv_valid'][rows].all(1)
    valid[target] = True
    supported = np.zeros(n, bool)
    supported[target] = eligible[loc]
    gain, harm = np.zeros(n), np.zeros(n)
    native_cost = np.asarray(cost[loc], np.float64)
    gain[target] = (native_cost[:, 0]-native_cost[:, 1])/preprocess['cost_scale']
    harm[target] = native_cost[:, 1]/preprocess['cost_scale']
    harm = np.maximum(harm, -gain)
    radius = np.median(data['scale'][tid[target]])
    xy = c['context_xy'][rows]
    edges = np.array([(i, j) for i in range(n) for j in range(i+1, n)
                      if valid[i] and valid[j] and np.sqrt(np.sum((xy[i]-xy[j])**2)) <= radius], int).reshape(-1, 2)
    table = np.empty((len(edges), 2, 2))
    for a, left in enumerate((b, cand)):
        for d, right in enumerate((b, cand)):
            delta = left[edges[:, 0]]-right[edges[:, 1]]
            distance = np.sqrt(delta[:, :, 0]**2+delta[:, :, 1]**2)
            table[:, a, d] = np.mean(np.maximum(0, 1-distance/(.1*radius))**2, axis=1)
    table = np.maximum(0, table-table[:, :1, :1])
    r = q['budgets']['half']
    pool = np.flatnonzero(supported)
    assert len(pool) == r['pool']
    source_ids = c['context_agent_ids'][rows]
    reference = pool[np.lexsort((source_ids[pool], -gain[pool]))[:r['count']]]
    np.testing.assert_allclose(harm[reference].sum()/n, r['predicted_harm_budget'], atol=1e-14, rtol=1e-12)
    candidates, feasible = 0, []
    for subset in itertools.combinations(pool, r['count']):
        candidates += 1
        bits = np.zeros(n, bool)
        bits[list(subset)] = True
        if np.mean(bits*harm) > r['predicted_harm_budget']+1e-10:
            continue
        i, j = bits[edges[:, 0]].astype(int), bits[edges[:, 1]].astype(int)
        neural_term = -np.mean(bits*gain)
        pair = np.mean(table[np.arange(len(edges)), i, j]) if len(edges) else 0.
        unary_pair = np.mean(table[:, 1, 0]*i+table[:, 0, 1]*j) if len(edges) else 0.
        feasible.append((neural_term+unary_pair, neural_term+pair))
    scores = np.asarray(feasible)
    assert len(scores)
    u, j = r['controls']['unary'], r['controls']['joint']
    for col, value in ((0, u['unary_objective']), (1, j['full_objective'])):
        np.testing.assert_allclose(scores[:, col].min(), value, rtol=1e-9, atol=1e-9)
    residual = scores[:, 1]-scores[:, 0]
    order = np.argsort(scores[:, 0])
    gap = None if len(scores) == 1 else float(scores[order[1], 0]-scores[order[0], 0])
    return dict(view=q['view'], recording=q['recording'], frame=q['frame'],
        pool=len(pool), count=r['count'], enumerated=candidates, feasible=len(scores),
        product_range=float(np.ptp(residual)), unary_runner_up_gap=gap,
        product_range_below_unary_gap=None if gap is None else bool(np.ptp(residual) < gap),
        objective_advantage=float(u['full_objective']-j['full_objective']),
        all_optima_checked=True, future_labels_used=False)


def main():
    cfg, data, context, views, gain, prior, identity = load()
    public = ROOT/cfg['reports']
    a = json.loads((public/'analysis.json').read_text())
    replay = json.loads((public/'decision_replay.json').read_text())
    manifest = json.loads((ROOT/cfg['output']/'decisions_complete.json').read_text())
    assert a['identity'] == identity and replay['all_checks_passed']
    assert a['decision_manifest_sha256'] == replay['decision_manifest_sha256'] == file_digest(ROOT/cfg['output']/'decisions_complete.json')
    n = len(data['sites'])
    choices = {s:np.zeros((n, len(ARMS)), bool) for s in cfg['seeds']}
    seen = {s:np.zeros(n, bool) for s in cfg['seeds']}
    queries = []
    for receipt in manifest['receipts']:
        path = ROOT/receipt['path']
        assert file_digest(path) == receipt['sha256']
        r = json.loads(path.read_text())
        assert file_digest(ROOT/r['path']) == r['sha256']
        seed = int(r['view'].rsplit('seed', 1)[1])
        with np.load(ROOT/r['path'], allow_pickle=False) as z:
            ids = z['ids']
            assert not seen[seed][ids].any()
            choices[seed][ids] = z['choices']
            seen[seed][ids] = True
        queries.extend(r['queries'])
    assert all(s.all() for s in seen.values())
    enumerated = []
    for key, meta in views.items():
        wanted = [q for q in queries if q['view'] == key and q['budgets']['half']['nonadditive_supported_edges'] > 0]
        ids, prediction, baseline, cost, eligible, pr = view_scores(key, meta, gain, data)
        index = np.full(n, -1, int)
        index[ids] = np.arange(len(ids))
        for record in context['records']:
            w = [q for q in wanted if q['recording'] == record['recording']]
            if not w:
                continue
            with np.load(ROOT/record['cache']['path'], allow_pickle=False) as z:
                c = {k:z[k] for k in z.files}
            for q in w:
                enumerated.append(exhaustive_query(q, c, data, index, prediction, baseline, cost, eligible, pr))
    assert len(enumerated) == a['mechanism']['half']['queries_with_nonadditive_opportunity']
    # Outcomes are read only after the separate, past-only exhaustive check.
    y, valid = np.empty((n, 12, 2), np.float32), np.zeros((n, 12), bool)
    for recording in np.unique(data['recordings']):
        root = ROOT/'data/stage_cvpr2027_experiments/sdd_auxiliary_v1/inputs'/recording
        use = data['recordings'] == recording
        y[use] = np.load(root/'target.npy', allow_pickle=False)
        valid[use] = np.load(root/'valid.npy', allow_pickle=False)
    cv, cf = distances(data['geometry'][:, 332:356].reshape(n, 12, 2), y, valid, data['scale'])
    masks = dict(complete=valid.all(1), zero_CV=valid.all(1) & (cv == 0),
                 hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    neural_ade = {s:np.full(n, np.nan) for s in cfg['seeds']}
    neural_fde = {s:np.full(n, np.nan) for s in cfg['seeds']}
    for key, meta in views.items():
        ids, prediction, _, _, eligible, pr = view_scores(key, meta, gain, data)
        seed = meta['seed']
        neural_ade[seed][ids], neural_fde[seed][ids] = distances(prediction, y[ids], valid[ids], data['scale'][ids])
        masks['hard'][ids] = cv[ids] >= pr['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= pr['positive_easy_cut'])
        for arm in ('full_independent', 'full_unary', 'full_joint'):
            np.testing.assert_array_equal(choices[seed][ids, ARMS.index(arm)], eligible)
    reductions, all_fde, unknowns = 0, {}, {}
    for j, arm in enumerate(ARMS):
        errors, endpoints = [], []
        unknowns[arm] = {}
        for seed in cfg['seeds']:
            bits = choices[seed][:, j]
            ade, fde = np.where(bits, neural_ade[seed], cv), np.where(bits, neural_fde[seed], cf)
            errors.append(ade)
            endpoints.append(fde)
            expected = a['summaries'][arm]['seeds'][str(seed)]
            for field, model, reference in (('ADE', ade, cv), ('FDE', fde, cf)):
                reductions += check_reduction(model, reference, data['sites'], cfg['sites'], expected[field])
            for key, mask in masks.items():
                reductions += check_reduction(ade[mask], cv[mask], data['sites'][mask], cfg['sites'], expected['subsets'][key])
            counts = dict(selected_rows=int(bits.sum()), unknown_ADE_selected_rows=int((bits & ~valid.any(1)).sum()),
                incomplete_risk_selected_rows=int((bits & ~valid.all(1)).sum()),
                complete_zero_CV_harmed_rows=int(np.count_nonzero(ade[masks['zero_CV']] > 0)))
            assert all(expected[k] == v for k, v in counts.items())
            unknowns[arm][str(seed)] = counts
        average = np.mean(errors, axis=0)
        expected = a['summaries'][arm]
        reductions += check_reduction(average, cv, data['sites'], cfg['sites'], expected['ADE'])
        for key, mask in masks.items():
            reductions += check_reduction(average[mask], cv[mask], data['sites'][mask], cfg['sites'], expected['subsets'][key])
        fde = np.mean(endpoints, axis=0)
        all_fde[arm] = {s:100*(1-fde[(data['sites'] == s) & np.isfinite(cf)].mean()/cf[(data['sites'] == s) & np.isfinite(cf)].mean()) for s in cfg['sites']}
    result = dict(analysis_sha256=file_digest(public/'analysis.json'), verifier_sha256=file_digest(Path(__file__)),
        all_checks_passed=True, source_bindings=len(identity['source_bindings']), scene_reductions=reductions,
        choices_replayed=replay['choices_replayed'], enumeration=enumerated,
        enumerated_combinations=sum(r['enumerated'] for r in enumerated),
        feasible_combinations=sum(r['feasible'] for r in enumerated),
        opportunities_with_unique_feasible_assignment=sum(r['feasible'] == 1 for r in enumerated),
        opportunities_with_variable_product_cost=sum(r['product_range'] > 1e-12 for r in enumerated),
        opportunities_with_product_range_below_unary_gap=sum(r['product_range_below_unary_gap'] is True for r in enumerated),
        seed_averaged_FDE_gain_by_scene=all_fde, selected_label_support=unknowns,
        independent_implementation_same_agent=True, independent_research_confirmation=False,
        future_labels_used_in_enumeration=False, future_labels_used_in_error_verification=True,
        new_training=False, new_policy_selection=False, deployment=False,
        scope='independent arithmetic and exhaustive finite optimization QA; no calibrated risk or physical safety claim')
    assert_current(identity)
    immutable_json(public/'independent_verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('enumeration', 'seed_averaged_FDE_gain_by_scene', 'selected_label_support')}, indent=2))


if __name__ == '__main__':
    main()
