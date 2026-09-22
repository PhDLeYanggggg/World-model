"""Separate ranking/choice, coordinate-error, reduction and bound verification."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_risk_ranking import load
from scripts.run_m3w_native_forecast import assert_current, immutable_json
from scripts.run_m3w_bounded_cost import read_arrays
from scripts.verify_m3w_temporal_intervention import manual_candidates
from scripts.verify_m3w_native_joint_controls import distances, check_reduction
from scripts.verify_m3w_tempered_cost import check_contrast
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def manual_rank(score, pool, ids, count, by_ratio):
    def key(i):
        g, h = map(float, score[i])
        value = (h/g if g > 0 else float('inf')) if by_ratio else h-g
        return value, int(ids[i])
    order = sorted(pool, key=key)
    use = np.zeros(len(ids), bool); use[order[:count]] = True
    return use


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, predictions, states, previous, identity = load()
    public = ROOT/cfg['reports']; report = json.loads((public/'analysis.json').read_text())
    assert report['identity'] == identity
    replay = json.loads((public/'replay.json').read_text())
    assert replay['analysis_sha256'] == file_digest(public/'analysis.json') and replay['all_checks_passed']
    assert file_digest(ROOT/cfg['output']/'decisions_complete.json') == report['decision_manifest_sha256']
    n = len(data['sites']); policies = cfg['policies']
    selected = {s: {p: np.zeros(n, bool) for p in policies} for s in cfg['seeds']}
    proposals = {s: np.empty((n, 12, 2)) for s in cfg['seeds']}
    scores = {}; decisions = 0
    for key, meta in views.items():
        old_archive = next(r for r in previous['archives'] if r['view'] == key)
        new_archive = next(r for r in report['archives'] if r['view'] == key)
        assert file_digest(ROOT/new_archive['path']) == new_archive['sha256']
        with np.load(ROOT/old_archive['path'], allow_pickle=False) as z: old = {k: z[k].copy() for k in z.files}
        with np.load(ROOT/new_archive['path'], allow_pickle=False) as z: new = {k: z[k].copy() for k in z.files}
        ids = old['ids']; np.testing.assert_array_equal(ids, new['ids'])
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        pool = list(np.flatnonzero((old['ramp_distance'] > 0) & np.any(past[:, -1] != past[:, -2], axis=1)))
        count = sum(bool(v) for v in old['ramp_forest'])
        for estimator in ('forest', 'neural'):
            score = old['ramp_'+estimator+'_score']; scores[key, estimator] = score
            for rule in ('strict', 'ratio', 'gain'):
                name = estimator+'_'+rule
                if rule == 'strict':
                    use = np.zeros(len(ids), bool)
                    for i in pool:
                        g, h = map(float, score[i]); use[i] = g > h and h <= .1*g
                else:
                    use = manual_rank(score, pool, ids, count, rule == 'ratio')
                np.testing.assert_array_equal(use, new[name])
                selected[meta['seed']][name][ids] = use; decisions += 1
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); p = z['prediction'].copy()
        b = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        proposals[meta['seed']][ids] = manual_candidates(b, p)[0]['ramp']
        print(json.dumps(dict(view=key, state='separate_rankings_verified')), flush=True)
    target, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    b = data['geometry'][:, 332:356].reshape(n, 12, 2); cv, cf = distances(b, target, valid, data['scale'])
    full = valid.all(1); masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool),
                                    positive_easy=np.zeros(n, bool))
    for key, meta in views.items():
        use = data['sites'] == meta['outer_site']; pr = states[key, 'ramp']['preprocess']
        masks['hard'][use] = cv[use] >= pr['hard_cut']
        masks['positive_easy'][use] = (cv[use] > 0) & (cv[use] <= pr['positive_easy_cut'])
    errors, bounds = {}, {}
    safe = np.where(valid[..., None], target, 0).astype(float)
    be = np.sqrt(np.square(b-safe).sum(-1))*data['scale'][:, None]
    for seed, p in proposals.items():
        errors[seed] = distances(p, target, valid, data['scale'])
        pe = np.sqrt(np.square(p-safe).sum(-1))*data['scale'][:, None]
        dd = np.sqrt(np.square(p-b).sum(-1))*data['scale'][:, None]
        observed = np.where(valid, be-pe, 0).sum(1)/12
        unknown = np.where(valid, 0, dd).sum(1)/12
        bounds[seed] = observed-unknown, observed+unknown
    reductions = 0
    for name in policies:
        ades, fdes = [], []
        for seed in cfg['seeds']:
            use = selected[seed][name]; ade = np.where(use, errors[seed][0], cv); fde = np.where(use, errors[seed][1], cf)
            ades.append(ade); fdes.append(fde); r = report['summaries'][name]['seeds'][str(seed)]
            assert r['selected'] == int(use.sum())
            assert r['selected_unknown'] == int((use & ~valid.any(1)).sum())
            assert r['selected_incomplete'] == int((use & ~full).sum())
            assert r['zero_CV_harmed'] == int((ade[masks['zero_CV']] > 0).sum())
            reductions += check_reduction(ade, cv, data['sites'], cfg['sites'], r['ADE'])
            reductions += check_reduction(fde, cf, data['sites'], cfg['sites'], r['FDE'])
            for group, mask in masks.items():
                reductions += check_reduction(ade[mask], cv[mask], data['sites'][mask], cfg['sites'], r['subsets'][group])
            for site in cfg['sites']:
                np.testing.assert_allclose([np.where(use, v, 0)[data['sites']==site].mean() for v in bounds[seed]],
                    r['full_grid_absolute_gain_bounds'][site], rtol=1e-10, atol=1e-8)
        r = report['summaries'][name]
        for vals, reference, field in ((ades, cv, 'ADE'), (fdes, cf, 'FDE')):
            reductions += check_reduction(np.mean(vals, 0), reference, data['sites'], cfg['sites'], r[field])
        for group, mask in masks.items():
            reductions += check_reduction(np.mean(ades, 0)[mask], cv[mask], data['sites'][mask], cfg['sites'], r['subsets'][group])
        checks = dict(each_seed_positive_cv=all(s['ADE']['equal_scene_gain_percent'] > 0 for s in r['seeds'].values()),
            aggregate_easy=all(s['subsets']['positive_easy']['equal_scene_gain_percent'] >= -2 for s in r['seeds'].values()),
            each_scene_seed_easy=all(v['gain_percent'] >= -2 for s in r['seeds'].values()
                for v in s['subsets']['positive_easy']['by_scene'].values()),
            exact_zero=not any(s['zero_CV_harmed'] for s in r['seeds'].values()))
        assert checks == report['descriptive_checks'][name]
    for pair, contrast in report['contrasts'].items():
        a, bname = pair.split('_minus_')
        check_contrast(report['summaries'][a]['ADE']['by_scene'], report['summaries'][bname]['ADE']['by_scene'], cfg['sites'], contrast)
    for row in report['overlaps']:
        meta = views[row['view']]; mask = data['sites'] == meta['outer_site']
        a = set(np.flatnonzero(selected[meta['seed']][row['policy_a']] & mask))
        bb = set(np.flatnonzero(selected[meta['seed']][row['policy_b']] & mask))
        expected = dict(a=len(a), b=len(bb), common=len(a & bb), a_only=len(a-bb), b_only=len(bb-a),
                        jaccard=None if not a | bb else len(a & bb)/len(a | bb))
        assert expected == row['counts']
    risk_rows = 0
    for row in report['conditional_quality']:
        key = row['view']; meta = views[key]; ids = np.flatnonzero(data['sites'] == meta['outer_site'])
        take = selected[meta['seed']][row['chosen_by']][ids] & full[ids]
        delta = cv[ids]-errors[meta['seed']][0][ids]
        score = scores[key, row['estimator']]; r = row['costs']['selected']
        assert int(take.sum()) == r['complete']
        if take.any():
            actual = np.maximum(-delta[take], 0); pred = score[take, 1]
            np.testing.assert_allclose([actual.mean(), pred.mean()], [r['realized_harm'], r['predicted_harm']], rtol=1e-10, atol=1e-8)
            assert bool(pred.mean() < actual.mean()) == r['harm_underpredicted']
        risk_rows += 1
    assert_current(identity)
    immutable_json(public/'separate_verification.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
        verifier_sha256=file_digest(Path(__file__)), all_checks_passed=True, decisions_checked=decisions,
        scene_reductions_checked=reductions, conditional_selected_harm_groups_checked=risk_rows,
        bounds_and_overlaps_checked=True, context_smoothness_turnover_replayed_only=True,
        same_agent_shared_arrays=True, independent_research_confirmation=False))
    print(json.dumps(dict(verified=True, decisions=decisions, reductions=reductions, risk_groups=risk_rows)))


if __name__ == '__main__': main()
