"""Independently reconstruct exact-count choices and scalar outcome reductions."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_matched_coverage import load
from scripts.verify_m3w_native_gain_harm import errors, check_mean
from scripts.run_m3w_native_forecast import array_hash, immutable_json
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def partition_choice(priority, eligible, ids, k):
    """Select by cutoff and boundary IDs, independently of the original lexsort."""
    use = np.zeros(len(ids), bool)
    values = priority[eligible]
    if k:
        cutoff = np.partition(values, len(values)-k)[len(values)-k]
        use = eligible & (priority > cutoff)
        boundary = np.flatnonzero(eligible & (priority == cutoff))
        use[boundary[np.argsort(ids[boundary])[:k-int(use.sum())]]] = True
    return use


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg, parent, data, views, old, identity = load()
    public = ROOT/reg['reports']
    analysis = json.loads((public/'analysis.json').read_text())
    replay = json.loads((public/'verification.json').read_text())
    assert analysis['identity'] == identity and replay['all_checks_passed']
    assert replay['analysis_sha256'] == file_digest(public/'analysis.json')
    prior = json.loads((ROOT/parent['reports']/'independent_verification.json').read_text())
    assert prior['verifier_sha256'] == file_digest(ROOT/'scripts/verify_m3w_native_gain_harm.py')
    choices_checked, reductions_checked, random_checks = 0, 0, 0
    for key, meta in views.items():
        site, seed = meta['outer_site'], meta['seed']
        scores_binding = next(r for r in old['score_archives'] if r['view'] == key)
        with np.load(ROOT/scores_binding['path'], allow_pickle=False) as z:
            ids = z['ids'].copy()
            a, m, ridge = z['underharm4'].copy(), z['mse'].copy(), z['ridge_raw'].copy()
        with np.load(ROOT/meta['outer_prediction']['prediction']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            prediction = z['prediction'].copy()
        baseline = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        eligible = np.any(prediction != baseline, axis=(1, 2))
        positive = (a[:, 0] > a[:, 1]) & eligible
        total = m.sum(1)
        ratio = np.zeros(len(m)); np.divide(m[:, 0]-m[:, 1], total, out=ratio, where=total != 0)
        salt = int.from_bytes(hashlib.sha256(f'native-coverage-v1:{site}:{seed}'.encode()).digest()[:8], 'little')
        shuffle = np.empty(len(ids)); shuffle[np.argsort(ids)] = np.random.default_rng(salt).permutation(len(ids))
        ranks = dict(mse_ratio=ratio, mse_net_gain=m[:, 0]-m[:, 1], asym_net_gain=a[:, 0]-a[:, 1],
            ridge_raw_net_gain=ridge[:, 0]-ridge[:, 1], hash_control=shuffle)
        cv, cf = errors(baseline, data['target'][ids], data['valid'][ids], data['scale'][ids])
        ne, nf = errors(prediction, data['target'][ids], data['valid'][ids], data['scale'][ids])
        known, complete = np.isfinite(cv), data['valid'][ids].all(1)
        record = next(r for r in old['training'] if r['view'] == key and r['arm'] == 'ridge')
        pr = torch.load(ROOT/record['checkpoint'], map_location='cpu', weights_only=False)['preprocess']
        masks = dict(complete=complete, zero_CV=complete & (cv == 0), hard=cv >= pr['hard_cut'],
            positive_easy=(cv > 0) & (cv <= pr['positive_easy_cut']))
        for budget in reg['budgets']:
            anchor = positive.copy()
            if budget == 'harm_fraction_0p1':
                anchor &= a[:, 1] <= .1*a[:, 0]
            k = int(anchor.sum())
            for policy in reg['policies']:
                use = anchor if policy == 'asym_rule' else partition_choice(ranks[policy], eligible, ids, k)
                rc = next(r for r in analysis['policy_receipts'] if r['view'] == key and r['budget'] == budget and r['policy'] == policy)
                assert int(use.sum()) == k == rc['selected_rows']
                assert array_hash(ids[use]) == rc['selected_ids_sha256']
                assert int((use & ~known).sum()) == rc['unknown_selected_rows']
                actual, end = np.where(use, ne, cv), np.where(use, nf, cf)
                r = analysis['summary'][budget][policy]['seeds'][str(seed)]
                check_mean(actual, cv, np.ones(len(ids), bool), r['ADE']['by_scene'][site]); reductions_checked += 1
                check_mean(end, cf, np.ones(len(ids), bool), r['FDE']['by_scene'][site]); reductions_checked += 1
                for name, mask in masks.items():
                    check_mean(actual, cv, mask, r['subsets'][name]['by_scene'][site]); reductions_checked += 1
                harmed = masks['zero_CV'] & (actual > 0)
                assert int(harmed.sum()) == rc['zero_CV_harmed_rows']
                np.testing.assert_allclose(actual[masks['zero_CV']].max(), rc['zero_CV_max_harm'], rtol=1e-12, atol=1e-12)
                choices_checked += 1
            random = next(r for r in analysis['uniform_random_expectations'] if r['view'] == key and r['budget'] == budget)
            p = k/eligible.sum() if eligible.any() else 0.
            for name, mask in dict(all=np.ones(len(ids), bool), **masks).items():
                use = mask & known
                # Total expectation: retain CV for every row, add the uniformly
                # sampled candidate-minus-CV cost only on eligible rows.
                value = (cv[use].sum()+p*(ne[use & eligible]-cv[use & eligible]).sum())/use.sum()
                np.testing.assert_allclose(value, random['groups'][name]['expected_ADE'], rtol=1e-12, atol=1e-12)
                random_checks += 1
        print(json.dumps(dict(view=key, independent_choices_and_errors='passed')), flush=True)
    result = dict(result_source='fresh_run_independent_partition_and_scalar_arithmetic',
        analysis_sha256=file_digest(public/'analysis.json'), verifier_sha256=file_digest(Path(__file__)),
        distance_helper_sha256=file_digest(ROOT/'scripts/verify_m3w_native_gain_harm.py'),
        policy_choices_checked=choices_checked, scene_reductions_checked=reductions_checked,
        random_expected_cost_reductions_checked=random_checks, all_checks_passed=True,
        new_training=False, new_threshold_search=False, independent_confirmation=False)
    immutable_json(public/'independent_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
