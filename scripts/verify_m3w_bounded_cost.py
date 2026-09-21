"""Separate finite-cost arithmetic, policy choice and training-support checks."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_bounded_cost import load, features, read_arrays, training_data
from scripts.run_m3w_native_joint_controls import view_scores
from scripts.run_m3w_native_forecast import array_hash, assert_current, immutable_json
from scripts.verify_m3w_native_joint_controls import distances, check_reduction
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_bounded_cost_head import ARMS
import numpy as np
import torch


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg, data, views, gain, identity = load()
    public = ROOT/cfg['reports']
    a = json.loads((public/'analysis.json').read_text())
    replay = json.loads((public/'verification_with_replay.json').read_text())
    assert replay['all_checks_passed'] and replay['analysis_sha256'] == file_digest(public/'analysis.json')
    assert a['identity'] == identity
    n = len(data['sites'])
    names = list(a['summaries'])
    choices = {s:{name:np.zeros(n, bool) for name in names} for s in cfg['seeds']}
    ne, nf = {s:np.full(n, np.nan) for s in cfg['seeds']}, {s:np.full(n, np.nan) for s in cfg['seeds']}
    lo, hi = {s:np.zeros(n) for s in cfg['seeds']}, {s:np.zeros(n) for s in cfg['seeds']}
    y, mask = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    b = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = distances(b, y, mask, data['scale'])
    full = mask.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    training_rows, choice_checks, score_rows = 0, 0, 0
    for key, meta in views.items():
        ids_train, x, targets, dtrain, pr = training_data(meta, data)
        known = full[ids_train]
        np.testing.assert_array_equal(known, pr['known'])
        assert meta['outer_site'] not in data['sites'][ids_train]
        previous_draws = None
        for arm in ARMS:
            record = next(r for r in a['training'] if r['view'] == key and r['arm'] == arm)
            assert file_digest(ROOT/record['checkpoint']) == record['checkpoint_sha256']
            cp = torch.load(ROOT/record['checkpoint'], map_location='cpu', weights_only=False)
            ti = cp['identity']
            assert ti['inputs_sha256'] == array_hash(ids_train, x, dtrain) and ti['labels_sha256'] == array_hash(targets)
            assert ti['complete_mask_sha256'] == array_hash(known) and cp['step'] == cfg['training']['steps']
            assert cp['draws'][~known].sum() == 0 and cp['draws'].sum() == cfg['training']['steps']*cfg['training']['batch_size']
            assert set(cp['preprocess']['training_sites']) == set(cfg['sites'])-{meta['outer_site']}
            for field in ('mean', 'std', 'known', 'weights', 'constant'):
                np.testing.assert_array_equal(cp['preprocess'][field], pr[field])
            if previous_draws is not None:
                np.testing.assert_array_equal(previous_draws, cp['draws'])
            previous_draws = cp['draws']
            training_rows += int(known.sum())
        ids, p, baseline, _, legacy, parent_pr = view_scores(key, meta, gain, data)
        seed = meta['seed']
        _, d, _ = features(data['geometry'][ids], p, data['scale'][ids])
        r = next(r for r in a['archives'] if r['view'] == key)
        assert file_digest(ROOT/r['path']) == r['sha256']
        with np.load(ROOT/r['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            past = data['geometry'][ids, :16].reshape(-1, 8, 2)
            allowed = (d > 0) & np.any(past[:, -1] != past[:, -2], axis=1)
            choices[seed]['uncontrolled'][ids] = True
            choices[seed]['legacy_stop_mse_strict'][ids] = legacy
            for arm in ARMS:
                score = z[arm]
                assert np.isfinite(score).all() and np.all(score >= 0)
                if arm != 'direct_native':
                    assert np.all(score.sum(1) <= d+2e-6*(1+d))
                net = allowed & (score[:, 0]-score[:, 1] > 0)
                strict = net & (10*score[:, 1] <= score[:, 0])
                pool = np.flatnonzero(allowed)
                ordered = sorted(pool, key=lambda j:(float(score[j, 1]-score[j, 0]), int(ids[j])))
                matched = np.zeros(len(ids), bool)
                matched[ordered[:int(legacy.sum())]] = True
                for policy, bits in (('net_stop', net), ('strict_stop', strict), ('matched_count', matched)):
                    np.testing.assert_array_equal(bits, z[arm+'_'+policy])
                    choices[seed][arm+'_'+policy][ids] = bits
                    choice_checks += 1
                score_rows += len(ids)
        ne[seed][ids], nf[seed][ids] = distances(p, y[ids], mask[ids], data['scale'][ids])
        yt = np.where(mask[ids, :, None], y[ids], 0).astype(float)
        bp = np.sqrt(np.sum((baseline.astype(float)-yt)**2, axis=-1))*data['scale'][ids, None]
        pp = np.sqrt(np.sum((p.astype(float)-yt)**2, axis=-1))*data['scale'][ids, None]
        disagreement = np.sqrt(np.sum((p.astype(float)-baseline.astype(float))**2, axis=-1))*data['scale'][ids, None]
        observed = np.where(mask[ids], bp-pp, 0).sum(1)/12
        unknown = np.where(mask[ids], 0, disagreement).sum(1)/12
        lo[seed][ids], hi[seed][ids] = observed-unknown, observed+unknown
        masks['hard'][ids] = cv[ids] >= parent_pr['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= parent_pr['positive_easy_cut'])
    reductions = 0
    for name in names:
        ades, fdes = [], []
        for seed in cfg['seeds']:
            use = choices[seed][name]
            ade, fde = np.where(use, ne[seed], cv), np.where(use, nf[seed], cf)
            ades.append(ade)
            fdes.append(fde)
            r = a['summaries'][name]['seeds'][str(seed)]
            assert int(use.sum()) == r['selected']
            assert int((use & ~mask.any(1)).sum()) == r['selected_unknown']
            assert int((use & ~full).sum()) == r['selected_incomplete']
            assert int((ade[masks['zero_CV']] > 0).sum()) == r['zero_CV_harmed']
            reductions += check_reduction(ade, cv, data['sites'], cfg['sites'], r['ADE'])
            reductions += check_reduction(fde, cf, data['sites'], cfg['sites'], r['FDE'])
            for site in cfg['sites']:
                interval = [np.where(use, x, 0)[data['sites'] == site].mean() for x in (lo[seed], hi[seed])]
                np.testing.assert_allclose(interval, r['full_grid_absolute_gain_bounds'][site], atol=1e-10, rtol=1e-10)
            for group, subgroup in masks.items():
                reductions += check_reduction(ade[subgroup], cv[subgroup], data['sites'][subgroup], cfg['sites'], r['subsets'][group])
        r = a['summaries'][name]
        reductions += check_reduction(np.mean(ades, 0), cv, data['sites'], cfg['sites'], r['ADE'])
        reductions += check_reduction(np.mean(fdes, 0), cf, data['sites'], cfg['sites'], r['FDE'])
        for group, subgroup in masks.items():
            reductions += check_reduction(np.mean(ades, 0)[subgroup], cv[subgroup], data['sites'][subgroup], cfg['sites'], r['subsets'][group])
    result = dict(analysis_sha256=file_digest(public/'analysis.json'), all_checks_passed=True,
        verifier_sha256=file_digest(Path(__file__)),
        reduction_helper_sha256=file_digest(ROOT/'scripts/verify_m3w_native_joint_controls.py'),
        training_complete_rows_checked_repeated=training_rows, policies_checked=choice_checks,
        score_rows_checked=score_rows, scene_reductions=reductions, matched_count_is_offline=True,
        independent_implementation_same_agent=True, independent_research_confirmation=False,
        full_grid_bounds_are_evaluation_only=True, zero_CV_rule_unchanged=True,
        selection_changes=False, deployment=False)
    assert_current(identity)
    immutable_json(public/'independent_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
