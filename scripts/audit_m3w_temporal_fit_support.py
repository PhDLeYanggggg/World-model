"""Replay fitting/held cost risk and moving-zero support without new training."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_temporal_intervention import load, load_states, training_arrays
from scripts.run_m3w_bounded_cost import read_arrays, features
from scripts.run_m3w_native_forecast import immutable_json, assert_current, array_hash
from scripts.verify_m3w_temporal_intervention import manual_candidates
from scripts.verify_m3w_native_joint_controls import distances
from src.world_model.m3w_bounded_cost_head import build, predict
from src.world_model.m3w_native_gain_harm import standardized
from src.evaluation.m3w_conditional_cost_audit import strict_bits
from src.evaluation.m3w_temporal_fit_support import population_summary, moving_zero_support, standardized_support
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, predictions, scalar, refs, identity = load()
    _, states = load_states(cfg, views, identity)
    public = ROOT/cfg['reports']; report = json.loads((public/'analysis.json').read_text())
    digest = file_digest(public/'analysis.json'); assert report['identity'] == identity
    bindings = {}
    for name in ('replay.json', 'separate_verification.json', 'action_choice_diagnosis.json'):
        r = json.loads((public/name).read_text()); assert r['analysis_sha256'] == digest
        if name != 'action_choice_diagnosis.json':
            assert r['all_checks_passed']
        bindings[name] = file_digest(public/name)
    n = len(data['sites']); allids = np.arange(n)
    future, valid = read_arrays(data, allids, 'target'), read_arrays(data, allids, 'valid')
    baseline = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, _ = distances(baseline, future, valid, data['scale']); full = valid.all(1)
    past = data['geometry'][:, :16].reshape(n, 8, 2)
    records, cases = [], []
    for key, meta in views.items():
        archive = next(a for a in report['archives'] if a['view'] == key)
        assert file_digest(ROOT/archive['path']) == archive['sha256']
        with np.load(ROOT/archive['path'], allow_pickle=False) as z:
            hi = z['ids'].copy()
            hscores = {a: z[a+'_score'].copy() for a in ('ramp', 'uniform')}
            hchoices = {a: z[a+'_strict'].copy() for a in ('ramp', 'uniform')}
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(hi, z['ids']); hp = z['prediction'].copy()
        hproposals, _ = manual_candidates(baseline[hi], hp)
        for arm in ('ramp', 'uniform'):
            ids, x, y, d, pr, _, weights, _ = training_arrays(meta, data, refs, key, cfg, arm)
            cp = states[key, arm]; model = build(x.shape[1], cfg['training']['width'], meta['seed'])
            model.load_state_dict(cp['model']); score = predict(model, x, d, pr, 'bounded_native')
            choice = strict_bits(score, past[ids], d)
            hx, hd, _ = features(data['geometry'][hi], hproposals[arm], data['scale'][hi])
            hs = predict(model, hx, hd, pr, 'bounded_native')
            np.testing.assert_array_equal(hs, hscores[arm])
            np.testing.assert_array_equal(strict_bits(hs, past[hi], hd), hchoices[arm])
            error, _ = distances(hproposals[arm], future[hi], valid[hi], data['scale'][hi])
            hy = np.column_stack((np.maximum(cv[hi]-error, 0), np.maximum(error-cv[hi], 0)))
            hy[~full[hi]] = np.nan
            for role, ii, xx, yy, dd, ss, cc, ww, draws in (
                ('fitting', ids, x, y, d, score, choice, weights, cp['draws']),
                ('held_source', hi, hx, hy, hd, hs, hchoices[arm], np.ones(len(hi)), None)):
                for site in sorted(set(data['sites'][ii])):
                    m = data['sites'][ii] == site
                    records.append(dict(view=key, arm=arm, role=role, site=str(site),
                        costs=population_summary(yy[m], ss[m], dd[m], cc[m], ww[m]),
                        reference_support=moving_zero_support(past[ii][m], cv[ii][m], full[ii][m], cc[m],
                                                             None if draws is None else draws[m]),
                        standardized_features=standardized_support(xx[m], pr['mean'], pr['std'], cc[m])))
            bad = hchoices[arm] & full[hi] & (cv[hi] == 0) & (error > 0)
            if bad.any():
                fit = np.flatnonzero(pr['known'] & (d > 0) & np.any(past[ids, -1] != past[ids, -2], axis=1))
                if len(fit) < 32:
                    raise ValueError('Insufficient fitting support for the fixed neighbor diagnostic')
                zfit = standardized(x[fit], pr).astype(float)
                for j in np.flatnonzero(bad):
                    zh = standardized(hx[j:j+1], pr).astype(float)[0]
                    squared = np.square(zfit-zh).mean(1)
                    nearest = fit[np.lexsort((ids[fit], squared))[:32]]
                    cases.append(dict(view=key, arm=arm, row_identity_sha256=array_hash(hi[j:j+1]),
                        selected_ade=float(error[j]), max_abs_standardized_feature=float(np.abs(zh).max()),
                        feature_columns_outside_fit_range=int(((hx[j] < x[pr['known']].min(0)) |
                                                              (hx[j] > x[pr['known']].max(0))).sum()),
                        fit_moving_zero_rows=int((pr['known'] & (cv[ids] == 0) & np.any(past[ids, -1] != past[ids, -2], axis=1)).sum()),
                        neighbors=32, neighbor_distance_rms_quantiles=np.quantile(np.sqrt(squared[np.lexsort((ids[fit], squared))[:32]]), [0, .5, 1]).tolist(),
                        neighbor_harmful=int((y[nearest, 1] > 0).sum()), neighbor_beneficial=int((y[nearest, 0] > 0).sum()),
                        neighbor_mean_harm=float(y[nearest, 1].mean()), neighbor_mean_predicted_harm=float(score[nearest, 1].mean()),
                        neighbor_mean_benefit=float(y[nearest, 0].mean()), neighbor_mean_predicted_benefit=float(score[nearest, 0].mean()),
                        neighbor_selected=int(choice[nearest].sum()),
                        neighbor_sites={str(s): int((data['sites'][ids[nearest]] == s).sum()) for s in pr['training_sites']}))
            print(json.dumps(dict(view=key, arm=arm, state='fit_and_held_diagnosed')), flush=True)
    summary = {}
    for arm in ('ramp', 'uniform'):
        for role in ('fitting', 'held_source'):
            r = [v for v in records if v['arm'] == arm and v['role'] == role]
            supported = [v['costs']['selected'] for v in r if v['costs']['selected']['complete']]
            ratios = [v['harm_ratio'] for v in supported if v['harm_ratio'] is not None]
            summary[arm+'_'+role] = dict(groups=len(r),
                groups_with_complete_selected_rows=len(supported),
                all_underprediction=sum(v['costs']['all']['harm_underpredicted'] for v in r),
                selected_underprediction=sum(v['costs']['selected'].get('harm_underpredicted', False) for v in r),
                selected_weighted_underprediction=sum(v['fit_weighted_harm'] > v['fit_weighted_predicted_harm']
                    for v in supported if v['fit_weighted_harm'] is not None),
                selected_harm_ratio_quantiles=np.quantile(ratios, [0, .5, 1]).tolist() if ratios else None)
    assert len(records) == 96 and len(cases) == 2
    out = dict(result_source='fresh_frozen_fit_and_held_support_diagnosis_no_new_fit',
        analysis_sha256=digest, verification_sources=bindings,
        code_sha256=file_digest(Path(__file__)), helper_sha256=file_digest(ROOT/'src/evaluation/m3w_temporal_fit_support.py'),
        tests_sha256=file_digest(ROOT/'tests/test_m3w_temporal_fit_support.py'),
        records=records, summary=summary, harmed_zero_reference_cases=cases,
        fitting_scores_in_sample_not_calibration=True, label_groups_for_diagnosis_only=True,
        nearest_neighbors_fixed_k32_in_existing_fit_standardized_features=True,
        neighboring_windows_not_independent=True, new_training=False, changed_decisions=False,
        independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity); immutable_json(public/'fit_support_diagnosis.json', out)
    print(json.dumps(dict(summary=summary, zero_cases=cases), indent=2))


if __name__ == '__main__':
    main()
