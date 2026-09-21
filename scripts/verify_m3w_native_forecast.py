"""Check completed real fits, paired sampling and independent prediction reductions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_forecast import load, specification, completed, assert_current, immutable_json
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_supervised_intervention import build_forecaster
from src.world_model.m3w_native_forecast import predict
import numpy as np
import torch


def independent_errors(p, y, valid, scale):
    x = p[:, :, 0].astype(float)-np.where(valid, y[:, :, 0], 0).astype(float)
    z = p[:, :, 1].astype(float)-np.where(valid, y[:, :, 1], 0).astype(float)
    distance = np.sqrt(x*x+z*z)*scale[:, None]
    count = valid.sum(1)
    ade = np.divide(np.sum(np.where(valid, distance, 0), axis=1), count,
                    out=np.full(len(p), np.nan), where=count > 0)
    return ade, np.where(valid[:, -1], distance[:, -1], np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true', help='Also replay fixed first/middle/last full inference batches')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg, data, rows, identity = load()
    out, public = ROOT/reg['output'], ROOT/reg['reports']
    report = json.loads((public/'analysis.json').read_text())
    if report['identity'] != identity or report['models'] != 24 or report['optimizer_updates'] != 96000:
        raise ValueError('Wrong completed registered matrix')
    by_trial = {r['trial']:r for r in report['predictions']}
    paired = {}; costs = {}; replay_rows = 0; weights_changed = 0
    for objective in reg['objectives']:
        for seed in reg['seeds']:
            n = len(data['sites'])
            ade, fde, cv, cfinal = [np.full(n, np.nan) for _ in range(4)]
            hard, seen = np.zeros(n, bool), np.zeros(n, bool)
            for site in reg['sites']:
                fold, ti, key = specification(reg, data, identity, site, objective, seed)
                receipt = completed(out/'trials'/key/'complete.json', ti, reg['training'])
                cp = torch.load(ROOT/receipt['checkpoint'], map_location='cpu', weights_only=False)
                assert cp['step'] == 4000 and cp['identity'] == ti
                assert int(cp['draws'].sum()) == 256000 and not cp['draws'][fold['held_ids']].any()
                np.testing.assert_array_equal(cp['factors'], fold['factors'])
                pair_key = site, seed
                if pair_key not in paired:
                    paired[pair_key] = (cp['draws'].copy(), cp['sampler_rng'].clone())
                else:
                    np.testing.assert_array_equal(cp['draws'], paired[pair_key][0])
                    torch.testing.assert_close(cp['sampler_rng'], paired[pair_key][1], atol=0, rtol=0)
                last = cp['model']['predictor.predictor.output.2.weight']
                if last.abs().sum() <= 0:
                    raise ValueError('Output weights stayed at zero initialization')
                weights_changed += 1
                pred = by_trial[key]
                if file_digest(ROOT/pred['path']) != pred['sha256']:
                    raise ValueError('Changed predictions')
                with np.load(ROOT/pred['path'], allow_pickle=False) as z:
                    ids, prediction = z['ids'], z['prediction']
                    np.testing.assert_array_equal(ids, fold['held_ids'])
                assert not seen[ids].any(); seen[ids] = True
                y, valid, scale = data['target'][ids], data['valid'][ids], data['scale'][ids]
                ade[ids], fde[ids] = independent_errors(prediction, y, valid, scale)
                cv[ids], cfinal[ids] = independent_errors(data['geometry'][ids, 332:356].reshape(-1, 12, 2), y, valid, scale)
                hard[ids] = cv[ids] >= fold['hard_cut']
                static = rows['static_history'][ids]
                np.testing.assert_array_equal(prediction[static], data['geometry'][ids[static], 332:356].reshape(-1, 12, 2))
                if args.replay:
                    torch.manual_seed(seed); model = build_forecaster(reg['architecture'])
                    model.load_state_dict(cp['model'])
                    for start in sorted({0, ((len(ids)//2)//128)*128, ((len(ids)-1)//128)*128}):
                        use = ids[start:start+128]
                        actual = predict(model, data, use)
                        np.testing.assert_array_equal(actual, prediction[start:start+128])
                        replay_rows += len(use)
                print(json.dumps(dict(trial=key, optimizer_steps=cp['step'], held_rows=len(ids),
                                      replay_rows_so_far=replay_rows)), flush=True)
            assert seen.all()
            costs[objective, seed] = dict(ade=ade, fde=fde, cv=cv, cfinal=cfinal, hard=hard)

    checked = 0
    def check(table, m, r, mask):
        nonlocal checked
        gains = []
        for site in reg['sites']:
            use = mask & (data['sites'] == site) & np.isfinite(r)
            expected = table['by_scene'][site]
            assert expected['rows'] == int(use.sum())
            if not use.any():
                assert expected['gain_percent'] is None
                continue
            mm, rr = np.mean(m[use], dtype=np.float64), np.mean(r[use], dtype=np.float64)
            np.testing.assert_allclose([expected['model_error'], expected['reference_error']], [mm, rr], rtol=1e-12, atol=1e-9)
            if rr == 0:
                assert expected['gain_percent'] is None
            else:
                gains.append(float(100*(rr-mm)/rr))
                np.testing.assert_allclose(expected['gain_percent'], gains[-1], rtol=1e-12, atol=1e-9)
            checked += 1
        if len(gains) == len(reg['sites']):
            np.testing.assert_allclose(table['equal_scene_gain_percent'], np.mean(gains), atol=1e-9)
            draws = np.random.default_rng(reg['bootstrap_seed']).integers(4, size=(reg['bootstrap_resamples'], 4))
            interval = np.quantile(np.asarray(gains)[draws].mean(1), [.025, .975])
            np.testing.assert_allclose(table['scene_bootstrap_ci95'], interval, atol=1e-9)
        else:
            assert table['equal_scene_gain_percent'] is table['scene_bootstrap_ci95'] is None

    all_rows = np.ones(len(data['sites']), bool)
    for objective in reg['objectives']:
        for seed in reg['seeds']:
            c = costs[objective, seed]
            metrics = report['summary'][objective]['seeds'][str(seed)]
            masks = dict(ADE=all_rows, complete_ADE=data['valid'].all(1),
                static_history_ADE=rows['static_history'], moving_history_ADE=~rows['static_history'],
                hard_training_q75_ADE=c['hard'], zero_CV_complete_easy_ADE=data['valid'].all(1) & (c['cv'] == 0))
            for key, mask in masks.items():
                check(metrics[key], c['ade'], c['cv'], mask)
            check(metrics['FDE'], c['fde'], c['cfinal'], all_rows)
        for metric, key, ref in [('mean_seed_ADE', 'ade', 'cv'), ('mean_seed_FDE', 'fde', 'cfinal')]:
            m = np.mean([costs[objective, seed][key] for seed in reg['seeds']], axis=0)
            check(report['summary'][objective][metric], m, costs[objective, 17][ref], all_rows)
    check(report['native_vs_matched_old_loss'],
          np.mean([costs['native_coordinate', seed]['ade'] for seed in reg['seeds']], axis=0),
          np.mean([costs['past_normalized', seed]['ade'] for seed in reg['seeds']], axis=0), all_rows)
    assert_current(identity)
    result = dict(result_source='fresh_run_checkpoint_and_independent_metric_verification',
        analysis_sha256=file_digest(public/'analysis.json'), verifier_sha256=file_digest(Path(__file__)),
        completed_fits=24, optimizer_updates=96000, matched_sampler_pairs=len(paired),
        learned_nonzero_output_heads=weights_changed, held_rows_sampled_by_training=0,
        source_query_predictions_checked=len(data['sites'])*6,
        exact_inference_replay_rows=replay_rows, scene_metric_reductions_checked=checked,
        all_checks_passed=True, new_training=False, independent_confirmation=False)
    immutable_json(public/('verification_with_replay.json' if args.replay else 'verification.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
