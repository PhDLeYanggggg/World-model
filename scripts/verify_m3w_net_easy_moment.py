"""Separate arithmetic, actual-query exhaustive checks and compact reporting."""
import csv
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_net_easy_moment_guarded import load, receipts
from scripts.run_m3w_native_forecast import file_digest, immutable_json, assert_current
import numpy as np
import torch


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, ap, tp, old, identity = pack = load()
    data = ap[1]; root = ROOT/cfg['output']; public = ROOT/cfg['reports']
    path = public/'analysis.json'; a = json.loads(path.read_text())
    assert a['experiment_sha256'] == file_digest(root/'identity.json')
    fitted = receipts(pack); n = len(data['sites']); arms = cfg['policies']
    dm = json.loads((root/'decisions_complete.json').read_text())
    groups = {(seed, action): dict(bits=np.zeros((n, len(arms)), bool),
        seen=np.zeros(n, bool), positive=np.zeros(n), net=np.zeros(n), denominator=np.zeros(n))
        for seed in cfg['seeds'] for action in cfg['actions']}
    optima = 0; skipped = 0; numeric = 0; seen_small = set(); solver_fallbacks = 0; count_failures = 0
    for ref in dm['receipts']:
        assert file_digest(ROOT/ref['path']) == ref['sha256']
        r = json.loads((ROOT/ref['path']).read_text()); assert file_digest(ROOT/r['path']) == r['sha256']
        seed = int(r['view'].rsplit('_seed', 1)[1]); group = groups[seed, r['action']]
        cut = fitted[r['view'], r['action']]['identity']['easy_cut']
        with np.load(ROOT/r['path'], allow_pickle=False) as z:
            z = {k: z[k].copy() for k in z.files}
        ids = z['ids']; assert not group['seen'][ids].any(); group['seen'][ids] = True
        group['bits'][ids] = z['choices']
        q = z['fractions']; d = z['distance']
        np.testing.assert_allclose(z['positive_risk'], q[:, 0]*d, rtol=0, atol=0)
        np.testing.assert_allclose(z['net_risk'], (q[:, 0]-q[:, 1])*d, rtol=0, atol=0)
        np.testing.assert_allclose(z['denominator'], q[:, 2]*cut, rtol=0, atol=0)
        for source, target in [('positive_risk', 'positive'), ('net_risk', 'net'), ('denominator', 'denominator')]:
            group[target][ids] = z[source]
        np.testing.assert_array_equal(z['choices'][:, 4], z['support'] & (z['positive_risk'] <= .02*z['denominator']))
        np.testing.assert_array_equal(z['choices'][:, 5], z['support'] & (z['net_risk'] <= .02*z['denominator']))
        assert not (z['choices'][:, 4] & ~z['choices'][:, 5]).any()
        for qr in r['queries']:
            rows = np.flatnonzero((data['recordings'][ids] == qr['recording']) & (data['frames'][ids] == qr['frame']))
            assert len(rows)
            gain = z['gain'][rows]; positive = z['positive_risk'][rows]; signed = z['net_risk'][rows]
            eligible = z['support'][rows]; bits = z['choices'][rows]; budget = .02*z['denominator'][rows].sum()
            for col, name, risk in [(6, 'positive', positive), (7, 'net', signed), (8, 'matched', signed)]:
                b = bits[:, col]; info = qr[name]
                assert not (b & ~eligible).any()
                assert int(b.sum()) == info['selected']
                np.testing.assert_allclose(risk[b].sum(), info['predicted_risk'], rtol=1e-12, atol=1e-12)
                assert math.fsum(risk[b]) <= budget
                if name == 'matched':
                    matched = b.sum() == bits[:, 6].sum()
                    assert matched == info['exact_count_pass']
                    count_failures += int(not matched)
                    if info['optimal']:
                        assert matched
                if not info['optimal']:
                    assert not b.any()
                    solver_fallbacks += 1
                numeric += 1
            if qr['positive']['optimal']:
                reference_gain = math.fsum(gain[bits[:, 6]])
                for col, name in [(7, 'net'), (8, 'matched')]:
                    if qr[name]['optimal']:
                        assert math.fsum(gain[bits[:, col]]) >= reference_gain - 1e-8*(1+abs(reference_gain))
            # Fixed audit selection depends on identities and causal support only.
            slot = (r['view'], r['action'], qr['recording'])
            if slot not in seen_small:
                seen_small.add(slot)
                if eligible.sum() > 16:
                    skipped += 1
                    continue
                active = np.flatnonzero(eligible)
                candidates = np.array(list(itertools.product([False, True], repeat=len(active))), bool)
                gains = np.array([math.fsum(gain[active[b]]) for b in candidates])
                for col, risk in [(6, positive), (7, signed), (8, signed)]:
                    name = {6:'positive', 7:'net', 8:'matched'}[col]
                    if not qr[name]['optimal']:
                        continue
                    feasible = np.array([math.fsum(risk[active[b]]) <= budget for b in candidates])
                    if col == 8:
                        feasible &= candidates.sum(1) == bits[:, 6].sum()
                    assert feasible.any()
                    np.testing.assert_allclose(gain[bits[:, col]].sum(), gains[feasible].max(), rtol=1e-7, atol=1e-9)
                    optima += 1
    assert all(g['seen'].all() for g in groups.values())
    records = []; reduced = 0; mean_scene = {}; quality = []
    for action in cfg['actions']:
        for seed in cfg['seeds']:
            ref = next(v for v in ap[3]['outcome_archives'] if v['path'].endswith(f'{action}_seed{seed}.npz'))
            assert file_digest(ROOT/ref['path']) == ref['sha256']
            with np.load(ROOT/ref['path'], allow_pickle=False) as z:
                o = {k: z[k].copy() for k in z.files}
            group = groups[seed, action]
            oldref = next(v for v in old['outcome_archives'] if v['path'].endswith(f'{action}_seed{seed}.npz'))
            with np.load(ROOT/oldref['path'], allow_pickle=False) as z:
                previous = z['choices'].copy()
            for j, oldj in enumerate((1, 2, 3, 5)):
                # Verify indices against the frozen public control schema too.
                from src.world_model.m3w_easy_allocation import ARMS
                name = ('net_stop', 'strict_stop', 'pointwise', 'aggregate_population')[j]
                np.testing.assert_array_equal(group['bits'][:, j], previous[:, ARMS.index(name)])
            for col, pol in enumerate(arms):
                e = np.where(group['bits'][:, col], o['candidate_ade'], o['cv'])
                f = np.where(group['bits'][:, col], o['candidate_fde'], o['cf'])
                result = a['summary'][action+'__'+pol]['seeds'][str(seed)]
                for subset in ('all', 'hard', 'positive_easy', 'complete', 'zero_CV'):
                    mask = np.ones(n, bool) if subset == 'all' else o[subset]
                    sm = result['ADE'] if subset == 'all' else result['subsets'][subset]
                    for site in cfg['sites']:
                        use = mask & (data['sites'] == site) & np.isfinite(o['cv'])
                        row = sm['by_scene'][site]
                        assert row['rows'] == int(use.sum())
                        if use.any():
                            mm, rr = float(e[use].mean()), float(o['cv'][use].mean())
                            np.testing.assert_allclose([row['model_error'], row['reference_error']], [mm, rr], rtol=1e-12)
                            gain = 100*(1-mm/rr) if rr > 0 else None
                            if gain is not None:
                                np.testing.assert_allclose(row['gain_percent'], gain, atol=1e-10)
                                mean_scene.setdefault((action, pol, subset, site), []).append(gain)
                        records.append(dict(action=action, policy=pol, seed=seed, site=site, subset=subset,
                            rows=row['rows'], gain_percent=row['gain_percent'], absolute_harm=row['absolute_harm']))
                        reduced += 1
                for site in cfg['sites']:
                    use = (data['sites'] == site) & np.isfinite(o['cf'])
                    np.testing.assert_allclose(result['FDE']['by_scene'][site]['model_error'], f[use].mean(), rtol=1e-12)
                    key = f'{site}_seed{seed}'; cut = fitted[key, action]['identity']['easy_cut']
                    easy = o['complete'] & (o['cv'] <= cut)
                    use = (data['sites']==site) & o['complete'] & group['bits'][:, col]
                    hp = np.where(easy, np.maximum(o['candidate_ade']-o['cv'], 0), 0)
                    bn = np.where(easy, np.maximum(o['cv']-o['candidate_ade'], 0), 0)
                    quality.append(dict(view=key, action=action, policy=pol, complete_selected=int(use.sum()),
                        predicted_positive=float(group['positive'][use].sum()), observed_positive=float(hp[use].sum()),
                        predicted_net=float(group['net'][use].sum()), observed_net=float((hp-bn)[use].sum()),
                        predicted_denominator=float(group['denominator'][use].sum()),
                        observed_denominator=float(np.where(easy, o['cv'], 0)[use].sum())))
    for (action, pol, subset, site), gains in mean_scene.items():
        result = a['summary'][action+'__'+pol]
        sm = result['ADE'] if subset == 'all' else result['subsets'][subset]
        np.testing.assert_allclose(sm['by_scene'][site]['gain_percent'], np.mean(gains), atol=1e-10)
    contrasts = 0
    for action, collection in a['contrasts'].items():
        for name, subsets in collection.items():
            left, right = name.split('_minus_')
            for subset, value in subsets.items():
                delta = np.array([np.mean(mean_scene[action, left, subset, s])-np.mean(mean_scene[action, right, subset, s]) for s in cfg['sites']])
                draws = np.random.default_rng(38113).choice(delta, size=(3000, 4)).mean(1)
                np.testing.assert_allclose(value['mean_gain_difference_pp'], delta.mean(), atol=1e-10)
                np.testing.assert_allclose(value['ci95_pp'], np.quantile(draws, [.025, .975]), atol=1e-10)
                contrasts += 1
    assert_current(identity)
    receipt = dict(all_checks_passed=True, analysis_sha256=file_digest(path),
        source_bindings={'scripts/verify_m3w_net_easy_moment.py': file_digest(Path(__file__))},
        query_constraint_checks=numeric, small_query_optima=optima,
        fixed_small_checks_skipped_large=skipped, scene_reductions=reduced, paired_contrasts=contrasts,
        explicitly_nonoptimal_solver_fallbacks=solver_fallbacks, exact_count_failures=count_failures,
        same_agent_separate_arithmetic=True, independent_research_confirmation=False)
    immutable_json(public/'independent_arithmetic.json', receipt)
    for name, rows in [('site_seed_results.csv', records), ('moment_reliability.csv', quality)]:
        with (public/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    compact = []; losses = []
    for name, result in a['summary'].items():
        action, pol = name.split('__')
        gains = [row['gain_percent'] for sd in result['seeds'].values() for row in sd['subsets']['positive_easy']['by_scene'].values() if row['gain_percent'] is not None]
        compact.append(dict(action=action, policy=pol, ADE_gain=result['ADE']['equal_scene_gain_percent'],
            CI=result['ADE']['scene_bootstrap_ci95'], FDE_gain=result['FDE']['equal_scene_gain_percent'],
            hard_gain=result['subsets']['hard']['equal_scene_gain_percent'],
            worst_site_seed_easy_degradation=max(0., -min(gains)),
            worst_site_ADE_gain=result['ADE']['worst_scene_gain_percent'],
            switch_rate=np.mean([sd['selected']/n for sd in result['seeds'].values()]),
            zero_CV_harmed_query_seed_instances=sum(sd['zero_CV_harmed'] for sd in result['seeds'].values()),
            incomplete_selected_query_seed_instances=sum(sd['selected_incomplete'] for sd in result['seeds'].values())))
    for r in a['fits']:
        losses.append(dict(view=r['view'], action=r['action'], seconds=r['fit']['seconds'],
            sampled_rows=r['fit']['sampled_rows'], effective_fit_rows=r['fit']['effective_fit_rows'],
            final_mse=r['fit']['trace'][-1]['fitting_mse_by_target']))
    immutable_json(public/'compact_results.json', dict(rows=compact, losses=losses, analysis_sha256=file_digest(path)))
    with (public/'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(compact[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(compact)
    lines = ['# Signed Easy-Risk Results', '',
        'Fresh 36-head training; frozen source-excluded forecasts and outcomes cached-verified. Four exposed sites, three seeds.',
        'Obs8/pred12 annotation pixels, equal-site gains vs CV. NOT t+50, confirmation or safety certification.',
        'Worst easy degradation is the maximum over individual site/seed views, not a pooled mean.', '',
        '| Predictor / policy | ADE gain % | Site CI95 | Hard gain % | Worst easy degradation % | Switch % | Zero-CV harmed |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in compact:
        lines.append(f"| {r['action']} / {r['policy']} | {r['ADE_gain']:.6f} | {r['CI']} | {r['hard_gain']:.6f} | {r['worst_site_seed_easy_degradation']:.6f} | {100*r['switch_rate']:.4f} | {r['zero_CV_harmed_query_seed_instances']} |")
    lines += ['', 'All counts, missing-label support, tails and partial-label bounds remain in analysis.json.',
        'Signed risk permits cancellation. Predicted-budget feasibility is not observed safety; all arms remain diagnostic.',
        'No external readout, new forecaster, threshold selection, deployment, Stage5C or SMC.']
    (public/'results.md').write_text('\n'.join(lines)+'\n')
    lines = ['# Fitting Losses', '', 'Targets: positive easy harm, easy benefit, easy denominator, easy probability.',
        'Weighted training MSE is not held-site performance. Seconds are fit-loop time, not whole-pipeline runtime.', '',
        '| View / predictor | Fit seconds | Draws | Effective rows | Final target MSE |', '|---|---:|---:|---:|---|']
    for r in losses:
        lines.append(f"| {r['view']} / {r['action']} | {r['seconds']:.3f} | {r['sampled_rows']} | {r['effective_fit_rows']} | {r['final_mse']} |")
    (public/'training_losses.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
