"""Full training-curve verification and gain/harm decomposition, never held scoring."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_continuation import load_config, build_data
import numpy as np
import torch
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    out, public = ROOT/reg['output'], ROOT/reg['reports']
    rp = public/'report.json'
    report = json.loads(rp.read_text())
    assert report['completed_branches'] == 6 and report['additional_updates'] == 48000
    assert report['unique_parent_models'] == 3 and report['inherited_unique_updates'] == 6000
    replay = json.loads((public/'replay.json').read_text())
    assert replay['identity'] == report['identity'] and replay['all_exact'] and replay['new_updates'] == 0
    assert len(replay['exact_milestones']) == 24
    data, ids, weights, scale, parents = build_data(reg)
    loc = ids-data.nmain
    target = data.target[loc].astype(float)
    cv = np.linalg.norm(target, axis=-1).mean(1)
    zero, moving = cv == 0, cv > 0
    hard = cv >= np.quantile(cv, .9)
    rows, slices, paths, paired, batch_baselines = [], [], [out/'identity.json'], [], {}
    lookup = {(t['schedule'], t['seed']): t for t in report['trials']}
    for seed in reg['seeds']:
        final_states = []
        for schedule in reg['schedules']:
            trial = lookup[schedule, seed]
            cp_path = ROOT/trial['checkpoint_path']
            assert file_digest(cp_path) == trial['checkpoint_sha256']
            cp = torch.load(cp_path, map_location='cpu', weights_only=False)
            assert cp['identity'] == trial['identity'] and cp['config'] == reg['training']
            assert cp['step'] == 10000 and trial['fit']['additional_updates'] == 8000
            assert cp['draw_counts'].sum() == 640000
            assert 0 <= cp['clipped_updates'] <= 8000 and np.isfinite(cp['maximum_gradient_norm'])
            assert all(torch.isfinite(t).all() for t in cp['model'].values())
            paths.extend([cp_path, out/'trials'/f"{trial['trial']}.json"])
            final_states.append(cp)
            for milestone in trial['milestones']:
                paths.extend(ROOT/milestone[k+'_path'] for k in ('checkpoint', 'prediction', 'receipt'))
                for kind in ('checkpoint', 'prediction', 'receipt'):
                    assert file_digest(ROOT/milestone[kind+'_path']) == milestone[kind+'_sha256']
                with np.load(ROOT/milestone['prediction_path'], allow_pickle=False) as a:
                    np.testing.assert_array_equal(ids, a['ids'])
                    pred = a['prediction'].astype(float)
                assert np.isfinite(pred).all() and pred.shape == target.shape
                bound = data.radius[loc]*np.linalg.svd(data.rotation[loc], compute_uv=False)[:, 0]
                assert np.all(np.linalg.norm(pred, axis=-1) <= bound[:, None]*(1+1e-5)+1e-8)
                assert not np.any(pred[~data.support[loc]])
                ade = np.linalg.norm(pred-target, axis=-1).mean(1)
                harm = ade-cv
                np.testing.assert_allclose(ade.mean(), milestone['metrics']['ade'], rtol=1e-12)
                np.testing.assert_allclose(cv.mean(), scale, rtol=1e-12)
                contribution_zero = float(harm[zero].sum()/len(ids))
                contribution_moving = float(harm[moving].sum()/len(ids))
                np.testing.assert_allclose(contribution_zero+contribution_moving, harm.mean(), rtol=1e-9, atol=1e-9)
                row = dict(trial=trial['trial'], schedule=schedule, seed=seed, step=milestone['step'],
                    training_ade=float(ade.mean()), cv_ade=float(cv.mean()),
                    gain_percent=100*float(1-ade.mean()/cv.mean()),
                    zero_rows=int(zero.sum()), moving_rows=int(moving.sum()),
                    easy_absolute_harm=float(ade[zero].mean()),
                    easy_pixel_harm=float((ade[zero]*data.native_scale[loc][zero]).mean()),
                    moving_gain_percent=100*float(1-ade[moving].mean()/cv[moving].mean()),
                    hard_gain_percent=100*float(1-ade[hard].mean()/cv[hard].mean()),
                    zero_harm_contribution=contribution_zero, moving_harm_contribution=contribution_moving,
                    predicted_path_norm_mean=float(np.linalg.norm(pred, axis=-1).mean()),
                    regret_binary_oracle=float((ade-np.minimum(cv, ade)).mean()),
                    training_binary_oracle_gain_percent=100*float(1-np.minimum(cv, ade).mean()/cv.mean()),
                    training_rows_helped=int((ade<cv).sum()), training_rows_harmed=int((ade>cv).sum()),
                    tail_ade_95=float(np.quantile(ade, .95)), tail_ade_99=float(np.quantile(ade, .99)),
                    percentage_easy_degradation=None,
                    continuation_clip_fraction=cp['clipped_updates']/8000 if milestone['step'] == 10000 else None)
                rows.append(row)
                for site in sorted(set(data.source_sites[loc])):
                    mask = data.source_sites[loc] == site
                    slices.append(dict(trial=trial['trial'], step=milestone['step'], site=site, rows=int(mask.sum()),
                        ade=float(ade[mask].mean()), cv_ade=float(cv[mask].mean()),
                        gain_percent=100*float(1-ade[mask].mean()/cv[mask].mean())))
        a, b = final_states
        np.testing.assert_array_equal(a['draw_counts'], b['draw_counts'])
        assert torch.equal(a['sampler_rng'], b['sampler_rng']) and torch.equal(a['torch_rng'], b['torch_rng'])
        assert a['optimizer']['param_groups'][0]['lr'] == .0003
        np.testing.assert_allclose(b['optimizer']['param_groups'][0]['lr'], .000003, rtol=1e-12)
        generator = torch.Generator()
        generator.set_state(parents[seed][1]['sampler_rng'])
        draws = parents[seed][1]['draw_counts'].copy()
        for step in range(2001, 10001):
            chosen = torch.multinomial(torch.as_tensor(weights, dtype=torch.float64), 64,
                                       replacement=True, generator=generator).numpy()
            np.add.at(draws, chosen, 1)
            if step == 2001 or step % 100 == 0:
                batch_baselines[seed, step] = float(cv[chosen].mean()/scale)
        np.testing.assert_array_equal(draws, a['draw_counts'])
        assert torch.equal(generator.get_state(), a['sampler_rng'])
        # Every scheduled evaluation uses the same cumulative sample stream.
        for ma, mb in zip(lookup['constant', seed]['milestones'], lookup['cosine', seed]['milestones']):
            sa, sb = [torch.load(ROOT/m['checkpoint_path'], map_location='cpu', weights_only=False) for m in (ma, mb)]
            np.testing.assert_array_equal(sa['draw_counts'], sb['draw_counts'])
            assert torch.equal(sa['sampler_rng'], sb['sampler_rng'])
            if ma['step'] == 2000:
                for name in sa['model']:
                    assert torch.equal(sa['model'][name], sb['model'][name])
        paired.append(dict(seed=seed, same_parent=True, same_cumulative_draw_counts=True,
            same_rng=True, mean_exposures_per_row=float(a['draw_counts'].mean()),
            minimum_exposures=int(a['draw_counts'].min()), maximum_exposures=int(a['draw_counts'].max())))
    parent_paths = [ROOT/r['checkpoint_path'] for r, _ in parents.values()]
    paths = sorted(set(paths+parent_paths))
    hashes = {str(p.relative_to(ROOT)):file_digest(p) for p in paths}
    report_hash = file_digest(rp)
    proc = subprocess.run([sys.executable, 'scripts/run_m3w_source_continuation.py', '--registration', str(args.registration)],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    event = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{')][-1]
    assert event['state'] == 'completed_resume_verified' and event['new_updates'] == event['new_branches'] == 0
    assert hashes == {str(p.relative_to(ROOT)):file_digest(p) for p in paths} and file_digest(rp) == report_hash
    summaries = []
    for schedule in reg['schedules']:
        for step in reg['training']['milestones']:
            cells = [r for r in rows if r['schedule'] == schedule and r['step'] == step]
            gains = [r['gain_percent'] for r in cells]
            summaries.append(dict(schedule=schedule, step=step, mean_gain_percent=float(np.mean(gains)),
                gain_seed_range=[min(gains), max(gains)],
                mean_moving_gain_percent=float(np.mean([r['moving_gain_percent'] for r in cells])),
                mean_hard_gain_percent=float(np.mean([r['hard_gain_percent'] for r in cells])),
                mean_easy_pixel_harm=float(np.mean([r['easy_pixel_harm'] for r in cells])),
                mean_zero_harm_contribution=float(np.mean([r['zero_harm_contribution'] for r in cells])),
                mean_moving_harm_contribution=float(np.mean([r['moving_harm_contribution'] for r in cells])),
                mean_training_binary_oracle_gain_percent=float(np.mean([r['training_binary_oracle_gain_percent'] for r in cells])),
                positive_training_seeds=sum(g > 0 for g in gains)))
    zero_by_step = (np.linalg.norm(target, axis=-1) == 0).sum(0)
    minority_bound = (2*zero_by_step-len(ids))/len(ids)
    evidence = dict(result_source='fresh_run_training_curve_verification_not_generalization',
        report_sha256=report_hash, exact_replays=24, matched_schedule_pairs=paired,
        unchanged_artifacts_including_parents=len(paths), artifact_hashes=hashes,
        completed_resume=event, summaries=summaries,
        zero_target_rows=int(zero.sum()), zero_target_fraction=float(zero.mean()),
        zero_target_count_by_step=zero_by_step.tolist(),
        constant_prediction_harm_lower_bound_coefficients=minority_bound.tolist(),
        constant_zero_optimal_ignoring_inputs=bool(np.all(minority_bound>0)),
        constant_prediction_proof='triangle_inequality_sum_norm(y-a)-sum_norm(y)>= (2*n_zero-n)*norm(a) per waypoint',
        proof_not_conditional_predictor_impossibility=True,
        binary_oracle_uses_future_labels_training_only_not_a_policy=True,
        training_rows=len(ids), generalization_established=False, new_deployment=False,
        held_rows_scored=0, main_rows_scored=0, independent_confirmation=False)
    json_write(public/'analysis.json', evidence)
    write_csv(public/'training_metrics.csv', rows)
    write_csv(public/'training_site_metrics.csv', slices)
    traces = []
    for trial in report['trials']:
        for point in trial['fit']['trace']:
            base = batch_baselines[trial['seed'], point['step']]
            traces.append(dict(trial=trial['trial'], **point, normalized_same_batch_cv=base,
                               normalized_excess_over_same_batch_cv=point['normalized_batch_ade']-base))
    write_csv(public/'batch_trace.csv', traces)
    lines = ['# Full Source-Training Continuation Results', '', '## Material Passport', '',
        'Fresh continuation of three verified parent models into six paired schedule branches.',
        '48000 new updates; shared parents contain6000 unique inherited updates. No held-source/main scoring.',
        'All15430 training rows retained. Mean and range of three seeds, not generalization confidence intervals.', '',
        '| Schedule | Step | Mean gain (%) [seed range] | Moving gain (%) | Hard gain (%) | Easy absolute pixel harm |',
        '| --- | ---: | --- | ---: | ---: | ---: |']
    for s in summaries:
        lo, hi = s['gain_seed_range']
        lines.append(f"| {s['schedule']} | {s['step']} | {s['mean_gain_percent']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {s['mean_moving_gain_percent']:+.6f} | {s['mean_hard_gain_percent']:+.6f} | {s['mean_easy_pixel_harm']:.8f} |")
    lines += ['', '## Verification and Limits', '',
        f"All24 milestone predictions replay exactly. Three schedule pairs retain matched parent states and sampler streams. Completed resume preserves{len(paths)} artifacts including parent checkpoints, plus the report, with zero new updates.",
        'No learning-rate winner is deployed. Easy percentage is undefined because its stationary CV error is zero.',
        'The supplied annotations include generated/interpolated positions. No seconds, metric, true3D or foundation claim.',
        'A schedule-induced reduction in output jitter is not automatically prediction of future motion.', '',
        '## Constant-Prediction Sanity Bound', '',
        f"Exactly{zero.sum()} of{len(ids)} training targets are entirely zero ({zero.mean():.2%}). At each waypoint the zero fraction exceeds one half. By the triangle inequality, an input-independent offset a increases the summed empirical distance by at least (2*n_zero-n)*norm(a). The smallest per-row coefficient across waypoints is{minority_bound.min():.6f}.",
        'Thus zero is the optimal input-independent path in the stored parent-normalized coordinates on this training population. This is not a constant local decoder code subsequently rotated/rescaled by each observed frame. It does not bound conditional predictors, establish an architectural limitation or rule out useful observed information; it makes output collapse versus actual conditional gain an important distinction.', '']
    (public/'results.md').write_text('\n'.join(lines))
    cache = out/'plot_runtime'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache/'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(cache/'xdg'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = 'm3w-source-continuation-v1'
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fields = [('gain_percent', 'Full training gain vs CV (%)'),
              ('moving_gain_percent', 'Moving-target training gain (%)'),
              ('easy_pixel_harm', 'Zero-target harm (annotation pixels)')]
    for ax, (field, ylabel) in zip(axes, fields):
        for schedule, color in [('constant', '#357289'), ('cosine', '#a45351')]:
            steps = reg['training']['milestones']
            values = np.array([[next(r[field] for r in rows if r['schedule']==schedule and r['seed']==seed and r['step']==step)
                                for step in steps] for seed in reg['seeds']])
            ax.plot(steps, values.mean(0), marker='o', color=color, label=schedule)
            ax.fill_between(steps, values.min(0), values.max(0), color=color, alpha=.15)
        ax.axhline(0, color='black', linewidth=.8, linestyle=':')
        ax.set_xlabel('Global optimizer step')
        ax.set_ylabel(ylabel)
        ax.set_xticks([2000, 4000, 6000, 10000], ['2k', '4k', '6k', '10k'])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Fixed full-source training complement: mean and range of three parent seeds')
    fig.text(.5, .01, 'Training-only optimization diagnostic. No held-source evaluation, model selection or generalization claim.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(public/'training_curves.svg', metadata={'Date':None})
    fig.savefig(cache/'training_curves.png', dpi=150)
    plt.close(fig)
    svg = public/'training_curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps({k:v for k,v in evidence.items() if k!='artifact_hashes'}, indent=2))


if __name__ == '__main__':
    main()
