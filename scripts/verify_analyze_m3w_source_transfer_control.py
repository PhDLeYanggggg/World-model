"""Verify matched fits and report every fixed exposed-source contrast, without selection."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_transfer_control import load_config, build_data
import numpy as np
import torch
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_recording_diagnostic import (
    recording_resamples, paired_gain_interval, error_summary,
)
from src.world_model.m3w_offline_visual_data import json_write


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    private, public = ROOT/reg['output'], ROOT/reg['reports']
    controls = read_json(public/'training_report.json')
    evaluation = read_json(public/'evaluation.json')
    replay = read_json(public/'replay.json')
    identity = controls['identity']
    assert evaluation['identity'] == replay['identity'] == identity
    assert controls['additional_updates'] == 48000 and len(controls['trials']) == 6
    assert replay['all_exact'] and len(replay['exact']) == len(set(replay['exact'])) == 42
    assert replay['new_updates'] == 0
    data, train, weights, held, scale, parents, rgb_report = build_data(reg)
    tr, loc = train-data.nmain, held-data.nmain
    target, native = data.target[loc].astype(float), data.native_scale[loc]
    cv = np.linalg.norm(target, axis=-1).mean(1)
    train_cv = np.linalg.norm(data.target[tr].astype(float), axis=-1).mean(1)
    hard_cut = float(np.quantile(train_cv, .9))
    assert hard_cut == evaluation['training_hard_cut']
    hard, zero, moving = cv >= hard_cut, cv == 0, cv > 0
    labels, inv, counts = recording_resamples(data.source_records[loc],
        reg['bootstrap_resamples'], reg['bootstrap_seed'])
    assert len(labels) == 7 and len(set(data.source_tracks[loc])) == 181
    assert not set(data.source_records[tr]) & set(data.source_records[loc])
    paths = {private/'identity.json', public/'training_report.json', public/'evaluation.json',
             public/'input_checks.json', public/'frozen_predictors.json', public/'replay.json'}
    old_analysis = read_json((ROOT/reg['rgb_report']).with_name('analysis.json'))
    for name, digest in old_analysis['artifact_hashes'].items():
        path = ROOT/name
        assert file_digest(path) == digest
        paths.add(path)
    paths.update(ROOT/reg[key] for key in ('rgb_report', 'parent_report', 'frozen_probability_report'))
    paths.update(ROOT/r['checkpoint_path'] for r, _ in parents.values())
    training_rows, pairs, final_states = [], [], {}
    lookup = {(arm, t['schedule'], t['seed']):t for arm, report in
              [('mask_only', controls), ('past_rgb', rgb_report)] for t in report['trials']}
    for (arm, schedule, seed), trial in lookup.items():
        checkpoint = ROOT/trial['checkpoint_path']
        assert file_digest(checkpoint) == trial['checkpoint_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['identity'] == trial['identity'] and state['arm'] == arm and state['step'] == 10000
        assert state['config'] == reg['training'] and state['normalizer'] == scale
        np.testing.assert_array_equal(state['train_ids'], train)
        assert state['draw_counts'].sum() == 640000
        assert 0 <= state['clipped_updates'] <= 8000 and np.isfinite(state['maximum_gradient_norm'])
        assert all(torch.isfinite(value).all() for value in state['model'].values())
        final_states[arm, schedule, seed] = state
        paths.add(checkpoint)
        if arm == 'mask_only':
            paths.add(private/'trials'/f"{trial['trial']}.json")
        for milestone in trial['milestones']:
            for kind in ('checkpoint', 'prediction', 'receipt'):
                path = ROOT/milestone[kind+'_path']
                assert file_digest(path) == milestone[kind+'_sha256']
                paths.add(path)
            with np.load(ROOT/milestone['prediction_path'], allow_pickle=False) as a:
                np.testing.assert_array_equal(a['ids'], train)
                prediction = a['prediction'].astype(float)
            assert prediction.shape == data.target[tr].shape and np.isfinite(prediction).all()
            bound = data.radius[tr]*np.linalg.svd(data.rotation[tr], compute_uv=False)[:, 0]
            assert np.all(np.linalg.norm(prediction, axis=-1) <= bound[:, None]*(1+1e-5)+1e-8)
            assert not np.any(prediction[~data.support[tr]])
            ade = np.linalg.norm(prediction-data.target[tr], axis=-1).mean(1)
            np.testing.assert_allclose(ade.mean(), milestone['metrics']['ade'], rtol=1e-12)
            training_rows.append(dict(arm=arm, schedule=schedule, seed=seed, step=milestone['step'],
                ade=float(ade.mean()), gain_percent=float(100*(1-ade.mean()/train_cv.mean())),
                zero_native_harm=float((ade[train_cv==0]*data.native_scale[tr][train_cv==0]).mean()),
                clip_fraction=state['clipped_updates']/8000 if milestone['step'] == 10000 else None))
    for seed in reg['seeds']:
        reference = final_states['past_rgb', 'constant', seed]
        for arm in ('mask_only', 'past_rgb'):
            for schedule in reg['schedules']:
                cp = final_states[arm, schedule, seed]
                np.testing.assert_array_equal(cp['draw_counts'], reference['draw_counts'])
                assert torch.equal(cp['sampler_rng'], reference['sampler_rng'])
                assert torch.equal(cp['torch_rng'], reference['torch_rng'])
                for milestone in lookup[arm, schedule, seed]['milestones']:
                    other = next(m for m in lookup['past_rgb', 'constant', seed]['milestones']
                                 if m['step'] == milestone['step'])
                    left, right = [torch.load(ROOT/m['checkpoint_path'], map_location='cpu', weights_only=False)
                                   for m in (milestone, other)]
                    np.testing.assert_array_equal(left['draw_counts'], right['draw_counts'])
                    assert torch.equal(left['sampler_rng'], right['sampler_rng'])
        generator = torch.Generator()
        ancestor = parents['past_rgb', seed][1]
        generator.set_state(ancestor['sampler_rng'])
        draws = ancestor['draw_counts'].copy()
        for _ in range(8000):
            batch = torch.multinomial(torch.as_tensor(weights, dtype=torch.float64), 64,
                                      replacement=True, generator=generator).numpy()
            np.add.at(draws, batch, 1)
        np.testing.assert_array_equal(draws, reference['draw_counts'])
        assert torch.equal(generator.get_state(), reference['sampler_rng'])
        pairs.append(dict(seed=seed, four_way_stream_exact=True, all_milestones_matched=True,
            mean_training_draws=float(draws.mean()), min_training_draws=int(draws.min()), max_training_draws=int(draws.max())))
    per_seed, arrays, summaries, group_rows = [], {}, [], []
    for result in evaluation['results']:
        path = ROOT/result['prediction_path']
        assert file_digest(path) == result['prediction_sha256']
        paths.add(path)
        with np.load(path, allow_pickle=False) as a:
            np.testing.assert_array_equal(a['ids'], held)
            prediction, probability = a['prediction'].astype(float), a['probability'].astype(float)
        assert np.isfinite(prediction).all() and prediction.shape == target.shape
        assert np.isfinite(probability).all() and np.all((probability>=0)&(probability<=1))
        bound = data.radius[loc]*np.linalg.svd(data.rotation[loc], compute_uv=False)[:, 0]
        assert np.all(np.linalg.norm(prediction, axis=-1) <= bound[:, None]*(1+1e-5)+1e-8)
        assert not np.any(prediction[~data.support[loc]])
        for mode in ('uncontrolled', 'fixed_probability_guard'):
            use = np.ones(len(held), bool) if mode == 'uncontrolled' else probability >= reg['fixed_probability_threshold']
            emitted = np.where(use[:, None, None], prediction, 0.)
            ade = np.linalg.norm(emitted-target, axis=-1).mean(1)
            fde = np.linalg.norm(emitted[:, -1]-target[:, -1], axis=-1)
            changed = np.any(emitted != 0, axis=(1, 2)).astype(float)
            metrics = error_summary(ade, fde, cv, native, changed, hard)
            np.testing.assert_allclose(metrics['ade'], result['metrics'][mode]['ade'], rtol=1e-12)
            np.testing.assert_allclose(changed.mean(), result['metrics'][mode]['actual_changed_rate'], rtol=1e-12)
            key = result['arm'], result['schedule'], mode, result['seed']
            arrays[key] = dict(ade=ade, fde=fde, changed=changed, use=use.astype(float))
            per_seed.append(dict(arm=key[0], schedule=key[1], mode=mode, seed=key[3],
                                 requested_rate=float(use.mean()), **metrics))
    means = {}
    for arm in ('mask_only', 'past_rgb'):
        for schedule in ('parent_2k', 'constant', 'cosine'):
            for mode in ('uncontrolled', 'fixed_probability_guard'):
                cells = [arrays[arm, schedule, mode, seed] for seed in reg['seeds']]
                values = {field:np.mean([c[field] for c in cells], axis=0) for field in cells[0]}
                means[arm, schedule, mode] = values
                summary = error_summary(values['ade'], values['fde'], cv, native, values['changed'], hard)
                seed_gains = [100*(1-c['ade'].mean()/cv.mean()) for c in cells]
                summary.update(arm=arm, schedule=schedule, mode=mode, requested_rate=float(values['use'].mean()),
                    gain_seed_range=[float(min(seed_gains)), float(max(seed_gains))],
                    gain_interval=paired_gain_interval(cv, values['ade'], cv, inv, counts),
                    hard_gain_interval=paired_gain_interval(cv, values['ade'], cv, inv, counts, hard))
                group_gains = []
                for kind, labels_array in [('recording', data.source_records[loc]), ('scoped_agent', data.source_tracks[loc])]:
                    grouped_ade, grouped_cv = [], []
                    for label in sorted(set(labels_array)):
                        mask = labels_array == label
                        cell = error_summary(values['ade'][mask], values['fde'][mask], cv[mask], native[mask],
                                             values['changed'][mask], hard[mask])
                        group_rows.append(dict(arm=arm, schedule=schedule, mode=mode, group_type=kind,
                            group=label, rows=int(mask.sum()), ade=cell['ade'], cv_ade=cell['cv_ade'],
                            gain_percent=cell['gain_percent'], harm_over_cv=cell['harm_over_cv'],
                            easy_pixel_harm=cell['easy_pixel_harm'], actual_changed_rate=cell['actual_changed_rate']))
                        grouped_ade.append(cell['ade'])
                        grouped_cv.append(cell['cv_ade'])
                        if kind == 'recording':
                            group_gains.append(cell['gain_percent'])
                    summary[f'equal_{kind}_gain_percent'] = float(100*(1-np.mean(grouped_ade)/np.mean(grouped_cv)))
                summary['worst_recording_gain_percent'] = min(x for x in group_gains if x is not None)
                summaries.append(summary)
    contrasts = []
    for mode in ('uncontrolled', 'fixed_probability_guard'):
        fixed = [(f'rgb_minus_mask:{s}', ('mask_only', s, mode), ('past_rgb', s, mode))
                 for s in ('parent_2k', 'constant', 'cosine')]
        fixed += [(f'endpoint_minus_parent:{a}:{s}', (a, 'parent_2k', mode), (a, s, mode))
                  for a in ('mask_only', 'past_rgb') for s in ('constant', 'cosine')]
        for name, reference, candidate in fixed:
            for subset, mask in [('all', np.ones(len(cv), bool)), ('hard', hard), ('moving', moving)]:
                contrasts.append(dict(contrast=name, mode=mode, subset=subset,
                    **paired_gain_interval(means[reference]['ade'], means[candidate]['ade'], cv, inv, counts, mask)))
    before = {str(p.relative_to(ROOT)):file_digest(p) for p in sorted(paths)}
    command = [sys.executable, 'scripts/run_m3w_source_transfer_control.py', '--registration', str(args.registration)]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    event = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{')][-1]
    assert event['state'] == 'completed_resume_verified' and event['new_updates'] == event['new_branches'] == 0
    subprocess.run(command+['--evaluate'], cwd=ROOT, capture_output=True, text=True, check=True)
    assert before == {str(p.relative_to(ROOT)):file_digest(p) for p in sorted(paths)}
    verification = dict(exact_replays=42, four_way_matches=pairs, unchanged_artifacts=len(paths),
        artifact_hashes=before, completed_resume=event, repeated_evaluation_unchanged=True,
        finite_bounded_predictions=True, unsupported_predictions_exactly_zero=True)
    evidence = dict(result_source='fresh_run_mask_training_and_fixed_inference_cached_verified_rgb',
        identity=identity, new_control_branches=6, new_updates=48000, reused_rgb_branches=6,
        reused_rgb_new_updates_from_prior_run=48000, unique_parent_states=6,
        inherited_unique_parent_updates=12000, summaries=summaries, contrasts=contrasts,
        verification=verification, training_rows=len(train), held_rows=len(held),
        source_recordings=len(labels), source_scoped_agents=len(set(data.source_tracks[loc])), physical_sites=1,
        training_hard_cut=hard_cut, easy_rows=int(zero.sum()), moving_rows=int(moving.sum()), hard_rows=int(hard.sum()),
        bootstrap_resamples=reg['bootstrap_resamples'], bootstrap_seed=reg['bootstrap_seed'],
        bootstrap_scope='conditional_on_seven_recordings_of_one_previously_explored_source_site',
        no_per_seed_selection=True, seed_aggregation='mean_per_row_error_not_averaged_prediction',
        independent_confirmation=False, main_primary_changed=False, sealed_roles_opened=False,
        new_deployment=False, stage5c_executed=False, smc_enabled=False)
    json_write(public/'analysis.json', evidence)
    write_csv(public/'training_metrics.csv', training_rows)
    write_csv(public/'held_seed_metrics.csv', per_seed)
    write_csv(public/'held_group_metrics.csv', group_rows)
    write_csv(public/'paired_contrasts.csv', contrasts)
    lines = ['# Matched Modality and Exposed-Source Results', '',
        'Fresh mask-only continuation:6 branches,48000 new updates. RGB branches and all parents are cached_verified.',
        'Fresh inference on6944 previously explored bookstore rows,7recordings,181 scopedagents,1physicalsite.',
        'All18 fixed predictor states retained. No winner selection, main-role scoring or deployment.', '',
        '| Arm | Endpoint | Policy | Gain vs CV (%) | Conditional recording95% CI | Seed range | Hard gain (%) | Zero-target harm (pixels) | Actual change rate |',
        '| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: |']
    for s in summaries:
        lo, hi = s['gain_interval']['conditional_recording_ci95']
        low, high = s['gain_seed_range']
        lines.append(f"| {s['arm']} | {s['schedule']} | {s['mode']} | {s['gain_percent']:+.6f} | [{lo:+.6f},{hi:+.6f}] | [{low:+.6f},{high:+.6f}] | {s['hard_gain_percent']:+.6f} | {s['easy_pixel_harm']:.8f} | {s['actual_changed_rate']:.3%} |")
    lines += ['', '## Paired Contrasts', '',
        'Gain difference uses the same stationary CV denominator, in percentage points. Positive favors the named candidate.',
        'Guarded RGB contrasts change both candidate and frozen classifier probabilities, not the image modality alone.', '',
        '| Contrast | Policy | Gain difference (pp) | Conditional recording95% CI |',
        '| --- | --- | ---: | --- |']
    for c in contrasts:
        if c['subset'] != 'all':
            continue
        lo, hi = c['conditional_recording_ci95']
        lines.append(f"| {c['contrast']} | {c['mode']} | {c['point_percent']:+.6f} | [{lo:+.6f},{hi:+.6f}] |")
    lines += ['', '## Evidence Limits', '',
        'These2000 paired recording-block draws preserve window weighting and first average seed errors, not predictions. Recordings share one physical site; this is not a physical-scene generalization interval. Main confirmation is not opened.',
        'Easy CV error is zero, so its percentage degradation is undefined. Absolute harm must not be converted into an invented passing percentage. Binary future oracles are diagnostics, never deployable policies.',
        'Eight supplied past annotation points to twelve stride12source steps (+144rawframes); no seconds or cross-dataset time equivalence. Source normalization and native annotation pixels are not metric coordinates. Offline/interpolated labels are not human intention gold.',
        f"Verification:42 exact replays,3four-way matched sampler streams,all finite bounded forecasts,unsupported contexts exactly zero,{len(paths)} artifacts unchanged on completed resume and repeated evaluation,zero new updates.", '']
    (public/'results.md').write_text('\n'.join(lines))
    cache = private/'plot_runtime'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache/'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(cache/'xdg'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = 'm3w-source-transfer-control-v1'
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for arm, color, offset in [('mask_only', '#357289', -.08), ('past_rgb', '#a45351', .08)]:
        cells = [next(s for s in summaries if s['arm']==arm and s['schedule']==schedule and s['mode']=='uncontrolled')
                 for schedule in ('parent_2k', 'constant', 'cosine')]
        x = np.arange(3)+offset
        y = np.array([s['gain_percent'] for s in cells])
        low, high = np.array([s['gain_interval']['conditional_recording_ci95'] for s in cells]).T
        axes[0].plot(x, y, 'o', color=color, label=arm)
        axes[0].vlines(x, low, high, color=color)
        axes[1].plot(x, [s['easy_pixel_harm'] for s in cells], 'o-', color=color, label=arm)
        for s in cells:
            step, schedule = (2000, 'constant') if s['schedule']=='parent_2k' else (10000, s['schedule'])
            values = [r['gain_percent'] for r in training_rows if r['arm']==arm and r['schedule']==schedule and r['step']==step]
            axes[2].scatter(np.mean(values), s['gain_percent'], color=color)
            axes[2].annotate(s['schedule'].replace('parent_2k', '2k'), (np.mean(values), s['gain_percent']),
                             xytext=(4, 3), textcoords='offset points', fontsize=7)
    for ax in axes[:2]:
        ax.set_xticks(range(3), ['Parent2k', 'Constant10k', 'Cosine10k'])
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Held-source gain vs CV (%)')
    axes[1].set_ylabel('Zero-target harm (annotation pixels)')
    axes[2].set_xlabel('Training gain vs CV (%)')
    axes[2].set_ylabel('Held-source gain vs CV (%)')
    axes[2].axvline(0, color='black', linewidth=.7, linestyle=':')
    for ax in axes:
        ax.axhline(0, color='black', linewidth=.7, linestyle=':')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.2)
    fig.suptitle('Matched modality continuation: previously explored bookstore source site')
    fig.text(.5, .015, 'Uncontrolled forecasts. Mean seed errors; conditional recording-block CI, not independent scene confirmation.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .96))
    fig.savefig(public/'matched_results.svg', metadata={'Date':None})
    fig.savefig(cache/'matched_results.png', dpi=150)
    plt.close(fig)
    svg = public/'matched_results.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps(dict(new_updates=48000, exact_replays=42, unchanged_artifacts=len(paths),
        summaries=[{k:v for k,v in s.items() if k in ('arm','schedule','mode','gain_percent','easy_pixel_harm','hard_gain_percent','gain_interval')}
                   for s in summaries]), indent=2))


if __name__ == '__main__':
    main()
