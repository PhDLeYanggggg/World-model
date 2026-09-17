"""Aggregate fixed residual-range comparison without selecting a deployable model."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    folder = ROOT/'outputs/publication_readiness_2026_09/residual_range'
    report = json.loads((folder/'report.json').read_text())
    replay = json.loads((folder/'replay.json').read_text())
    if not report['complete'] or replay['identity'] != report['identity'] or len(replay['trials']) != 72:
        raise ValueError('Complete replay required')
    if not all(r['prediction_exact'] for r in replay['trials']):
        raise ValueError('Replay failed')
    source = ROOT/'data/stage_cvpr2027_experiments/offline_visual_forecast/inputs'
    if file_digest(source/'data_manifest.json') != report['identity']['source_identity']['source_manifest_sha256']:
        raise ValueError('Source changed')
    manifest = json.loads((source/'data_manifest.json').read_text())
    a = {}
    for n in ('targets', 'baselines', 'geometry', 'scale', 'folds'):
        if file_digest(source/(n+'.npy')) != manifest['arrays'][n+'.npy']:
            raise ValueError('Source array changed')
        a[n] = np.load(source/(n+'.npy'), mmap_mode='r')
    metadata = json.loads((source/'rows.json').read_text())
    records = np.array([m['recording'] for m in metadata])
    cv = manifest['baseline_names'].index('constant_velocity_causal_fd')
    baseline = a['baselines'][:, cv].astype(float)
    reference = np.linalg.norm(baseline-a['targets'], axis=-1).mean(1)
    static = np.all(a['geometry'][:, :16] == 0, axis=1)
    moves = static & np.any(a['targets'] != 0, axis=(1, 2))
    stays = static & ~moves
    details, conditions = {}, {}
    for trial in report['trials']:
        pp, cp = ROOT/trial['prediction_path'], ROOT/trial['checkpoint_path']
        if file_digest(pp) != trial['prediction_sha256'] or file_digest(cp) != trial['checkpoint_sha256']:
            raise ValueError('Fixed artifact changed')
        with np.load(pp) as saved:
            ids, prediction = saved['held_indices'], saved['prediction'].astype(float)
        if not np.array_equal(ids, np.flatnonzero(a['folds'] == trial['fold'])):
            raise ValueError('Fold alignment changed')
        error = np.linalg.norm(prediction-a['targets'][ids], axis=-1).mean(1)
        slices = {}
        for name, mask in [('static_moves', moves[ids]), ('static_stays', stays[ids]), ('moving_history', ~static[ids])]:
            if not mask.any():
                slices[name] = dict(rows=0); continue
            e, b = error[mask].mean(), reference[ids[mask]].mean()
            slices[name] = dict(rows=int(mask.sum()), ADE=float(e), CV_ADE=float(b),
                                gain_percent=float(100*(1-e/b)) if b > 1e-12 else None)
        native = {}
        for rec in np.unique(records[ids]):
            mask = records[ids] == rec
            b = (reference[ids[mask]]*a['scale'][ids[mask]]).mean()
            e = (error[mask]*a['scale'][ids[mask]]).mean()
            native[rec] = dict(rows=int(mask.sum()), native_ADE=float(e), native_CV_ADE=float(b),
                               gain_percent=float(100*(1-e/b)) if b > 1e-12 else None)
        details[trial['trial']] = dict(slices=slices, per_recording_native_diagnostic=native)
    for key, values in report['summary'].items():
        chosen = [t for t in report['trials'] if t['variant']+'_'+t['arm'] == key]
        easy = [v for v in values['easy_degradation_percent'] if v is not None]
        conditions[key] = dict(values,
            easy_degradation_range_percent=[min(easy), max(easy)],
            undefined_easy_ratios=len(chosen)-len(easy),
            train_gain_mean_percent=float(np.mean(values['training_gain_percent'])),
            train_static_gain_percent=[100*(1-t['training_static_ADE']/t['training_static_CV_ADE'])
                                      for t in chosen if t['training_static_CV_ADE'] and t['training_static_CV_ADE'] > 0],
            train_saturated_coordinates=sum(t['fit'].get('saturated_coordinates', 0) for t in chosen),
            final_minibatch_training_loss=[t['fit']['losses'][-1]['loss'] for t in chosen])
    result = dict(result_source='fresh_run_analysis_cached_verified_forecasts',
        report_sha256=file_digest(folder/'report.json'), replay_sha256=file_digest(folder/'replay.json'),
        analysis_code_sha256=file_digest(Path(__file__)), conditions=conditions, trial_diagnostics=details,
        fixed_fresh_fits=54, cached_verified_controls=18, exact_replays=72,
        fresh_fit_seconds=sum(v['fresh_fit_seconds'] for v in conditions.values()),
        inference_inputs_changed=False, primary_metric_changed=False, new_source_admitted=False,
        sealed_roles_opened=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    json_write(folder/'analysis.json', result)
    private = ROOT/'data/stage_cvpr2027_experiments/residual_range'
    cache = private/'plot_runtime'; cache.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache); os.environ['XDG_CACHE_HOME'] = str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'svg.hashsalt':'m3w-residual-range-v1'})
    arms = ['linear_log','sinh_log','linear_asinh','sinh_asinh']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for shift, v, color in [(-.18,'quality_control','#3977a8'),(.18,'directed','#b96435')]:
        axes[0].bar(np.arange(4)+shift, [conditions[v+'_'+arm]['train_gain_mean_percent'] for arm in arms],
                    .35, label=v, color=color)
        axes[1].bar(np.arange(4)+shift, [conditions[v+'_'+arm]['vs_CV']['gain_percent'] for arm in arms],
                    .35, label=v, color=color)
    for ax, title in zip(axes, ('Fit cohorts: mean diagnostic gain', 'Held fit scenes: primary gain')):
        ax.set_title(title); ax.set_xticks(np.arange(4), arms, rotation=20)
        ax.axhline(0, color='#333333', linewidth=.7)
        ax.set_ylabel('Gain vs causal CV (%)'); ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle('Output range and target transform: matched 2x2 control')
    fig.text(.5,.01,'Same inputs, seeds, row sampler and 4,000 updates. Exposed fit scenes only; no independent confirmation.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94))
    fig.savefig(folder/'range_comparison.svg', metadata={'Date':None})
    fig.savefig(private/'range_comparison.png', dpi=140)
    plt.close(fig)
    print(json.dumps({k:{'gain':v['vs_CV']['gain_percent'], 'train_gain':v['train_gain_mean_percent'],
                          'easy_range':v['easy_degradation_range_percent'], 'safe_fits':v['safe_positive_fits']}
                      for k,v in conditions.items()}, indent=2))


if __name__ == '__main__':
    main()
