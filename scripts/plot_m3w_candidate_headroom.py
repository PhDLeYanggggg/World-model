"""Aggregate-only figures and primary-error decomposition for oracle diagnosis."""
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
    public = ROOT/'outputs/publication_readiness_2026_09/candidate_headroom'
    private = ROOT/'data/stage_cvpr2027_experiments/candidate_headroom'
    report = json.loads((public/'report.json').read_text())
    verification = json.loads((public/'independent_verification.json').read_text())
    if verification['report_sha256'] != file_digest(public/'report.json'):
        raise ValueError('Changed verified diagnostic')
    source = ROOT/'data/stage_cvpr2027_experiments/offline_visual_forecast/inputs'
    if file_digest(source/'data_manifest.json') != report['identity']['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    manifest = json.loads((source/'data_manifest.json').read_text())
    arrays = {}
    for name in ('baselines', 'targets', 'geometry', 'folds'):
        if file_digest(source/(name+'.npy')) != manifest['arrays'][name+'.npy']:
            raise ValueError('Changed source array')
        arrays[name] = np.load(source/(name+'.npy'), mmap_mode='r')
    cv = manifest['baseline_names'].index('constant_velocity_causal_fd')
    ref = np.linalg.norm(arrays['baselines'][:, cv].astype(float)-arrays['targets'], axis=-1).mean(1)
    static = np.all(arrays['geometry'][:, :16] == 0, axis=1)
    denominator = np.mean([ref[arrays['folds'] == f].mean() for f in range(3)])
    contribution = np.mean([(ref*static)[arrays['folds'] == f].mean() for f in range(3)])
    result = dict(result_source='fresh_run_label_error_decomposition_cached_verified_inputs',
        report_sha256=file_digest(public/'report.json'), code_sha256=file_digest(Path(__file__)),
        all_rows=len(ref), static_rows=int(static.sum()), equal_scene_CV_ADE=float(denominator),
        static_share_of_equal_scene_CV_error=float(contribution/denominator),
        past_scale_and_primary_unchanged=True, independent_support_added=False,
        model_trained=False, raw_units_pooled=False)
    json_write(public/'error_decomposition.json', result)
    cache = private/'plot_runtime'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache)
    os.environ['XDG_CACHE_HOME'] = str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'svg.hashsalt':'m3w-action-ceiling-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    labels = ['Row', 'Scene', 'Track', 'Event/track', 'Pool of 8']
    modes = ['row_uniform', 'scene_uniform', 'scene_track', 'scene_event_track']
    for offset, variant, color in [(-.18, 'quality_control', '#3977a8'), (.18, 'directed', '#b96435')]:
        names = [variant+'_'+m for m in modes]
        values = [report['conditions'][n]['feasible_loss']['gain_percent'] for n in names]
        axes[0].bar(np.arange(4)+offset, values, .35, color=color, label=variant)
    pooled = report['conditions']['pooled_eight_candidates_per_seed']['feasible_loss']['gain_percent']
    bar = axes[0].bar([4], [pooled], .6, color='#5b8164', label='Pooled oracle')
    axes[0].bar_label(bar, fmt='%.3f', padding=3)
    axes[0].set_xticks(np.arange(5), labels, rotation=15)
    axes[0].set_ylabel('Future-label scaling oracle gain vs CV (%)')
    axes[0].set_ylim(0, pooled*1.25)
    curves = report['budget_curves']['pooled_eight_candidates_per_seed']
    for seed, curve, color in zip((17, 29, 43), curves, ('#3977a8', '#b96435', '#5b8164')):
        axes[1].plot([100*p['fraction'] for p in curve], [p['feasible_gain_percent'] for p in curve],
                     marker='o', color=color, label=f'Seed {seed}')
    axes[1].set_xlabel('Relaxed maximum intervention fraction (%)')
    axes[1].set_ylabel('Future-label pooled oracle gain vs CV (%)')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle('Action-class ceiling, not a learned forecasting result')
    fig.text(.5, .01, 'Frozen forecasts; one scalar per whole path. Future labels choose actions. No causal policy or safety guarantee.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(public/'oracle_ceiling.svg', metadata={'Date':None})
    fig.savefig(private/'oracle_ceiling.png', dpi=140)
    plt.close(fig)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
