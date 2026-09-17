"""Descriptive sampling diagnostics after fixed fits; no candidate selection."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    directory = ROOT/'outputs/publication_readiness_2026_09/track_event_sampling'
    path = directory/'report.json'
    report = json.loads(path.read_text())
    reg_path = ROOT/'configs/m3w_track_event_sampling.json'
    if report['identity']['registration_sha256'] != file_digest(reg_path):
        raise ValueError('Changed experiment registration')
    reg = json.loads(reg_path.read_text())
    for name, digest in reg['bindings'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Bound dependency changed')
    if not report['complete'] or len(report['trials']) != 72:
        raise ValueError('Full fixed comparison required')
    replay = json.loads((directory/'replay.json').read_text())
    if replay['identity'] != report['identity'] or len(replay['trials']) != 72 or not all(
            t['prediction_exact'] for t in replay['trials']):
        raise ValueError('All control/fresh checkpoints must replay exactly')
    for t in report['trials']:
        if (file_digest(ROOT/t['checkpoint_path']) != t['checkpoint_sha256']
                or file_digest(ROOT/t['prediction_path']) != t['prediction_sha256']):
            raise ValueError('Completed artifact changed')
    keys = sorted(report['summary'])
    diagnostics = {}
    for key in keys:
        chosen = [t for t in report['trials'] if t['variant']+'_'+t['mode'] == key]
        strata = {}
        names = sorted({name for t in chosen for name in t['slices']})
        for name in names:
            values = [t['slices'][name] for t in chosen if t['slices'].get(name, {}).get('rows', 0)]
            if not values:
                continue
            ade = np.mean([v['primary_ADE'] for v in values])
            ref = np.mean([v['reference_ADE'] for v in values])
            recording_specific = name.startswith(('eth_', 'ucy_'))
            native = float(np.mean([v['native_ADE_diagnostic'] for v in values])) if recording_specific else None
            strata[name] = dict(rows_not_independent=sum(v['rows'] for v in values)//3,
                recording_local_tracks=sum(v['tracks'] for v in values)//3,
                supported_scene_seed_cells=len(values), primary_ADE=float(ade), reference_ADE=float(ref),
                gain_percent=float(100*(1-ade/ref)) if ref > 1e-12 else None,
                absolute_harm=float(ade-ref), native_ADE_diagnostic=native,
                native_units_pooled_across_recordings=False)
        exposure = {str(f):next(t['training_distribution'] for t in chosen if t['fold'] == f) for f in range(3)}
        diagnostics[key] = dict(strata=strata, first_seed_exposure_by_held_fold=exposure,
            fold_seed_easy_absolute_harm=[t['vs_CV']['easy_absolute_harm'] for t in chosen],
            fold_seed_tail_p95=[t['vs_CV']['tail_ADE_p95'] for t in chosen],
            fold_seed_train_gain=[t['training_equal_scene_gain_percent'] for t in chosen],
            fold_seed_strongest_gain=[t['vs_train_selected_strongest']['improvement_percent'] for t in chosen])
    result = dict(result_source='fresh_analysis_of_hash_verified_fixed_fit_predictions',
        report_sha256=file_digest(path), replay_sha256=file_digest(directory/'replay.json'),
        analysis_code_sha256=file_digest(Path(__file__)), conditions=diagnostics,
        independent_confirmation=False, conditions_selected_for_deployment=False,
        no_new_optimizer_updates=True, event_subsets_are_supervised_proxies_not_registered_primary=True)
    json_write(directory/'diagnostics.json', result)
    lines = ['# Sampling Diagnostics', '',
        'All conditions, not selection of the best held result. Future-derived event categories are descriptive evaluation labels, never inputs.', '',
        '| Feature/sampling | Static-to-move gain (%) | Moving/stops gain (%) | Moving/turns gain (%) | Easy harm range (normalized ADE) |',
        '| --- | ---: | ---: | ---: | --- |']
    def fmt(v):
        return 'undefined/no support' if v is None else f'{v:.4f}'
    for key, item in diagnostics.items():
        gains = [fmt(item['strata'].get(name, {}).get('gain_percent')) for name in ('static_moves', 'moving_stops', 'moving_turns')]
        harm = [v for v in item['fold_seed_easy_absolute_harm'] if v is not None]
        lines.append('| '+key+' | '+' | '.join(gains)+f' | {min(harm):.6f} to {max(harm):.6f} |')
    lines += ['', 'Category means use only physical scenes with support, equal scene and seed weight; they are not the primary all-scene endpoint.',
        'Track IDs can occur in multiple categories and folds; these counts are not independent people or event replications.', '']
    (directory/'diagnostics.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'svg.hashsalt':'m3w-track-event-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = ['Row', 'Scene', 'Scene/track', 'Scene/event/track']
    x = np.arange(4)
    for offset, variant, color in [(-.18, 'quality_control', '#3977a8'), (.18, 'directed', '#b96435')]:
        values = [report['summary'][variant+'_'+m]['vs_CV']['gain_percent'] for m in reg['modes']]
        bars = axes[0].bar(x+offset, values, .35, color=color, label=variant)
        axes[0].bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
        easy = [np.median(report['summary'][variant+'_'+m]['easy_degradation_percent']) for m in reg['modes']]
        axes[1].plot(x, easy, marker='o', color=color, label=variant)
    axes[0].axhline(0, color='black', lw=.7)
    axes[0].set_ylabel('Equal-scene primary gain vs CV (%)')
    axes[1].axhline(2, color='black', lw=.8, ls='--', label='2% easy limit')
    axes[1].set_ylabel('Median easy degradation across 9 fits (%)')
    for ax in axes:
        ax.set_xticks(x, labels, rotation=15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle('Track/event sampling: fixed model, loss and update budget')
    fig.text(.5, .01, '54 fresh fits + 18 verified controls; three historically used fit scenes, not independent confirmation.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(directory/'sampling_results.svg', metadata={'Date':None})
    fig.savefig(ROOT/reg['output']/'sampling_results.png', dpi=140)
    plt.close(fig)
    print(json.dumps(dict(conditions=len(diagnostics), fresh_training_seconds=sum(
        v['fresh_fit_seconds'] for v in report['summary'].values()), exact_replays=len(replay['trials']))))


if __name__ == '__main__':
    main()
