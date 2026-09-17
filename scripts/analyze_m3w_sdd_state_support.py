"""Distinguish sampled controls from controls inside the observed time span."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.audit_m3w_sdd_state_support import load_source
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_sdd_state_support import past_window_indices


def control_span_counts(control_frames, queries, stride):
    controls, queries = np.asarray(control_frames), np.asarray(queries)
    return (np.searchsorted(controls, queries, side='right')
            -np.searchsorted(controls, queries-7*stride, side='left'))


def main():
    reports = ROOT/'outputs/publication_readiness_2026_09/sdd_state_support'
    audit_path = reports/'audit.json'
    audit = json.loads(audit_path.read_text())
    if not audit['complete_inventory']:
        raise ValueError('Require full source-support audit')
    for name, digest in audit['identity']['code_sha256'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Changed source-support implementation')
    manifest = json.loads((ROOT/'outputs/publication_readiness_2026_09/sdd_media_alignment/diagnostic_media_links.json').read_text())
    total, pedestrians = {}, {}
    for entry in manifest['records']:
        if file_digest(ROOT/entry['annotations_path']) != entry['annotations_sha256']:
            raise ValueError('Changed source')
        rows, labels = load_source(ROOT/entry['annotations_path'])
        for indices in np.split(np.arange(len(rows)), np.flatnonzero(np.diff(rows[:, 0]))+1):
            track = rows[indices]
            controls = track[track[:, 8] == 0, 5].astype(np.int64)
            for stride in (1, 6, 12, 30):
                query, _ = past_window_indices(track, stride)
                counts = control_span_counts(controls, query, stride)
                values = dict(past_eligible=len(query), no_controls_in_history_span=int((counts == 0).sum()),
                    one_control_in_history_span=int((counts == 1).sum()),
                    at_least_two_controls_in_history_span=int((counts >= 2).sum()))
                total.setdefault(str(stride), Counter()).update(values)
                if labels[indices[0]] == 'Pedestrian':
                    pedestrians.setdefault(str(stride), Counter()).update(values)
        print(entry['annotation_key'], flush=True)
    for stride in total:
        if (total[stride]['past_eligible'] != audit['support_by_stride'][stride]['past_eligible']
                or pedestrians[stride]['past_eligible'] != audit['by_agent_type']['Pedestrian']['support'][stride]['past_eligible']):
            raise ValueError('Coverage census changed past eligibility')
    result = dict(result_source='fresh_run_source_control_span_census',
        audit_sha256=file_digest(audit_path), code_sha256=file_digest(Path(__file__)),
        by_stride=total, pedestrian_by_stride=pedestrians,
        controls_are_source_nongenerated_rows_not_human_gold=True,
        includes_unsampled_control_rows_inside_past_span=True,
        label_or_sampling_choice_changed=False, training_admitted=False)
    json_write(reports/'control_span_coverage.json', result)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'svg.hashsalt':'m3w-sdd-source-support-v1'})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    strides = ['1', '6', '12', '30']
    ped = audit['by_agent_type']['Pedestrian']['support']
    x = np.arange(4)
    for offset, key, name, color in (
        (-.25, 'exact_static_to_movement_windows', 'Overlapping windows', '#3977a8'),
        (0, 'exact_static_to_movement_tracks', 'Recording-local tracks', '#c06433'),
        (.25, 'exact_static_to_movement_disjoint_spans', 'Disjoint spans (not IID)', '#50975b')):
        bars = axes[0].bar(x+offset, [ped[s][key] for s in strides], .24, label=name, color=color)
        axes[0].bar_label(bars, padding=2, fontsize=8)
    axes[0].set(title='Pedestrian static-to-movement proxy', ylabel='Count',
                xticks=x, xticklabels=strides, ylim=(0, 440))
    axes[0].legend(frameon=False, fontsize=8, loc='upper right')
    a = [100*ped[s]['histories_with_at_least_two_controls']/ped[s]['past_eligible'] for s in strides]
    b = [100*pedestrians[s]['at_least_two_controls_in_history_span']/ped[s]['past_eligible'] for s in strides]
    axes[1].plot(x, a, marker='o', color='#3977a8', label='At least 2 sampled non-generated points')
    axes[1].plot(x, b, marker='s', color='#c06433', label='At least 2 source controls in history span')
    axes[1].set(title='Sampling controls is not the same as having controls',
                ylabel='Percent of eligible pedestrian histories', xticks=x, xticklabels=strides, ylim=(0, 105))
    axes[1].legend(frameon=False, fontsize=8, loc='upper left')
    for ax in axes:
        ax.set_xlabel('Sampling interval (raw annotation frames)')
        ax.spines[['top', 'right']].set_visible(False)
    figure.suptitle('SDD source-support census, not a model result', fontsize=13)
    figure.text(.5, .01, 'Intervals change the task span. Counts are not independent people, physical time, or a training-stride recommendation.',
                ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .05, 1, .94))
    figure.savefig(reports/'source_support.svg', metadata={'Date':None})
    private = ROOT/'data/stage_cvpr2027_experiments/sdd_state_support/source_support.png'
    figure.savefig(private, dpi=140)
    plt.close(figure)


if __name__ == '__main__':
    main()
