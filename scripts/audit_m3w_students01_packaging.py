"""Trace packaged UCY rows to continuous local annotations without model scores."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest


def inspect_packaging(continuous, packaged):
    lookup = defaultdict(set)
    for frame, agent, x, y in continuous:
        lookup[(int(frame), *np.round([x, y], 3))].add(int(agent))
    mappings = [lookup.get((int(f), x, y), set()) for f, _, x, y in packaged]
    if any(len(ids) != 1 for ids in mappings):
        raise ValueError('Every packaged row must uniquely match its original frame and rounded coordinate')
    recovered = np.array([next(iter(ids)) for ids in mappings])
    chunk_map, original_to_chunks = {}, defaultdict(list)
    for identifier in np.unique(packaged[:, 1]).astype(int):
        mask = packaged[:, 1] == identifier
        ids = np.unique(recovered[mask])
        if len(ids) != 1:
            raise ValueError('One packaged track mixes original physical identities')
        chunk_map[int(identifier)] = int(ids[0])
        original_to_chunks[int(ids[0])].append(int(identifier))
    _, lengths = np.unique(continuous[:, 1], return_counts=True)
    _, chunk_lengths = np.unique(packaged[:, 1], return_counts=True)
    if not np.all(chunk_lengths == 20):
        raise ValueError('Unexpected packaged window length')
    prefix_match = True
    for identifier in np.unique(continuous[:, 1]).astype(int):
        full = continuous[continuous[:, 1] == identifier]
        full = full[np.argsort(full[:, 0])]
        chunks = packaged[recovered == identifier]
        chunks = chunks[np.argsort(chunks[:, 0])]
        expected = full[:len(full) // 20 * 20]
        prefix_match &= len(chunks) == len(expected) and np.array_equal(chunks[:, 0], expected[:, 0])
        if len(chunks) == len(expected):
            prefix_match &= np.array_equal(chunks[:, 2:], np.round(expected[:, 2:], 3))
    support = {}
    for name, points in (('continuous', continuous), ('packaged', packaged)):
        by_frame, complete = defaultdict(int), 0
        for identifier in np.unique(points[:, 1]):
            frames = np.sort(points[points[:, 1] == identifier, 0])
            if len(frames) > 1 and not np.all(np.diff(frames) == 10):
                raise ValueError('Unexpected source clock discontinuity')
            for frame in frames[7:]:
                by_frame[int(frame)] += 1
            complete += max(0, len(frames) - 19)
        support[name] = {'past_supported_agent_queries': sum(by_frame.values()),
                         'complete_8to12_windows': int(complete), 'by_frame': dict(by_frame)}
    all_frames = set(support['continuous']['by_frame']) | set(support['packaged']['by_frame'])
    losses = [support['continuous']['by_frame'].get(f, 0) - support['packaged']['by_frame'].get(f, 0)
              for f in all_frames]
    for item in support.values():
        item.pop('by_frame')
    return {'continuous_rows': len(continuous), 'packaged_rows': len(packaged),
        'matched_rows_same_native_frame_and_rounded_xy': len(packaged),
        'continuous_physical_track_ids': len(lengths), 'packaged_track_ids': len(chunk_lengths),
        'original_tracks_split_into_multiple_ids': sum(len(v) > 1 for v in original_to_chunks.values()),
        'original_tracks_missing_from_packaged': len(lengths) - len(original_to_chunks),
        'removed_rows': len(continuous) - len(packaged),
        'discarded_length_remainders_sum': int((lengths % 20).sum()),
        'entire_tracks_shorter_than_20': int((lengths < 20).sum()),
        'exact_full_track_prefix_floor_length_over_20_reconstruction': bool(prefix_match),
        'support': support, 'frames_with_less_past_support': sum(x > 0 for x in losses),
        'max_lost_past_supported_agents_at_one_frame': int(max(losses)),
        'frame_clock_mismatch_detected': False,
        'upstream_future_availability_conditioning': bool(prefix_match),
        'future_endpoint_in_current_model_payload': False,
        'continuous_annotation_generation_causality': 'not_verified_by_this_packaging_audit'}


def main():
    paths = [ROOT / 'external_data/OpenTraj/datasets/UCY/students01' / name
             for name in ('students001.txt', 'students001-trajnet.txt')]
    result = inspect_packaging(*(np.loadtxt(p) for p in paths))
    result.update({'result_source': 'fresh_run_source_mapping_audit_no_model_scores',
        'source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in paths},
        'code_sha256': file_digest(Path(__file__)), 'data_role': 'development_exposed',
        'fit_recordings_affected': False, 'independent_confirmation': False,
        'metric_or_seconds_claim': False})
    output = ROOT / 'outputs/publication_readiness_2026_09/students01_packaging_audit'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Students01 Upstream Packaging Audit', '',
        'Fresh source-to-source row mapping; no neural scores or threshold search.', '',
        f"The continuous file has {result['continuous_rows']} rows / {result['continuous_physical_track_ids']} identities; "
        f"the packaged file has {result['packaged_rows']} rows / {result['packaged_track_ids']} identities.",
        'Every packaged position uniquely matches the same native frame and the continuous coordinate rounded to three decimals.',
        'There is no detected frame reset or coordinate mismatch. This is identity fragmentation and availability conditioning, not arbitrary fabricated clocks.', '',
        f"The package exactly retains floor(track_length/20)*20 points: {result['exact_full_track_prefix_floor_length_over_20_reconstruction']}.",
        f"It removes {result['removed_rows']} tail/short-track rows and all {result['entire_tracks_shorter_than_20']} tracks shorter than 20 points.",
        'A new ID starts each 20-point chunk. Determining whether to retain a chunk uses its later availability.',
        'Therefore a reader that uses only past rows cannot on its own prove causal eligibility of the upstream context population.',
        'This is distinct from passing a future endpoint to the neural network or leaking a train/test duplicate.', '',
        '| Source | Past-supported queries | Complete obs8/pred12 labels |', '| --- | ---: | ---: |']
    for name, s in result['support'].items():
        lines.append(f"| {name} | {s['past_supported_agent_queries']} | {s['complete_8to12_windows']} |")
    lines += ['', '## Consequence', '',
        'The v1-v4 Students01 results use this packaged observation universe and cannot certify full-scene causal deployment.',
        'Training recordings are unchanged by this finding. Preserve completed/partial fits and historical results, but do not continue toward a positive claim using the affected evaluation context.',
        'Repair in a new version: use the continuous local source with original IDs and precision, retain short/past-only agents for context, and read future availability only for scoring.',
        'Do not rewrite existing cache files, protocol hashes or checkpoint identities. Historical exposure remains development-only; the repair does not create an untouched test set.',
        'The continuous file is still an annotation product: this audit does not verify its upstream interpolation, physical calibration, seconds, or sensor-as-of causality.', '']
    (output / 'audit.md').write_text('\n'.join(lines))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
