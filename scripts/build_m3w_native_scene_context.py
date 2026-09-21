"""Repair neighbor identities and retain every visible causal scene-context row."""
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_forecast import json_write, immutable_json, array_hash, assert_current
from scripts.run_m3w_native_nested import write_arrays
from scripts.audit_m3w_sdd_state_support import load_source
from src.data_unification.m3w_native_scene_context import SourceSceneContext
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/native_scene_context_v2'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/native_scene_context_v2'
AUDIT = ROOT/'outputs/publication_readiness_2026_09/native_scene_alignment_v1/analysis.json'


def load_past_queries(bindings):
    cfg = json.loads((ROOT/'configs/m3w_source_population_v1.json').read_text())
    mp = ROOT/cfg['input_manifest']
    if file_digest(mp) != cfg['input_manifest_sha256']:
        raise ValueError('Changed source population')
    manifest = json.loads(mp.read_text())
    data = {k:[] for k in ('recordings', 'tracks', 'frames', 'geometry')}
    for receipt in manifest['records']:
        rec = receipt['recording']
        if rec.split('/')[0] not in cfg['allowed_sites']:
            continue
        if receipt['original_split'] != 'train' or receipt['data_role'] != 'supervised_auxiliary_training':
            raise ValueError('Closed recording role')
        arrays = {}
        for name in ('geometry', 'query_keys'):
            path = mp.parent/rec/(name+'.npy'); sha = file_digest(path)
            if sha != receipt['arrays'][path.name]:
                raise ValueError('Changed past input')
            bindings[str(path.relative_to(ROOT))] = sha
            arrays[name] = np.load(path, allow_pickle=False)
        data['geometry'].append(arrays['geometry']); keys = arrays['query_keys']
        data['frames'].append(keys[:, 0]); data['recordings'].append(np.array([rec]*len(keys)))
        data['tracks'].append(np.array([f'{rec}:{a}' for a in keys[:, 1]]))
    return {k:np.concatenate(v) for k,v in data.items()}


def main():
    start = time.monotonic(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    def beat(**v):
        value = dict(pid=os.getpid(), updated_unix=time.time(), **v)
        json_write(PRIVATE/'heartbeat.json', value); print(json.dumps(value), flush=True)
    beat(state='verifying_source_scope')
    audit = json.loads(AUDIT.read_text()); bindings = dict(audit['source_bindings'])
    data = load_past_queries(bindings)
    if array_hash(data['recordings'], data['tracks'], data['frames']) != audit['cache']['population_sha256']:
        raise ValueError('Past-only population alignment changed')
    for path in (AUDIT, Path(__file__), ROOT/'src/data_unification/m3w_native_scene_context.py',
                 ROOT/'tests/test_m3w_native_scene_context.py', ROOT/'scripts/audit_m3w_sdd_state_support.py',
                 ROOT/'outputs/publication_readiness_2026_09/sdd_media_alignment/diagnostic_media_links.json'):
        bindings[str(path.relative_to(ROOT))] = file_digest(path)
    assert_current(dict(source_bindings=bindings))
    cp = ROOT/audit['cache']['path']
    if file_digest(cp) != audit['cache']['sha256']:
        raise ValueError('Changed input-only alignment index')
    bindings[str(cp.relative_to(ROOT))] = file_digest(cp)
    with np.load(cp, allow_pickle=False) as z:
        cache = {k:z[k].copy() for k in z.files}
    links = json.loads((ROOT/'outputs/publication_readiness_2026_09/sdd_media_alignment/diagnostic_media_links.json').read_text())
    entries = {v['annotation_key']:v for v in links['records']}
    records = []
    for rec in np.unique(data['recordings']):
        beat(state='source_identity_reconstruction', recording=str(rec))
        ids = np.flatnonzero(data['recordings'] == rec); entry = entries[str(rec)]
        source_path = ROOT/entry['annotations_path']; source_sha = file_digest(source_path)
        if source_sha != entry['annotations_sha256']:
            raise ValueError('Changed source annotation')
        bindings[entry['annotations_path']] = source_sha
        out = PRIVATE/str(rec); rp = out/'receipt.json'
        input_identity = dict(source_sha256=source_sha, audit_sha256=file_digest(AUDIT),
            code_sha256=file_digest(Path(__file__)), module_sha256=file_digest(ROOT/'src/data_unification/m3w_native_scene_context.py'),
            ids_sha256=array_hash(ids), recordings=str(rec))
        if rp.exists():
            receipt = json.loads(rp.read_text())
            if receipt['identity'] != input_identity or file_digest(ROOT/receipt['cache']['path']) != receipt['cache']['sha256']:
                raise ValueError('Changed completed context cache')
            records.append(receipt); beat(state='cached_verified', recording=str(rec)); continue
        rows, labels = load_source(source_path); source = SourceSceneContext(rows, labels)
        keys = np.column_stack((data['frames'][ids], [int(v.rsplit(':', 1)[1]) for v in data['tracks'][ids]]))
        value = source.build(keys, cache['origin'][ids], cache['rotation'][ids], cache['annotation_scale'][ids], data['geometry'][ids], ids)
        array_values = {k:v for k,v in value.items() if isinstance(v, np.ndarray)}
        array_values['query_ids'] = ids
        order = {int(f):np.flatnonzero(value['context_frame_ids'] == f) for f in np.unique(keys[:, 0])}
        checked, truncation_checks = 0, 0
        # Independent scalar ID lookup on fixed sampled queries; no outcome access.
        for i in np.unique(np.linspace(0, len(ids)-1, min(9, len(ids)), dtype=int)):
            f, agent = map(int, keys[i]); selected = (rows[:, 5] == f) & (rows[:, 6] == 0) & (rows[:, 0] != agent)
            v = rows[selected]; xy = (v[:, 1:3]+v[:, 3:5])/2
            nearest = sorted(range(len(v)), key=lambda j:(float(np.linalg.norm(xy[j]-cache['origin'][ids[i]])), int(v[j, 0])))[:8]
            np.testing.assert_array_equal(value['neighbor_agent_ids'][i, :len(nearest)], v[nearest, 0])
            assert np.all(value['neighbor_agent_ids'][i, len(nearest):] == -1)
            checked += 1
        for f in np.unique(keys[:, 0])[[0, len(order)//2, -1]]:
            # Truncating the raw source at the query cannot change input context.
            truncated = SourceSceneContext(rows[rows[:, 5] <= f], labels[rows[:, 5] <= f])
            loc = truncated.at_frame(int(f)); ci = order[int(f)]
            np.testing.assert_array_equal(value['context_agent_ids'][ci], truncated.agent[loc])
            for stored, actual in zip(('context_history', 'context_history_offsets', 'context_history_mask'), truncated.history(loc)):
                np.testing.assert_array_equal(value[stored][ci], actual)
            truncation_checks += 1
        full_history = value['context_history_mask'].all(1)
        contiguous = full_history & np.all(np.diff(value['context_history_offsets'], axis=1) == 12, axis=1)
        pedestrian = value['context_agent_type'] == 'Pedestrian'
        eligible = contiguous & pedestrian
        has_prediction = value['context_target_rows'] >= 0
        np.testing.assert_array_equal(eligible, has_prediction)
        np.testing.assert_array_equal(np.sort(value['context_target_rows'][has_prediction]), ids)
        old = cache['target_neighbor_rows'][ids]; new = value['neighbor_target_rows']; observed = value['neighbor_observed']
        wrong = (old >= 0) & (old != new)
        ambiguous = old == -2
        true_target = new >= 0
        path = out/'context.npz'; write_arrays(path, array_values)
        receipt = dict(identity=input_identity, recording=str(rec), target_rows=len(ids),
            scene_queries=len(order), context_rows=len(has_prediction),
            context_without_prediction=int((~has_prediction).sum()),
            non_pedestrian_context=int((~pedestrian).sum()),
            pedestrian_insufficient_history=int((pedestrian & ~contiguous).sum()),
            context_CV_unsupported_rows=int((~value['context_cv_valid'].all(1)).sum()),
            context_type_counts={k:int((value['context_agent_type'] == k).sum()) for k in sorted(set(value['context_agent_type']))},
            neighbor_slots=int(observed.sum()), true_target_neighbor_slots=int(true_target.sum()),
            false_positive_positional_identity=int(wrong.sum()),
            ambiguous_positional_slots=int(ambiguous.sum()),
            ambiguous_resolved_to_target=int((ambiguous & true_target).sum()),
            ambiguous_resolved_to_nontarget=int((ambiguous & ~true_target).sum()),
            previously_unresolved_now_target=int((observed & (old == -1) & true_target).sum()),
            compared_neighbor_history_points=value['compared_neighbor_history_points'],
            max_normalized_neighbor_error=value['max_normalized_neighbor_error'],
            irregular_neighbor_histories=value['irregular_neighbor_histories'],
            scalar_identity_checks=checked, raw_future_truncation_checks=truncation_checks,
            no_future_target_access=True,
            cache=dict(path=str(path.relative_to(ROOT)), sha256=file_digest(path), bytes=path.stat().st_size,
                arrays_sha256={k:array_hash(v) for k,v in array_values.items()}))
        immutable_json(rp, receipt); records.append(receipt)
        beat(state='context_record_verified', recording=str(rec), context_rows=len(has_prediction),
            false_positional_links=int(wrong.sum()), resolved_ambiguous=int(ambiguous.sum()))
    sums = ('target_rows','scene_queries','context_rows','context_without_prediction','non_pedestrian_context',
        'pedestrian_insufficient_history','context_CV_unsupported_rows','neighbor_slots','true_target_neighbor_slots',
        'false_positive_positional_identity','ambiguous_positional_slots','ambiguous_resolved_to_target',
        'ambiguous_resolved_to_nontarget','previously_unresolved_now_target','compared_neighbor_history_points',
        'irregular_neighbor_histories','scalar_identity_checks','raw_future_truncation_checks')
    result = dict(result_source='fresh_run_raw_annotation_identity_repair_cached_verified_frozen_cohort',
        source_bindings=bindings, records=records, totals={k:sum(r[k] for r in records) for k in sums},
        cache_bytes=sum(r['cache']['bytes'] for r in records),
        normalized_geometry_QA_tolerance=dict(atol=1e-7, rtol=2e-7, changes_safety_metric=False),
        max_normalized_neighbor_error=max(r['max_normalized_neighbor_error'] for r in records),
        actual_full_scene_prediction=False, full_visible_past_context_built=True,
        new_training=False, new_forecasts=False, new_selection=False, new_readout=False,
        future_target_arrays_loaded=False, future_labels_used_for_context_or_identity=False,
        original_val_test_opened=False,
        implementation_revision='v2 uses an explicit geometry/key-only loader; v1 inherited unused target arrays and its target-read metadata was inaccurate',
        independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(dict(source_bindings=bindings)); immutable_json(PUBLIC/'analysis.json', result)
    beat(state='complete', seconds=time.monotonic()-start, analysis_sha256=file_digest(PUBLIC/'analysis.json'))


if __name__ == '__main__':
    main()
