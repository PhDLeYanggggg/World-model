"""Independent reductions and identity links over the source-only scene cache."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_forecast import assert_current, array_hash, immutable_json
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np


def main():
    public = ROOT/'outputs/publication_readiness_2026_09/native_scene_context_v2'
    a = json.loads((public/'analysis.json').read_text())
    audit = json.loads((ROOT/'outputs/publication_readiness_2026_09/native_scene_alignment_v1/analysis.json').read_text())
    assert_current(dict(source_bindings=a['source_bindings']))
    mp = ROOT/'data/stage_cvpr2027_experiments/sdd_auxiliary_v1/inputs'
    with np.load(ROOT/audit['cache']['path'], allow_pickle=False) as z:
        transforms = {k:z[k] for k in ('origin', 'rotation', 'annotation_scale')}
    total = dict(target_rows=0, context_rows=0, neighbor_slots=0, true_target_neighbor_slots=0,
                 context_CV_unsupported_rows=0, context_without_prediction=0)
    mask_checks = 0; max_xy_error = 0.; max_neighbor_normalized_error = 0.
    for r in a['records']:
        path = ROOT/r['cache']['path']
        assert file_digest(path) == r['cache']['sha256']
        with np.load(path, allow_pickle=False) as z:
            c = {k:z[k] for k in z.files}
        for k, v in c.items(): assert array_hash(v) == r['cache']['arrays_sha256'][k]
        rec = r['recording']; keys = np.load(mp/rec/'query_keys.npy', allow_pickle=False)
        g = np.load(mp/rec/'geometry.npy', allow_pickle=False)
        targets = c['context_target_rows']; has = targets >= 0
        total['target_rows'] += len(keys); total['context_rows'] += len(targets)
        total['context_without_prediction'] += int((~has).sum())
        assert len(set(zip(c['context_frame_ids'], c['context_agent_ids']))) == len(targets)
        context_lookup = {tuple(k):i for i,k in enumerate(zip(c['context_frame_ids'], c['context_agent_ids']))}
        target_lookup = {tuple(k):int(i) for k,i in zip(keys, c['query_ids'])}
        lookup_target = np.array([target_lookup.get(k, -1) for k in zip(c['context_frame_ids'], c['context_agent_ids'])])
        np.testing.assert_array_equal(targets, lookup_target)
        h, times, mask = c['context_history'], c['context_history_offsets'], c['context_history_mask']
        assert mask[:, -1].all() and np.all(times[mask] <= 0)
        np.testing.assert_array_equal(h[:, -1], c['context_xy'])
        assert not h[~mask].any() and not times[~mask].any()
        eligible = mask.all(1) & np.all(np.diff(times, axis=1) == 12, axis=1) & (c['context_agent_type'] == 'Pedestrian')
        np.testing.assert_array_equal(eligible, has)
        valid_velocity = mask[:, -2:].all(1)
        np.testing.assert_array_equal(c['context_cv_valid'], np.repeat(valid_velocity[:, None], 12, axis=1))
        total['context_CV_unsupported_rows'] += int((~valid_velocity).sum())
        velocity = (h[valid_velocity, -1]-h[valid_velocity, -2]) / (-times[valid_velocity, -2, None])
        expected = h[valid_velocity, -1, None] + velocity[:, None]*(np.arange(1, 13)*12)[None, :, None]
        error = np.abs(expected-c['context_cv_rollout'][valid_velocity])
        max_xy_error = max(max_xy_error, float(error.max()) if len(error) else 0.)
        np.testing.assert_allclose(expected, c['context_cv_rollout'][valid_velocity], rtol=1e-12, atol=1e-10)
        # Resolve every cached neighbor slot through (recording, frame, source ID),
        # independent of the builder's nearest-neighbor search and target table.
        observed = c['neighbor_observed']; owners, slots = np.where(observed)
        neighbor_context = np.array([context_lookup[(int(keys[i, 0]), int(c['neighbor_agent_ids'][i, j]))] for i,j in zip(owners, slots)])
        np.testing.assert_array_equal(c['neighbor_target_rows'][observed], targets[neighbor_context])
        assert not np.any(c['neighbor_agent_ids'][observed] == keys[owners, 1])
        global_owners = c['query_ids'][owners]
        delta = h[neighbor_context]-transforms['origin'][global_owners, None]
        rot = transforms['rotation'][global_owners]
        # Explicit scalar coordinate formula rather than the builder's einsum.
        x = (delta[..., 0]*rot[:, 0, 0, None]+delta[..., 1]*rot[:, 1, 0, None]) / transforms['annotation_scale'][global_owners, None]
        y = (delta[..., 0]*rot[:, 0, 1, None]+delta[..., 1]*rot[:, 1, 1, None]) / transforms['annotation_scale'][global_owners, None]
        xy = np.where(mask[neighbor_context, :, None], np.stack((x, y), -1), 0.)
        stored = g[:, 38:166].reshape(-1, 8, 8, 2)[observed]
        np.testing.assert_allclose(xy.astype(np.float32), stored, rtol=2e-7, atol=1e-7)
        max_neighbor_normalized_error = max(max_neighbor_normalized_error, float(np.max(np.abs(xy-stored))) if len(xy) else 0.)
        np.testing.assert_array_equal(g[:, 230:294].reshape(-1, 8, 8)[observed], mask[neighbor_context])
        np.testing.assert_array_equal(g[:, 166:230].reshape(-1, 8, 8)[observed], (times[neighbor_context]/144).astype(np.float32))
        mask_checks += int(mask[neighbor_context].sum())
        total['neighbor_slots'] += len(owners); total['true_target_neighbor_slots'] += int((c['neighbor_target_rows'] >= 0).sum())
    for k, v in total.items(): assert a['totals'][k] == v
    assert mask_checks == a['totals']['compared_neighbor_history_points']
    assert not a['future_target_arrays_loaded'] and not a['future_labels_used_for_context_or_identity']
    receipt = dict(analysis_sha256=file_digest(public/'analysis.json'), all_checks_passed=True,
        checked_totals=total, compared_neighbor_history_points=mask_checks,
        context_cv_max_absolute_error=max_xy_error, max_normalized_neighbor_error=max_neighbor_normalized_error,
        cv_arithmetic_QA_tolerance=dict(atol=1e-10, rtol=1e-12, safety_rule_unchanged=True),
        public_source_recordings=len(a['records']), future_labels_loaded=False,
        verifier_sha256=file_digest(Path(__file__)), new_training=False, independent_implementation_same_agent=True,
        scope='cache identity, all current neighbor links, causal history masks/times, explicit forecast support and native CV arithmetic; not independent research confirmation')
    immutable_json(public/'verification.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
