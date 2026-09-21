"""Build a verified past-only scene index and audit the frozen control's support.

No training, threshold selection, new policy, or closed-role readout. Row-level
geometry remains private; public artifacts contain counts, hashes and diagnostics.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_geometric_risk import load
from scripts.run_m3w_native_forecast import array_hash, immutable_json, json_write, assert_current
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import native_errors
from src.evaluation.m3w_native_scene_alignment import (
    past_transforms, restore, scene_index, resolve_target_neighbors, support_counts,
)
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/native_scene_alignment_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/native_scene_alignment_v1'


def quantiles(x):
    return np.quantile(x, [0, .25, .5, .75, .95, 1]).tolist() if len(x) else None


def main():
    started = time.monotonic()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    def beat(**v):
        value = dict(pid=os.getpid(), time_unix=time.time(), **v)
        json_write(PRIVATE/'heartbeat.json', value); print(json.dumps(value), flush=True)
    beat(state='verifying_source_bindings')
    reg, parent, data, views, gain, previous, identity = load()
    prior_path = ROOT/reg['reports']/'analysis.json'
    prior = json.loads(prior_path.read_text())
    replay = json.loads((ROOT/reg['reports']/'verification.json').read_text())
    if (prior['identity'] != identity or not replay['all_checks_passed']
            or replay['analysis_sha256'] != file_digest(prior_path)):
        raise ValueError('Verified frozen factorial/control evidence required')
    bindings = dict(identity['source_bindings'])
    for path in (prior_path, ROOT/reg['reports']/'verification.json', Path(__file__),
                 ROOT/'src/evaluation/m3w_native_scene_alignment.py', ROOT/'tests/test_m3w_native_scene_alignment.py'):
        bindings[str(path.relative_to(ROOT))] = file_digest(path)
    cfg = json.loads((ROOT/'configs/m3w_source_population_v1.json').read_text())
    mp = ROOT/cfg['input_manifest']; manifest = json.loads(mp.read_text())
    n = len(data['sites']); origin = np.empty((n, 2)); rotation = np.empty((n, 2, 2))
    scales = np.empty(n); current_boxes = np.empty((n, 4))
    alignment, seen = [], np.zeros(n, bool)
    for receipt in manifest['records']:
        rec = receipt['recording']
        if rec.split('/')[0] not in cfg['allowed_sites']:
            continue
        if receipt['original_split'] != 'train' or receipt['data_role'] != 'supervised_auxiliary_training':
            raise ValueError('Closed source role')
        ids = np.flatnonzero(data['recordings'] == rec)
        values = {}
        for name in ('query_keys', 'image_rows', 'crop_keys', 'annotation_boxes'):
            path = mp.parent/rec/(name+'.npy')
            digest = file_digest(path)
            if digest != receipt['arrays'][path.name]:
                raise ValueError('Changed past provenance: '+str(path))
            bindings[str(path.relative_to(ROOT))] = digest
            values[name] = np.load(path, allow_pickle=False)
        np.testing.assert_array_equal(data['frames'][ids], values['query_keys'][:, 0])
        np.testing.assert_array_equal(data['tracks'][ids], [f'{rec}:{a}' for a in values['query_keys'][:, 1]])
        tr = past_transforms(values['query_keys'], values['image_rows'], values['crop_keys'],
            values['annotation_boxes'], data['geometry'][ids], data['scale'][ids])
        origin[ids], rotation[ids], scales[ids], current_boxes[ids] = tr['origin'], tr['rotation'], tr['annotation_scale'], tr['current_box']
        assert not seen[ids].any(); seen[ids] = True
        alignment.append(dict(recording=rec, rows=len(ids), **{k:v for k,v in tr.items() if k.startswith('max_')}))
        beat(state='past_reconstruction_verified', recording=rec, rows=len(ids))
    assert seen.all() and len(alignment) == cfg['expected_recordings']
    index = scene_index(data['recordings'], data['frames'], data['tracks'])
    neighbors = resolve_target_neighbors(data['geometry'], origin, rotation, scales, index)
    counts = np.diff(index['offsets']); groups = index['group']; first = index['order'][index['offsets'][:-1]]
    visible = data['geometry'][:, 300]+1
    np.testing.assert_array_equal(visible, np.rint(visible))
    np.testing.assert_array_equal(visible, visible[first][groups])
    assert np.all(counts <= visible[first])
    native_history = restore(data['geometry'][:, :16].reshape(n, 8, 2), origin, rotation, scales)
    # Check linked neighbor histories where their raw timestamps coincide.
    g = data['geometry']; nb = g[:, 38:166].reshape(n, 8, 8, 2)
    mask = g[:, 230:294].reshape(n, 8, 8).astype(bool)
    times = g[:, 166:230].reshape(n, 8, 8)*144
    links = neighbors['target_row']; checked = 0; past_max = 0.
    reconstructed = restore(nb, origin, rotation, scales)
    for slot in range(8):
        rows = np.flatnonzero(links[:, slot] >= 0); targets = links[rows, slot]
        for t in range(8):
            at = np.rint(times[rows, slot, t]/12).astype(int)+7
            use = mask[rows, slot, t] & (at >= 0) & (at < 8)
            valid_rows, target_rows, at = rows[use], targets[use], at[use]
            if len(valid_rows):
                error = np.linalg.norm(reconstructed[valid_rows, slot, t]-native_history[target_rows, at], axis=-1)
                # Nearby but distinct agents can share a current location. Do not
                # turn a positional link into a certified identity in that case.
                past_max = max(past_max, float(error.max())); checked += len(error)
    mapping_usable = past_max <= 1e-3
    cache = dict(origin=origin, rotation=rotation, annotation_scale=scales,
        stored_metric_scale=data['scale'], current_boxes=current_boxes,
        group=groups, group_order=index['order'], group_offsets=index['offsets'],
        group_recordings=index['recordings'], group_frames=index['frames'],
        target_neighbor_rows=links, neighbor_observed=neighbors['observed'])
    path = PRIVATE/'past_scene_index.npz'; write_arrays(path, cache)
    cache_receipt = dict(path=str(path.relative_to(ROOT)), sha256=file_digest(path), bytes=path.stat().st_size,
        population_sha256=array_hash(data['recordings'], data['tracks'], data['frames']),
        arrays_sha256={k:array_hash(v) for k,v in cache.items()},
        target_neighbor_links_identity_verified=mapping_usable)
    immutable_json(PRIVATE/'past_scene_index.json', cache_receipt)
    coverage = []
    for site in parent['sites']:
        use = data['sites'] == site; gi = np.flatnonzero(np.isin(index['recordings'], np.unique(data['recordings'][use])))
        ls = links[use]; obs = neighbors['observed'][use]
        coverage.append(dict(site=site, query_rows=int(use.sum()), scene_queries=len(gi),
            groups_with_two_or_more_targets=int((counts[gi] >= 2).sum()),
            groups_covering_all_current_visible=int((counts[gi] == visible[first][gi]).sum()),
            target_count_quantiles=quantiles(counts[gi]), visible_count_quantiles=quantiles(visible[first][gi]),
            observed_neighbor_slots=int(obs.sum()), linked_target_slots=int((ls >= 0).sum()),
            ambiguous_slots=int((ls == -2).sum()), non_target_or_unresolved_slots=int((obs & (ls == -1)).sum())))
    beat(state='scene_index_built', scene_queries=len(counts), rows=n, neighbor_links_identity_verified=mapping_usable)
    baseline = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, _ = native_errors(baseline, data['target'], data['valid'], data['scale'])
    full = data['valid'].all(1); stop = np.all(data['geometry'][:, :16].reshape(n, 8, 2)[:, -1] == data['geometry'][:, :16].reshape(n, 8, 2)[:, -2], axis=1)
    all_selected = {seed:np.zeros(n, bool) for seed in parent['seeds']}
    conditional, per_recording, selected_geometry_checks = [], [], []
    for key, meta in views.items():
        site, seed = meta['outer_site'], meta['seed']
        pp = meta['outer_prediction']['prediction']
        if file_digest(ROOT/pp['path']) != pp['sha256']:
            raise ValueError('Changed frozen neural candidate')
        with np.load(ROOT/pp['path'], allow_pickle=False) as z:
            ids, pred = z['ids'].copy(), z['prediction'].copy()
        score = next(r for r in gain['score_archives'] if r['view'] == key)
        if file_digest(ROOT/score['path']) != score['sha256']:
            raise ValueError('Changed frozen gain scores')
        with np.load(ROOT/score['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); costs = z['mse'].copy()
        same = np.all(pred == baseline[ids], axis=(1, 2))
        use = (costs[:, 0] > costs[:, 1]) & (costs[:, 1] <= .1*costs[:, 0]) & ~same & ~stop[ids]
        cap = next(r for r in prior['capacities'] if r['view'] == key)
        assert array_hash(ids[use]) == cap['selected_ids_sha256']['stop_mse_strict']
        all_selected[seed][ids] = use
        err, _ = native_errors(pred, data['target'][ids], data['valid'][ids], data['scale'][ids])
        restored_pred = restore(pred, origin[ids], rotation[ids], data['scale'][ids])
        restored_target = restore(data['target'][ids], origin[ids], rotation[ids], data['scale'][ids])
        native_distance = np.linalg.norm(restored_pred-restored_target, axis=-1)
        count = data['valid'][ids].sum(1)
        independently_restored = np.divide(np.where(data['valid'][ids], native_distance, 0).sum(1), count,
            out=np.full(len(ids), np.nan), where=count > 0)
        np.testing.assert_allclose(err, independently_restored, rtol=1e-10, atol=1e-10, equal_nan=True)
        selected_geometry_checks.append(dict(view=key, restored_cost_max_error=float(np.nanmax(np.abs(err-independently_restored)))))
        for subset, sel in dict(all=np.ones(len(ids), bool), no_neighbor=g[ids, 300] == 0,
                has_neighbor=g[ids, 300] > 0, multiple_target=counts[groups[ids]] >= 2,
                only_target=counts[groups[ids]] == 1,
                all_visible_targets=counts[groups[ids]] == visible[ids],
                scale_below_1=data['scale'][ids] < 1,
                scale_1_to_10=(data['scale'][ids] >= 1) & (data['scale'][ids] < 10),
                scale_10_to_100=(data['scale'][ids] >= 10) & (data['scale'][ids] < 100),
                scale_at_least_100=data['scale'][ids] >= 100).items():
            known = use & sel & np.isfinite(cv[ids]); complete = use & sel & full[ids]
            zero = complete & (cv[ids] == 0)
            conditional.append(dict(view=key, subset=subset, population_rows=int(sel.sum()),
                **support_counts(use & sel, data['valid'][ids], groups[ids], data['tracks'][ids]),
                complete_zero_CV_selected=int(zero.sum()), complete_zero_CV_harmed=int((zero & (err > 0)).sum()),
                supported_native_ADE_harm_mean=float(np.mean(err[known]-cv[ids][known])) if known.any() else None,
                complete_native_ADE_harm_mean=float(np.mean(err[complete]-cv[ids][complete])) if complete.any() else None))
        for rec in np.unique(data['recordings'][ids]):
            sel = data['recordings'][ids] == rec
            per_recording.append(dict(seed=seed, recording=rec, population_rows=int(sel.sum()),
                **support_counts(use & sel, data['valid'][ids], groups[ids], data['tracks'][ids])))
        beat(state='fixed_control_support_audited', view=key, selected=int(use.sum()))
    union = np.any(list(all_selected.values()), axis=0)
    repeated = [support_counts(use, data['valid'], groups, data['tracks']) for use in all_selected.values()]
    selected_groups = []
    for seed, use in all_selected.items():
        selected_per_group = np.bincount(groups, weights=use, minlength=len(counts)).astype(int)
        selected_groups.append(dict(seed=seed, any_switch_scene_queries=int((selected_per_group > 0).sum()),
            multi_switch_scene_queries=int((selected_per_group >= 2).sum()),
            selected_count_quantiles=quantiles(selected_per_group[selected_per_group > 0]),
            incomplete_context_switch_groups=int(((selected_per_group > 0) & (counts < visible[first])).sum())))
    result = dict(result_source='fresh_run_alignment_and_support_audit_cached_verified_frozen_models',
        source_bindings=bindings, policy='unchanged_stop_mse_strict', rows=n,
        source_recordings=len(alignment), source_sites=parent['sites'], scene_queries=len(counts),
        alignment=alignment, coverage=coverage, cache=cache_receipt,
        neighbor_identity=dict(current_unique_match_max_error=neighbors['max_unique_match_error'],
            compared_past_points=checked, linked_history_max_error=past_max,
            all_unique_current_links_history_verified=mapping_usable,
            ambiguous_links_remain_unresolved=True),
        native_cost_invariance_checks=selected_geometry_checks,
        selected_repeated_instances={k:sum(r[k] for r in repeated) for k in ('selected_rows', 'complete_selected', 'partial_selected', 'unknown_selected')},
        selected_unique_queries=support_counts(union, data['valid'], groups, data['tracks']),
        scene_switch_opportunities=selected_groups, conditional_support=conditional,
        per_recording_support=per_recording,
        new_model_training=False, new_policy_evaluation=False, joint_selection_experiment=False,
        selection_uses_future_mask=False, independent_calibration=False, independent_confirmation=False,
        closed_roles_opened=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(dict(source_bindings=bindings))
    immutable_json(PUBLIC/'analysis.json', result)
    beat(state='complete', analysis_sha256=file_digest(PUBLIC/'analysis.json'), seconds=time.monotonic()-started)


if __name__ == '__main__':
    main()
