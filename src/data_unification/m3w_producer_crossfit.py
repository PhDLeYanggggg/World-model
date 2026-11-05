"""Source-only out-of-fold forecast donors inside each existing producer A."""
import json
from pathlib import Path

import numpy as np

from src.data_unification.m3w_moving_forecast_data import (
    SPEC, array_hash, baseline_numpy, BASELINES, encode_scene, mot_record,
    native_errors, past_inputs, target_labels, load_admission, require_fit_record,
)


def plan(registry):
    jobs = registry['jobs']
    result = []
    for fold in (0, 1):
        parents = [j for j in jobs if j['producer_fold'] == fold]
        producer = sorted(parents[0]['producer_sites'])
        if any(sorted(j['producer_sites']) != producer for j in parents):
            raise ValueError('Parent producer definitions disagree')
        donor = next(j for j in jobs if sorted(j['critic_sites']) == producer)
        parts = registry['partitions'][donor['key']]
        names = [s for p in parts for s in p['names']]
        if sorted(names) != producer or len(names) != len(set(names)):
            raise ValueError('Existing inner partitions must cover producer A exactly')
        for part in parts:
            held = sorted(part['names'])
            fit = sorted(set(producer) - set(held))
            if not fit or not held:
                raise ValueError('Nonempty fitting and donor sources required')
            for p in parents:
                forbidden = set(p['critic_sites']) | set(p['evaluation_sites'])
                if set(producer) & forbidden:
                    raise ValueError('Parent A overlaps outer C/H')
            result.append(dict(key=f'eqmotion_A{fold}_inner{part["part"]}_seed17',
                producer_fold=fold, part=part['part'], fit_sites=fit,
                donor_sites=held, producer_sites=producer,
                parent_contexts=[p['key'] for p in parents]))
    return result


def eligibility(registry):
    rows = []
    for target in registry['jobs']:
        for donor in registry['jobs']:
            if target['key'] == donor['key']:
                continue
            sites = registry['partitions'][donor['key']][0]['names']
            reasons = []
            if set(sites) & set(target['evaluation_sites']):
                reasons.append('donor_is_target_outer_H')
            if set(target['critic_sites']) & set(donor['producer_sites']):
                reasons.append('donor_teacher_trained_on_target_critic_sources')
            if donor['forecast_key'] != target['forecast_key']:
                reasons.append('different_teacher')
            if donor['easy_cut'] != target['easy_cut']:
                reasons.append('different_training_defined_target_cut')
            rows.append(dict(target=target['key'], donor=donor['key'],
                source_groups=sites, reasons=reasons, blind_pooling_allowed=not reasons))
    return rows


def selected_records(admission, structure, allowed):
    """Read metadata only for skipped sources; keep original global row/query IDs."""
    lookup = {r['clip']: r for r in structure['recordings']}
    row, query = 0, 0
    selected = []
    for index, node in enumerate(sorted(admission['recordings'])):
        meta = admission['recordings'][node]
        info = lookup[meta['clip']]
        n = info['window_support']['obs8_pred12_native_stride1']['history_complete_agent_instances']
        frames = info['image_frames']
        if meta['exclusion_group'] in allowed:
            selected.append(dict(index=index, node=node, meta=meta, rows=n,
                                 frames=frames, row_offset=row, query_offset=query))
        row += n
        query += max(frames - 12 - SPEC.history_span, 0)
    if set(r['meta']['exclusion_group'] for r in selected) != set(allowed):
        raise ValueError('Every allowed producer source needs admitted metadata')
    return selected


def pack(root, admission_path, allowed, heartbeat=lambda **_: None):
    root = Path(root)
    admission = load_admission(root, root/admission_path)
    structure = json.loads((root/'outputs/publication_readiness_2026_09/movingdronecrowd_intake_v1/analysis.json').read_text())
    records = selected_records(admission, structure, allowed)
    n = sum(r['rows'] for r in records)
    specs = dict(geometry=(np.float32, (n, 476)), shared_context=(np.float32, (n, 48)),
        target=(np.float32, (n, 12, 2)), target_eval=(np.float64, (n, 12, 2)),
        valid=(bool, (n, 12)), history=(np.float64, (n, 8, 2)), origin=(np.float64, (n, 2)),
        sites=('<U40', (n,)), recordings=(np.int32, (n,)), frames=(np.int32, (n,)),
        agents=(np.int64, (n,)), queries=(np.int32, (n,)), global_ids=(np.int64, (n,)),
        baseline_ade=(np.float64, (n, len(BASELINES))), baseline_fde=(np.float64, (n, len(BASELINES))))
    out = {k: np.empty(shape, dtype=dtype) for k, (dtype, shape) in specs.items()}
    cursor = 0
    for r in records:
        meta = r['meta']
        path = root/meta['source_annotation_path']
        require_fit_record(admission, r['node'], 'fit', path)
        rec = mot_record(path.read_bytes(), meta['clip'], range(1, r['frames']+1))
        begin = cursor
        for qi, q in enumerate(range(SPEC.history_span, rec.valid.shape[1]-12)):
            inputs = past_inputs(rec, q, SPEC)
            g, local, shared = encode_scene(inputs['positions'], inputs['history_mask'], inputs['agent_ids'])
            count = len(local)
            dest = slice(cursor, cursor+count)
            if count:
                labels = target_labels(rec, inputs, SPEC)
                h = inputs['positions'][local]
                y, valid = labels['future_positions'][local], labels['future_mask'][local]
                out['geometry'][dest], out['shared_context'][dest] = g, shared
                out['history'][dest], out['origin'][dest] = h, h[:, -1]
                out['target'][dest] = np.where(valid[..., None], y-h[:, -1, None], 0)
                out['target_eval'][dest], out['valid'][dest] = y, valid
                out['sites'][dest], out['recordings'][dest] = meta['exclusion_group'], r['index']
                out['frames'][dest], out['agents'][dest] = q, inputs['agent_ids'][local]
                out['queries'][dest] = r['query_offset']+qi
                out['global_ids'][dest] = r['row_offset']+np.arange(cursor-begin, cursor-begin+count)
                for k in range(len(BASELINES)):
                    a, f = native_errors(baseline_numpy(h, k), y, valid, np.ones(count))
                    out['baseline_ade'][dest, k], out['baseline_fde'][dest, k] = a, f
            cursor += count
        if cursor-begin != r['rows']:
            raise ValueError('Subset rows differ from pinned intake')
        heartbeat(state='packing_A_only', record=r['node'], rows=cursor)
    if cursor != n or len(np.unique(out['global_ids'])) != n:
        raise ValueError('Population or global alignment mismatch')
    return out, dict(rows=n, recordings=len(records), groups=sorted(allowed),
        complete_rows=int(out['valid'].all(1).sum()), known_rows=int(out['valid'].any(1).sum()),
        unknown_rows=int((~out['valid'].any(1)).sum()),
        arrays={k: dict(shape=list(v.shape), dtype=v.dtype.str, bytes=v.nbytes,
                       sha256=array_hash(v)) for k, v in out.items()},
        input_keys=['geometry'], labels_never_inputs=True,
        global_ids_match_original_population_order=True,
        result_source='fresh_run', outer_readout=False,
        role='parent_A_only_out_of_fold_training_donors',
        coordinate_unit='image_pixel', time_unit='native_annotation_step_unverified_seconds')
