"""Existing past-context proximity proxy applied to fixed temporal policy choices."""
import json
from pathlib import Path
import numpy as np
from src.evaluation.m3w_native_scene_alignment import restore
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_joint_intervention import past_proximity_edges, proximity_cost_table
from src.world_model.m3w_temporal_intervention import POLICIES, policy_arm

ROOT = Path(__file__).resolve().parents[2]


def audit(cfg, data, proposals, choices, beat):
    context = json.loads((ROOT/'outputs/publication_readiness_2026_09/native_scene_context_v2/analysis.json').read_text())
    baseline = restore(data['geometry'][:, 332:356].reshape(-1, 12, 2), data['origin'], data['rotation'], data['scale'])
    results = {}
    for seed in cfg['seeds']:
        native = {a: restore(p, data['origin'], data['rotation'], data['scale']) for a, p in proposals[seed].items()}
        rows = {s: dict(queries=0, known_edges=0, unknown_edges=0, unsupported_context_rows=0,
                       policy_excess_sum={p: 0. for p in POLICIES}) for s in cfg['sites']}
        for record in context['records']:
            rec = record['recording']; site = rec.split('/')[0]
            assert file_digest(ROOT/record['cache']['path']) == record['cache']['sha256']
            with np.load(ROOT/record['cache']['path'], allow_pickle=False) as z:
                c = {k: z[k].copy() for k in z.files}
            for frame in np.unique(c['context_frame_ids']):
                ix = np.flatnonzero(c['context_frame_ids'] == frame)
                target = c['context_target_rows'][ix] >= 0
                tid = c['context_target_rows'][ix][target]
                if not len(tid):
                    raise ValueError('Expected past-eligible targets in registered query')
                b = c['context_cv_rollout'][ix].astype(float); b[target] = baseline[tid]
                valid = c['context_cv_valid'][ix].all(1); valid[target] = True
                radius = float(np.median(data['scale'][tid]))
                edges = past_proximity_edges(c['context_xy'][ix], radius=radius)
                known = valid[edges].all(1); safe_edges = edges[known]
                r = rows[site]; r['queries'] += 1; r['known_edges'] += int(known.sum())
                r['unknown_edges'] += int((~known).sum()); r['unsupported_context_rows'] += int((~valid).sum())
                tables = {}
                for arm in ('ramp', 'uniform'):
                    q = b.copy(); q[target] = native[arm][tid]
                    tables[arm] = proximity_cost_table(b, q, safe_edges, distance_threshold=.1*radius)
                for name in POLICIES:
                    chosen = np.zeros(len(ix), bool); chosen[target] = choices[seed][name][tid]
                    table = tables[policy_arm(name)]
                    cost = table[np.arange(len(safe_edges)), chosen[safe_edges[:, 0]].astype(int), chosen[safe_edges[:, 1]].astype(int)]
                    r['policy_excess_sum'][name] += float(cost.sum())
            beat(state='past_context_proxy', seed=seed, recording=rec)
        assert sum(r['queries'] for r in rows.values()) == 20932
        results[str(seed)] = rows
    return dict(seeds=results, radius='median_target_past_scale', threshold_fraction=.1,
                future_labels_used=False, physical_safety_certified=False, new_joint_optimization=False)
