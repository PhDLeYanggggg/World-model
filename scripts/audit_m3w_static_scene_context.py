"""Audit static image-region context on original fitting rows, never reserved roles."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_multiaction_cost as parent
from src.world_model.m3w_static_scene_context import scene_features, source_polygons
import numpy as np

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_static_scene_audit_v1'
ARCHIVE = ROOT/'external_data/EuropeanSquares_source_audit/02.Resources.zip'
PACKED = ROOT/'data/stage_cvpr2027_experiments/european_source_forecast_v1/packed/receipt.json'
FILES = (Path(__file__), ROOT/'src/world_model/m3w_static_scene_context.py',
         ROOT/'tests/test_m3w_static_scene_context.py', PUBLIC/'protocol.md', ARCHIVE, PACKED)


def main(phase):
    cfg, reg, _, data, jobs, identity = parent.admission()
    packed = json.loads(PACKED.read_text()); polygons = source_polygons(ARCHIVE, packed['records'])
    bindings = {str(p.relative_to(ROOT)):parent.sha(p) for p in FILES}
    registration = dict(bindings=bindings, parent_registration_sha256=parent.sha(parent.REG),
        scope='original_fitting_recordings_only', optimizer_updates=0,
        raw_camera_mapping=True, independent_roles_opened=False)
    parent.once(PUBLIC/'registration.json', registration)
    if phase == 'register':
        return
    for path, digest in bindings.items():
        if ROOT/path not in (ARCHIVE, PACKED):
            parent.committed(ROOT/path)
        if parent.sha(ROOT/path) != digest:
            raise ValueError('Static audit input changed')
    parent.committed(PUBLIC/'registration.json')
    start = time.monotonic(); rows, saved = [], set()
    for context, items in sorted(reg['source_contexts'].items()):
        scopes = parent.prepare_context(context, items, data, jobs, identity, lambda **_:None, include_held=False)
        for source, packet in sorted(scopes['fit'].items()):
            for ri in np.unique(data['recordings'][packet['ids']]):
                ids = packet['ids'][data['recordings'][packet['ids']] == ri]
                record = packed['records'][int(ri)]; polygon = polygons[record]['vertices']
                history = data['history'][ids]
                rollout = parent.baseline_numpy(history, 1)
                features = scene_features(history=history, polygon=polygon, rollout=rollout)
                bounds = (history[:, -1]-polygon.min(0))/np.ptp(polygon, axis=0)
                where = np.isin(packet['ids'], ids)
                known = np.isfinite(packet['floor_ade'][where])
                bins = [-np.inf, -.1, 0., .02, .05, np.inf]
                groups = []
                for lo, hi in zip(bins[:-1], bins[1:]):
                    at = (features[:, 9] >= lo) & (features[:, 9] < hi)
                    k = at & known
                    groups.append(dict(interval=[str(lo), str(hi)], rows=int(at.sum()),
                        unknown_rows=int((at & ~known).sum()),
                        floor_ADE_sum=float(packet['floor_ade'][where][k].sum())))
                rows.append(dict(context=context, source=source, recording=int(ri), rows=len(ids),
                    source_member=record, polygon_member=polygons[record]['member'],
                    normalized_position_min=bounds.min(0).tolist(), normalized_position_max=bounds.max(0).tolist(),
                    outside_expanded_polygon_bbox=int(((bounds < -.5) | (bounds > 1.5)).any(1).sum()),
                    inside_polygon=int((features[:, 9] >= 0).sum()),
                    all_features_finite=bool(np.isfinite(features).all()), bins=groups))
                saved.update(ids.tolist())
        print(json.dumps(dict(context=context, record_views=len(rows))), flush=True)
    result = dict(result_source='fresh_run_static_context_train_diagnosis', record_views=len(rows),
        unique_recordings=len(set(r['recording'] for r in rows)), unique_rows=len(saved), rows=rows,
        total_rows_over_contexts=sum(r['rows'] for r in rows),
        outside_expanded_bbox_over_contexts=sum(r['outside_expanded_polygon_bbox'] for r in rows),
        registered_feature_fields=18, optimizer_updates=0, independent_roles_opened=False,
        mapping_is_not_verified_metric_or_walkability=True, registration_sha256=parent.sha(PUBLIC/'registration.json'))
    if phase == 'verify':
        if result != json.loads((PUBLIC/'audit.json').read_text()):
            raise ValueError('Static context audit differs')
        parent.once(PUBLIC/'verification.json', dict(exact_replay=True, audit_sha256=parent.sha(PUBLIC/'audit.json')))
    else:
        parent.once(PUBLIC/'audit.json', result)
    print(json.dumps(dict(phase=phase, seconds=time.monotonic()-start, record_views=len(rows))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run', 'verify'))
    main(parser.parse_args().phase)
