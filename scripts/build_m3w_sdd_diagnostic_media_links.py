"""Freeze explicit local media links without admitting a new training source."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    reports = ROOT/'outputs/publication_readiness_2026_09/sdd_media_alignment'
    paths = [reports/name for name in ('audit.json', 'resize_mapping.json', 'nexus_identity.json')]
    audit, resize, nexus = [json.loads(p.read_text()) for p in paths]
    geometry = {r['recording']:r['geometry'] for r in resize['records']}
    override = {r['annotation_key']:r for r in nexus['records']}
    rows, used_media = [], set()
    for source in audit['records']:
        key = source['identity']['key']
        old = source['identity']['source_hashes']
        entry = dict(annotation_key=key, scene_id=key.split('/')[0],
            annotations_path='external_data/StanfordDroneDataset/annotations/'+key+'/annotations.txt',
            annotations_sha256=old['annotations'],
            reference_path='external_data/StanfordDroneDataset/annotations/'+key+'/reference.jpg',
            reference_sha256=old['reference'],
            video_path='external_data/StanfordDroneDataset/video/'+key+'/video.mp4',
            video_sha256=old['video'], geometry=geometry[key], coverage=source['coverage'],
            correspondence='same_name_with_reference_image_resize')
        if key in override:
            fixed = override[key]
            entry.update(video_path=fixed['video_path'], video_sha256=fixed['video_sha256'],
                geometry=fixed['geometry'], coverage=fixed['coverage'],
                correspondence='fixed_lexical_reindex_hypothesis_supported_by_index_and_image_checks')
        for name in ('annotations', 'reference', 'video'):
            if file_digest(ROOT/entry[name+'_path']) != entry[name+'_sha256']:
                raise ValueError('Changed source at diagnostic link export')
        if not entry['coverage']['decoded_range_covers_annotations'] or entry['video_path'] in used_media:
            raise ValueError('Missing frames or duplicate source video mapping')
        used_media.add(entry['video_path'])
        rows.append(entry)
    result = dict(schema_version=1, data_role='source_audit_only',
        report_bindings={str(p.relative_to(ROOT)):file_digest(p) for p in paths},
        code_sha256=file_digest(Path(__file__)), records=rows,
        annotation_ids_and_labels_unchanged=True, raw_files_modified=False,
        new_training_source_admitted=False, must_not_use_as_training_registration=True,
        official_eval_allowed=False, semantic_alignment_all_rows_certified=False,
        observed_crop_rule='require frame<=query, decoded-frame availability, explicit partial/occluded/padded support',
        state_coordinates='original_annotation_pixel', physical_time_or_metric_certified=False,
        stage5c_executed=False, smc_enabled=False)
    json_write(reports/'diagnostic_media_links.json', result)
    print(json.dumps(dict(records=len(rows), unique_media_files=len(used_media),
        all_annotation_indices_covered=True, training_admitted=False)))


if __name__ == '__main__':
    main()
