"""Test a fixed lexical-renumbering hypothesis on media, not forecast labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_sdd_source_links import lexical_reindex_links
from src.world_model.m3w_sdd_image_coordinates import SDDImageCoordinates
from src.world_model.m3w_sdd_media_audit import decoded_coverage, reference_comparison
from scripts.audit_m3w_sdd_resize_mapping import draw_comparison


def main():
    reports = ROOT/'outputs/publication_readiness_2026_09/sdd_media_alignment'
    audit = json.loads((reports/'audit.json').read_text())
    resize = json.loads((reports/'resize_mapping.json').read_text())
    old = {r['identity']['key'].split('/')[1]:r for r in audit['records']
           if r['identity']['key'].startswith('nexus/')}
    previous = {r['recording']:r for r in resize['records']}
    links = lexical_reindex_links(old)
    # The mapping is fixed from filename ordering, not selected by image or prediction scores.
    parent = json.loads((ROOT/'configs/m3w_offline_visual_forecast.json').read_text())
    sys.path.insert(0, str(ROOT/parent['decoder_path']))
    import av
    directory = ROOT/'external_data/StanfordDroneDataset'
    private = ROOT/'data/stage_cvpr2027_experiments/sdd_nexus_identity'
    results = []
    for annotation_name, media_name in links.items():
        annotations = directory/'annotations/nexus'/annotation_name/'annotations.txt'
        reference_path = annotations.parent/'reference.jpg'
        media = directory/'video/nexus'/media_name/'video.mp4'
        if (file_digest(annotations) != old[annotation_name]['identity']['source_hashes']['annotations']
                or file_digest(media) != old[media_name]['identity']['source_hashes']['video']
                or file_digest(reference_path) != old[annotation_name]['identity']['source_hashes']['reference']):
            raise ValueError('Changed source identity')
        rows = np.loadtxt(annotations, usecols=range(9), dtype=float, ndmin=2)
        count = old[media_name]['coverage']['decoded_frames']
        coverage = decoded_coverage(rows[:, 5], count, old[media_name]['coverage']['header_frames'])
        wanted = old[media_name]['sampled_frame_indices']
        images = {}
        with av.open(str(media)) as container:
            container.streams.video[0].codec_context.thread_count = 4
            for i, frame in enumerate(container.decode(video=0)):
                if i in wanted:
                    image = frame.to_ndarray(format='rgb24')
                    if hashlib.sha256(image.tobytes()).hexdigest() != old[media_name]['sampled_frame_pixel_sha256'][str(i)]:
                        raise ValueError('Independently decoded pixels differ')
                    images[i] = image
                if i >= max(wanted):
                    break
        if len(images) != len(wanted):
            raise ValueError('Incomplete matched decode')
        reference = Image.open(reference_path).convert('RGB')
        height, width = images[0].shape[:2]
        mapper = SDDImageCoordinates(reference.width, reference.height, width, height)
        compare = reference_comparison(images[0], np.asarray(reference.resize((width,height), Image.Resampling.BILINEAR)))
        sheet = private/'private_visual_checks'/(annotation_name+'_to_'+media_name+'.png')
        draw_comparison(images, rows, mapper, sheet, 'nexus '+annotation_name+' -> '+media_name)
        result = dict(annotation_key='nexus/'+annotation_name, media_key='nexus/'+media_name,
            annotation_path=str(annotations.relative_to(ROOT)), video_path=str(media.relative_to(ROOT)),
            annotation_sha256=file_digest(annotations), video_sha256=file_digest(media),
            reference_path=str(reference_path.relative_to(ROOT)), reference_sha256=file_digest(reference_path),
            coverage=coverage, geometry=mapper.metadata(), first_frame_reference=compare,
            previous_same_name_reference_correlation=previous['nexus/'+annotation_name]['first_frame_resized_reference_comparison']['pixel_correlation'],
            replay_exact_frames=sorted(images), private_visual_evidence=str(sheet.relative_to(ROOT)),
            private_visual_evidence_sha256=file_digest(sheet),
            visual_review_status='not_reviewed', training_admitted=False)
        results.append(result)
        print(json.dumps(dict(annotation=annotation_name, media=media_name,
            index_coverage=coverage['decoded_range_covers_annotations'], first_frame_correlation=compare['pixel_correlation'])), flush=True)
    out = dict(result_source='fresh_run_fixed_source_link_hypothesis_validation',
        hypothesis='lexicographic_video_names_enumerated_into_numeric_video_names_within_nexus',
        hypothesis_is_not_verified_compression_history=True,
        candidate_mapping_not_chosen_by_forecast_outcomes=True,
        raw_files_renamed_or_edited=False, other_scenes_not_reordered=True,
        code_sha256={str(p.relative_to(ROOT)):file_digest(p) for p in
                    (Path(__file__), ROOT/'src/world_model/m3w_sdd_source_links.py')},
        source_audit_sha256=file_digest(reports/'audit.json'), records=results,
        index_coverage_pass=sum(r['coverage']['decoded_range_covers_annotations'] for r in results),
        independently_replayed_frames=sum(len(r['replay_exact_frames']) for r in results),
        future_forecast_targets_built=False, training_source_admitted=False,
        sealed_evaluation_roles_opened=False, semantic_alignment_all_rows_certified=False,
        physical_time_or_metric_certified=False, stage5c_executed=False, smc_enabled=False)
    json_write(reports/'nexus_identity.json', out)


if __name__ == '__main__':
    main()
