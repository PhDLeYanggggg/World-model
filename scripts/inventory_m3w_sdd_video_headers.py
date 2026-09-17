"""Metadata-only local SDD media inventory; no annotation or training admission."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    parent = json.loads((ROOT/'configs/m3w_offline_visual_forecast.json').read_text())
    sys.path.insert(0, str(ROOT/parent['decoder_path']))
    import av
    source = ROOT/'external_data/StanfordDroneDataset/video'
    records = []
    for path in sorted(source.glob('*/*/video.mp4')):
        record = dict(path=str(path.relative_to(ROOT)), scene=path.parent.parent.name,
                      video=path.parent.name, bytes=path.stat().st_size, sha256=file_digest(path))
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                record.update(header_readable=True, codec=stream.codec_context.name,
                    width=stream.width, height=stream.height, header_frames=stream.frames,
                    header_average_rate=str(stream.average_rate), time_base=str(stream.time_base))
        except Exception as error:
            record.update(header_readable=False, error=repr(error))
        records.append(record)
    result = dict(result_source='fresh_run_local_media_headers_and_hashes',
        files=len(records), bytes=sum(r['bytes'] for r in records), records=records,
        decoded_all_frames=False, annotation_files_opened=False, frame_annotation_alignment_verified=False,
        physical_time_verified=False, original_dataset_completeness_verified=False,
        new_training_source_admitted=False, source_admission='pending_scientific_protocol_decision',
        role_boundaries_unchanged=True, model_trained=False,
        historical_sdd_results_are_independent_confirmation=False,
        stage5c_executed=False, smc_enabled=False)
    destination = ROOT/'outputs/publication_readiness_2026_09/sdd_media_inventory'
    json_write(destination/'headers.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))


if __name__ == '__main__':
    main()
