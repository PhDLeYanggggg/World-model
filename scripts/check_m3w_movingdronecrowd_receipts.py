"""Repeatable read-only source/window replay; no downloads or predictive scores."""
import sys
sys.dont_write_bytecode=True

import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import intake_m3w_movingdronecrowd as intake
from scripts import review_m3w_movingdronecrowd_splits as review
from src.evaluation.m3w_source_reservations import REGISTRY,SourceReservations


def main():
    start=time.monotonic();public=intake.PUBLIC
    reg=json.loads((public/'registration.json').read_text())
    for name,digest in reg['bindings'].items():
        if intake.sha(ROOT/name)!=digest:raise ValueError('Changed frozen source implementation')
    SourceReservations(ROOT)
    if intake.sha(ROOT/REGISTRY)!=reg['reservation_sha256']:
        raise ValueError('Changed source reservations')
    verification=json.loads((public/'verification.json').read_text())
    if (not verification['all_recording_structures_exact']
            or verification['analysis_sha256']!=intake.sha(public/'analysis.json')):
        raise ValueError('Missing or stale original full structural replay')
    source=intake.cached_source()
    original=json.loads((public/'analysis.json').read_text())
    actual=intake.analyze(reg['configuration'],source,lambda **_:None)
    if actual!=original:raise ValueError('Full native-window structure changed')
    split=json.loads((public/'split_interpretation.json').read_text())
    receipt=json.loads((public/'split_verification.json').read_text())
    if (not receipt['all_split_summaries_exact']
            or receipt['split_interpretation_sha256']!=intake.sha(public/'split_interpretation.json')):
        raise ValueError('Missing or stale split interpretation replay')
    for name,digest in split['bindings'].items():
        if intake.sha(ROOT/name)!=digest:raise ValueError('Changed split interpretation implementation')
    if intake.sha(intake.PRIVATE/'publisher_dataset_reader.py')!=split['publisher_reader_sha256']:
        raise ValueError('Changed pinned author loader snapshot')
    if review.calculate(source,original)!=split['protocols']:
        raise ValueError('Changed publisher split summaries')
    print(json.dumps(dict(result_source='cached_verified_read_only_full_structural_replay',
        seconds=time.monotonic()-start,recordings=actual['processed_recordings'],rows=actual['rows'],
        native8_12_windows=actual['complete_obs8_pred12_windows'],all_structures_and_split_summaries_exact=True,
        application_artifact_files_written=0,bytecode_cache_writes_disabled=True,
        network_requests=0,new_training=False,forecast_errors_read=False,
        roles_assigned=False,source_reservations_changed=False)))


if __name__=='__main__':main()
