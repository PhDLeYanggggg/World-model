"""Resolve mixed-level publisher lists; preserve the original structural audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import intake_m3w_movingdronecrowd as intake
from src.evaluation.m3w_movingdronecrowd_splits import expand_publisher_list,compare_publisher_splits

PUBLIC,PRIVATE=intake.PUBLIC,intake.PRIVATE
REVISION='df2b44d788b1f4bbb270139c52247ae68f699130'
URL=f'https://raw.githubusercontent.com/fyw1999/MovingDroneCrowd/{REVISION}/cusdatasets/dataset.py'
FILES=(Path(__file__),ROOT/'src/evaluation/m3w_movingdronecrowd_splits.py',
       ROOT/'tests/test_m3w_movingdronecrowd_splits.py')


def calculate(source,analysis):
    clips={v['clip']:v for v in analysis['recordings']}
    result={}
    for label,prefix in (('extended',''),('conference_subset','MDC_')):
        splits={}
        for role in ('train','val','test'):
            name=prefix+role+'.txt'
            records=[v for k,v in source['members'].items() if Path(k).name==name]
            if len(records)!=1:raise ValueError('Publisher split file missing or ambiguous')
            value=expand_publisher_list((ROOT/records[0]['path']).read_text(),clips)
            rows=[clips[key] for key in value['clips']]
            value.update(clips_count=len(rows),scene_labels_count=len(value['scene_labels']),
                annotation_rows=sum(v['rows'] for v in rows),
                recording_local_agents=sum(v['agents'] for v in rows),
                complete_obs8_pred12_windows=sum(v['window_support']['obs8_pred12_native_stride1']['target_agent_windows'] for v in rows),
                endpoint_support={str(h):{field:sum(v['endpoint_support'][str(h)][field] for v in rows)
                    for field in ('endpoint_label_windows','complete_interval_windows')} for h in (10,25,50,100)})
            splits[role]=value
        union=set().union(*(set(v['clips']) for v in splits.values()))
        result[label]=dict(splits=splits,comparison=compare_publisher_splits(splits),
            total_clips=len(union),total_scene_labels=len(set().union(*(set(v['scene_labels']) for v in splits.values()))),
            clips_not_in_this_publisher_protocol=sorted(set(clips)-union))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    start=time.monotonic()
    replay=json.loads((PUBLIC/'verification.json').read_text())
    if not replay['all_recording_structures_exact'] or replay['analysis_sha256']!=intake.sha(PUBLIC/'analysis.json'):
        raise ValueError('Complete source structural replay first')
    for path in FILES:
        import subprocess
        raw=subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if hashlib.sha256(raw).hexdigest()!=intake.sha(path):raise ValueError('Commit the split interpretation before execution')
    endpoint=PUBLIC/('split_verification.json' if args.verify else 'split_execution.json')
    if endpoint.exists():raise ValueError('Completed split interpretation is immutable')
    reader=PRIVATE/'publisher_dataset_reader.py'
    source_receipt=PUBLIC/'publisher_reader_source.json'
    if args.verify:
        before=json.loads(source_receipt.read_text())
        if before['sha256']!=intake.sha(reader):raise ValueError('Changed pinned publisher reader')
    else:
        raw=intake.small_get(URL,1024**2)
        if reader.exists() and reader.read_bytes()!=raw:raise ValueError('Changed reader snapshot')
        if not reader.exists():reader.write_bytes(raw)
        intake.once(source_receipt,dict(url=URL,revision=REVISION,path=str(reader.relative_to(ROOT)),
            sha256=intake.sha(reader),bytes=len(raw),executed=False,
            reviewed_lines=[26,44],frame_mapping_reviewed_lines=[227,255]))
    source=intake.cached_source();analysis=json.loads((PUBLIC/'analysis.json').read_text())
    value=dict(result_source='post_intake_publisher_split_interpretation_not_new_outcome_evaluation',
        analysis_sha256=intake.sha(PUBLIC/'analysis.json'),source_manifest_sha256=intake.sha(PUBLIC/'source_manifest.json'),
        publisher_reader_sha256=intake.sha(reader),bindings={str(p.relative_to(ROOT)):intake.sha(p) for p in FILES},
        protocols=calculate(source,analysis),roles_assigned=False,predictive_admission=False,
        independent_sites_certified=False,forecast_errors_read=False,existing_reservations_changed=False)
    intake.once(PUBLIC/'split_interpretation.json',value)
    intake.once(endpoint,dict(pid=os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        result_source='cached_verified_reinterpretation_no_new_network' if args.verify else 'fresh_interpretation_of_cached_source_with_pinned_reader_review',
        all_split_summaries_exact=args.verify,split_interpretation_sha256=intake.sha(PUBLIC/'split_interpretation.json')))
    print(json.dumps({label:{role:{k:v[k] for k in ('clips_count','scene_labels_count','annotation_rows','complete_obs8_pred12_windows')}
        for role,v in block['splits'].items()} for label,block in value['protocols'].items()}))


if __name__=='__main__':main()
