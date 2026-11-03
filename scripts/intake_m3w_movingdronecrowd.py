"""Acquire sparse author-release annotations and test native-step window support."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import resource
import shutil
import subprocess
import sys
import time
from urllib.parse import quote
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import requests
from src.data_unification.m3w_movingdronecrowd_intake import (
    PublicRangeZip,clip_key,endpoint_support,member_path,mot_record,
)
from src.evaluation.m3w_dronecrowd_windows import WindowSpec,audit_support,verify_prefix_invariance
from src.evaluation.m3w_source_reservations import REGISTRY,SourceReservations

NAME='movingdronecrowd_intake_v1'
PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG=ROOT/'configs/m3w_movingdronecrowd_intake_v1.json'
FILES=(Path(__file__),CONFIG,ROOT/'src/data_unification/m3w_movingdronecrowd_intake.py',
       ROOT/'tests/test_m3w_movingdronecrowd_intake.py',
       ROOT/'src/evaluation/m3w_dronecrowd_windows.py',
       ROOT/'src/evaluation/m3w_dronecrowd_image_intake.py',PUBLIC/'protocol.md')
TEXT_NAMES={'train.txt','val.txt','test.txt','scene_labels.txt',
            'MDC_train.txt','MDC_val.txt','MDC_test.txt','MDC_scene_labels.txt'}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:raise ValueError('Changed immutable receipt: '+str(path))
    else:
        with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    pending.replace(path)


def register():
    cfg=json.loads(CONFIG.read_text())
    if (cfg['data_role']!='diagnostic_only' or cfg['predictive_admission']
            or cfg['download_images'] or cfg['download_weights'] or cfg['execute_third_party_code']):
        raise ValueError('Only the registered non-predictive annotation intake is allowed')
    reservations=SourceReservations(ROOT)
    if sum(r['dataset']=='DroneCrowd' and r['reserved_role']=='confirmation'
           for r in reservations.records.values()) !=112:
        raise ValueError('Preserve the complete DroneCrowd confirmation reservation')
    value=dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        reservation_sha256=sha(ROOT/REGISTRY),configuration=cfg,
        acquisition_authority='outputs/publication_readiness_2026_09/delegated_research_authorization_20260923.json',
        result_scope='source_feasibility_not_forecasting',roles_assigned=False,
        predictions_run=False,source_time_causality_proved=False,stage5c_executed=False,smc_enabled=False)
    once(PUBLIC/'registration.json',value)
    return cfg


def small_get(url,limit):
    with requests.get(url,headers={'Accept-Encoding':'identity'},stream=True,timeout=(10,45)) as r:
        r.raise_for_status();data=r.raw.read(limit+1)
        if len(data)>limit:raise ValueError('Metadata exceeds the source-intake budget')
        return data


def inventory(archive):
    entries=archive.infolist()
    if len(entries)>250000:raise ValueError('Oversized archive inventory')
    rows=[];seen=set()
    for item in entries:
        key=str(member_path(item.filename))
        if key in seen:raise ValueError('Canonical archive path collision')
        seen.add(key)
        rows.append(dict(name=item.filename,bytes=item.file_size,compressed_bytes=item.compress_size,
                         crc32=item.CRC,offset=item.header_offset,compression=item.compress_type))
    return sorted(rows,key=lambda r:r['name'])


def source_layout(entries):
    annotations={};images={};texts={}
    for item in entries:
        path=PurePosixPath(item['name'])
        if '__MACOSX' in path.parts or path.name.startswith('._'):continue
        key=clip_key(item['name'],'annotations')
        if key is not None:
            if key in annotations:raise ValueError('Ambiguous annotation clip namespace')
            annotations[key]=item
        key=clip_key(item['name'],'frames')
        if key is not None:
            values=images.setdefault(key,set())
            frame=int(path.stem)
            if frame in values:raise ValueError('Two images map to one clip/frame')
            values.add(frame)
        if path.name in TEXT_NAMES:
            if path.name in texts:raise ValueError('Ambiguous publisher split filename')
            texts[path.name]=item
    if not annotations or not images:raise ValueError('No published MOT annotations/frame inventory found')
    return annotations,images,texts


def acquire(cfg,resume,beat):
    api=f'https://huggingface.co/api/datasets/{cfg["repository"]}/revision/{cfg["revision"]}?blobs=true'
    raw=small_get(api,4*1024**2);meta=json.loads(raw)
    metadata_transferred=len(raw)
    if meta['sha']!=cfg['revision'] or meta.get('gated') or meta.get('disabled'):
        raise ValueError('Changed or inaccessible official dataset revision')
    if meta.get('cardData',{}).get('license')!=cfg['dataset_card_license']:
        raise ValueError('Publisher license declaration differs from discovery')
    entry=next(v for v in meta['siblings'] if v['rfilename']==cfg['archive'])
    if entry['size']!=cfg['archive_bytes'] or entry['lfs']['sha256']!=cfg['publisher_lfs_sha256_not_locally_verified']:
        raise ValueError('Changed pinned archive identity')
    metadata_path=PRIVATE/'publisher_metadata.json'
    # API activity counters may vary. Keep the first source snapshot and bind it;
    # scientific archive identity, revision and license are checked every call.
    if not metadata_path.exists():metadata_path.write_bytes(raw)
    base=f'https://huggingface.co/datasets/{cfg["repository"]}/resolve/{cfg["revision"]}/'
    receipts_path=PRIVATE/'member_receipts.json'
    receipts=json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    if receipts and not resume:raise ValueError('Explicit unchanged source resume required')
    outside={}
    for name in ('README.md','MDC_scene_labels.txt'):
        body=small_get(base+quote(name,safe=''),2*1024**2)
        metadata_transferred+=len(body)
        upstream=next(v for v in meta['siblings'] if v['rfilename']==name)
        blob=hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest()
        if blob!=upstream['blobId']:raise ValueError('Publisher small-file blob changed')
        path=PRIVATE/'outside'/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=body:raise ValueError('Changed source small file')
        if not path.exists():path.write_bytes(body)
        outside[name]=dict(path=str(path.relative_to(ROOT)),bytes=len(body),sha256=sha(path))
    url=base+quote(cfg['archive'],safe='')
    transferred=0
    with PublicRangeZip(url,cfg['archive_bytes'],maximum_bytes=cfg['maximum_network_bytes']-metadata_transferred) as stream:
        with zipfile.ZipFile(stream) as archive:
            entries=inventory(archive);annotations,images,texts=source_layout(entries)
            digest=hashlib.sha256(encoded(entries)).hexdigest()
            once(PRIVATE/'archive_inventory.json',entries)
            chosen={item['name']:item for item in [*annotations.values(),*texts.values()]}
            if sum(v['bytes'] for v in chosen.values())>cfg['maximum_total_extracted_bytes']:
                raise ValueError('Selected annotations exceed the declared extraction budget')
            for number,(name,item) in enumerate(sorted(chosen.items()),1):
                beat(state='acquiring_annotation_metadata',member=number,total=len(chosen))
                if item['bytes']>cfg['maximum_member_bytes']:raise ValueError('Oversized source member')
                path=PRIVATE/'members'/member_path(name)
                path.parent.mkdir(parents=True,exist_ok=True)
                if not path.resolve().is_relative_to((PRIVATE/'members').resolve()):
                    raise ValueError('Cached member escapes the source tree')
                if name in receipts:
                    previous=receipts[name]
                    if (not path.is_file() or sha(path)!=previous['sha256'] or path.stat().st_size!=item['bytes']
                            or previous['crc32']!=item['crc32']):raise ValueError('Changed cached source member')
                    continue
                body=archive.read(name)  # ZipFile checks each selected member CRC.
                if len(body)!=item['bytes']:raise ValueError('Changed expanded source size')
                if path.exists() and path.read_bytes()!=body:raise ValueError('Unreceipted member differs from source')
                if not path.exists():path.write_bytes(body)
                receipts[name]=dict(path=str(path.relative_to(ROOT)),bytes=len(body),sha256=sha(path),crc32=item['crc32'])
                atomic(receipts_path,receipts)
            transferred=stream.transferred_bytes
            requests_used=stream.range_requests
    with PublicRangeZip(url,cfg['archive_bytes'],maximum_bytes=cfg['maximum_network_bytes']-metadata_transferred-transferred) as stream:
        with zipfile.ZipFile(stream) as archive:repeated=inventory(archive)
        if repeated!=entries:raise ValueError('Remote central directory changed after acquisition')
        transferred+=stream.transferred_bytes;requests_used+=stream.range_requests
    source=dict(repository=cfg['repository'],revision=cfg['revision'],archive_url=url,
        archive_bytes=cfg['archive_bytes'],full_archive_sha256_verified=False,
        publisher_lfs_sha256=entry['lfs']['sha256'],dataset_card_license=cfg['dataset_card_license'],
        publisher_metadata=dict(path=str(metadata_path.relative_to(ROOT)),sha256=sha(metadata_path)),
        inventory_sha256=digest,inventory_entries=len(entries),
        annotation_clips=len(annotations),image_clips=len(images),image_inventory_frames=sum(len(v) for v in images.values()),
        outside=outside,members=receipts,network_bytes_this_successful_invocation=transferred+metadata_transferred,
        archive_range_bytes_this_successful_invocation=transferred,metadata_bytes_this_successful_invocation=metadata_transferred,
        range_requests_this_successful_invocation=requests_used,remote_inventory_recheck_exact=True,
        downloaded_images=0,downloaded_weights=0,third_party_code_executed=False,
        predictive_admission=False,existing_reservations_unchanged=True)
    once(PUBLIC/'source_manifest.json',source)
    return source


def cached_source():
    source=json.loads((PUBLIC/'source_manifest.json').read_text())
    entries=json.loads((PRIVATE/'archive_inventory.json').read_text())
    if hashlib.sha256(encoded(entries)).hexdigest()!=source['inventory_sha256']:
        raise ValueError('Cached archive inventory changed')
    for value in [source['publisher_metadata'],*source['outside'].values(),*source['members'].values()]:
        path=ROOT/value['path']
        if not path.resolve().is_relative_to(PRIVATE.resolve()) or sha(path)!=value['sha256']:
            raise ValueError('Changed downloaded source or metadata')
    return source


def split_table(source,clips):
    rows={}
    for name in ('train.txt','val.txt','test.txt','MDC_train.txt','MDC_val.txt','MDC_test.txt'):
        found=[v for n,v in source['members'].items() if PurePosixPath(n).name==name]
        if len(found)!=1:
            rows[name]=dict(status='not_present');continue
        values=[];unknown=[]
        for line in (ROOT/found[0]['path']).read_text().splitlines():
            if not line.strip():continue
            text=line.strip().strip('/')
            if text not in clips:unknown.append(text)
            else:values.append(text)
        scene_roles={v.split('/')[0] for v in values}
        rows[name]=dict(status='parsed' if not unknown else 'unresolved_publisher_identifiers',
                        clips=values,declared_lines=len(values)+len(unknown),unmatched_identifiers=unknown,
                        repeated_identifiers=len(values)-len(set(values)),scene_labels=sorted(scene_roles))
    return rows


def analyze(cfg,source,beat):
    entries=json.loads((PRIVATE/'archive_inventory.json').read_text())
    annotations,images,_=source_layout(entries)
    profiles=[WindowSpec('obs8_pred12_native_stride1',8,1,tuple(range(1,13))),
              *(WindowSpec(f'obs8_native_offset{h}',8,1,(h,)) for h in cfg['endpoint_offsets'])]
    results=[];flags=[]
    for number,(key,item) in enumerate(sorted(annotations.items()),1):
        beat(state='native_window_structure',recording=number,total=len(annotations))
        if key not in images:
            flags.append(dict(clip=key,error='no_matching_image_inventory'));continue
        try:
            value=source['members'][item['name']]
            record=mot_record((ROOT/value['path']).read_bytes(),key,images[key],maximum_cells=cfg['maximum_dense_recording_cells'])
            lengths=record.valid.sum(1)
            counts={spec.name:audit_support(record,spec) for spec in profiles}
            # No pixel bounds were acquired; do not publish all-false bounds as exclusions.
            for support in counts.values():support.pop('target_agent_windows_all_boxes_in_bounds')
            result=dict(clip=key,publisher_scene_label=key.split('/')[0],rows=int(record.valid.sum()),
                agents=len(record.agent_ids),image_frames=record.valid.shape[1],
                annotation_frame_range=[int(np.flatnonzero(record.valid.any(0))[0]),int(np.flatnonzero(record.valid.any(0))[-1])],
                track_visible_length_quantiles=np.quantile(lengths,[0,.25,.5,.75,1]).tolist(),
                window_support=counts,endpoint_support={str(h):endpoint_support(record,8,h) for h in cfg['endpoint_offsets']},
                prefix_checks=verify_prefix_invariance(record) if record.valid.shape[1]>=8 else
                    dict(query_prefixes_checked=0,transformations_per_prefix=2,input_mismatches=0,
                         source_time_causality_proved=False,reason='clip_shorter_than_eight_observations'),
                image_bounds_status='not_measured')
            results.append(result)
        except (ValueError,UnicodeError,zipfile.BadZipFile) as error:
            flags.append(dict(clip=key,error=str(error)))
    splits=split_table(source,{v['clip'] for v in results})
    conflicts={}
    for prefix in ('','MDC_'):
        splitsets={role:set(splits[prefix+role+'.txt'].get('clips',[])) for role in ('train','val','test')}
        overlaps={a+'_'+b:sorted(splitsets[a]&splitsets[b]) for a,b in (('train','val'),('train','test'),('val','test'))}
        scene_roles={scene:sorted(role for role,ids in splitsets.items() if any(v.split('/')[0]==scene for v in ids))
                     for scene in sorted({v.split('/')[0] for ids in splitsets.values() for v in ids})}
        conflicts[prefix or 'extended']=dict(clip_overlaps=overlaps,
            publisher_scene_labels_crossing_splits={s:r for s,r in scene_roles.items() if len(r)>1})
    return dict(result_source='fresh_source_structural_audit_not_forecasting',
        source_manifest_sha256=sha(PUBLIC/'source_manifest.json'),recordings=results,quality_flags=flags,
        processed_recordings=len(results),annotation_clips=len(annotations),
        rows=sum(v['rows'] for v in results),recording_local_agents=sum(v['agents'] for v in results),
        publisher_scene_labels=len({v['publisher_scene_label'] for v in results}),
        complete_obs8_pred12_windows=sum(v['window_support'][profiles[0].name]['target_agent_windows'] for v in results),
        endpoint_support={str(h):{field:sum(v['endpoint_support'][str(h)][field] for v in results)
            for field in ('endpoint_label_windows','complete_interval_windows')} for h in cfg['endpoint_offsets']},
        prefix_queries=sum(v['prefix_checks']['query_prefixes_checked'] for v in results),
        prefix_transformations=sum(v['prefix_checks']['query_prefixes_checked']*2 for v in results),
        input_mismatches=sum(v['prefix_checks']['input_mismatches'] for v in results),
        publisher_splits=splits,split_conflicts=conflicts,
        coordinate_unit='image_pixel',frame_unit='native_annotation_step_not_verified_raw_video_frame',
        effective_seconds=None,homography_verified=False,metric_scale_verified=False,
        structural_support_ready=not flags and len(results)==len(annotations),
        independent_sites_certified=False,data_role='diagnostic_only',predictive_admission=False,
        new_model_training=False,forecast_errors_read=False,existing_reservations_unchanged=True,
        stage5c_executed=False,smc_enabled=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','acquire','verify'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=register()
        if args.phase=='register':return
        for path in (*FILES,PUBLIC/'registration.json'):
            blob=subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(blob).hexdigest()!=sha(path):raise ValueError('Commit source definition before acquisition')
        end=PUBLIC/('verification.json' if args.phase=='verify' else 'execution.json')
        if end.exists():raise ValueError('Completed phase is immutable')
        if args.phase=='verify' and not (PUBLIC/'execution.json').exists():raise ValueError('Complete source intake first')
        start=time.monotonic()
        def beat(**kw):
            if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Keep existing10GiB reserve')
            if time.monotonic()-start>43200:raise TimeoutError('12h invocation limit; resume unchanged receipts')
            event=dict(pid=os.getpid(),elapsed_seconds=time.monotonic()-start,**kw)
            atomic(PRIVATE/'heartbeat.json',event);print(json.dumps(event),flush=True)
        beat(state='source_intake_ready',predictive_admission=False)
        if args.phase=='verify':source=cached_source()
        elif (PUBLIC/'source_manifest.json').exists():
            if not args.resume:raise ValueError('Explicit unchanged resume required')
            source=cached_source()
        else:source=acquire(cfg,args.resume,beat)
        result=analyze(cfg,source,beat)
        once(PUBLIC/'analysis.json',result)
        if sha(ROOT/REGISTRY)!=json.loads((PUBLIC/'registration.json').read_text())['reservation_sha256']:
            raise ValueError('Source reservations changed')
        once(end,dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            observed_utc=datetime.now(timezone.utc).isoformat(),analysis_sha256=sha(PUBLIC/'analysis.json'),
            source_manifest_sha256=sha(PUBLIC/'source_manifest.json'),
            result_source='cached_verified_structural_replay_no_new_network' if args.phase=='verify' else 'fresh_run_source_acquisition_and_structure',
            all_recording_structures_exact=args.phase=='verify',new_model_training=False,forecast_errors_read=False))
        print(json.dumps(dict(complete=str(end),recordings=result['processed_recordings'],rows=result['rows'])),flush=True)


if __name__=='__main__':main()
