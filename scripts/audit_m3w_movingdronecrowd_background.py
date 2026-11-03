"""Resumable complete new-source/background audit; never predictive evaluation."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from scripts import intake_m3w_movingdronecrowd as intake
from src.data_unification.m3w_movingdronecrowd_intake import PublicRangeZip,clip_key
from src.evaluation.m3w_dronecrowd_grouping import extract_features,compare_clip_views
from src.evaluation.m3w_movingdronecrowd_background import (
    selected_frames,moving_node,frame_boxes,prepare_raster,features_from_raster,
    exact_features,raster_hash,pair_roster,conservative_exclusions,
)
from src.evaluation.m3w_source_reservations import REGISTRY,SourceReservations

NAME='movingdronecrowd_background_v1'
PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG=ROOT/'configs/m3w_movingdronecrowd_background_v1.json'
OLD_PRIVATE=ROOT/'data/stage_cvpr2027_experiments/dronecrowd_grouping_v2'
OLD_GROUPS=ROOT/'outputs/publication_readiness_2026_09/dronecrowd_grouping_v2/exclusion_constraints.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',
    ROOT/'src/evaluation/m3w_movingdronecrowd_background.py',
    ROOT/'tests/test_m3w_movingdronecrowd_background.py',
    ROOT/'src/evaluation/m3w_dronecrowd_grouping.py',ROOT/'src/evaluation/m3w_dronecrowd_image_geometry.py',
    ROOT/'src/data_unification/m3w_movingdronecrowd_intake.py',
    ROOT/'configs/m3w_movingdronecrowd_intake_v1.json',
    ROOT/'scripts/intake_m3w_movingdronecrowd.py')
sha,once,atomic=intake.sha,intake.once,intake.atomic


def stable(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def admission():
    cfg=json.loads(CONFIG.read_text())
    old=json.loads((OLD_PRIVATE/'identity.json').read_text())
    matcher_path=ROOT/cfg['matcher_config'];matcher=json.loads(matcher_path.read_text())
    if (old['config_sha256']!=sha(matcher_path) or old['opencv']!=cv2.__version__
            or old['numpy']!=np.__version__):raise ValueError('Protected descriptor runtime/config changed')
    for name in ('src/evaluation/m3w_dronecrowd_grouping.py','src/evaluation/m3w_dronecrowd_image_geometry.py'):
        if old['code_sha256'][name]!=sha(ROOT/name):raise ValueError('Protected descriptor algorithm changed')
    source=intake.cached_source()
    verified=json.loads((intake.PUBLIC/'verification.json').read_text())
    if (not verified['all_recording_structures_exact']
            or verified['analysis_sha256']!=sha(intake.PUBLIC/'analysis.json')
            or verified['source_manifest_sha256']!=sha(intake.PUBLIC/'source_manifest.json')):
        raise ValueError('Complete new-source structural replay required')
    SourceReservations(ROOT)
    source_cfg=json.loads(intake.CONFIG.read_text())
    entries=json.loads((intake.PRIVATE/'archive_inventory.json').read_text())
    annotations,images,_=intake.source_layout(entries)
    by_image={}
    for row in entries:
        key=clip_key(row['name'],'frames')
        if key is not None:by_image[key,int(Path(row['name']).stem)]=row
    selection=[]
    for key in sorted(annotations):
        for slot,frame in enumerate(selected_frames(images[key]),1):
            row=by_image[key,frame]
            selection.append(dict(clip=key,node=moving_node(key),slot=slot,image_frame_one_based=frame,
                annotation_frame_zero_based=frame-1,member=row,annotation_member=annotations[key]['name']))
    if len(annotations)!=cfg['moving_clips'] or len(selection)!=cfg['moving_views']:
        raise ValueError('Keep all original clips and three distinct views')
    if sum(r['member']['compressed_bytes'] for r in selection)!=cfg['selected_compressed_member_bytes']:
        raise ValueError('Keep the complete fixed sparse image selection')
    groups=json.loads(OLD_GROUPS.read_text())['conservative_components']
    protected=sorted({node for group in groups for node in group})
    if len(protected)!=112 or len(groups)!=45:raise ValueError('Protected source inventory changed')
    definition=dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        configuration=cfg,matcher_configuration_sha256=sha(matcher_path),
        source_manifest_sha256=sha(intake.PUBLIC/'source_manifest.json'),
        source_analysis_sha256=sha(intake.PUBLIC/'analysis.json'),
        source_split_interpretation_sha256=sha(intake.PUBLIC/'split_interpretation.json'),
        protected_identity_sha256=sha(OLD_PRIVATE/'identity.json'),protected_run_id=old['run_id'],
        protected_constraints_sha256=sha(OLD_GROUPS),reservation_sha256=sha(ROOT/REGISTRY),
        authorization_sha256=sha(ROOT/'outputs/publication_readiness_2026_09/delegated_research_authorization_20260923.json'),
        opencv=cv2.__version__,numpy=np.__version__,machine=platform.machine(),
        view_selection=selection,compressed_selected_member_bytes=sum(r['member']['compressed_bytes'] for r in selection),
        publisher_scene_constraints=True,physical_independence_certified=False,
        roles_assigned=False,predictive_outcomes_read=False,stage5c_executed=False,smc_enabled=False)
    once(PUBLIC/'registration.json',stable(definition))
    matcher=dict(matcher,selected_frames_one_based=[1,2,3])
    return cfg,source,source_cfg,selection,protected,groups,matcher


def feature_path(node,slot):return PRIVATE/'features'/node/f'view{slot}.npz'


def feature_value(arrays):
    result=dict(xy=arrays['xy'],descriptors=arrays['descriptors'],background_fraction=float(arrays['background_fraction']))
    if result['xy'].ndim!=2 or result['xy'].shape[1]!=2 or result['descriptors'].shape!=(len(result['xy']),128):
        raise ValueError('Malformed source feature arrays')
    if not np.isfinite(result['xy']).all() or not np.isfinite(result['descriptors']).all():
        raise ValueError('Nonfinite source features')
    if not len(result['descriptors']):result['descriptors']=None
    return result


def read_view(path,expected,matcher,reextract=False):
    with np.load(path,allow_pickle=False) as arrays:
        meta=json.loads(str(arrays['metadata_json']))
        for key,value in expected.items():
            if meta.get(key)!=value:raise ValueError('Changed source/view cache identity')
        gray,mask=arrays['gray'],arrays['mask']
        if raster_hash(gray,mask)!=meta['raster_sha256']:raise ValueError('Changed prepared pixels/mask')
        value=feature_value(arrays)
        if reextract:exact_features(value,features_from_raster(gray,mask,matcher))
    return value,meta


def acquire(cfg,source,source_cfg,selection,matcher,resume,guard,beat):
    if (PUBLIC/'acquisition.json').exists():raise ValueError('Completed acquisition is immutable')
    if list((PRIVATE/'features').glob('*/*.npz')) and not resume:
        raise ValueError('Explicit unchanged acquisition resume required')
    expected_inventory=json.loads((intake.PRIVATE/'archive_inventory.json').read_text())
    rows=[];fresh=0
    boxes_by_clip={}
    stream=PublicRangeZip(source['archive_url'],source_cfg['archive_bytes'],
        maximum_bytes=cfg['maximum_network_bytes_per_acquisition_invocation'])
    started=time.monotonic()
    try:
        with zipfile.ZipFile(stream) as archive:
            if intake.inventory(archive)!=expected_inventory:raise ValueError('Changed remote source inventory')
            for number,row in enumerate(selection,1):
                guard();path=feature_path(row['node'],row['slot'])
                expected=dict(registration_sha256=sha(PUBLIC/'registration.json'),
                    clip=row['clip'],node=row['node'],slot=row['slot'],
                    image_frame_one_based=row['image_frame_one_based'],
                    annotation_frame_zero_based=row['annotation_frame_zero_based'],
                    source_member=row['member']['name'],source_crc32=row['member']['crc32'])
                if path.exists():
                    value,meta=read_view(path,expected,matcher)
                else:
                    item=row['member']
                    if item['bytes']>cfg['maximum_raw_member_bytes']:raise ValueError('Oversized source image')
                    body=archive.read(item['name'])
                    if len(body)!=item['bytes']:raise ValueError('Changed source JPEG size')
                    with Image.open(io.BytesIO(body)) as image:
                        width,height=image.size
                        if width*height>cfg['maximum_image_pixels']:raise ValueError('Decoded image exceeds pixel bound')
                        rgb=np.asarray(image.convert('RGB'))
                    if row['clip'] not in boxes_by_clip:
                        annotation=source['members'][row['annotation_member']]
                        raw=(ROOT/annotation['path']).read_bytes()
                        if hashlib.sha256(raw).hexdigest()!=annotation['sha256']:raise ValueError('Changed source head masks')
                        boxes_by_clip[row['clip']]=raw
                    boxes=frame_boxes(boxes_by_clip[row['clip']],row['annotation_frame_zero_based'])
                    centers=(boxes[:,:2]+boxes[:,2:])/2
                    gray,mask=prepare_raster(rgb,boxes,matcher)
                    value=features_from_raster(gray,mask,matcher)
                    exact_features(extract_features(rgb,boxes,matcher),value)
                    meta=dict(**expected,source_JPEG_sha256=hashlib.sha256(body).hexdigest(),
                        source_JPEG_bytes=len(body),source_dimensions=[width,height],heads_masked=len(boxes),
                        head_boxes_outside_image=int(((boxes[:,0]<0)|(boxes[:,1]<0)|(boxes[:,2]>width)|(boxes[:,3]>height)).sum()),
                        head_centers_outside_image=int(((centers[:,0]<0)|(centers[:,1]<0)|(centers[:,0]>=width)|(centers[:,1]>=height)).sum()),
                        annotation_sha256=source['members'][row['annotation_member']]['sha256'],
                        raster_sha256=raster_hash(gray,mask),RGB_to_prepared_features_exact=True,
                        retained_raw_JPEG=False,keypoints=len(value['xy']),background_fraction=value['background_fraction'])
                    path.parent.mkdir(parents=True,exist_ok=True)
                    temp=path.with_suffix('.pending')
                    with temp.open('wb') as f:
                        np.savez_compressed(f,gray=gray,mask=mask,xy=value['xy'],
                            descriptors=value['descriptors'] if value['descriptors'] is not None else np.empty((0,128),np.float32),
                            background_fraction=np.asarray(value['background_fraction']),
                            metadata_json=np.asarray(json.dumps(meta,sort_keys=True)))
                    temp.replace(path);fresh+=1
                    del body,rgb,gray,mask
                rows.append(dict(**meta,path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size))
                beat(state='source_views',complete=number,total=len(selection),fresh_views=fresh,
                     transferred_bytes=stream.transferred_bytes)
        transferred,requests_used=stream.transferred_bytes,stream.range_requests
    finally:stream.close()
    with PublicRangeZip(source['archive_url'],source_cfg['archive_bytes'],
            maximum_bytes=cfg['maximum_network_bytes_per_acquisition_invocation']-transferred) as recheck:
        with zipfile.ZipFile(recheck) as archive:
            if intake.inventory(archive)!=expected_inventory:raise ValueError('Remote inventory changed after images')
        transferred+=recheck.transferred_bytes;requests_used+=recheck.range_requests
    manifest=dict(result_source='fresh_source_view_acquisition_with_explicit_resume_reuse',
        registration_sha256=sha(PUBLIC/'registration.json'),views=rows,
        remote_inventory_rechecked_exact=True,full_archive_sha256_verified=False,
        network_payload_range_bytes_this_invocation=transferred,range_requests_this_invocation=requests_used,
        new_views_this_invocation=fresh,cached_views_this_invocation=len(rows)-fresh,
        raw_source_JPEG_retained=False,new_model_training=False,predictive_outcomes_read=False)
    once(PUBLIC/'image_manifest.json',stable(manifest))
    once(PUBLIC/'acquisition.json',dict(pid=os.getpid(),seconds=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        image_manifest_sha256=sha(PUBLIC/'image_manifest.json'),new_views=fresh,cached_views=len(rows)-fresh))


def load_features(selection,protected,matcher,reextract=False):
    manifest=json.loads((PUBLIC/'image_manifest.json').read_text())
    lookup={(v['node'],v['slot']):v for v in manifest['views']}
    if len(lookup)!=360:raise ValueError('Complete moving view roster required')
    features={};frames={};receipts={}
    for row in selection:
        v=lookup[row['node'],row['slot']];path=ROOT/v['path']
        if sha(path)!=v['sha256']:raise ValueError('Changed frozen moving descriptor/raster')
        value,meta=read_view(path,{k:v[k] for k in ('registration_sha256','node','clip','slot','source_member','source_crc32')},matcher,reextract)
        if meta!={k:v[k] for k in meta}:raise ValueError('Changed source view metadata')
        features.setdefault(row['node'],{})[row['slot']]=value
        frames.setdefault(row['node'],{})[row['slot']]=row['image_frame_one_based']
        receipts[str(path.relative_to(ROOT))]=v['sha256']
    old=json.loads((OLD_PRIVATE/'identity.json').read_text())
    for clip in protected:
        node='drone_'+clip
        for slot,frame in enumerate((1,150,300),1):
            path=OLD_PRIVATE/'features'/f'{clip}_{frame:03d}.npz'
            receipt=json.loads(path.with_suffix('.json').read_text())
            if receipt['run_id']!=old['run_id'] or sha(path)!=receipt['sha256']:
                raise ValueError('Protected cached descriptors changed')
            with np.load(path,allow_pickle=False) as arrays:value=feature_value(arrays)
            features.setdefault(node,{})[slot]=value;frames.setdefault(node,{})[slot]=frame
            receipts[str(path.relative_to(ROOT))]=receipt['sha256']
    return features,frames,receipts


def match_pair(row,features,frames,matcher):
    value=compare_clip_views(features[row['left']],features[row['right']],matcher)
    for support in value['support']:
        support['left_actual_image_frame']=frames[row['left']][support['left_frame']]
        support['right_actual_image_frame']=frames[row['right']][support['right_frame']]
    return stable(dict(registration_sha256=sha(PUBLIC/'registration.json'),**row,**value))


def pair_path(row):return PRIVATE/'pairs'/(row['left']+'___'+row['right']+'.json')


def priors(selection,groups):
    by_scene={}
    for row in selection:by_scene.setdefault(row['clip'].split('/')[0],set()).add(row['node'])
    edges=[]
    for group in [*map(sorted,by_scene.values()),*(['drone_'+v for v in g] for g in groups)]:
        edges.extend(zip(group,group[1:]))
    return edges


def summarize(results,selection,protected,groups,feature_receipts):
    edges=priors(selection,groups)
    nodes=sorted({r['node'] for r in selection}|{'drone_'+v for v in protected})
    grouping=conservative_exclusions(nodes,results,edges)
    checks={str(pair_path(r).relative_to(ROOT)):sha(pair_path(r)) for r in results}
    image_manifest=json.loads((PUBLIC/'image_manifest.json').read_text())
    unresolved=sorted({v['node'] for v in image_manifest['views']
        if sum(w['keypoints']>=25 and w['background_fraction']>=.04
               for w in image_manifest['views'] if w['node']==v['node'])<2})
    return stable(dict(scope='source_overlap_not_forecast_evaluation',
        registration_sha256=sha(PUBLIC/'registration.json'),clip_pairs=len(results),
        within_moving_pairs=sum(r['kind']=='within_moving' for r in results),
        cross_source_pairs=sum(r['kind']=='cross_source' for r in results),
        frame_pairs_attempted=sum(r['frame_pairs_attempted'] for r in results),
        status_counts={kind:{status:sum(r['kind']==kind and r['status']==status for r in results)
            for status in ('strong_multiview','ambiguous','unsupported')} for kind in ('within_moving','cross_source')},
        supported_pairs=[{k:v for k,v in r.items() if k not in ('registration_sha256','match_seconds')}
                         for r in results if r['status']!='unsupported'],
        prior_edges=edges,**grouping,insufficient_background_moving_nodes=unresolved,
        feature_receipt_sha256=hashlib.sha256(intake.encoded(feature_receipts)).hexdigest(),
        pair_receipt_sha256=hashlib.sha256(intake.encoded(checks)).hexdigest(),
        publisher_scene_constraints_preserved=True,protected45group_constraints_preserved=True,
        metric_homography_claim=False,scientific_roles_assigned=False,predictive_outcomes_read=False,
        DUT_image_overlap_not_screened_in_this_unit=True,stage5c_executed=False,smc_enabled=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','acquire','pilot','run','verify'))
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cv2.setNumThreads(1)
        cfg,source,source_cfg,selection,protected,groups,matcher=admission()
        if args.phase=='register':return
        for path in (*FILES,PUBLIC/'registration.json'):
            raw=subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(raw).hexdigest()!=sha(path):raise ValueError('Commit source-audit definition first')
        started=time.monotonic()
        def guard():
            import shutil
            if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Keep original10GiB reserve')
            if time.monotonic()-started>cfg['maximum_invocation_seconds']:raise TimeoutError('12h boundary; resume unchanged work')
        def beat(**kw):
            guard()
            size=sum(v.stat().st_size for v in (PRIVATE/'features').glob('*/*.npz'))
            if size>cfg['maximum_new_cache_bytes']:raise OSError('Prepared source cache exceeds registered512MiB cap')
            usage=resource.getrusage(resource.RUSAGE_SELF)
            event=dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,
                       cpu_user_seconds=usage.ru_utime,cpu_system_seconds=usage.ru_stime,
                       peak_RSS_bytes=usage.ru_maxrss,**kw)
            atomic(PRIVATE/'heartbeat.json',event);print(json.dumps(event),flush=True)
        beat(state='source_audit_ready',phase=args.phase)
        if args.phase=='acquire':
            acquire(cfg,source,source_cfg,selection,matcher,args.resume,guard,beat);return
        if not (PUBLIC/'acquisition.json').exists():raise ValueError('Complete360view acquisition first')
        end=PUBLIC/({'pilot':'pilot.json','run':'execution.json','verify':'verification.json'}[args.phase])
        if end.exists():raise ValueError('Completed phase is immutable')
        if args.phase=='verify' and not (PUBLIC/'execution.json').exists():raise ValueError('Complete all pairs first')
        features,frames,feature_receipts=load_features(selection,protected,matcher,reextract=args.phase=='verify')
        moving=sorted({v['node'] for v in selection});old=['drone_'+v for v in protected]
        roster=pair_roster(moving,old,cfg['roster_seed'])
        if len(roster)!=cfg['within_moving_pairs']+cfg['cross_source_pairs']:raise ValueError('Complete paired scope required')
        limit=2*cfg['pilot_pairs_per_kind'] if args.phase=='pilot' else len(roster)
        results=[];fresh=0
        for i,row in enumerate(roster[:limit]):
            guard();path=pair_path(row)
            if path.exists():
                if args.phase!='verify' and not args.resume:raise ValueError('Explicit unchanged pair resume required')
                result=json.loads(path.read_text())
                for k,v in dict(registration_sha256=sha(PUBLIC/'registration.json'),**row).items():
                    if result.get(k)!=v:raise ValueError('Changed pair identity')
            else:
                if args.phase=='verify':raise ValueError('Missing complete pair receipt')
                begin=time.monotonic();result=match_pair(row,features,frames,matcher)
                result['match_seconds']=time.monotonic()-begin;once(path,result);fresh+=1
            if args.phase=='verify' and i<cfg['verification_fresh_pair_prefix']:
                repeated=match_pair(row,features,frames,matcher)
                if repeated!={k:v for k,v in result.items() if k!='match_seconds'}:
                    raise ValueError('Fresh fixed pair-prefix replay mismatch')
            results.append(result)
            if (i+1)%50==0 or i+1==limit:beat(state='pair_screen',complete=i+1,total=len(roster),new_pairs=fresh)
        if args.phase=='pilot':
            timings={kind:float(np.mean([r['match_seconds'] for r in results if r['kind']==kind]))
                for kind in ('within_moving','cross_source')}
            once(end,dict(pid=os.getpid(),seconds=time.monotonic()-started,clip_pairs=len(results),
                per_kind_mean_seconds=timings,estimated_full_matching_seconds=
                    timings['within_moving']*7140+timings['cross_source']*13440,
                full_screen_complete=False,new_pairs=fresh,registration_sha256=sha(PUBLIC/'registration.json')))
            return
        summary=summarize(results,selection,protected,groups,feature_receipts)
        once(PUBLIC/'analysis.json',summary)
        once(end,dict(pid=os.getpid(),seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            result_source='cached_verified_all_receipts_plus_fresh_features_and_fixed50pairs' if args.phase=='verify' else 'fresh_run_complete_new_pair_roster_with_checkpoint_reuse',
            new_pairs=fresh,cached_pairs=len(results)-fresh,analysis_sha256=sha(PUBLIC/'analysis.json'),
            all_new_prepared_features_reextracted_exact=args.phase=='verify',
            freshly_recomputed_pair_prefix=cfg['verification_fresh_pair_prefix'] if args.phase=='verify' else 0,
            all_pair_receipts_and_graph_exact=args.phase=='verify',predictive_outcomes_read=False))
        print(json.dumps(dict(complete=str(end),clip_pairs=len(results),status_counts=summary['status_counts'])),flush=True)


if __name__=='__main__':main()
