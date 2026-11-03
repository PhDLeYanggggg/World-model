"""Complete DUT background bridge; no predictive labels, fitting or role grants."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scripts import check_m3w_movingdronecrowd_background as parent
from scripts.fetch_m3w_dut_annotations import CLIPS,REPO,COMMIT,TREE,fetch,verify_bytes
from src.evaluation.m3w_dut_background_bridge import (
    select_backgrounds,pair_roster,compare_available_views,conservative_bridge)
from src.evaluation.m3w_movingdronecrowd_orientation import canonical_rgb_boxes

base=parent.base
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/dut_background_bridge_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/dut_background_bridge_v1'
CONFIG=ROOT/'configs/m3w_dut_background_bridge_v1.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',
       ROOT/'src/evaluation/m3w_dut_background_bridge.py',ROOT/'tests/test_m3w_dut_background_bridge.py',
       ROOT/'src/evaluation/m3w_movingdronecrowd_orientation.py',
       ROOT/'outputs/publication_readiness_2026_09/external_source_use_disposition_20260923.md',
       ROOT/'outputs/publication_readiness_2026_09/delegated_research_authorization_20260923.json')
sha,once,atomic=base.sha,base.once,base.atomic


def bytes_once(path,body):
    if path.exists():
        if path.read_bytes()!=body:raise ValueError('Changed existing source bytes')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        pending=Path(stream.name);stream.write(body)
    os.replace(pending,path)


def register(allow_metadata_fetch=False):
    cfg=json.loads(CONFIG.read_text());reg,_,_=parent.frozen_context()
    if cfg['upstream_commit']!=COMMIT or cfg['upstream_tree']!=TREE:
        raise ValueError('Preserve already audited author commit/tree')
    path=PRIVATE/'upstream_tree.json';network_metadata_bytes=0
    if not path.exists():
        if not allow_metadata_fetch:raise ValueError('Register metadata before acquiring images')
        body=fetch(f'https://api.github.com/repos/{REPO}/git/trees/{TREE}?recursive=1',2000000)
        tree=json.loads(body);network_metadata_bytes=len(body)
        select_backgrounds(tree,CLIPS,TREE,cfg['expected_background_bytes'])
        once(path,dict(tree=tree,successful_metadata_payload_bytes=network_metadata_bytes))
    snapshot=json.loads(path.read_text());tree=snapshot['tree']
    entries=select_backgrounds(tree,CLIPS,TREE,cfg['expected_background_bytes'])
    if len(entries)!=cfg['expected_backgrounds']:raise ValueError('Require all28backgrounds')
    pp=parent.version.PUBLIC
    analysis=json.loads((pp/'analysis.json').read_text());verified=json.loads((pp/'verification.json').read_text())
    if (verified['analysis_sha256']!=sha(pp/'analysis.json') or not verified['all_pair_receipts_and_graph_exact']):
        raise ValueError('Complete parent source-overlap replay first')
    matcher_path=ROOT/cfg['matcher_config']
    if sha(matcher_path)!=json.loads((parent.version.PARENT_PUBLIC/'registration.json').read_text())['matcher_configuration_sha256']:
        raise ValueError('Keep the frozen matcher configuration')
    value=dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},configuration=cfg,
        upstream_tree_snapshot_sha256=sha(path),background_entries=entries,
        parent_registration_sha256=sha(pp/'registration.json'),parent_analysis_sha256=sha(pp/'analysis.json'),
        parent_verification_sha256=sha(pp/'verification.json'),
        parent_image_manifest_sha256=sha(pp/'image_manifest.json'),
        reservation_sha256=sha(ROOT/base.REGISTRY),matcher_configuration_sha256=sha(matcher_path),
        opencv=base.cv2.__version__,numpy=np.__version__,source_roles_assigned=False,
        no_predictive_outcomes_read=True,stage5c_executed=False,smc_enabled=False)
    once(PUBLIC/'registration.json',base.stable(value))
    return cfg,entries,analysis


def raw_path(entry):return PRIVATE/'source_backgrounds'/Path(entry['path']).name
def view_path(node):return PRIVATE/'features'/(node+'.npz')
def pair_path(row):return PRIVATE/'pairs'/(row['left']+'___'+row['right']+'.json')


def source_node(entry):return 'dut_'+Path(entry['path']).name.removesuffix('_background_GSOC.png')


def decode(entry,body,cfg):
    verify_bytes(body,entry)
    with Image.open(io.BytesIO(body)) as source:
        if source.format!='PNG' or source.width*source.height>cfg['maximum_image_pixels']:
            raise ValueError('Only bounded original PNG backgrounds accepted')
        rgb=np.asarray(source.convert('RGB'))
    canonical,boxes,transform,rotation=canonical_rgb_boxes(rgb,np.empty((0,4)))
    return rgb,canonical,boxes,transform,rotation


def acquire(cfg,entries,matcher,resume,guard,beat):
    if (PUBLIC/'acquisition.json').exists():raise ValueError('Complete acquisition is immutable')
    if list((PRIVATE/'features').glob('*.npz')) and not resume:raise ValueError('Explicit unchanged resume required')
    views=[];new_images=0;new_features=0;payload=0;started=time.monotonic()
    for i,entry in enumerate(entries,1):
        guard();node=source_node(entry);raw=raw_path(entry)
        if raw.exists():body=raw.read_bytes()
        else:
            if payload+entry['size']>cfg['maximum_source_payload_bytes']:raise OSError('Source payload budget')
            body=fetch(f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{entry["path"]}',entry['size'])
            verify_bytes(body,entry);bytes_once(raw,body);payload+=len(body);new_images+=1
        rgb,canonical,boxes,transform,rotation=decode(entry,body,cfg)
        source_sha=hashlib.sha256(body).hexdigest();path=view_path(node)
        expected=dict(registration_sha256=sha(PUBLIC/'registration.json'),node=node,
                      source_git_blob=entry['sha'],source_PNG_sha256=source_sha)
        if path.exists():_,meta=base.read_view(path,expected,matcher)
        else:
            gray,mask=base.prepare_raster(canonical,boxes,matcher)
            features=base.features_from_raster(gray,mask,matcher)
            base.exact_features(base.extract_features(canonical,boxes,matcher),features)
            width,height=rgb.shape[1],rgb.shape[0];cw,ch=canonical.shape[1],canonical.shape[0]
            meta=dict(**expected,source_member=entry['path'],source_bytes=len(body),
                source_dimensions=[width,height],canonical_dimensions=[cw,ch],
                pixel_center_transform=transform.tolist(),rotation_counterclockwise_degrees=rotation,
                axis_resize_scale_ratio=max(960/cw,540/ch)/min(960/cw,540/ch),
                source_raw_path=str(raw.relative_to(ROOT)),raster_sha256=base.raster_hash(gray,mask),
                keypoints=len(features['xy']),background_fraction=features['background_fraction'],
                mask_kind='unmasked_author_background',image_kind=cfg['DUT_image_kind'],
                actual_temporal_frame=None,predictive_image_input_allowed=False,
                original_RGB_prepared_features_exact=True)
            path.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
                pending=Path(stream.name)
                np.savez_compressed(stream,gray=gray,mask=mask,xy=features['xy'],
                    descriptors=features['descriptors'] if features['descriptors'] is not None else np.empty((0,128),np.float32),
                    background_fraction=np.asarray(features['background_fraction']),
                    metadata_json=np.asarray(json.dumps(meta,sort_keys=True)))
            pending.replace(path);new_features+=1
        views.append(dict(meta,path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size))
        if i%4==0 or i==len(entries):beat(state='background_acquisition',complete=i,total=len(entries),successful_image_payload_bytes=payload)
    manifest=dict(views=views,registration_sha256=sha(PUBLIC/'registration.json'),
        new_images_this_invocation=new_images,new_features_this_invocation=new_features,
        cached_images_this_invocation=len(entries)-new_images,successful_image_payload_bytes_this_invocation=payload,
        source_media_private_not_committed=True,predictive_inputs=False)
    once(PUBLIC/'image_manifest.json',manifest)
    once(PUBLIC/'acquisition.json',dict(pid=os.getpid(),seconds=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        image_manifest_sha256=sha(PUBLIC/'image_manifest.json'),
        new_images=new_images,new_features=new_features,successful_image_payload_bytes=payload))


def load_features(cfg,entries,matcher,reextract):
    reg,groups,protected=parent.frozen_context()
    base.PUBLIC,base.PRIVATE=parent.version.PUBLIC,parent.version.PRIVATE
    features,frames,receipts=base.load_features(reg['view_selection'],protected,matcher,reextract=False)
    manifest=json.loads((PUBLIC/'image_manifest.json').read_text())
    if manifest['registration_sha256']!=sha(PUBLIC/'registration.json'):raise ValueError('Changed DUT view identity')
    lookup={r['node']:r for r in manifest['views']}
    if set(lookup)!={source_node(e) for e in entries}:raise ValueError('Require all28DUTbackgrounds')
    for entry in entries:
        node=source_node(entry);row=lookup[node];path=ROOT/row['path'];body=raw_path(entry).read_bytes()
        verify_bytes(body,entry)
        if sha(path)!=row['sha256'] or hashlib.sha256(body).hexdigest()!=row['source_PNG_sha256']:
            raise ValueError('Changed cached source background or features')
        value,meta=base.read_view(path,{k:row[k] for k in ('node','registration_sha256','source_PNG_sha256','source_git_blob')},matcher,reextract)
        if meta!={k:row[k] for k in meta}:raise ValueError('Changed saved background metadata')
        if reextract:
            _,rgb,boxes,_,_=decode(entry,body,cfg)
            base.exact_features(base.extract_features(rgb,boxes,matcher),value)
        features[node]={1:value};frames[node]={1:None};receipts[str(path.relative_to(ROOT))]=row['sha256']
        receipts[str(raw_path(entry).relative_to(ROOT))]=row['source_PNG_sha256']
    return features,frames,receipts


def match_pair(row,features,frames,matcher):
    value=compare_available_views(features[row['left']],features[row['right']],matcher)
    for evidence in value['support']:
        evidence['left_actual_image_frame']=frames[row['left']][evidence['left_slot']]
        evidence['right_actual_image_frame']=frames[row['right']][evidence['right_slot']]
    return base.stable(dict(registration_sha256=sha(PUBLIC/'registration.json'),**row,**value))


def summarize(results,analysis,dut_nodes,receipts):
    inventory={str(pair_path(r).relative_to(ROOT)):sha(pair_path(r)) for r in results}
    value=dict(scope='source_overlap_only_not_prediction',clip_pairs=len(results),
        registration_sha256=sha(PUBLIC/'registration.json'),
        frame_pairs_attempted=sum(r['frame_pairs_attempted'] for r in results),
        status_counts={k:{s:sum(r['kind']==k and r['status']==s for r in results) for s in ('ambiguous','unsupported')}
                       for k in ('moving_to_dut','drone_to_dut','within_dut')},
        supported_pairs=[{k:v for k,v in r.items() if k not in ('match_seconds','registration_sha256')}
                         for r in results if r['status']!='unsupported'],
        feature_receipt_sha256=hashlib.sha256(base.intake.encoded(receipts)).hexdigest(),
        pair_receipt_sha256=hashlib.sha256(base.intake.encoded(inventory)).hexdigest(),
        no_model_training=True,no_predictive_outcomes_read=True,metric_or_seconds_claim=False,
        stage5c_executed=False,smc_enabled=False)
    value.update(conservative_bridge(analysis['conservative_exclusion_components'],dut_nodes,results))
    return base.stable(value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','acquire','pilot','run','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.cv2.setNumThreads(1)
        cfg,entries,analysis=register(args.phase=='register')
        if args.phase=='register':return
        for path in (*FILES,PUBLIC/'registration.json'):
            body=subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(body).hexdigest()!=sha(path):raise ValueError('Commit source bridge definition first')
        started=time.monotonic()
        def guard():
            if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Keep original10GiBreserve')
            if time.monotonic()-started>cfg['maximum_invocation_seconds']:raise TimeoutError('12hboundary;resume unchanged')
            cache=sum(p.stat().st_size for sub in ('features','source_backgrounds') for p in (PRIVATE/sub).glob('*') if p.is_file())
            if cache>cfg['maximum_incremental_cache_bytes']:raise OSError('Registered incremental cache budget')
        def beat(**values):
            guard();usage=resource.getrusage(resource.RUSAGE_SELF)
            event=dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,
                       cpu_user_seconds=usage.ru_utime,cpu_system_seconds=usage.ru_stime,peak_RSS_bytes=usage.ru_maxrss,**values)
            atomic(PRIVATE/'heartbeat.json',event);print(json.dumps(event),flush=True)
        beat(state='ready',phase=args.phase)
        matcher=json.loads((ROOT/cfg['matcher_config']).read_text())
        if args.phase=='acquire':acquire(cfg,entries,matcher,args.resume,guard,beat);return
        if not (PUBLIC/'acquisition.json').exists():raise ValueError('Acquire all28backgrounds first')
        end=PUBLIC/({'pilot':'pilot.json','run':'execution.json','verify':'verification.json'}[args.phase])
        if end.exists():raise ValueError('Completed phase is immutable')
        if args.phase=='verify' and not (PUBLIC/'execution.json').exists():raise ValueError('Complete all6874pairs first')
        features,frames,receipts=load_features(cfg,entries,matcher,reextract=args.phase=='verify')
        moving=sorted(n for n in features if n.startswith('moving_'));drone=sorted(n for n in features if n.startswith('drone_'))
        dut=sorted(source_node(e) for e in entries);roster=pair_roster(moving,drone,dut,cfg['roster_seed'])
        if len(roster)!=cfg['total_pairs']:raise ValueError('Keep the complete fixed roster')
        limit=3*cfg['pilot_pairs_per_kind'] if args.phase=='pilot' else len(roster)
        results=[];fresh=0
        for i,row in enumerate(roster[:limit]):
            guard();path=pair_path(row)
            if path.exists():
                if args.phase!='verify' and not args.resume:raise ValueError('Explicit unchanged pair resume required')
                result=json.loads(path.read_text())
                if any(result.get(k)!=v for k,v in dict(registration_sha256=sha(PUBLIC/'registration.json'),**row).items()):
                    raise ValueError('Changed saved pair identity')
            else:
                if args.phase=='verify':raise ValueError('Missing complete pair receipt')
                begin=time.monotonic();result=match_pair(row,features,frames,matcher)
                result['match_seconds']=time.monotonic()-begin;once(path,result);fresh+=1
            if args.phase=='verify' and i<cfg['verification_fresh_pair_prefix']:
                if match_pair(row,features,frames,matcher)!={k:v for k,v in result.items() if k!='match_seconds'}:
                    raise ValueError('Fixed fresh pair-prefix mismatch')
            results.append(result)
            if (i+1)%50==0 or i+1==limit:beat(state='pair_screen',complete=i+1,total=len(roster),new_pairs=fresh)
        if args.phase=='pilot':
            timings={k:float(np.mean([r['match_seconds'] for r in results if r['kind']==k]))
                     for k in ('moving_to_dut','drone_to_dut','within_dut')}
            once(end,dict(pid=os.getpid(),seconds=time.monotonic()-started,clip_pairs=len(results),
                per_kind_mean_seconds=timings,estimated_full_matching_seconds=sum(timings[k]*cfg[k+'_pairs'] for k in timings),
                full_screen_complete=False,new_pairs=fresh))
        else:
            summary=summarize(results,analysis,dut,receipts)
            once(PUBLIC/'analysis.json',summary)
            once(end,dict(pid=os.getpid(),seconds=time.monotonic()-started,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                result_source='cached_verified_all_receipts_plus28fresh_feature_replays_and60fresh_pairs' if args.phase=='verify'
                              else 'fresh_run_full_roster_with_fixed_pilot_checkpoint_reuse',
                new_pairs=fresh,cached_pairs=len(results)-fresh,analysis_sha256=sha(PUBLIC/'analysis.json'),
                all28raw_and_prepared_features_reextracted_exact=args.phase=='verify',
                all696parent_feature_views_hash_verified=True,
                all6874receipts_and_graph_exact=args.phase=='verify',
                freshly_recomputed_pair_prefix=cfg['verification_fresh_pair_prefix'] if args.phase=='verify' else 0,
                source_roles_assigned=False,new_model_training=False,predictive_outcomes_read=False))
        beat(state='complete',phase=args.phase,clip_pairs=len(results))


if __name__=='__main__':main()
