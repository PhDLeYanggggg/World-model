"""Forward portrait-axis adapter for the frozen complete source-audit pipeline."""
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scripts import audit_m3w_movingdronecrowd_background as base
from src.evaluation.m3w_movingdronecrowd_orientation import canonical_rgb_boxes

PARENT_PUBLIC,PARENT_PRIVATE=base.PUBLIC,base.PRIVATE
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/movingdronecrowd_background_v2'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/movingdronecrowd_background_v2'
CONFIG=ROOT/'configs/m3w_movingdronecrowd_background_v2.json'
OLD_PRIVATE=base.OLD_PRIVATE
sha,intake,moving_node=base.sha,base.intake,base.moving_node
FILES=(*base.FILES,Path(__file__),CONFIG,PUBLIC/'protocol.md',PARENT_PUBLIC/'geometry_failure.md',
       ROOT/'src/evaluation/m3w_movingdronecrowd_orientation.py',ROOT/'tests/test_m3w_movingdronecrowd_orientation.py')


def registration(parent_values):
    cfg=json.loads(CONFIG.read_text())
    manifest=json.loads((PARENT_PUBLIC/'image_manifest.json').read_text())
    portraits=[r for r in manifest['views'] if r['source_dimensions'][0]<r['source_dimensions'][1]]
    if (len(portraits)!=cfg['portrait_views_to_reextract']
            or sum(r['source_JPEG_bytes'] for r in portraits)!=cfg['portrait_source_JPEG_bytes']):
        raise ValueError('Only the observed fixed portrait subset may change')
    parent_reg=json.loads((PARENT_PUBLIC/'registration.json').read_text())
    if sha(base.ROOT/base.REGISTRY)!=parent_reg['reservation_sha256']:
        raise ValueError('Preserve source reservations')
    value=dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},configuration=cfg,
        parent_registration_sha256=sha(PARENT_PUBLIC/'registration.json'),
        parent_image_manifest_sha256=sha(PARENT_PUBLIC/'image_manifest.json'),
        parent_acquisition_sha256=sha(PARENT_PUBLIC/'acquisition.json'),
        parent_timing_pilot_sha256=sha(PARENT_PUBLIC/'pilot.json'),
        view_selection=parent_reg['view_selection'],portrait_nodes=sorted({r['node'] for r in portraits}),
        source_roles_and_matching_criteria_unchanged=True,old_pair_receipts_reused=False,
        landscape_views_exact_reuse=339,portrait_views_new_extraction=21,
        parent_incomplete_matching_not_promoted=True,new_model_training=False,
        predictive_outcomes_read=False,stage5c_executed=False,smc_enabled=False)
    base.once(PUBLIC/'registration.json',base.stable(value))
    return (cfg,*parent_values[1:])


def acquire_corrected(cfg,source,source_cfg,selection,matcher,resume,guard,beat):
    if (PUBLIC/'acquisition.json').exists():raise ValueError('Completed v2 acquisition is immutable')
    if list((PRIVATE/'features').glob('*/*.npz')) and not resume:
        raise ValueError('Explicit unchanged portrait resume required')
    parent=json.loads((PARENT_PUBLIC/'image_manifest.json').read_text())
    previous={(r['node'],r['slot']):r for r in parent['views']}
    expected_inventory=json.loads((intake.PRIVATE/'archive_inventory.json').read_text())
    rows=[];fresh=0;reused_landscape=0;started=time.monotonic()
    with base.PublicRangeZip(source['archive_url'],source_cfg['archive_bytes'],
            maximum_bytes=cfg['maximum_network_bytes_per_acquisition_invocation']) as stream:
        with zipfile.ZipFile(stream) as archive:
            if intake.inventory(archive)!=expected_inventory:raise ValueError('Changed remote source inventory')
            for number,row in enumerate(selection,1):
                guard();old=previous[row['node'],row['slot']];old_path=ROOT/old['path']
                if sha(old_path)!=old['sha256']:raise ValueError('Parent source view changed')
                if old['source_dimensions'][0]>=old['source_dimensions'][1]:
                    rows.append(dict(old,preparation_variant='parent_landscape_exact_reuse'))
                    reused_landscape+=1
                else:
                    path=base.feature_path(row['node'],row['slot'])
                    expected=dict(registration_sha256=sha(PUBLIC/'registration.json'),
                        node=row['node'],slot=row['slot'],canonical_rotation_counterclockwise_degrees=90)
                    if path.exists():
                        value,meta=base.read_view(path,expected,matcher)
                    else:
                        raw=archive.read(row['member']['name'])
                        if hashlib.sha256(raw).hexdigest()!=old['source_JPEG_sha256']:
                            raise ValueError('Portrait bytes differ from frozen v1 source')
                        with Image.open(io.BytesIO(raw)) as image:
                            if image.width*image.height>cfg['maximum_image_pixels']:raise ValueError('Oversized portrait source')
                            rgb=np.asarray(image.convert('RGB'))
                        annotation=source['members'][row['annotation_member']]
                        body=(ROOT/annotation['path']).read_bytes()
                        if hashlib.sha256(body).hexdigest()!=annotation['sha256']:raise ValueError('Changed head masks')
                        boxes=base.frame_boxes(body,row['annotation_frame_zero_based'])
                        canonical,rectangles,transform,rotation=canonical_rgb_boxes(rgb,boxes)
                        gray,mask=base.prepare_raster(canonical,rectangles,matcher)
                        value=base.features_from_raster(gray,mask,matcher)
                        base.exact_features(base.extract_features(canonical,rectangles,matcher),value)
                        _,parent_meta=base.read_view(old_path,{},matcher)
                        meta=dict(parent_meta,**expected,raster_sha256=base.raster_hash(gray,mask),
                            canonical_source_dimensions=[canonical.shape[1],canonical.shape[0]],
                            canonical_pixel_center_transform=transform.tolist(),
                            parent_view_payload_sha256=old['sha256'],
                            keypoints=len(value['xy']),background_fraction=value['background_fraction'],
                            RGB_to_prepared_features_exact=True)
                        path.parent.mkdir(parents=True,exist_ok=True);pending=path.with_suffix('.pending')
                        with pending.open('wb') as f:
                            np.savez_compressed(f,gray=gray,mask=mask,xy=value['xy'],
                                descriptors=value['descriptors'] if value['descriptors'] is not None else np.empty((0,128),np.float32),
                                background_fraction=np.asarray(value['background_fraction']),
                                metadata_json=np.asarray(json.dumps(meta,sort_keys=True)))
                        pending.replace(path);fresh+=1
                    rows.append(dict(**meta,path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size,
                                     preparation_variant='portrait_axis_corrected'))
                if number%15==0 or number==len(selection):
                    beat(state='portrait_geometry_repair',complete=number,total=len(selection),
                         new_portrait_views=fresh,landscape_reused=reused_landscape,transferred_bytes=stream.transferred_bytes)
        transferred,requests_used=stream.transferred_bytes,stream.range_requests
    with base.PublicRangeZip(source['archive_url'],source_cfg['archive_bytes'],
            maximum_bytes=cfg['maximum_network_bytes_per_acquisition_invocation']-transferred) as check:
        with zipfile.ZipFile(check) as archive:
            if intake.inventory(archive)!=expected_inventory:raise ValueError('Changed remote inventory after portraits')
        transferred+=check.transferred_bytes;requests_used+=check.range_requests
    manifest=dict(result_source='fresh21portrait_reextractions_with339exact_landscape_reuse',
        registration_sha256=sha(PUBLIC/'registration.json'),views=rows,
        parent_image_manifest_sha256=sha(PARENT_PUBLIC/'image_manifest.json'),
        remote_inventory_rechecked_exact=True,full_archive_sha256_verified=False,
        network_payload_range_bytes_this_invocation=transferred,range_requests_this_invocation=requests_used,
        new_views_this_invocation=fresh,cached_views_this_invocation=len(rows)-fresh,
        landscape_views_reused_exact=reused_landscape,raw_source_JPEG_retained=False,
        new_model_training=False,predictive_outcomes_read=False)
    base.once(PUBLIC/'image_manifest.json',base.stable(manifest))
    base.once(PUBLIC/'acquisition.json',dict(pid=os.getpid(),seconds=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        image_manifest_sha256=sha(PUBLIC/'image_manifest.json'),new_views=fresh,cached_views=len(rows)-fresh))


def main():
    # Explicit process-local reuse; no frozen parent file or cache is modified.
    values=base.admission()
    base.PUBLIC,base.PRIVATE,base.CONFIG,base.FILES=PUBLIC,PRIVATE,CONFIG,FILES
    base.admission=lambda:registration(values)
    base.acquire=acquire_corrected
    base.main()


if __name__=='__main__':main()
