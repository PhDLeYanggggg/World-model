"""Static training-camera correspondences, not metric trajectory validation."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import platform
import time
import zipfile

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 numerical environment required')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/source_projection_audit_v1'
ARCHIVE=ROOT/'external_data/EuropeanSquares_source_audit/02.Resources.zip'
OLD=ROOT/'outputs/publication_readiness_2026_09/european_static_scene_audit_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolution(name):
    for token,size in (('QHD+',(2688,1520)),('UHD',(3840,2160)),('QHD',(2560,1440)),
                       ('FHD',(1920,1080)),('HD+',(1280,1024)),('HD',(1280,720))):
        if token in PurePosixPath(name).name:
            return np.array(size,float)
    raise ValueError('Unknown metadata resolution; do not guess raw-video dimensions')


def points(content):
    # QGIS CRS comments contain a non-UTF8 degree symbol; numeric CSV is ASCII.
    text=content.decode('latin1')
    data=list(csv.DictReader(io.StringIO('\n'.join(x for x in text.splitlines() if not x.startswith('#')))))
    active=[r for r in data if float(r['enable'])>0]
    image=np.array([[float(r['sourceX']),-float(r['sourceY'])] for r in active])
    mapped=np.array([[float(r['mapX']),float(r['mapY'])] for r in active])
    if len(active)<4 or not np.isfinite(image).all() or not np.isfinite(mapped).all():
        raise ValueError('At least four enabled finite publisher pairs required')
    return image,mapped,dict(total=len(data),enabled=len(active),
        declared_EPSG3857='ID["EPSG",3857]' in text,
        comment_encoding='lossless_latin1_numeric_fields_ASCII',source_y_negated=True)


def project(x,h):
    value=np.column_stack((x,np.ones(len(x))))@h.T
    if not np.isfinite(value).all() or np.any(np.abs(value[:,2])<1e-12):
        raise ValueError('Singular/nonfinite projective denominator')
    return value[:,:2]/value[:,2,None]


def homography(image,mapped):
    origin=mapped.mean(0)
    h,_=cv2.findHomography(image.astype(float),(mapped-origin).astype(float),method=0)
    if h is None or not np.isfinite(h).all() or np.linalg.matrix_rank(h)!=3:
        raise ValueError('No nondegenerate point-fit homography')
    # Keep destinations near zero rather than quantize large Mercator ordinates.
    return h,origin


def stats(values):
    x=np.asarray(values,float);ok=np.isfinite(x)
    return dict(required=len(x),finite=int(ok.sum()),undefined=int((~ok).sum()),
        median=float(np.median(x[ok])) if ok.any() else None,
        maximum=float(x[ok].max()) if ok.any() else None,
        RMSE=float(np.sqrt(np.square(x[ok]).mean())) if ok.any() else None)


def polygon(content,columns):
    data=list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
    return np.array([[float(r[c]) for c in columns] for r in data])


def unique(members,camera):
    found=[n for n in members if int(PurePosixPath(n).name.split('_',1)[0])==camera]
    if len(found)!=1:
        raise ValueError(f'Camera{camera}: expected one exact metadata member, got{len(found)}')
    return found[0]


def audit():
    old=json.loads((OLD/'audit.json').read_text());check=json.loads((OLD/'verification.json').read_text())
    if not check['exact_replay'] or check['audit_sha256']!=sha(OLD/'audit.json'):
        raise ValueError('Verified existing fitting-camera identity required')
    cameras=sorted({int(PurePosixPath(r['source_member']).name.split('_',1)[0]) for r in old['rows']})
    rows=[];cv2.setNumThreads(4)
    with zipfile.ZipFile(ARCHIVE) as z:
        names=[n for n in z.namelist() if '__MACOSX' not in n]
        point_members=[n for n in names if n.endswith('.points')]
        raw=[n for n in names if PurePosixPath(n).parent.as_posix()=='02.Resources/02.Place_polygons' and n.endswith('.csv')]
        geo=[n for n in names if '/georeferenced/' in n and n.endswith('.csv')]
        for camera in cameras:
            pm,rm,gm=[unique(pool,camera) for pool in (point_members,raw,geo)]
            image,mapped,info=points(z.read(pm));h,origin=homography(image,mapped)
            error=np.linalg.norm(project(mapped-origin,np.linalg.inv(h))-image,axis=1)
            roundtrip=np.linalg.norm(project(project(image,h),np.linalg.inv(h))-image,axis=1)
            loo=[]
            for i in range(len(image)):
                keep=np.arange(len(image))!=i
                if keep.sum()<4:
                    loo.append(np.nan);continue
                try:
                    hh,oo=homography(image[keep],mapped[keep])
                    loo.append(float(np.linalg.norm(project((mapped[i]-oo)[None],np.linalg.inv(hh))[0]-image[i])))
                except (ValueError,np.linalg.LinAlgError,cv2.error):
                    loo.append(np.nan)
            poly=polygon(z.read(rm),('x','y'));pair=polygon(z.read(gm),('x','y','real_world_x','real_world_y'))
            if poly.shape!=pair[:,:2].shape or not np.array_equal(poly,pair[:,:2]):
                raise ValueError('Raw/publisher-georeferenced polygon vertices do not align')
            native_size=resolution(pm);poly_size=resolution(rm)
            hpoly=h@np.diag([native_size[0]/poly_size[0],native_size[1]/poly_size[1],1.])
            polygon_error=np.linalg.norm(project(pair[:,2:]-origin,np.linalg.inv(hpoly))-poly,axis=1)
            rows.append(dict(camera_id=camera,point_member=pm,polygon_member=rm,georeferenced_polygon_member=gm,
                member_sha256={n:hashlib.sha256(z.read(n)).hexdigest() for n in (pm,rm,gm)},
                point_info=info,point_filename_frame_size=native_size.astype(int).tolist(),
                polygon_filename_frame_size=poly_size.astype(int).tolist(),
                filename_resolution_mismatch=bool(np.any(native_size!=poly_size)),
                point_fit_inverse_error_native_pixels=stats(error),
                leave_one_point_out_inverse_error_native_pixels=stats(loo),
                matrix_roundtrip_error_native_pixels=stats(roundtrip),
                supplied_polygon_match_error_polygon_pixels=stats(polygon_error),
                raw_trajectory_frame_size_verified=False,metric_trajectory_claim_allowed=False))
    return dict(result_source='fresh_static_metadata_point_and_polygon_reprojection_audit',
        bindings=dict(script_sha256=sha(__file__),archive_sha256=sha(ARCHIVE),existing_fitting_audit_sha256=sha(OLD/'audit.json')),
        scope='fourteen_existing_SOURCE_fitting_cameras_metadata_only_no_trajectory_label_read',
        cameras=len(rows),rows=rows,opencv_version=cv2.__version__,numpy_version=np.__version__,
        original_units_or_schema_changed=False,forecast_or_policy_evaluated=False,
        metric_trajectory_claim_allowed=False,effective_seconds_claim_allowed=False,
        independent_roles_opened=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)


def render(d):
    lines=['# Static Camera Projection Audit','','## Material Passport','',
        'Fresh_run: enabled publisher control-point fits, leave-one-point checks and',
        'reprojection against supplied polygons for all14 previously admitted fitting',
        'cameras. Cached_verified: original fitting-camera identity audit. Not_run:',
        'trajectory forecasting, policy scoring, raw-video resolution verification,',
        'physical scale validation, new dynamics training or independent-role access.','',
        'QGIS sourceY is negated and point-file coordinates are rescaled using explicit',
        'filename resolution to the polygon filename frame. OpenCV least squares uses',
        'centered destination coordinates, no RANSAC or target-selected tuning. CRS',
        'EPSG3857 is publisher metadata, not verified ground-truth metric accuracy.','',
        '| Camera | Enabled points | Point / polygon frame | Point inverse RMSE px | LOO inverse RMSE px (undefined) | Supplied polygon inverse RMSE px |',
        '|---|---:|---|---:|---:|---:|']
    fmt=lambda x:'not_estimable' if x is None else f'{x:.6g}'
    for r in d['rows']:
        a=r['point_fit_inverse_error_native_pixels'];b=r['leave_one_point_out_inverse_error_native_pixels'];c=r['supplied_polygon_match_error_polygon_pixels']
        lines.append(f"| {r['camera_id']} | {r['point_info']['enabled']} | {r['point_filename_frame_size']} / {r['polygon_filename_frame_size']} | {fmt(a['RMSE'])} | {fmt(b['RMSE'])} ({b['undefined']}) | {fmt(c['RMSE'])} |")
    lines += ['','These are internal static resource checks, not independent surveyed geometry.',
        'The raw benchmark uses bounding-box midpoints; the publisher processing code',
        'applies mapping to box bottoms. Raw recording frame dimensions are not supplied',
        'by this audit. A point/polygon fit does not resolve that observation-convention',
        'difference or authorize converting reported trajectory errors to metres.','',
        'Next: verify raw recording coordinate resolution and observation convention',
        'before a separately registered projective forecast proposal. Do not silently',
        'replace existing scene features, labels, normalizers or benchmarks. No new',
        'forecast result, metric/effective-second, true-3D or foundation claim follows.','',
        'The publisher implementation documents resolution rescaling, sourceY sign and',
        'box-bottom mapping: [primary source](https://github.com/kaktusracing/pedestrian_trajectories/blob/main/csv_functions_v2/calc_geo_df.py).',
        'Its code was read, not executed or copied into the repository. Stage5C and SMC',
        'remain off, and independent roles and deployment remain unchanged.','']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('run','verify'));args=parser.parse_args()
    started=time.monotonic();d=audit();PUBLIC.mkdir(parents=True,exist_ok=True);p=PUBLIC/'audit.json'
    if args.phase=='verify':
        if d!=json.loads(p.read_text()):raise ValueError('Exact static projection audit differs')
        (PUBLIC/'verification.json').write_text(json.dumps(dict(exact_replay=True,cameras=d['cameras'],audit_sha256=sha(p)),indent=2)+'\n')
    else:
        if p.exists():raise ValueError('Completed static audit immutable')
        p.write_text(json.dumps(d,indent=2)+'\n');(PUBLIC/'readout.md').write_text(render(d))
    print(json.dumps(dict(phase=args.phase,pid=os.getpid(),seconds=time.monotonic()-started,cameras=d['cameras'],forecast_evaluated=False)))
