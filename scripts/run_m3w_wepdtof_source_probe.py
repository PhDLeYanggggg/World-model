"""Qualify WEPDTOF support and run a causal camera-only pilot."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import platform
import ssl
import sys
import time
from zipfile import ZipFile
from zlib import crc32

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 .venv-pytorch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data_unification import m3w_wepdtof_source as data
from src.data_unification.m3w_public_zip_range import RangeArchive
from scripts.run_m3w_waterloo_raw_prefix import WEIGHTS, immutable, sha


URL = 'https://www.bu.edu/vip/files/WEPDTOF.zip'
LENGTH = 5435073011
VERSION = 'cegTXuCBDaUeFXl25NZl4jnyjdL3TS54'
OUT = ROOT/'outputs/publication_readiness_2026_09/wepdtof_source_probe_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/OUT.name
LOCAL = ROOT/'external_data/WEPDTOF_source_probe'
PILOT, STRIDE, WINDOWS = 'convenience_store', 4, 12
MIN_HISTORIES, MIN_QUERIES = 8, 3
REG = OUT/'registration_v3.json'


def archive(cafile):
    return RangeArchive(URL, LENGTH, context=ssl.create_default_context(cafile=cafile),
                        block_size=1 << 18, byte_limit=256 << 20)


def check_version(a):
    if a.version != VERSION:
        raise ValueError('Unexpected publisher archive version')


def cached_member(z, a, name, path):
    info = z.getinfo(name)
    if not path.exists():
        b = z.read(info)
        check_version(a)
        tmp = path.with_suffix('.pending')
        tmp.write_bytes(b)
        tmp.replace(path)
    else:
        b = path.read_bytes()
    if len(b) != info.file_size or crc32(b) != info.CRC:
        raise ValueError(f'Archive CRC mismatch: {name}')
    return b


def register(cafile):
    with archive(cafile) as a, ZipFile(a) as z:
        check_version(a)
        clips = data.catalog(z.namelist())
        scenes = {data.source(k) for k in clips}
        if len(clips) != 16 or len(scenes) != 14 or sum(map(len,clips.values())) != 10544:
            raise ValueError('Published archive population changed')
        starts = data.pilot_starts(clips[PILOT], STRIDE, WINDOWS)
        names = [[clips[PILOT][s+i*STRIDE] for i in range(8)] for s in starts]
        record = dict(archive_url=URL, archive_bytes=LENGTH, archive_version=VERSION, clips=16, sources=14,
            frame_count=10544, pilot=PILOT, stride=STRIDE, starts=starts, image_members=names,
            selection='even quantiles of disjoint 20-step frame intervals from archive names only',
            weight_sha256=sha(WEIGHTS), support_minima=[MIN_HISTORIES, MIN_QUERIES], cpu_threads=4, workers=0,
            code={str(x.relative_to(ROOT)):sha(x) for x in (Path(__file__),Path(data.__file__),
                  ROOT/'scripts/run_m3w_waterloo_raw_prefix.py', ROOT/'src/world_model/m3w_streaming_boxes.py',
                  ROOT/'src/data_unification/m3w_public_zip_range.py', OUT/'protocol.md',
                  OUT/'protocol_amendment.md', OUT/'frame_amendment.md')},
            previous_registration_sha256=[sha(OUT/k) for k in ('registration.json','registration_v2.json')],
            original_2pct=True, independent_confirmation=False)
        if not record['weight_sha256'].startswith('907ea3f9'):
            raise ValueError('Require the frozen MobileNet detector')
        initial = json.loads((OUT/'registration.json').read_text())
        if any(initial[k]!=record[k] for k in ('image_members','starts','stride','pilot','weight_sha256')):
            raise ValueError('Original camera schedule changed')
        immutable(REG,record)
    return record


def audit(cafile):
    rows = []
    begin = time.monotonic()
    with archive(cafile) as a, ZipFile(a) as z:
        check_version(a)
        clips = data.catalog(z.namelist())
        for clip, frames in sorted(clips.items()):
            path = LOCAL/f'{clip}.json'
            b = cached_member(z,a,f'WEPDTOF/annotations/{clip}.json',path)
            people, quality = data.annotation_table(json.loads(b),clip,frames)
            row = dict(clip=clip, source=data.source(clip), frames=len(frames), extent=[min(frames),max(frames)],
                track_ids=len(people), rows=sum(map(len,people.values())), annotation_sha256=sha(path), quality=quality,
                support={str(s):data.support(people,frames,s) for s in (1,4,8)})
            rows.append(row)
            beat(phase='archive_support', clip=clip, done=len(rows), total=len(clips), seconds=time.monotonic()-begin)
    total = {k:{q:sum(r['support'][k][q] for r in rows) for q in rows[0]['support'][k]} for k in ('1','4','8')}
    immutable(OUT/'support.json',dict(registration_sha256=sha(REG), clips=rows, total=total,
        camera_images_in_audit=False, geometry_used_in_audit=False, fps_verified=False, original_2pct=True,
        nonpilot_model_scored=False, independent_confirmation=False))
    print(total,flush=True)


def beat(**v):
    z = dict(pid=os.getpid(), **v)
    temp = PRIVATE/'heartbeat.pending'
    temp.write_text(json.dumps(z)+'\n')
    temp.replace(PRIVATE/'heartbeat.json')
    print(z,flush=True)


def run_camera(cafile, reg, cold):
    import cv2
    import torch
    from scripts.run_m3w_waterloo_raw_prefix import detect, detector, file_digest_array
    from src.world_model.m3w_streaming_boxes import StreamingTracks

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    start = time.monotonic()
    reg_sha = sha(REG)
    saved = []
    net = None
    with archive(cafile) as a, ZipFile(a) as z:
        check_version(a)
        schedule = list(zip(reg['starts'],reg['image_members'],strict=True))
        for k, (first,names) in enumerate(schedule[:1] if cold else schedule,1):
            checkpoint = PRIVATE/f'window_{first:06}.json'
            frozen = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
            if frozen is not None and (frozen['registration'] != reg_sha or frozen['start'] != first
                                       or [f['member'] for f in frozen['frames']] != names):
                raise ValueError('Incompatible sensor cache')
            if frozen is not None and not cold:
                row = frozen
            else:
                if net is None:
                    net = detector()
                tracker = StreamingTracks()
                row = dict(registration=reg_sha,start=first,frames=[],paths=[])
                for j,name in enumerate(names):
                    b = cached_member(z,a,name,LOCAL/Path(name).name)
                    image = cv2.imdecode(np.frombuffer(b,dtype=np.uint8),cv2.IMREAD_COLOR)
                    if image is None or image.shape != (720,960,3):
                        raise ValueError('Unexpected Convenience Store image geometry')
                    d = detect(net,image)
                    tracks, forecasts = tracker.step(j,d[d[:,-1]==1])
                    row['frames'].append(dict(member=name,frame=first+j*STRIDE,
                        pixel_sha256=file_digest_array(image),rows=tracks.tolist()))
                    if j == 7:
                        row['paths'] = [[int(i), cv.tolist(), ols.tolist()] for i,cv,ols in forecasts]
                if cold:
                    if frozen != row:
                        raise ValueError('Cold detector prefix disagrees with frozen predictions')
                else:
                    immutable(checkpoint,row)
            saved.append(dict(start=first,frames=len(row['frames']),forecasts=len(row['paths']),
                detections=sum(len(f['rows']) for f in row['frames']),cache_sha256=sha(checkpoint)))
            beat(phase='camera_cold' if cold else 'camera_frozen', done=k,total=1 if cold else len(schedule),
                 seconds=time.monotonic()-start,reused=frozen is not None and not cold, archive_bytes_read=a.total)
    name = 'cold_replay.json' if cold else 'camera_freeze.json'
    immutable(OUT/name,dict(registration_sha256=reg_sha,annotation_geometry_read=False,
        cold_prefix=cold, windows=saved, independent_confirmation=False))


def associate(rows, labels):
    ids = sorted(labels)
    if not len(rows) or not ids:
        return set()
    b = np.stack([labels[k] for k in ids])
    c = .5*(rows[:,1:3]+rows[:,3:5])
    d = np.linalg.norm(c[:,None]-b[None,:,:2],axis=2)
    r = d / (.5*np.linalg.norm(b[:,2:4],axis=1)).clip(1.)
    i,j = linear_sum_assignment(np.where(r <= 1.,r,1e6))
    return {(int(rows[a,0]),ids[t]) for a,t in zip(i,j,strict=True) if r[a,t] <= 1.}


def measures(samples):
    # Equal weights for matched queries, including those with partial/absent futures.
    mass = np.zeros(2)
    num = np.zeros(2)
    errors, endpoints = [], []
    counts = defaultdict(int)
    for y, v, p in samples:
        if not len(v):
            continue
        e = np.linalg.norm(p-y[:,None],axis=-1)
        g = e[:,0]-e[:,1]
        gap = np.linalg.norm(p[:,0]-p[:,1],axis=-1)
        known = v.any(1)
        complete = v.all(1)
        counts['matched'] += len(v)
        counts['complete'] += int(complete.sum())
        counts['absent'] += int((~known).sum())
        counts['partial'] += int((known&~complete).sum())
        counts['endpoint'] += int(v[:,-1].sum())
        mass[0] += np.where(v,e[:,0],0.).sum()/len(v)
        num += np.array([np.where(v,g,0.).sum(),np.where(v,g,-gap).sum()])/len(v)
        if known.any():
            by = np.where(v[:,None],e,0.).sum(-1)/v.sum(1).clip(1)[:,None]
            errors.append(by[known].mean(0))
        if v[:,-1].any():
            endpoints.append(e[v[:,-1],:,-1].mean(0))
    scores = lambda x: np.mean(x,axis=0).tolist() if x else None
    return dict(counts=dict(counts), available_ade_cv_ols4_px=scores(errors),
        available_fde_cv_ols4_px=scores(endpoints),
        observed_and_lower_ols4_vs_cv_pp=(100.*num/mass[0]).tolist() if mass[0]>0. else None)


def evaluate(reg):
    from scripts.run_m3w_acme_gmu_camera import rollout

    reg_sha = sha(REG)
    f = OUT/'camera_freeze.json'
    freeze = json.loads(f.read_text())
    audit = json.loads((OUT/'support.json').read_text())
    if (freeze['registration_sha256'] != reg_sha or audit['registration_sha256'] != reg_sha
            or freeze['annotation_geometry_read'] or [z['start'] for z in freeze['windows']] != reg['starts']):
        raise ValueError('Require a complete outcome-blind camera freeze')
    pilot, = [c for c in audit['clips'] if c['clip'] == PILOT]
    path = LOCAL/f'{PILOT}.json'
    if sha(path) != pilot['annotation_sha256'] or pilot['quality']['unlisted_images'] != 0:
        raise ValueError('Pilot labels changed since source audit')
    doc = json.loads(path.read_text())
    frames = {data.frame_number(i['file_name'],PILOT):i['file_name'] for i in doc['images']}
    people, _ = data.annotation_table(doc,PILOT,frames,geometry=True)
    rows, scored = [], []
    for item in freeze['windows']:
        s = item['start']
        p = PRIVATE/f'window_{s:06}.json'
        if sha(p) != item['cache_sha256']:
            raise ValueError('Frozen prediction bytes changed')
        w = json.loads(p.read_text())
        pairs, labels = [], []
        for j,f0 in enumerate(w['frames']):
            t = s+j*STRIDE
            if f0['frame'] != t:
                raise ValueError('Camera and reference indices differ')
            ann = {k:z[t] for k,z in people.items() if t in z}
            pairs.append(associate(np.asarray(f0['rows'],dtype=float).reshape(-1,7),ann))
            labels.append(set(ann))
        stable = sorted(set.intersection(*pairs))
        pred = {i:np.stack((c,o)) for i,c,o in w['paths']}
        if any(i not in pred for i,_ in stable):
            raise ValueError('Stable track has no frozen forecast')
        rows.append(dict(start=s, reference_histories=len(set.intersection(*labels)), online_histories=len(pred),
                         matched_histories=len(stable), frame_pair_counts=[len(z) for z in pairs]))
        scored.append((s,stable,pred))
    n = sum(r['matched_histories'] for r in rows)
    q = sum(r['matched_histories']>0 for r in rows)
    gate = n >= MIN_HISTORIES and q >= MIN_QUERIES
    scores = None
    if gate:
        score = dict(online=[],label_history=[])
        for s,stable,p in scored:
            if not stable:
                continue
            future = [[people[j].get(s+k*STRIDE) for k in range(8,20)] for _,j in stable]
            y = np.array([[z[:2] if z is not None else np.zeros(2) for z in row] for row in future])
            v = np.array([[z is not None for z in row] for row in future])
            score['online'].append((y,v,np.stack([p[i] for i,_ in stable])))
            score['label_history'].append((y,v,np.stack([rollout([people[j][s+k*STRIDE][:2] for k in range(8)])
                                                       for _,j in stable])))
        scores = {k:measures(v) for k,v in score.items()}
    r = dict(registration_sha256=reg_sha,camera_freeze_sha256=sha(f),windows=rows, matched_histories=n,
        matched_windows=q,support_gate=gate, scores=scores, future_geometry_scored=gate,
        risk_evaluation='not_run', original_2pct=True, independent_confirmation=False)
    immutable(OUT/'pilot_readout.json',r)
    print({k:v for k,v in r.items() if k!='windows'},flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('audit','produce','verify','evaluate'))
    p.add_argument('--cafile',default='/etc/ssl/cert.pem')
    a = p.parse_args()
    for directory in (OUT,PRIVATE,LOCAL):
        directory.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'probe.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reg = register(a.cafile)
        if a.phase == 'audit':
            audit(a.cafile)
        elif a.phase == 'evaluate':
            evaluate(reg)
        else:
            run_camera(a.cafile,reg,cold=a.phase=='verify')


if __name__ == '__main__':
    main()
