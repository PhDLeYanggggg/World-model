"""Freeze image-only WEPDTOF external histories before any outcome readout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import ssl
import sys
import threading
import time
from urllib.error import URLError
from zipfile import ZipFile
from zlib import crc32

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_wepdtof_rapid_probe as detector
from src.world_model.m3w_streaming_boxes import StreamingTracks
from scripts.run_m3w_waterloo_raw_prefix import file_digest_array

base, np = detector.base, detector.np
OUT = base.OUT.parent/'wepdtof_external_camera_v1'
PRIVATE = base.PRIVATE.parent/OUT.name
LOCAL = base.LOCAL/'external_image_prefixes'
IO_WORKERS = 6
LOOKAHEAD = 3


def register():
    reg, _ = detector.register()
    plan_path = base.OUT/'blind_camera_plan.json'
    roles_path = base.OUT/'external_roles.json'
    plan, roles = map(detector.load,(plan_path,roles_path))
    if (plan['role_sha256']!=base.sha(roles_path) or not roles['roles_frozen']
            or roles['external_labels_opened_for_scoring'] or plan['outcomes_read'] or plan['pixels_read']
            or plan['archive_version']!=base.VERSION or plan['observe']!=8 or plan['predict']!=12
            or plan['external_sources']!=13 or not roles['original_2pct']):
        raise ValueError('Require approved source roles and blind camera plan')
    expected = {c:s for s,r in roles['groups'].items() if r['role']=='frozen_external_once' for c in r['clips']}
    actual = {r['clip']:r['source'] for r in plan['clips']}
    if actual!=expected or len(actual)!=len(plan['clips']) or base.PILOT in actual:
        raise ValueError('External clip roster differs from approved sources')
    jobs = [(r['source'],r['clip'],w) for r in plan['clips'] for w in r['windows']]
    names = [m['member'] for _,c,w in jobs for m in w['past']
             if m['member'].startswith(f'WEPDTOF/frames/{c}/')]
    if (len(jobs)!=plan['windows'] or len(names)!=8*len(jobs)
            or len(set(names))!=len(names) or len(names)!=plan['image_prefix_frames']):
        raise ValueError('External windows do not have unique observed frames')
    for _,clip,w in jobs:
        actual = [base.data.frame_number(m['member'],clip) for m in w['past']]
        if actual != [w['start']+i*reg['stride'] for i in range(8)]:
            raise ValueError('Past frame grid changed')
    p = OUT/'registration.json'
    base.immutable(p,dict(plan_sha256=base.sha(plan_path), roles_sha256=base.sha(roles_path),
        pilot_detector_sha256=base.sha(detector.REG), code_sha256=base.sha(Path(__file__)),
        history_steps=8, forecast_steps=12, stride=reg['stride'], backend='arm64 torch CPU',
        torch_threads=4, dataloader_workers=0, network_threads=IO_WORKERS,
        no_annotation_io=True, original_2pct=True, scores_opened=False))
    return base.sha(p),jobs


class FramePool:
    """One verified, bounded range handle per network thread."""
    def __init__(self,cafile):
        self.context = ssl.create_default_context(cafile=cafile)
        self.state = threading.local()
        self.handles = []
        self.lock = threading.Lock()

    def handle(self):
        if not hasattr(self.state,'archive'):
            a = base.RangeArchive(base.URL,base.LENGTH,context=self.context,block_size=1<<18,byte_limit=512<<20)
            z = ZipFile(a)
            base.check_version(a)
            self.state.archive = (a,z)
            with self.lock:
                self.handles.append((a,z))
        return self.state.archive

    def read(self,item):
        name = item['member']
        path = LOCAL/Path(name).name
        if not path.exists():
            for retry in range(3):
                try:
                    a,z = self.handle()
                    info = z.getinfo(name)
                    if (info.file_size, f'{info.CRC:08x}')!=(item['bytes'],item['crc32']):
                        raise ValueError('Image member differs from frozen catalog')
                    b = base.cached_member(z,a,name,path)
                    break
                except (OSError,URLError):
                    if retry==2:
                        raise
                    time.sleep(1.+retry)
        else:
            b = path.read_bytes()
        if len(b)!=item['bytes'] or f'{crc32(b):08x}'!=item['crc32']:
            raise ValueError('Image failed frozen checksum')
        return b

    def close(self):
        for a,z in self.handles:
            z.close()
            a.close()


def checkpoint_path(clip,start):
    return PRIVATE/f'{clip}_{start:07}.json'


def check_row(row,key,group,clip,window):
    if (row['registration']!=key or row['source']!=group or row['clip']!=clip or row['start']!=window['start']
            or [r['member'] for r in row['frames']]!=[m['member'] for m in window['past']]):
        raise ValueError('Frozen window does not match camera plan')


def observe(predict,key,group,clip,window,buffers):
    import cv2
    track = StreamingTracks()
    z = dict(registration=key,source=group,clip=clip,start=window['start'],frames=[],paths=[])
    for j,(m,b) in enumerate(zip(window['past'],buffers,strict=True)):
        im = cv2.imdecode(np.frombuffer(b,dtype=np.uint8),cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"Cannot decode {m['member']}")
        rows,pred = track.step(j,predict(im))
        z['frames'].append(dict(member=m['member'],frame=base.data.frame_number(m['member'],clip),
            pixel_sha256=file_digest_array(im),rows=rows.tolist()))
        if j==7:
            for ident,cv,ols in pred:
                boxes = [next(r[1:5] for r in f['rows'] if r[0]==ident) for f in z['frames']]
                h = np.asarray(boxes)
                z['paths'].append(dict(id=int(ident), history=(.5*(h[:,:2]+h[:,2:])).tolist(),
                                       cv=cv.tolist(), ols4=ols.tolist()))
    return z


def beat(**v):
    z = dict(pid=os.getpid(),**v)
    temp = PRIVATE/'heartbeat.pending'
    temp.write_text(json.dumps(z)+'\n')
    temp.replace(PRIVATE/'heartbeat.json')
    print(z,flush=True)


def run(key,jobs,cafile,cold):
    import cv2
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    start = time.monotonic()
    net, scheduled, summary = None, {}, []
    pool = FramePool(cafile)
    active = jobs[:1] if cold else jobs
    try:
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as io:
            def queue(i):
                if i<len(active):
                    _,c,w = active[i]
                    if cold or not checkpoint_path(c,w['start']).exists():
                        scheduled[i] = [io.submit(pool.read,m) for m in w['past']]

            for i in range(LOOKAHEAD):
                queue(i)
            for i,(g,c,w) in enumerate(active):
                p = checkpoint_path(c,w['start'])
                cached = detector.load(p) if p.exists() else None
                if cached is not None:
                    check_row(cached,key,g,c,w)
                if cached is None or cold:
                    if net is None:
                        net = detector.rapid()
                    row = observe(net,key,g,c,w,[f.result() for f in scheduled.pop(i)])
                    if cold and cached!=row:
                        raise ValueError('Cold replay changed predictions')
                    if not cold:
                        base.immutable(p,row)
                else:
                    row = cached
                queue(i+LOOKAHEAD)
                summary.append(dict(source=g,clip=c,start=w['start'],cache_sha256=base.sha(p),
                    frames=len(row['frames']),forecasts=len(row['paths']),
                    detections=sum(len(f['rows']) for f in row['frames'])))
                beat(phase='cold' if cold else 'observed',done=i+1,total=len(active),source=g,
                     forecast_instances=len(row['paths']),reused=cached is not None and not cold,
                     seconds=round(time.monotonic()-start,3))
    finally:
        pool.close()
    name = 'cold_replay.json' if cold else 'camera_freeze.json'
    totals = {k:sum(z[k] for z in summary) for k in ('frames','forecasts','detections')}
    base.immutable(OUT/name,dict(registration=key,records=summary,totals=totals,
        source_groups=len({z['source'] for z in summary}),windows_with_forecasts=sum(z['forecasts']>0 for z in summary),
        cold=cold, annotations_read=False, labels_scored=False, original_2pct=True,
        independent_confirmation=False))
    beat(phase='complete',done=len(summary),totals=totals,cold=cold)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('produce','verify'))
    p.add_argument('--cafile',default='/etc/ssl/cert.pem')
    a = p.parse_args()
    for d in (OUT,PRIVATE,LOCAL):
        d.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'producer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        key,jobs = register()
        run(key,jobs,a.cafile,a.phase=='verify')


if __name__=='__main__':
    main()
