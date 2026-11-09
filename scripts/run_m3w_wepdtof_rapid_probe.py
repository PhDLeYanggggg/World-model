"""Compare a frozen fisheye detector on the same WEPDTOF pilot frames."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_wepdtof_source_probe as base
from src.world_model.m3w_streaming_boxes import StreamingTracks

np = base.np
CODE = base.LOCAL/'RAPiD'
WEIGHTS = base.LOCAL/'pL1_MWHB1024_Mar11_4000.ckpt'
WEIGHT_URL = 'https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/'+WEIGHTS.name
REV = '3e55294e0e419166bf152a15271c6b2fbd75b5cd'
PRIVATE = base.PRIVATE/'rapid'
REG = base.OUT/'rapid_registration.json'


def load(path):
    return json.loads(path.read_text())


def beat(**values):
    value = dict(pid=os.getpid(), **values)
    tmp = PRIVATE/'heartbeat.pending'
    tmp.write_text(json.dumps(value)+'\n')
    tmp.replace(PRIVATE/'heartbeat.json')
    print(value,flush=True)


def rapid():
    import torch
    from PIL import Image
    # Load the pinned frame-only model with no training imports from this project.
    sys.path.insert(0,str(CODE))
    from api import Detector
    from models.rapid import RAPiD
    if (Path.cwd()/'weights/dark53_imgnet.pth').exists():
        raise ValueError('Unregistered optional RAPiD pretraining file in current directory')
    m = RAPiD(backbone='dark53')
    m.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True)['model'])
    net = Detector(model=m.eval(),input_size=1024,conf_thres=.3)

    def predict(image):
        p = np.asarray(net.detect_one(pil_img=Image.fromarray(image[:,:,::-1].copy())),dtype=float).reshape(-1,6)
        a = np.deg2rad(p[:,4:5])
        r = .5*(p[:,2:4]*np.abs(np.cos(a))+p[:,[3,2]]*np.abs(np.sin(a)))
        return np.column_stack((p[:,:2]-r,p[:,:2]+r,p[:,5],np.ones(len(p))))
    return predict


def register():
    import importlib.metadata
    ref = load(base.REG)
    f = base.OUT/'camera_freeze.json'
    mobile = load(f)
    if mobile['registration_sha256']!=base.sha(base.REG) or mobile['annotation_geometry_read']:
        raise ValueError('Require outcome-blind MobileNet camera freeze')
    g = lambda *a: subprocess.check_output(('git','-C',str(CODE),*a),text=True).strip()
    if g('rev-parse','HEAD') != REV or g('status','--porcelain','--untracked-files=no'):
        raise ValueError('Unexpected publisher source checkout')
    d = dict(parent_camera_freeze=base.sha(f), camera_registration=base.sha(base.REG),
        code_sha256=base.sha(Path(__file__)), weight_url=WEIGHT_URL, weight_sha256=base.sha(WEIGHTS),
        upstream_commit=REV, publisher_weight_training='COCO + MW-R + HABBOF, no WEPDTOF per publisher',
        dependency_versions={k:importlib.metadata.version(k) for k in ('torch','torchvision','pycocotools','Pillow')},
        fixed_confidence=.3, input_size=1024, threads=4, workers=0,
        online_input='one current JPEG at a time; same tracker and association as MobileNet',
        no_model_fitting=True, no_policy_selection=True, original_2pct=True, independent_confirmation=False)
    base.immutable(REG,d)
    return ref,mobile


def verify_source(item):
    f = base.PRIVATE/f"window_{item['start']:06}.json"
    if base.sha(f)!=item['cache_sha256']:
        raise ValueError('MobileNet frame receipt changed')
    return load(f)


def produce(mobile, cold):
    import cv2
    import torch
    from scripts.run_m3w_waterloo_raw_prefix import file_digest_array

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    start = time.monotonic()
    key = base.sha(REG)
    detector, out = None, []
    wanted = mobile['windows'][:1] if cold else mobile['windows']
    for k,item in enumerate(wanted,1):
        orig = verify_source(item)
        p = PRIVATE/f"window_{item['start']:06}.json"
        cached = load(p) if p.exists() else None
        if cached is not None and cached['registration'] != key:
            raise ValueError('Cached RAPiD registration changed')
        if cached is None or cold:
            if detector is None:
                detector = rapid()
            track = StreamingTracks()
            result = dict(start=item['start'],registration=key,frames=[],paths=[])
            for j,src in enumerate(orig['frames']):
                image = cv2.imdecode(np.frombuffer((base.LOCAL/Path(src['member']).name).read_bytes(),
                                                    dtype=np.uint8),cv2.IMREAD_COLOR)
                if image is None or file_digest_array(image)!=src['pixel_sha256']:
                    raise ValueError('Camera image differs across detector arms')
                d = detector(image)
                rows,pred = track.step(j,d)
                result['frames'].append(dict(member=src['member'],frame=src['frame'],
                    pixel_sha256=src['pixel_sha256'],rows=rows.tolist()))
                if j == 7:
                    result['paths'] = [[int(i), c.tolist(), o.tolist()] for i,c,o in pred]
            if cold and cached != result:
                raise ValueError('RAPiD cold inference did not reproduce first window')
            if not cold:
                base.immutable(p,result)
        z = load(p)
        if [(f['frame'],f['pixel_sha256']) for f in z['frames']] != [
                (f['frame'],f['pixel_sha256']) for f in orig['frames']]:
            raise ValueError('Camera frame alignment changed')
        out.append(dict(start=item['start'],cache_sha256=base.sha(p),forecasts=len(z['paths']),
                        detections=sum(len(f['rows']) for f in z['frames'])))
        beat(phase='rapid_cold' if cold else 'rapid_frozen', done=k, total=len(wanted),
             reused=cached is not None and not cold,seconds=time.monotonic()-start)
    base.immutable(base.OUT/('rapid_cold.json' if cold else 'rapid_freeze.json'),
        dict(registration=key, windows=out, annotation_geometry_read=False, independent_confirmation=False))


def match(w, people):
    sets = []
    for f in w['frames']:
        t = f['frame']
        labels = {j:z[t] for j,z in people.items() if t in z}
        sets.append(base.associate(np.asarray(f['rows']).reshape(-1,7),labels))
    pairs = set.intersection(*sets)
    pred = {i:np.stack((c,o)) for i,c,o in w['paths']}
    return {j:pred[i] for i,j in pairs}


def score(windows, people):
    batches = []
    for s,p in windows:
        ids = sorted(p)
        if not ids:
            continue
        labels = [[people[j].get(s+k*base.STRIDE) for k in range(8,20)] for j in ids]
        y = np.array([[z[:2] if z is not None else np.zeros(2) for z in x] for x in labels])
        v = np.array([[z is not None for z in x] for x in labels])
        batches.append((y,v,np.stack([p[j] for j in ids])))
    return base.measures(batches)


def evaluate(mobile):
    fr = base.OUT/'rapid_freeze.json'
    f = load(fr)
    if f['registration']!=base.sha(REG) or f['annotation_geometry_read']:
        raise ValueError('RAPiD predictions not frozen before scoring')
    p = base.LOCAL/f'{base.PILOT}.json'
    support = load(base.OUT/'support.json')
    pilot, = [x for x in support['clips'] if x['clip']==base.PILOT]
    if base.sha(p)!=pilot['annotation_sha256'] or pilot['quality']['unlisted_images']:
        raise ValueError('Pilot annotations changed')
    d = load(p)
    frames = {base.data.frame_number(i['file_name'],base.PILOT):i['file_name'] for i in d['images']}
    people,_ = base.data.annotation_table(d,base.PILOT,frames,geometry=True)
    rows, ma, ra, mc, rc = [], [], [], [], []
    for a,b in zip(mobile['windows'],f['windows'],strict=True):
        s = a['start']
        path = PRIVATE/f'window_{s:06}.json'
        if s!=b['start'] or base.sha(path)!=b['cache_sha256']:
            raise ValueError('Different forecast roster')
        w0,w1 = verify_source(a),load(path)
        m,r = match(w0,people),match(w1,people)
        both = set(m)&set(r)
        ma.append((s,m)); ra.append((s,r))
        mc.append((s,{j:m[j] for j in both})); rc.append((s,{j:r[j] for j in both}))
        rows.append(dict(start=s, mobile=len(m),rapid=len(r),matched=len(both),
                         online_mobile=len(w0['paths']),online_rapid=len(w1['paths'])))
    n = {k:sum(z[k] for z in rows) for k in ('mobile','rapid','matched')}
    q = {k:sum(z[k]>0 for z in rows) for k in n}
    pass_gate = {k:n[k]>=base.MIN_HISTORIES and q[k]>=base.MIN_QUERIES for k in n}
    result = dict(registration=base.sha(REG),freeze_sha256=base.sha(fr), matched_histories=n, matched_windows=q,
        support_gate=pass_gate, by_window=rows, independent_confirmation=False, original_2pct=True,
        mobile_scores=score(ma,people) if pass_gate['mobile'] else None,
        rapid_scores=score(ra,people) if pass_gate['rapid'] else None,
        common_scores={k:score(a,people) for k,a in (('mobile',mc),('rapid',rc))} if pass_gate['matched'] else None,
        risk_evaluation='not_run', neural_forecasting='not_run')
    base.immutable(base.OUT/'rapid_comparison.json',result)
    print({k:v for k,v in result.items() if k!='by_window'},flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('produce','verify','evaluate'))
    phase = p.parse_args().phase
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'rapid.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _, mobile = register()
        if phase == 'evaluate':
            evaluate(mobile)
        else:
            produce(mobile,cold=phase=='verify')


if __name__ == '__main__':
    main()
