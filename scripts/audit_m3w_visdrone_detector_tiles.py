"""Compare COCO tile scales while replaying the same frozen camera transforms."""

import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image

from scripts import audit_m3w_visdrone_association as association
from scripts.audit_m3w_visdrone_camera_pilot import FRAMES, OBS, PastImages, label_frames, measure, source, write_json
from scripts.audit_m3w_waterloo_person_tiles import detect as fine_detect
from scripts.run_m3w_waterloo_raw_prefix import WEIGHTS, detect, detector, torch, tv
from src.world_model.m3w_camera_association import MotionAssociation

OUT = source.OUT.parent/'visdrone_detector_tiles_v1'
PRIVATE = association.PILOT_CACHE.parent/'visdrone_detector_tiles_v1'
ARMS = ('coarse', 'fine', 'pooled')


def person_boxes(d):
    d = np.asarray(d, dtype=np.float64).reshape(-1, 6)
    return d[d[:, 5] == 1]


def pool(coarse, fine):
    d = np.concatenate((coarse, fine))
    keep = tv.ops.nms(torch.from_numpy(d[:, :4].copy()), torch.from_numpy(d[:, 4].copy()), .5)
    return d[keep.numpy()]


def rows_for(clip, original, detections, camera):
    past_frames = list(FRAMES[:OBS])
    if [r['frame'] for r in original] != past_frames or [r['frame'] for r in camera] != past_frames[1:]:
        raise ValueError('Frozen frame roster mismatch')
    variants = {k: dict(clip=clip, past=[]) for k in ARMS}
    trackers = {k: MotionAssociation(False) for k in ARMS}
    for i, (row, predictions) in enumerate(zip(original, detections, strict=True)):
        coarse, fine = (person_boxes(predictions[k]) for k in ARMS[:2])
        if not np.array_equal(coarse[:, :4], association.detection(row)[:, :4]):
            raise ValueError('New inference must reproduce the frozen coarse boxes and order')
        for k, d in zip(ARMS, (coarse, fine, pool(coarse, fine)), strict=True):
            motion = np.asarray(camera[i-1]['matrix']) if i else None
            online = trackers[k].step(i, d, motion)
            variants[k]['past'].append(dict(frame=row['frame'], online=online[:, :5].tolist(),
                detections=np.column_stack((np.arange(len(d)), d[:, :4])).tolist()))
    return variants


def evidence():
    pilot = (source.OUT/'camera_freeze.json').read_bytes()
    camera = (association.OUT/'freeze.json').read_bytes()
    previous = (association.OUT/'readout.json').read_bytes()
    manifest = (source.DATA/'source_manifest.json').read_bytes()
    pilot_reg = (source.OUT/'camera_registration.json').read_bytes()
    assoc_reg = (association.OUT/'registration.json').read_bytes()
    p, c, r, m, pr, ar = map(json.loads, (pilot, camera, previous, manifest, pilot_reg, assoc_reg))
    if (source.digest(pilot_reg) != p['registration_sha256'] or
            source.digest(pilot) != ar['parent_freeze_sha256'] or
            source.digest(assoc_reg) != c['registration_sha256'] or
            source.digest(camera) != r['freeze_sha256'] or
            source.digest(manifest) != pr['source_manifest_sha256'] or
            sorted(p['prefixes']) != pr['clips'] or sorted(c['camera_steps']) != pr['clips']):
        raise ValueError('Frozen evidence chain changed')
    weight_hash = source.digest(WEIGHTS.read_bytes())
    if weight_hash != pr['detector_weight_sha256']:
        raise ValueError('Detector checkpoint changed')
    paths = [Path(__file__), Path(association.__file__), ROOT/'scripts/audit_m3w_waterloo_person_tiles.py',
             ROOT/'scripts/run_m3w_waterloo_raw_prefix.py', ROOT/'src/world_model/m3w_camera_association.py']
    reg = dict(pilot_freeze_sha256=source.digest(pilot), camera_freeze_sha256=source.digest(camera),
               parent_readout_sha256=source.digest(previous), detector_weight_sha256=weight_hash,
               clips=pr['clips'], past_frames=pr['past_frames'], reference_frames=pr['reference_frames'],
               coarse_grid=[4, 3], fine_grid=[8, 5], overlap=.2, min_score=.35, nms_iou=.5,
               pool='score NMS across coarse and fine post-NMS boxes, person class only',
               arm_shared='unchanged frozen camera matrices; no per-arm camera re-estimation',
               implementation={f.name: source.digest(f.read_bytes()) for f in paths},
               cpu_threads=4, workers=0, role='reused_training_windows_only', scientific_evaluation='not_run')
    return reg, m, p, c, r


def run(verify, recompute):
    subprocess.run(['git', 'check-ignore', '--quiet', str(PRIVATE/'heartbeat.json')], cwd=ROOT, check=True)
    if any(p.is_symlink() for p in (PRIVATE, *PRIVATE.parents)):
        raise ValueError('Refuse symlinked cache path')
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    reg, metadata, pilot, camera, previous = evidence()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock, ExitStack() as stack:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(OUT/'registration.json', reg, verify)
        bound = source.digest((OUT/'registration.json').read_bytes())
        images, net = PastImages(metadata, stack, verify=True), None
        all_rows, hashes = {}, {}
        for clip in reg['clips']:
            original = (association.PILOT_CACHE/f'{clip}_prefix.json').read_bytes()
            if source.digest(original) != pilot['prefixes'][clip]['prefix_sha256']:
                raise ValueError('Original detection bundle changed')
            prefix = json.loads(original)['past']
            history = []
            for row in prefix:
                frame = row['frame']
                raw = images.read(clip, frame)
                if source.digest(raw) != row['jpeg_sha256']:
                    raise ValueError('Past input differs from original detection')
                binding = dict(frame=frame, jpeg_sha256=row['jpeg_sha256'], registration_sha256=bound)
                path = PRIVATE/f'{clip}_{frame:07}.json'
                if recompute or (not verify and not path.exists()):
                    if net is None:
                        net = detector()
                    with Image.open(BytesIO(raw)) as im:
                        bgr = np.asarray(im.convert('RGB'))[:, :, ::-1].copy()
                    fresh = dict(binding=binding, coarse=person_boxes(detect(net, bgr)).tolist(),
                                 fine=person_boxes(fine_detect(net, bgr)).tolist())
                    write_json(path, fresh, verify)
                content = path.read_bytes()
                frozen = json.loads(content)
                if frozen['binding'] != binding:
                    raise ValueError('Detection checkpoint has a different input binding')
                history.append(frozen)
                hashes[path.name] = source.digest(content)
                msg = dict(pid=os.getpid(), phase='prefix', completed=len(hashes), total=32, clip=clip, frame=frame)
                pending = PRIVATE/'heartbeat.pending'
                pending.write_text(json.dumps(msg)+'\n')
                pending.replace(PRIVATE/'heartbeat.json')
            all_rows[clip] = rows_for(clip, prefix, history, camera['camera_steps'][clip])
            print(json.dumps(msg), flush=True)
        paths = PRIVATE/'prefixes.json'
        write_json(paths, all_rows, verify)
        write_json(OUT/'freeze.json', dict(registration_sha256=bound, detections=hashes,
                   prefix_sha256=source.digest(paths.read_bytes()), labels_opened_this_run=False), verify)
        records, totals = [], {k: Counter() for k in ARMS}
        ceilings = Counter()
        expected = {r['clip']: r['arms']['camera'] for r in previous['records']}
        for clip, arms in json.loads(paths.read_text()).items():
            payload = (source.DATA/f'{clip}.txt').read_bytes()
            if source.digest(payload) != metadata['labels'][clip]:
                raise ValueError('Training annotation changed')
            labels = label_frames(payload, metadata['frames'][clip])
            stats = {k: measure(bundle, labels) for k, bundle in arms.items()}
            limits = {k: association.possible_reference_support(bundle, labels) for k, bundle in arms.items()}
            if stats['coarse'] != expected[clip]:
                raise ValueError('Original camera arm did not replay exactly')
            for k, score in stats.items():
                totals[k].update({f: n for f, n in score.items() if f != 'clip'})
            ceilings.update(limits)
            records.append(dict(clip=clip, arms=stats, eligible_reference_upper_bound=limits))
        readout = dict(result_source='fresh_run', images_and_camera='cached_verified', registration_sha256=bound,
                       records=records, totals=totals, eligible_reference_upper_bound=ceilings,
                       freeze_sha256=source.digest((OUT/'freeze.json').read_bytes()), validation_or_test_opened=False,
                       independent_confirmation=False, scientific_evaluation='not_run', deployment_changed=False)
        write_json(OUT/'readout.json', readout, verify)
        print(json.dumps(dict(totals=totals, eligible_reference_upper_bound=ceilings), sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Replay frozen detections and recompute tracking and support')
    parser.add_argument('--recompute', action='store_true', help='Also rerun both detector grids and require exact caches')
    args = parser.parse_args()
    if args.recompute and not args.verify:
        parser.error('--recompute requires --verify')
    run(args.verify, args.recompute)
