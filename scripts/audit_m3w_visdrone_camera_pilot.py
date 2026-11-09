"""Probe whole-roster label support from causal camera prefixes on publisher train."""

import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from zipfile import ZipFile

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch

from scripts import audit_m3w_visdrone_mot_train as source
from scripts.audit_m3w_ht21_detection_support import FIELDS, associate, groups, support, write_json
from scripts.run_m3w_waterloo_raw_prefix import WEIGHTS, detect, detector
from src.world_model.m3w_streaming_boxes import StreamingTracks

PRIVATE = ROOT/'data/stage_cvpr2027_experiments/visdrone_mot_camera_pilot_v1'
OBS, FUTURE, STRIDE = 8, 12, 4
FRAMES = tuple(1+STRIDE*i for i in range(OBS+FUTURE))


def select_clips(roster, limit=4):
    choices = {}
    for clip in sorted(roster):
        match = re.fullmatch(source.CLIP, clip)
        if match is None:
            raise ValueError('Unexpected source name')
        if set(FRAMES).issubset(roster[clip]):
            choices.setdefault(match[1], clip)
    if len(choices) < limit:
        raise ValueError('Not enough distinct filename groups for the fixed pilot')
    return list(choices.values())[:limit]


def label_frames(payload, roster):
    a = source.parse(payload, roster)
    a = a[(a[:, 6] == 1) & (a[:, 7] == 1)]
    xyxy = np.column_stack((a[:, :4], a[:, 2:4]+a[:, 4:6]))
    by_frame = groups(xyxy)
    return [by_frame.get(t, np.empty((0, 6)))[:, 1:] for t in FRAMES]


def registration(manifest):
    bound_paths = [Path(__file__), Path(source.__file__), ROOT/'src/evaluation/m3w_remote_zip.py',
                   ROOT/'scripts/run_m3w_waterloo_raw_prefix.py', ROOT/'src/world_model/m3w_streaming_boxes.py',
                   ROOT/'scripts/audit_m3w_ht21_detection_support.py']
    weight_hash = source.digest(WEIGHTS.read_bytes())
    if not weight_hash.startswith('907ea3f9'):
        raise ValueError('Unexpected COCO detector weights')
    source_receipt = (source.OUT/'source_receipt.json').read_bytes()
    if source.digest(manifest) != json.loads(source_receipt)['source_manifest_sha256']:
        raise ValueError('Source manifest differs from range intake receipt')
    return dict(source_manifest_sha256=source.digest(manifest), source_receipt_sha256=source.digest(source_receipt),
                clips=select_clips(json.loads(manifest)['frames']), past_frames=FRAMES[:OBS],
                reference_frames=FRAMES[OBS:], selection='first eligible clip per sorted filename group; first window',
                detector='COCO Faster R-CNN MobileNetV3 Large 320 FPN; 4x3 tiles; confidence .35; NMS .5',
                detector_weight_sha256=weight_hash, cpu_threads=4, workers=0, source_role='training_source_audit',
                reference='included VisDrone category 1; category 2 is outside strict pedestrian support',
                association='same identity in all eight frames; one-to-one IoU >= .10 and center <= half label diagonal',
                code_sha256={p.name: source.digest(p.read_bytes()) for p in bound_paths},
                prediction_evaluation=False, validation_payload_read=False, test_payload_read=False)


class PastImages:
    def __init__(self, metadata, stack, verify):
        self.metadata, self.stack, self.verify = metadata, stack, verify
        self.archive, self.remote = None, None

    def read(self, clip, frame):
        if frame not in FRAMES[:OBS] or clip not in select_clips(self.metadata['frames']):
            raise ValueError('Only registered past images may be read')
        dst = PRIVATE/clip/f'{frame:07}.jpg'
        receipt = dst.with_suffix('.json')
        if receipt.exists():
            raw = dst.read_bytes()
            saved = json.loads(receipt.read_text())
            if saved != self.binding(clip, frame, raw):
                raise ValueError('Cached JPEG disagrees with its receipt')
            return raw
        if self.verify:
            raise FileNotFoundError('Offline replay requires the previously cached past frames')
        if self.archive is None:
            repo, revision = source.MIRRORS[0]
            url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{source.ARCHIVE}?download=true'
            self.remote = self.stack.enter_context(source.RangeFile(url, source.SIZE))
            self.archive = self.stack.enter_context(ZipFile(self.remote))
            _, _, directory_hash = source.inventory(self.archive)
            if directory_hash != self.metadata['mirrors'][0]['inventory_sha256']:
                raise ValueError('Archive directory changed')
        info = self.archive.getinfo(f'{source.PREFIX}/sequences/{clip}/{frame:07}.jpg')
        if max(info.file_size, info.compress_size) > 2_000_000:
            raise ValueError('Oversized pilot image')
        self.remote.span(info.header_offset, min(source.SIZE-info.header_offset, info.compress_size+2048))
        raw = self.archive.read(info)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix('.pending')
        tmp.write_bytes(raw)
        tmp.replace(dst)
        write_json(receipt, self.binding(clip, frame, raw), False)
        return raw

    def binding(self, clip, frame, raw):
        return dict(clip=clip, frame=frame, bytes=len(raw), sha256=source.digest(raw),
                    zip_directory_sha256=self.metadata['mirrors'][0]['inventory_sha256'])


def replay_prefix(images, net, clip, reg):
    tracker = StreamingTracks()
    rows = []
    for i, frame in enumerate(FRAMES[:OBS]):
        raw = images.read(clip, frame)
        with Image.open(BytesIO(raw)) as image:
            if image.format != 'JPEG':
                raise ValueError('Expected JPEG')
            bgr = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
        d = detect(net, bgr)
        d = d[d[:, -1] == 1]
        online, _ = tracker.step(i, d)
        rows.append(dict(frame=frame, jpeg_sha256=source.digest(raw), online=online[:, :5].tolist(),
                         detections=np.column_stack((np.arange(len(d)), d[:, :4])).tolist()))
    return dict(clip=clip, registration_sha256=reg, past=rows)


def measure(bundle, labels):
    past = [np.asarray(r['online'], dtype=float).reshape(-1, 5) for r in bundle['past']]
    raw = [np.asarray(r['detections'], dtype=float).reshape(-1, 5) for r in bundle['past']]
    stats = support(past, labels)
    complete_reference = set.intersection(*(set(a[:, 0].astype(int)) for a in labels))
    boxes_found = set.intersection(*({label_id for _, label_id in associate(d, a)}
                                    for d, a in zip(raw, labels[:OBS], strict=True)))
    stats['reference_complete_detected_every_past_frame'] = len(complete_reference & boxes_found)
    stats['past_detections'] = sum(map(len, raw))
    return dict(clip=bundle['clip'], **{k: stats[k] for k in (*FIELDS,
                'reference_complete_detected_every_past_frame', 'past_detections')})


def run(verify):
    subprocess.run(['git', 'check-ignore', '--quiet', str(PRIVATE/'heartbeat.json')], cwd=ROOT, check=True)
    if any(p.is_symlink() for p in (PRIVATE, *PRIVATE.parents)):
        raise ValueError('Refuse symlinked private workspace')
    manifest = (source.DATA/'source_manifest.json').read_bytes()
    metadata = json.loads(manifest)
    reg = registration(manifest)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'lock').open('a') as lock, ExitStack() as stack:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registration_path = source.OUT/'camera_registration.json'
        write_json(registration_path, reg, verify)
        reg_hash = source.digest(registration_path.read_bytes())
        images, net = PastImages(metadata, stack, verify), detector()
        bundles = {}
        for clip in reg['clips']:
            path = PRIVATE/f'{clip}_prefix.json'
            bundle = replay_prefix(images, net, clip, reg_hash)
            write_json(path, bundle, verify)
            bundles[clip] = dict(prefix_sha256=source.digest(path.read_bytes()),
                                 images=[r['jpeg_sha256'] for r in bundle['past']])
            msg = dict(pid=os.getpid(), phase='past_prefix', clips_completed=len(bundles), clip=clip)
            (PRIVATE/'heartbeat.json').write_text(json.dumps(msg)+'\n')
            print(json.dumps(msg), flush=True)
        freeze = source.OUT/'camera_freeze.json'
        write_json(freeze, dict(registration_sha256=reg_hash, prefixes=bundles, reference_payload_opened=False), verify)
        counts = []
        for clip, binding in bundles.items():
            prefix = (PRIVATE/f'{clip}_prefix.json').read_bytes()
            payload = (source.DATA/f'{clip}.txt').read_bytes()
            if source.digest(prefix) != binding['prefix_sha256'] or source.digest(payload) != metadata['labels'][clip]:
                raise ValueError('Frozen prefix or reference labels changed')
            counts.append(measure(json.loads(prefix), label_frames(payload, metadata['frames'][clip])))
        totals = Counter()
        for row in counts:
            totals.update({k: v for k, v in row.items() if k != 'clip'})
        result = dict(result_source='fresh_run', source_role='training_source_audit', clips=counts, totals=totals,
                      freeze_sha256=source.digest(freeze.read_bytes()), independent_confirmation=False,
                      scientific_evaluation='not_run', validation_payload_read=False, test_payload_read=False,
                      units='image pixels and release-frame indices; no verified seconds')
        write_json(source.OUT/'camera_support.json', result, verify)
        (PRIVATE/'heartbeat.json').write_text(json.dumps(dict(pid=os.getpid(), phase='complete'))+'\n')
        print(json.dumps(totals, sort_keys=True), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', action='store_true', help='Recompute prefixes and support without network access')
    run(p.parse_args().verify)
