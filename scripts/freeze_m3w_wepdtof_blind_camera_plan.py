"""Pin external camera prefixes from image names only, before any outcome readout."""
import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_wepdtof_source_probe as probe


def windows(frames, stride):
    """Build disjoint 20-step windows, never traversing a gap in released frames."""
    start = last = None
    for t in sorted(frames):
        if last is None or t != last+1:
            start = t
        if t-start == 19*stride:
            yield start, [frames[start+i*stride] for i in range(8)]
            start = t+1
        last = t


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cafile',default='/etc/ssl/cert.pem')
    a = p.parse_args()
    root = probe.OUT
    role_path = root/'external_roles.json'
    roles = json.loads(role_path.read_text())
    if (not roles['roles_frozen'] or roles['external_labels_opened_for_scoring'] or not roles['original_2pct']
            or roles['external_sources']!=13 or roles['groups'][probe.PILOT]['role']!='development'):
        raise ValueError('Require the approved external role freeze')
    meta = {}
    for group,r in roles['groups'].items():
        if r['role']=='frozen_external_once':
            for clip in r['clips']:
                if clip in meta or probe.data.source(clip)!=group or clip==probe.PILOT:
                    raise ValueError('External clip or source reused')
                meta[clip] = group
    with probe.archive(a.cafile) as ar, ZipFile(ar) as z:
        probe.check_version(ar)
        clips = probe.data.catalog(z.namelist())
        result = []
        for clip,group in sorted(meta.items()):
            w = []
            for start,names in windows(clips[clip],probe.STRIDE):
                items = [z.getinfo(k) for k in names]
                w.append(dict(start=start,past=[dict(member=i.filename,crc32=f'{i.CRC:08x}',bytes=i.file_size)
                                                 for i in items]))
            result.append(dict(clip=clip,source=group,windows=w))
    n = sum(len(r['windows']) for r in result)
    manifest = dict(role_sha256=probe.sha(role_path), code_sha256=probe.sha(Path(__file__)), archive_version=probe.VERSION,
        cadence='stride four on released image grid; not a claim in seconds', observe=8, predict=12,
        external_sources=13, external_clips=len(result), windows=n, image_prefix_frames=8*n,
        clips=result, outcomes_read=False, pixels_read=False, inference_run=False,
        ready_for_official_scoring=False, original_2pct=True)
    probe.immutable(root/'blind_camera_plan.json',manifest)
    print({k:manifest[k] for k in ('external_sources','external_clips','windows','image_prefix_frames','outcomes_read')})


if __name__=='__main__':
    main()
