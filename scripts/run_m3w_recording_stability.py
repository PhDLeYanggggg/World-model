"""Memory-only source forecast reuse with nested deletion and exact replay."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native .venv-pytorch arm64, not Rosetta')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from scripts import verify_m3w_selected_set_readout as readout
from src.world_model import m3w_recording_stability as api

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_recording_stability_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_recording_stability_v1'
CONFIG = ROOT/'configs/m3w_european_recording_stability_v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == raw, 'Immutable output differs'
    else:
        with path.open('x') as f:
            f.write(raw)


def registration():
    prior = json.loads((readout.PUBLIC/'registration.json').read_text())
    for name, h in prior['remote_code'].items():
        assert sha(ROOT/name) == h, 'Changed frozen dependency: '+name
    assert json.loads(readout.READOUT_REGISTRATION.read_text()) == readout.register()
    files = [Path(__file__), ROOT/'src/world_model/m3w_recording_stability.py',
             ROOT/'tests/test_m3w_recording_stability.py', CONFIG, PUBLIC/'protocol.md']
    return dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in files},
        prior_scientific_registration_sha256=sha(readout.PUBLIC/'registration.json'),
        prior_readout_registration_sha256=sha(readout.READOUT_REGISTRATION),
        source_input_manifest_sha256=sha(readout.PUBLIC/'input_manifest.json'),
        parent_freeze_sha256=sha(readout.PARENT/'calibration_freeze.json'),
        new_neural_updates=0, local_numeric_array_cache=False, independent_roles_read=False)


def load_parents():
    out = {}
    for row in json.loads((readout.PARENT/'calibration_freeze.json').read_text())['groups']:
        path = ROOT/row['path']
        assert sha(path) == row['sha256']
        out[path.stem] = json.loads(path.read_text())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'pilot', 'run'])
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    reg = registration()
    if args.phase == 'register':
        once(PUBLIC/'registration.json', reg)
        print(json.dumps(dict(registration_sha256=sha(PUBLIC/'registration.json'))))
        return
    assert json.loads((PUBLIC/'registration.json').read_text()) == reg
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', str(PUBLIC/'registration.json')], check=True, cwd=ROOT)
    subprocess.run(['git', 'cat-file', '-e', 'HEAD:'+str((PUBLIC/'registration.json').relative_to(ROOT))], check=True, cwd=ROOT)
    cfg = json.loads(CONFIG.read_text())
    assert not (PUBLIC/'complete.json').exists(), 'Already completed; inspect existing evidence'
    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bundle, nbytes = readout.fetch()
        parents = load_parents()
        names = [e['group'] for e in bundle['manifest']['packets']]
        assert len(names) == cfg['source_heads'] == len(parents)
        if args.phase == 'pilot':
            names = sorted(names, key=lambda n: (-len(parents[n]['final']['recordings']), n))[:1]
        groups, refs = [], []
        used = 0
        compute_seconds = 0.
        for i, name in enumerate(names):
            if time.monotonic()-start > cfg['hard_runtime_limit_seconds']:
                raise RuntimeError('Registered 12-hour bound reached; preserve groups for resume')
            z = readout.load_packet(base64.b64decode(bundle['inputs'][name]))
            original = json.loads(bundle['outputs'][name])
            readout.verify_group(z, original, parents[name])
            target = PUBLIC/'groups'/(name+'.json')
            t = time.monotonic()
            if args.phase == 'run' and target.exists() and not args.resume:
                raise RuntimeError('Existing group requires --resume')
            # Deterministic calibration is rerun, including when resuming a group.
            got = api.audit_head(z, parents[name], original, cfg)
            replay = api.audit_head(z, parents[name], original, cfg)
            assert got == replay, 'Full nested-calibration replay differs'
            compute_seconds += time.monotonic()-t
            size = len(json.dumps(got, indent=2, allow_nan=False).encode())+1
            used += size
            assert used < cfg['aggregate_output_cap_bytes']
            if args.phase == 'run':
                once(target, got)
                refs.append(dict(path=str(target.relative_to(ROOT)), sha256=sha(target), bytes=size))
            groups.append(got)
            beat = dict(pid=os.getpid(), phase=args.phase, groups=i+1, seconds=time.monotonic()-start,
                compute_and_replay_seconds=compute_seconds,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
            (PRIVATE/'heartbeat.json').write_text(json.dumps(beat)+'\n')
            print(json.dumps(beat), flush=True)
        receipt = dict(beat, input_output_bundle_bytes=nbytes, numerical_arrays_saved_locally=False,
            exact_nested_replay=True, registration_sha256=sha(PUBLIC/'registration.json'),
            new_neural_updates=0, torch_version=torch.__version__, architecture=platform.machine(),
            cpu_threads=cfg['cpu_threads'], num_workers=0)
        if args.phase == 'pilot':
            receipt['pilot_group'] = names[0]
            receipt['source_recordings'] = groups[0]['source_recordings']
            receipt['projected_72_group_compute_upper_seconds_if_each_like_pilot'] = compute_seconds*72
            receipt['projected_group_aggregate_bytes_if_each_like_pilot'] = used*72
            once(PUBLIC/'pilot.json', receipt)
        else:
            summary = api.summarize(groups, cfg)
            once(PUBLIC/'summary.json', summary)
            assert used+(PUBLIC/'summary.json').stat().st_size < cfg['aggregate_output_cap_bytes']
            once(PUBLIC/'complete.json', dict(receipt, groups=refs, summary_sha256=sha(PUBLIC/'summary.json')))
            print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
