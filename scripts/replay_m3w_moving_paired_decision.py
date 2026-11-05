"""Replay a completed paired-decision experiment without replacing its receipts."""
import json
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_paired_decision as run

source = run.source


def readonly_receipt(path, value):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Completed receipt required for replay: '+str(path))
    if source.read(path) != value:
        raise ValueError('Replay disagrees with immutable receipt: '+str(path))


def manifest(public):
    return {str(p): source.sha(p) for p in sorted(public.rglob('*')) if p.is_file()}


def require_completed(public):
    for name in ('pilot_execution', 'train_execution', 'evaluate_execution', 'verify_execution',
                 'training_freeze', 'analysis', 'verification'):
        if not (public/(name+'.json')).is_file():
            raise ValueError('Complete original run required before replay: '+name)
    freeze = source.read(public/'training_freeze.json')
    if freeze['complete_heads'] != 9 or not freeze['all_before_validation']:
        raise ValueError('Nine models frozen before readout required')
    verification = source.read(public/'verification.json')
    run.api.base.exact(verification['analysis_sha256'], source.sha(public/'analysis.json'))
    analysis = source.read(public/'analysis.json')
    for receipt in analysis.get('group_receipts', []):
        group = source.read(source.checked(receipt))
        source.checked(group['cache'])


def main():
    require_completed(run.PUBLIC)
    before = manifest(run.PUBLIC)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        run.old.previous.teacher.lineage.PRIVATE = run.PRIVATE
        started, guard, beat = run.old.previous.teacher.lineage.runtime({}, source.read(run.CONFIG), 'completed_replay')
        original_once = source.once
        source.once = readonly_receipt
        try:
            cfg, values = run.register(beat)
            run.evaluate(cfg, values, beat, guard, verify=True)
            run.api.base.exact(before, manifest(run.PUBLIC))
        finally:
            source.once = original_once
        result = dict(pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            public_files_unchanged=len(before), analysis_sha256=source.sha(run.PUBLIC/'analysis.json'),
            training=False, new_scientific_readout=False, exact_replay=True)
        beat(state='complete', **{k: v for k, v in result.items() if k != 'pid'})
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
