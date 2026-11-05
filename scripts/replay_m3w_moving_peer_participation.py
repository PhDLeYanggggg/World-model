"""Read-only replay of all twelve frozen factorial cells."""
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 environment required before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_peer_participation as run
from scripts.replay_m3w_moving_paired_decision import manifest, readonly_receipt


def require_completed(public):
    for name in ('pilot_execution', 'train_execution', 'evaluate_execution', 'verify_execution',
                 'training_freeze', 'analysis', 'verification'):
        if not (public/(name+'.json')).is_file():
            raise ValueError('Complete original run required before replay: '+name)
    freeze = run.source.read(public/'training_freeze.json')
    if freeze['fresh_heads'] != 6 or freeze['cached_verified_heads'] != 6 or not freeze['all_before_validation']:
        raise ValueError('Twelve frozen factorial cells required')
    run.api.base.exact(run.source.read(public/'verification.json')['analysis_sha256'], run.source.sha(public/'analysis.json'))


def main():
    require_completed(run.PUBLIC)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        run.source.fcntl.flock(lock, run.source.fcntl.LOCK_EX | run.source.fcntl.LOCK_NB)
        before = manifest(run.PUBLIC)
        runtime = run.old.old.old.previous.teacher.lineage; runtime.PRIVATE = run.PRIVATE
        started, guard, beat = runtime.runtime({}, run.source.read(run.CONFIG), 'completed_replay')
        original = run.source.once; run.source.once = readonly_receipt
        try:
            cfg, values = run.register(beat)
            run.evaluate(cfg, values, beat, guard, verify=True)
            run.api.base.exact(before, manifest(run.PUBLIC))
        finally:
            run.source.once = original
        result = dict(pid=run.source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=run.source.resource.getrusage(run.source.resource.RUSAGE_SELF).ru_maxrss,
            public_files_unchanged=len(before), analysis_sha256=run.source.sha(run.PUBLIC/'analysis.json'),
            training=False, new_scientific_readout=False, exact_replay=True)
        beat(state='complete', **{k:v for k,v in result.items() if k != 'pid'})
        run.source.once(run.PRIVATE/'completed_replays'/(str(result['pid'])+'.json'), result)
        print(run.source.json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
