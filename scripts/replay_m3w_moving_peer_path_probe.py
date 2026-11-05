"""Replay saved probe predictions without any fitting or public-file modification."""
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 environment required')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_peer_path_probe as run
from scripts.replay_m3w_moving_paired_decision import manifest, readonly_receipt


def require_complete():
    for name in ('pilot_execution', 'run_execution', 'verify_execution', 'analysis', 'verification'):
        if not (run.PUBLIC/(name+'.json')).is_file():
            raise ValueError('Complete study required: '+name)
    paths = list((run.PUBLIC/'fits').glob('*.json'))
    if len(paths) != 48:
        raise ValueError('All48saved fits required')
    for path in paths:
        run.source.checked(run.source.read(path)['checkpoint'])
    run.rel.base.exact(run.source.read(run.PUBLIC/'verification.json')['analysis_sha256'], run.source.sha(run.PUBLIC/'analysis.json'))


def forbidden_fit(*args, **kwargs):
    raise RuntimeError('Read-only replay cannot train or repair missing checkpoints')


def main():
    require_complete()
    with (run.PRIVATE/'process.lock').open('a') as lock:
        run.source.fcntl.flock(lock, run.source.fcntl.LOCK_EX | run.source.fcntl.LOCK_NB)
        before = manifest(run.PUBLIC)
        runtime = run.old.run.old.old.previous.teacher.lineage; runtime.PRIVATE = run.PRIVATE
        started, guard, beat = runtime.runtime({}, run.source.read(run.CONFIG), 'completed_replay')
        once, fit = run.source.once, run.method.fit_predict
        run.source.once = readonly_receipt; run.method.fit_predict = forbidden_fit
        try:
            cfg, values = run.register(beat)
            run.execute(cfg, values, guard, beat, resume=True, verify=False)
            run.rel.base.exact(before, manifest(run.PUBLIC))
        finally:
            run.source.once = once; run.method.fit_predict = fit
        result = dict(pid=run.source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=run.source.resource.getrusage(run.source.resource.RUSAGE_SELF).ru_maxrss,
            all48_saved_predictions_exact=True, fitting_forbidden=True, public_files_unchanged=len(before),
            analysis_sha256=run.source.sha(run.PUBLIC/'analysis.json'), new_V_or_H=False)
        once(run.PRIVATE/'completed_replays'/(str(result['pid'])+'.json'), result)
        beat(state='complete'); print(run.source.json.dumps(result))


if __name__ == '__main__':
    main()
