"""Replay completed probes without changing public results or neural models."""
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_m3w_candidate_neighbors as d


def main():
    source, api = d.source, d.api
    root = d.PRIVATE/'completed_replays'; root.mkdir(parents=True, exist_ok=True)
    with (d.PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        before = {str(p.relative_to(d.PUBLIC)):source.sha(p) for p in d.PUBLIC.rglob('*') if p.is_file()}
        reg = source.read(d.PUBLIC/'registration_v3.json')
        for path, sha in reg['files'].items():
            api.base.exact(source.sha(ROOT/path), sha)
        d.run.old.old.previous.teacher.lineage.PRIVATE = root
        started, guard, beat = d.run.old.old.previous.teacher.lineage.runtime({}, source.read(d.run.CONFIG), 'readonly_replay')
        _, values = d.run.register(beat)
        for i, job in enumerate(values[3]):
            row = d.audit(job, values, guard, True)
            api.base.exact(source.stable(row), source.read(d.PUBLIC/(job['key']+'.json')))
            beat(state='context_verified', complete=i+1, total=3)
        after = {str(p.relative_to(d.PUBLIC)):source.sha(p) for p in d.PUBLIC.rglob('*') if p.is_file()}
        api.base.exact(before, after)
        receipt = dict(pid=source.os.getpid(), seconds=time.monotonic()-started,
            public_files_unchanged=len(before), all3_exact=True, neural_training=False, new_V_or_H=False,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            analysis_sha256=source.sha(d.PUBLIC/'analysis.json'))
        source.once(root/(str(source.os.getpid())+'.json'), receipt)
        print(source.json.dumps(receipt))


if __name__ == '__main__':
    main()
