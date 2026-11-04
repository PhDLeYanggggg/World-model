"""Repeat frozen C inference without modifying existing public study artifacts."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_source_separated_critics as run

source = run.source


class ReadOnlyReceipts:
    def __getattr__(self, name):
        return getattr(source, name)

    def once(self, path, value):
        if not path.is_file():
            raise ValueError('Read-only replay cannot create a missing study receipt')
        if source.stable(source.read(path)) != source.stable(value):
            raise ValueError('Read-only replay disagrees with the saved scientific result')


def snapshot():
    return {str(p.relative_to(run.PUBLIC)): source.sha(p) for p in run.PUBLIC.rglob('*') if p.is_file()}


def main():
    reg, verification = source.read(run.REG), source.read(run.PUBLIC / 'verification.json')
    if (not verification['all216_unique_nested_C_predictions_and_all108policy_blocks_exact']
            or verification['analysis_sha256'] != source.sha(run.PUBLIC / 'analysis.json')):
        raise ValueError('Complete the original whole-population verification first')
    for name, digest in reg['bindings'].items():
        if source.sha(ROOT / name) != digest:
            raise ValueError('The original registered scientific definition changed')
    before = snapshot()
    with (run.PRIVATE / 'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        run.previous.teacher.lineage.PRIVATE = run.PRIVATE
        spec = source.read(run.CONFIG)
        start, guard, beat = run.previous.teacher.lineage.runtime({}, spec, 'public_artifact_readonly_replay')
        run.source = ReadOnlyReceipts()
        try:
            spec, cfg, data, jobs, refs, parts = run.register(beat)
            run.evaluate(spec, cfg, data, jobs, refs, parts, beat, guard, verify=True)
        finally:
            run.source = source
        if before != snapshot():
            raise ValueError('Existing public study artifacts changed during read-only replay')
        beat(state='complete_existing_public_artifacts_unchanged')
        print(source.json.dumps(dict(existing_public_artifacts_unchanged=True,
            all216_predictions108policy_blocks1620cost_cells_exact=True, no_training=True, new_H_prediction=False,
            pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss), indent=2), flush=True)


if __name__ == '__main__': main()
