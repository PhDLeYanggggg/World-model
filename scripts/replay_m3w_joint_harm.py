"""Repeat all frozen joint-harm inference without rewriting public receipts."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_joint_harm as run
from scripts.replay_m3w_source_separated_critics import ReadOnlyReceipts

source = run.source


def snapshot():
    return {str(p.relative_to(run.PUBLIC)): source.sha(p) for p in run.PUBLIC.rglob('*') if p.is_file()}


def main():
    verification = source.read(run.PUBLIC / 'verification.json')
    if (not verification['all108_policy_blocks_and_prediction_hashes_exact']
            or verification['analysis_sha256'] != source.sha(run.PUBLIC / 'analysis.json')):
        raise ValueError('Complete and verify the original experiment first')
    for path, digest in source.read(run.REG)['bindings'].items():
        if source.sha(ROOT / path) != digest:
            raise ValueError('Changed registered scientific definition')
    before = snapshot()
    with (run.PRIVATE / 'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        run.old.previous.teacher.lineage.PRIVATE = run.PRIVATE
        start, guard, beat = run.old.previous.teacher.lineage.runtime(
            {}, source.read(run.CONFIG), 'public_artifact_readonly_replay')
        run.source = ReadOnlyReceipts()
        try:
            values = run.register(beat)
            run.evaluate(*values, beat, guard, verify=True)
        finally:
            run.source = source
        if before != snapshot():
            raise ValueError('Read-only replay changed public artifacts')
        beat(state='complete_existing_public_artifacts_unchanged')
        print(source.json.dumps(dict(existing_public_artifacts_unchanged=True,
            all108_policy_blocks_and_prediction_hashes_exact=True, no_training=True,
            no_new_H=True, pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss), indent=2), flush=True)


if __name__ == '__main__':
    main()
