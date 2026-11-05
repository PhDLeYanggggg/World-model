"""Replay the completed neighbor audit without replacing immutable receipts."""
from pathlib import Path
import platform
import sys

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_m3w_neighbor_order as run
from scripts.replay_m3w_moving_paired_decision import manifest, readonly_receipt


def receipt(path, value):
    if Path(path) == run.PUBLIC/'verify_execution.json':
        if not Path(path).is_file() or set(value) != {'pid', 'seconds', 'peak_RSS_bytes'}:
            raise ValueError('Completed verification and exact runtime fields required')
        print(run.source.json.dumps(dict(completed_replay_runtime=value)))
        return
    readonly_receipt(path, value)


def main():
    for name in ('analysis', 'verification', 'verify_execution'):
        if not (run.PUBLIC/(name+'.json')).is_file():
            raise ValueError('Completed audit required before replay')
    before = manifest(run.PUBLIC)
    original, argv = run.source.once, sys.argv
    run.source.once, sys.argv = receipt, [str(Path(run.__file__)), '--verify']
    try:
        run.main()
        run.api.base.exact(before, manifest(run.PUBLIC))
    finally:
        run.source.once, sys.argv = original, argv
    print(run.source.json.dumps(dict(exact_replay=True, public_file_contents_unchanged=len(before))))


if __name__ == '__main__':
    main()
