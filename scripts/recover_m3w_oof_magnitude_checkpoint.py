"""Restore this study's archived, receipt-incomplete checkpoint before --resume."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_oof_magnitude_v1'


def recover(home, root, *, restore=False):
    home, root = Path(home), Path(root)
    if home.is_symlink() or root.resolve() not in home.resolve().parents:
        raise ValueError('Only this run-owned directory may be recovered')
    if (home/'complete.json').exists() or (home/'checkpoint.pt').exists():
        return None
    path = home/'checkpoint.pt.gz'
    if not path.is_file() or path.is_symlink(): raise ValueError('Regular archived checkpoint required')
    with gzip.open(path, 'rb') as handle: payload = handle.read()
    if restore:
        tmp = home/'checkpoint.recovery.tmp'
        with tmp.open('wb') as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, home/'checkpoint.pt')
    return dict(directory=str(home), uncompressed_sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload), restored=restore, original_archive_preserved=True,
        resume_identity_validation_still_required=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--restore', action='store_true'); args = parser.parse_args()
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows = [r for path in sorted(PRIVATE.glob('**/checkpoint.pt.gz'))
            if (r := recover(path.parent, PRIVATE, restore=args.restore)) is not None]
    print(json.dumps(dict(incomplete_archives=rows, restore=args.restore), indent=2))


if __name__ == '__main__': main()
