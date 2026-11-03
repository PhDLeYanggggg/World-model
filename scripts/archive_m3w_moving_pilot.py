"""Lossless storage compaction for completed pilot copies, not model reserialization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_source_forecast_supported_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_source_forecast_supported_v1'
MEMBERS = tuple(f'pilot/{family}/{kind}/checkpoint.pt'
    for family in ('pct', 'eqmotion') for kind in ('resumed', 'direct'))
ARCHIVE = PRIVATE/'pilot_verified.tar.xz'
MANIFEST = PUBLIC/'pilot_storage_archive.json'


def digest(stream):
    value = hashlib.sha256()
    while chunk := stream.read(1024*1024):
        value.update(chunk)
    return value.hexdigest()


def file_hash(path):
    with path.open('rb') as stream:
        return digest(stream)


def verify(archive, manifest):
    if file_hash(archive) != manifest['archive_sha256']:
        raise ValueError('Archive bytes changed')
    with tarfile.open(archive, 'r:xz') as source:
        if set(source.getnames()) != set(MEMBERS):
            raise ValueError('Archive must contain exactly the four completed pilot copies')
        for member in MEMBERS:
            record = manifest['members'][member]
            item = source.getmember(member)
            if not item.isfile() or item.size != record['bytes']:
                raise ValueError('Changed checkpoint member identity')
            with source.extractfile(member) as stream:
                if digest(stream) != record['sha256']:
                    raise ValueError('Checkpoint does not round-trip byte exactly')


def create(archive, directory):
    records = {}
    for member in MEMBERS:
        path = directory/member
        if not path.is_file() or path.is_symlink():
            raise ValueError('Only complete regular original pilot files may be packed')
        records[member] = dict(bytes=path.stat().st_size, sha256=file_hash(path))
    pending = archive.with_suffix('.pending.xz')
    if archive.exists() or pending.exists():
        raise ValueError('Do not replace a completed or partial archive without inspection')
    with tarfile.open(pending, 'x:xz', preset=9) as output:
        for member in MEMBERS:
            output.add(directory/member, arcname=member, recursive=False)
    value = dict(members=records, archive_sha256=file_hash(pending),
        archive_bytes=pending.stat().st_size, original_bytes=sum(r['bytes'] for r in records.values()),
        paths_and_original_raw_checkpoint_bytes_restorable=True,
        raw_data_or_primary_model_files_affected=False, new_model_training=False,
        stage5c_executed=False, smc_enabled=False)
    verify(pending, value)
    for member in MEMBERS:
        if file_hash(directory/member) != records[member]['sha256']:
            raise ValueError('Original changed while archiving; keep every standalone copy')
    pending.replace(archive)
    return value


def compact(archive, directory, manifest):
    verify(archive, manifest)
    for path in (archive, archive.parent):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    paths = [directory/member for member in MEMBERS]
    for member, path in zip(MEMBERS, paths):
        if path.exists() and (path.is_symlink() or file_hash(path) != manifest['members'][member]['sha256']):
            raise ValueError('Do not compact a changed or unrelated file')
    for path in paths:
        if path.exists():
            path.unlink()


def restore(archive, directory, manifest):
    verify(archive, manifest)
    required = sum(r['bytes'] for key,r in manifest['members'].items() if not (directory/key).exists())
    if shutil.disk_usage(directory).free < 10*1024**3+required:
        raise OSError('Restore needs the original10GiB reserve plus all missing raw pilot bytes')
    with tarfile.open(archive, 'r:xz') as source:
        for member in MEMBERS:
            path = directory/member
            if path.exists():
                if path.is_symlink() or file_hash(path) != manifest['members'][member]['sha256']:
                    raise ValueError('Never overwrite a changed existing checkpoint')
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = path.with_suffix('.restoring')
            with source.extractfile(member) as incoming, pending.open('xb') as output:
                shutil.copyfileobj(incoming, output)
            if file_hash(pending) != manifest['members'][member]['sha256']:
                raise ValueError('Do not expose a changed restoration')
            pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('build', 'verify', 'compact', 'restore'))
    args = parser.parse_args()
    training = json.loads((PUBLIC/'training.json').read_text())
    if training['complete_primary_models'] != 18 or not training['matched_sampling_all9pairs_exact']:
        raise ValueError('Finish original18-model training before pilot storage compaction')
    if args.phase == 'build':
        if MANIFEST.exists():
            raise ValueError('Preserve complete storage receipt')
        for row in json.loads((PUBLIC/'pilot.json').read_text())['rows']:
            if (not row['resume_direct_checkpoint_and_outputs_exact']
                    or file_hash(ROOT/row['prefix_path']) != row['prefix_sha256']):
                raise ValueError('Original exact pilot prefix must still match its frozen receipt')
        manifest = create(ARCHIVE, PRIVATE)
        manifest.update(archive_path=str(ARCHIVE.relative_to(ROOT)),
            logical_bytes_reclaimable=manifest['original_bytes']-manifest['archive_bytes'],
            result_source='fresh_lossless_archive_and_all4raw_member_hash_replay_not_training')
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    else:
        manifest = json.loads(MANIFEST.read_text())
        if args.phase == 'compact':
            for path in (MANIFEST, MANIFEST.parent):
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            compact(ARCHIVE, PRIVATE, manifest)
        elif args.phase == 'restore':
            restore(ARCHIVE, PRIVATE, manifest)
        else:
            verify(ARCHIVE, manifest)
    print(json.dumps(dict(phase=args.phase, archive_sha256=manifest['archive_sha256'],
        archive_bytes=manifest['archive_bytes'], original_bytes=manifest['original_bytes'],
        logical_bytes_reclaimable=manifest['original_bytes']-manifest['archive_bytes'],
        original_checkpoint_bytes_preserved=True)), flush=True)


if __name__ == '__main__':
    main()
