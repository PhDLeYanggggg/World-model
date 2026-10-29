"""Bounded local cache for exact existing TRAIN bytes, not new role admission."""
import hashlib
import os
from pathlib import Path
import re
import shutil


def verified_packet_bytes(ref, directory, fetch, *, allow_fetch=True,
                          minimum_free_bytes=10*2**30, cache_cap_bytes=4*2**30):
    sha, size = ref['sha256'], ref['bytes']
    if not re.fullmatch('[0-9a-f]{64}', sha) or not isinstance(size, int) or size <= 0:
        raise ValueError('Frozen packet SHA256 and positive byte count required')
    directory = Path(directory)
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError('Private cache cannot traverse symlinks')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/(sha+'.npz')

    def check(raw):
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError('TRAIN packet bytes differ from the original frozen manifest')
        return raw

    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise ValueError('Expected regular cached packet')
        return check(path.read_bytes()), 'cached_verified'
    if not allow_fetch:
        raise FileNotFoundError('Exact replay requires the existing verified local packet')
    if sum(p.stat().st_size for p in directory.glob('*.npz'))+size > cache_cap_bytes:
        raise OSError('Private TRAIN cache cap exceeded; do not materialize more data')
    if shutil.disk_usage(directory).free < minimum_free_bytes+size:
        raise OSError('Preserve local disk reserve before fetching a packet')
    raw = check(fetch())
    temp = path.with_suffix('.tmp.'+str(os.getpid()))
    try:
        with temp.open('xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return raw, 'fresh_read_existing_remote_TRAIN'
