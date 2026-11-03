import json
from types import SimpleNamespace

import pytest

from scripts.archive_m3w_moving_pilot import MEMBERS, compact, create, file_hash, restore, verify


def test_lossless_raw_checkpoint_archive_before_compaction(tmp_path, monkeypatch):
    for index, member in enumerate(MEMBERS):
        path = tmp_path/member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((b'raw-checkpoint-'+bytes([index]))*100)
    archive = tmp_path/'pilot.tar.xz'
    manifest = create(archive, tmp_path)
    assert json.loads(json.dumps(manifest)) == manifest
    verify(archive, manifest)
    before = {member:file_hash(tmp_path/member) for member in MEMBERS}
    compact(archive, tmp_path, manifest)
    assert not any((tmp_path/member).exists() for member in MEMBERS)
    verify(archive, manifest)
    assert before == {k:v['sha256'] for k,v in manifest['members'].items()}
    monkeypatch.setattr('scripts.archive_m3w_moving_pilot.shutil.disk_usage',
        lambda _:SimpleNamespace(free=20*1024**3))
    restore(archive, tmp_path, manifest)
    assert before == {member:file_hash(tmp_path/member) for member in MEMBERS}


def test_changed_copy_is_never_removed(tmp_path):
    for member in MEMBERS:
        path = tmp_path/member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original')
    archive = tmp_path/'pilot.tar.xz'
    manifest = create(archive, tmp_path)
    (tmp_path/MEMBERS[-1]).write_bytes(b'user-modified')
    with pytest.raises(ValueError, match='changed or unrelated'):
        compact(archive, tmp_path, manifest)
    assert all((tmp_path/member).exists() for member in MEMBERS)
