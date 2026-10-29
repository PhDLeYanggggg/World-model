import hashlib
import json
from pathlib import Path
import pytest
from src.evaluation.m3w_metric_bundle import pack, verify


def fixture(root):
    public = root/'outputs'/'study'; path = public/'groups'/'a.json'
    path.parent.mkdir(parents=True); value = b'{"aggregate": 17}\n'; path.write_bytes(value)
    (public/'complete.json').write_text(json.dumps(dict(groups=[dict(path=str(path.relative_to(root)),
        sha256=hashlib.sha256(value).hexdigest())])))
    return public, path, value


def test_archive_is_exact_repeatable_and_can_restore_missing_aggregates(tmp_path):
    public, path, value = fixture(tmp_path)
    first = pack(tmp_path, public); assert pack(tmp_path, public) == first
    path.unlink(); assert verify(tmp_path, public)['groups'] == 1
    verify(tmp_path, public, restore=True); assert path.read_bytes() == value


def test_archive_never_overwrites_changed_local_work(tmp_path):
    public, path, _ = fixture(tmp_path); pack(tmp_path, public)
    path.write_text('different')
    with pytest.raises(ValueError, match='do not overwrite'):
        verify(tmp_path, public, restore=True)


def test_only_registered_aggregate_paths_are_packaged(tmp_path):
    public, _, _ = fixture(tmp_path)
    (public/'complete.json').write_text(json.dumps(dict(groups=[dict(path='data/raw.npz', sha256='none')])))
    with pytest.raises(ValueError, match='Only named aggregate'):
        pack(tmp_path, public)
