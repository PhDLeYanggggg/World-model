import json
import pytest
from src.evaluation.m3w_verification_receipt import retain_verified_receipt


def test_new_runtime_retains_the_original_identical_scientific_receipt(tmp_path):
    path = tmp_path/'receipt.json'
    original = dict(pid=1, seconds=2., exact=True, scientific_sha256='frozen')
    write = lambda path,value:path.write_text(json.dumps(value))
    retain_verified_receipt(path, original, fields=('exact', 'scientific_sha256'), flag='exact', immutable_write=write)
    before = path.read_bytes()
    retain_verified_receipt(path, dict(original, pid=2, seconds=3.),
        fields=('exact', 'scientific_sha256'), flag='exact', immutable_write=write)
    assert path.read_bytes() == before


def test_runtime_replay_cannot_waive_a_changed_hash_or_failed_check(tmp_path):
    path = tmp_path/'receipt.json'
    original = dict(pid=1, exact=True, scientific_sha256='frozen')
    write = lambda path,value:path.write_text(json.dumps(value))
    kwargs = dict(fields=('exact', 'scientific_sha256'), flag='exact', immutable_write=write)
    retain_verified_receipt(path, original, **kwargs)
    with pytest.raises(ValueError):
        retain_verified_receipt(path, dict(original, scientific_sha256='changed'), **kwargs)
    with pytest.raises(ValueError):
        retain_verified_receipt(path, dict(original, exact=False), **kwargs)
