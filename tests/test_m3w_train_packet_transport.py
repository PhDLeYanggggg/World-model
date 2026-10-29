import hashlib
import pytest
from src.evaluation.m3w_train_packet_transport import verified_packet_bytes


def ref(raw):
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def test_verified_existing_bytes_cached_and_replayed_without_network(tmp_path):
    raw = b'existing frozen SOURCE TRAIN payload'
    calls = []
    fetch = lambda:(calls.append(1) or raw)
    args = ref(raw), tmp_path, fetch
    assert verified_packet_bytes(*args, minimum_free_bytes=0) == (raw, 'fresh_read_existing_remote_TRAIN')
    assert verified_packet_bytes(*args, allow_fetch=False) == (raw, 'cached_verified')
    assert calls == [1]


def test_corruption_is_not_replaced_or_credited_as_verified(tmp_path):
    raw = b'original'
    args = ref(raw), tmp_path, lambda:raw
    verified_packet_bytes(*args, minimum_free_bytes=0)
    (tmp_path/(ref(raw)['sha256']+'.npz')).write_bytes(b'changed')
    with pytest.raises(ValueError):
        verified_packet_bytes(*args)


def test_missing_replay_wrong_manifest_and_cache_cap_fail_closed(tmp_path):
    raw = b'original'
    with pytest.raises(FileNotFoundError):
        verified_packet_bytes(ref(raw), tmp_path, lambda:raw, allow_fetch=False)
    with pytest.raises(ValueError):
        verified_packet_bytes(ref(raw), tmp_path, lambda:b'wrong', minimum_free_bytes=0)
    assert not list(tmp_path.iterdir())
    with pytest.raises(OSError):
        verified_packet_bytes(ref(raw), tmp_path, lambda:raw, minimum_free_bytes=0, cache_cap_bytes=1)
