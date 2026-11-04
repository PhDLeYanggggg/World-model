import json
import pytest

from scripts.replay_m3w_source_separated_critics import ReadOnlyReceipts


def test_readonly_receipt_never_creates_or_replaces_public_result(tmp_path):
    checker = ReadOnlyReceipts()
    path = tmp_path / 'receipt.json'
    with pytest.raises(ValueError, match='cannot create'):
        checker.once(path, dict(score=3))
    assert not path.exists()
    original = json.dumps(dict(score=3)).encode()
    path.write_bytes(original)
    checker.once(path, dict(score=3))
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match='disagrees'):
        checker.once(path, dict(score=4))
    assert path.read_bytes() == original
