import gzip
import pytest
from scripts.recover_m3w_oof_magnitude_checkpoint import recover


def test_orphan_archive_recovery_preserves_bytes_and_existing_files(tmp_path):
    home = tmp_path/'inner'/'head'; home.mkdir(parents=True)
    payload = b'full optimizer rng and model state'
    (home/'checkpoint.pt.gz').write_bytes(gzip.compress(payload))
    assert not recover(home, tmp_path)['restored']
    assert not (home/'checkpoint.pt').exists()
    assert recover(home, tmp_path, restore=True)['restored']
    assert (home/'checkpoint.pt').read_bytes() == payload
    assert gzip.decompress((home/'checkpoint.pt.gz').read_bytes()) == payload
    assert recover(home, tmp_path, restore=True) is None
    (home/'checkpoint.pt').unlink(); (home/'complete.json').write_text('{}')
    assert recover(home, tmp_path, restore=True) is None


def test_recovery_rejects_outside_and_corrupt_archives(tmp_path):
    home = tmp_path/'head'; home.mkdir()
    (home/'checkpoint.pt.gz').write_bytes(b'corrupt')
    with pytest.raises(ValueError): recover(home, tmp_path/'elsewhere', restore=True)
    with pytest.raises(gzip.BadGzipFile): recover(home, tmp_path, restore=True)
    assert not (home/'checkpoint.pt').exists()
