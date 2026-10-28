import ast
import json
from types import SimpleNamespace

import pytest

from scripts import run_m3w_task_risk_local as api


def test_remote_reader_only_streams_registered_train_and_has_no_execution():
    ast.parse(api.PACKET)
    assert 'source_TRAIN_only' in api.PACKET and 'manifest_hash' in api.PACKET
    assert 'sbatch' not in api.PACKET and 'torch' not in api.PACKET
    assert "open('rb')" in api.PACKET


def test_local_storage_retains_reserve(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'PRIVATE',tmp_path)
    monkeypatch.setattr(api.shutil,'disk_usage',lambda _:SimpleNamespace(free=9*2**30))
    with pytest.raises(OSError):api.storage(json.loads(api.manager.CONFIG.read_text()))
    monkeypatch.setattr(api.shutil,'disk_usage',lambda _:SimpleNamespace(free=24*2**30))
    assert api.storage(json.loads(api.manager.CONFIG.read_text()))['checkpoint_bytes']==0


def test_stream_checksum_failures_do_not_enter_training(monkeypatch):
    monkeypatch.setattr(api.base,'ssh_args',lambda:['ssh','example'])
    monkeypatch.setattr(api.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=b'bad'))
    with pytest.raises(ValueError,match='byte/hash'):
        api.load_packet(dict(group='train',bytes=4,sha256='x'),
                        dict(input_manifest_sha256='x',parent_registration_sha256='x'))
