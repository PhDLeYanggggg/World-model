import json
from types import SimpleNamespace

import pytest

from scripts import run_m3w_scaled_risk as api


def test_storage_reserve_and_cap(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'PRIVATE',tmp_path)
    cfg=json.loads(api.CONFIG.read_text())
    monkeypatch.setattr(api.shutil,'disk_usage',lambda _:SimpleNamespace(free=9*2**30))
    with pytest.raises(OSError):api.storage(cfg)
    monkeypatch.setattr(api.shutil,'disk_usage',lambda _:SimpleNamespace(free=20*2**30))
    assert api.storage(cfg)['checkpoint_bytes']==0


def test_settings_do_not_change_model_budget_or_risk():
    cfg=json.loads(api.CONFIG.read_text());parent=json.loads(api.heads.manager.CONFIG.read_text())
    assert cfg['head_training']==parent['head_training']
    assert cfg['objective']==parent['objective']
    assert cfg['risk_budget']==.02 and not cfg['independent_roles_read']
    assert not cfg['threshold_search'] and cfg['neural_fits']==144
