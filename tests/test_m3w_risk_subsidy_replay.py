import json
import pytest
from scripts import run_m3w_risk_subsidy as runner


def make(tmp_path, monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    root=tmp_path/'private';root.mkdir();identity={'version':'test'}
    (root/'identity.json').write_text(json.dumps(identity));ish=runner.file_digest(root/'identity.json')
    path=root/'decision.npz';path.write_bytes(b'identity-check-fixture-not-numpy')
    receipt={'experiment_sha256':ish,'path':'private/decision.npz','sha256':runner.file_digest(path)}
    rp=root/'decision.json';rp.write_text(json.dumps(receipt))
    manifest={'experiment_sha256':ish,'queries':188388,'receipts':[{'path':'private/decision.json','sha256':runner.file_digest(rp)}]}
    (root/'decisions_complete.json').write_text(json.dumps(manifest))
    return root,identity


def test_existing_decision_replay_is_read_only(tmp_path,monkeypatch):
    root,identity=make(tmp_path,monkeypatch)
    before={p.name:p.read_bytes() for p in root.iterdir()}
    runner.require_replay(root,{},'decide',identity)
    assert before=={p.name:p.read_bytes() for p in root.iterdir()}


@pytest.mark.parametrize('name',['identity.json','decisions_complete.json','decision.json','decision.npz'])
def test_missing_replay_evidence_is_not_recreated(tmp_path,monkeypatch,name):
    root,identity=make(tmp_path,monkeypatch);(root/name).unlink()
    with pytest.raises(ValueError):runner.require_replay(root,{},'decide',identity)
    assert not (root/name).exists()


def test_changed_replay_identity_fails(tmp_path,monkeypatch):
    root,identity=make(tmp_path,monkeypatch)
    with pytest.raises(ValueError):runner.require_replay(root,{},'decide',{'version':'other'})


def test_no_aggregate_replay_before_first_readout(tmp_path,monkeypatch):
    root,identity=make(tmp_path,monkeypatch)
    with pytest.raises(ValueError):runner.require_replay(root,{'reports':'reports'},'evaluate',identity)
