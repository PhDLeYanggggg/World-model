import hashlib
from types import SimpleNamespace

import pytest

from src.evaluation.m3w_frozen_runtime import prepare_runtime,RelocatedCodeContract,SUPERVISED,MIRROR_FILES
from src.evaluation.m3w_experiment_contract import ContractError,ExperimentContract,file_digest
from test_m3w_experiment_contract import development_only_protocol,approve


def test_exact_historical_code_is_relocated_not_waived(tmp_path):
    p = development_only_protocol(tmp_path)
    current = tmp_path/SUPERVISED;current.parent.mkdir(parents=True);current.write_text('new-version')
    frozen = tmp_path/'mirror'/SUPERVISED;frozen.parent.mkdir(parents=True);frozen.write_text('old-version')
    p['bindings'] = {SUPERVISED:file_digest(frozen)};approve(p)
    with pytest.raises(ContractError,match='binding'):
        ExperimentContract(p,tmp_path)
    runtime = {'relocation':{SUPERVISED:str(frozen.relative_to(tmp_path))}}
    contract = RelocatedCodeContract(p,tmp_path,[],runtime=runtime)
    assert contract._path(SUPERVISED) == frozen
    assert current.read_text() == 'new-version'
    assert contract.digest == p['approval']['protocol_sha256']
    frozen.write_text('tampered')
    with pytest.raises(ContractError,match='binding'):
        contract._assert_frozen()


def test_no_data_or_unbounded_runtime_relocation(tmp_path):
    p = development_only_protocol(tmp_path)
    with pytest.raises(ValueError,match='Only the explicitly'):
        RelocatedCodeContract(p,tmp_path,[],runtime={'relocation':{'a':'b'}})
    with pytest.raises(ContractError,match='escapes'):
        p['bindings'] = {SUPERVISED:'irrelevant'};approve(p)
        RelocatedCodeContract(p,tmp_path,[],runtime={'relocation':{SUPERVISED:'../escape.py'}})


def test_recovery_checks_bytes_before_writing_and_rejects_changed_snapshot(tmp_path,monkeypatch):
    import src.evaluation.m3w_frozen_runtime as m
    for name in MIRROR_FILES:
        p = tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'current')
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=b'old'))
    protocol = {'bindings':{SUPERVISED:hashlib.sha256(b'old').hexdigest()}}
    directory = tmp_path/'snapshot'
    result = prepare_runtime(tmp_path,protocol,directory,revision='known',install=False)
    assert not result['hash_checks_relaxed'] and (tmp_path/SUPERVISED).read_bytes() == b'current'
    assert prepare_runtime(tmp_path,protocol,directory,revision='known',install=False) == result
    (directory/SUPERVISED).write_bytes(b'altered')
    with pytest.raises(ValueError,match='snapshot changed'):
        prepare_runtime(tmp_path,protocol,directory,revision='known',install=False)
    protocol['bindings'][SUPERVISED] = 'wrong'
    other = tmp_path/'other'
    with pytest.raises(ValueError,match='does not match'):
        prepare_runtime(tmp_path,protocol,other,revision='known',install=False)
    assert not (other/SUPERVISED).exists()
