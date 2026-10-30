import pytest
from scripts import audit_m3w_source_label_provenance as run


def test_scalar_checkpoint_creates_parent_and_rejects_changed_completed_receipt(tmp_path):
    path=tmp_path/'records'/'record1.json';value=dict(rows=3,mask_mismatches=0)
    run.once(path,value);before=path.read_bytes();run.once(path,value)
    assert path.read_bytes()==before and not path.with_suffix('.json.tmp').exists()
    with pytest.raises(ValueError):run.once(path,dict(value,rows=4))
