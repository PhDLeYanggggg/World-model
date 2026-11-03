import pytest

from scripts.check_m3w_movingdronecrowd_background import check_bindings
from scripts.audit_m3w_movingdronecrowd_background import sha


def test_readonly_binding_check_rejects_changes_missing_files_and_path_escape(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    path=root/'definition.json';path.write_text('{"scope":"source_only"}')
    digest=sha(path);before=path.read_bytes()
    check_bindings(root,{'definition.json':digest})
    assert path.read_bytes()==before
    with pytest.raises(ValueError,match='frozen binding'):
        check_bindings(root,{'definition.json':'0'*64})
    with pytest.raises(ValueError,match='frozen binding'):
        check_bindings(root,{'absent.json':digest})
    outside=tmp_path/'outside.json';outside.write_bytes(before)
    with pytest.raises(ValueError,match='frozen binding'):
        check_bindings(root,{'../outside.json':sha(outside)})
