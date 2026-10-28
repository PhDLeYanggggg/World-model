import ast
import inspect

from scripts import manage_m3w_task_risk_port as api
from scripts import train_m3w_task_risk as old
from scripts import train_m3w_task_risk_portable as new


def test_only_owned_root_comparison_changes_not_other_verification():
    before=inspect.getsource(old.verify)
    after=inspect.getsource(new.verify)
    expected=before.replace('root = root.resolve();','root = root.absolute();')
    assert ast.dump(ast.parse(expected))==ast.dump(ast.parse(after))
    assert new.train is old.train and new.quota is old.quota


def test_submission_and_collection_keep_original_checks():
    for kind in ('submit','observe'):
        code=api.remote_code(kind);ast.parse(code)
        assert 'from scripts.train_m3w_task_risk_portable import' in code
        assert 'controller_success' in code
    assert 'check_port(root)' in api.remote_code('submit')
    assert ' -m scripts.train_m3w_task_risk_portable ' in api.remote_code('submit')
    ast.parse(api.PORT_INSTALL)
