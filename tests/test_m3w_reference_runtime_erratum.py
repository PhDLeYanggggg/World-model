import pytest
from scripts import run_m3w_reference_cost_refit_readout as erratum


def test_original_scope_defect_is_reproduced_before_policy_work(monkeypatch):
    monkeypatch.setattr(erratum.run,'admission',lambda:({},None))
    with pytest.raises(UnboundLocalError,match='runtime'):erratum.run.evaluate()


def test_only_three_receipt_names_change_and_runtime_helper_is_called(monkeypatch):
    fixed,binding=erratum.corrected_evaluation()
    assert binding['changed_name_occurrences']==3
    assert binding['original_AST_restored_exactly_after_inverse_rename']
    assert 'runtime' in erratum.run.evaluate.__code__.co_varnames
    assert 'runtime' not in fixed.__code__.co_varnames
    monkeypatch.setitem(fixed.__globals__,'admission',lambda:({},None))
    class ReachedGlobalRuntime(Exception):pass
    def runtime(cfg):raise ReachedGlobalRuntime()
    monkeypatch.setitem(fixed.__globals__,'runtime',runtime)
    with pytest.raises(ReachedGlobalRuntime):fixed()
