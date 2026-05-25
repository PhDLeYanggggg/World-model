from src import m3w_neural_goal_completion_audit as audit


def test_status_helper():
    assert audit._status(True) == "complete"
    assert audit._status(False, partial=True) == "partial"
    assert audit._status(False) == "incomplete"


def test_metric_reads_numbers_safely():
    assert audit._metric({"x": 1}, "missing") == 0.0
    assert audit._metric({"x": 1}, "x") == 1.0
    assert audit._metric({"x": "bad"}, "x", 2.0) == 2.0


def test_gate_map_extracts_passed_flags():
    gates = {"gates": [{"gate": "a", "passed": True}, {"gate": "b", "passed": False}]}
    assert audit._gate_map(gates) == {"a": True, "b": False}


def test_pytest_passed_accepts_current_count():
    assert audit._pytest_passed("- result: `251 passed in 60.73s`")
    assert audit._pytest_passed("251 passed in 60.73s")
    assert not audit._pytest_passed("- result: `1 failed, 250 passed`")
