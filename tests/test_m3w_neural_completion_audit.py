from src import m3w_neural_completion_audit as audit


def test_m3w_neural_completion_audit_status_helpers() -> None:
    assert audit._status(True) == "complete"
    assert audit._status(False, partial=True) == "partial"
    assert audit._status(False) == "incomplete"


def test_m3w_neural_completion_audit_output_path() -> None:
    assert str(audit.OUT_DIR).endswith("outputs/m3w_neural_v1")
