from src import m3w_neural_completion_audit as audit
from src.stage14_pipeline import read_json


def test_m3w_neural_completion_audit_status_helpers() -> None:
    assert audit._status(True) == "complete"
    assert audit._status(False, partial=True) == "partial"
    assert audit._status(False) == "incomplete"


def test_m3w_neural_completion_audit_output_path() -> None:
    assert str(audit.OUT_DIR).endswith("outputs/m3w_neural_v1")


def test_m3w_neural_completion_audit_tracks_composite_tail_candidate() -> None:
    report = read_json("outputs/m3w_neural_v1/completion_audit_m3w_neural_v1.json", {})
    assert "composite-tail" in report.get("current_best_deployable", "")
    assert report.get("composite_tail_bounded_neural_evidence_summary", {}).get("evidence_pass") is True
    assert report.get("composite_tail_bounded_neural_multiseed_summary", {}).get("replication_pass") is True
    assert report.get("pure_ucy_source_heldout_validation_summary", {}).get("pure_ucy_source_heldout_gate") is True
