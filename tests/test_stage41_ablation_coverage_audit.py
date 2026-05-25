from src import stage41_ablation_coverage_audit as audit


def test_status_labels_cross_protocol_when_weaker():
    assert audit._status(True, weaker=False) == "complete"
    assert audit._status(True, True, weaker=True) == "complete_but_cross_protocol"
    assert audit._status(True, False, weaker=False) == "partial"
    assert audit._status(False, False, weaker=False) == "missing"


def test_delta_reads_direct_or_nested():
    assert audit._delta({"all_delta": 0.1}) == 0.1
    assert audit._delta({"delta_vs_full": {"all_delta": -0.2}}) == -0.2
    assert audit._delta({}) is None
