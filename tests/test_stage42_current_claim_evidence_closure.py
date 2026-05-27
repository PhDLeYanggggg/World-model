from src.stage42_current_claim_evidence_closure import _gate


def _payload(**summary_overrides):
    summary = {
        "supported_claim_count": 6,
        "blocked_claim_count": 7,
        "module_lock_gate_passed": True,
        "claim_linter_violations": 0,
        "t100_row_replay_rows": 47458,
        "t100_row_replay_gate_passed": True,
        "source_terms_conversion_ready": 0,
        "source_terms_converted_now": 0,
        "source_terms_evaluated_now": 0,
    }
    summary.update(summary_overrides)
    return {
        "summary": summary,
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def _patch_required_files(monkeypatch, tmp_path):
    from src import stage42_current_claim_evidence_closure as mod

    master = tmp_path / "master.md"
    matrix = tmp_path / "matrix.md"
    master.write_text("x", encoding="utf-8")
    matrix.write_text("x", encoding="utf-8")
    monkeypatch.setattr(mod, "MASTER_README", master)
    monkeypatch.setattr(mod, "PAPER_MATRIX_MD", matrix)
    monkeypatch.setattr(mod, "_text_has", lambda path, needles: True)


def test_gate_passes_for_supported_and_blocked_claim_closure(monkeypatch, tmp_path):
    _patch_required_files(monkeypatch, tmp_path)
    gate = _gate(_payload())
    assert gate["verdict"] == "stage42_ic_current_claim_evidence_closure_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_on_metric_seconds_overclaim(monkeypatch, tmp_path):
    _patch_required_files(monkeypatch, tmp_path)
    payload = _payload()
    payload["claim_boundary"]["metric_or_seconds_claim"] = True
    gate = _gate(payload)
    assert gate["gates"]["no_metric_seconds_claim"] is False
    assert gate["verdict"] == "stage42_ic_current_claim_evidence_closure_partial"


def test_gate_fails_if_source_terms_claim_ready(monkeypatch, tmp_path):
    _patch_required_files(monkeypatch, tmp_path)
    gate = _gate(_payload(source_terms_conversion_ready=1))
    assert gate["gates"]["source_terms_still_block_conversion"] is False


def test_gate_fails_if_t100_replay_missing(monkeypatch, tmp_path):
    _patch_required_files(monkeypatch, tmp_path)
    gate = _gate(_payload(t100_row_replay_rows=0))
    assert gate["gates"]["t100_replay_large_enough"] is False
