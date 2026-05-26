from __future__ import annotations

from pathlib import Path

from src import stage42_context_rescue_decision as db


def _metric(all_v: float, t50: float, hard: float, easy: float = 0.0) -> dict[str, float]:
    return {
        "all_improvement": all_v,
        "t50_improvement": t50,
        "hard_failure_improvement": hard,
        "easy_degradation": easy,
    }


def _payload_with_variants() -> dict:
    base = _metric(0.3, 0.3, 0.3, -0.2)
    return {
        "source": "fresh_run",
        "stage42_x_gate": {"verdict": "pass"},
        "variants": {
            "baseline_family_control": {"protected": base},
            "context_bad": {"protected": _metric(0.28, 0.25, 0.27, -0.18)},
            "context_easy_harm": {"protected": _metric(0.32, 0.32, 0.32, -0.15)},
            "context_bad_2": {"protected": _metric(0.27, 0.24, 0.26, -0.18)},
            "context_bad_3": {"protected": _metric(0.26, 0.23, 0.25, -0.18)},
            "context_bad_4": {"protected": _metric(0.25, 0.22, 0.24, -0.18)},
        },
    }


def test_variant_rows_compare_against_baseline_family() -> None:
    rows = db._variant_rows("protocol", _payload_with_variants())
    assert len(rows) >= 2
    assert rows[0]["delta_vs_baseline_family_control"]["all_improvement"] < 0
    assert rows[1]["safe_positive_increment"] is False


def test_summarize_stops_repeating_when_no_safe_positive_context() -> None:
    payloads = {
        "goal_scene_gated": _payload_with_variants(),
        "neighbor_interaction_gated": _payload_with_variants(),
        "sequence_context": {
            "source": "fresh_run",
            "baseline_family_only": {"protected": _metric(0.3, 0.3, 0.3, -0.2)},
            "sequence_variants": {"sequence_bad": {"protected": _metric(0.29, 0.22, 0.28, -0.19)}},
        },
        "graph_context": {
            "source": "fresh_run",
            "baseline_family_only": {"protected": _metric(0.3, 0.3, 0.3, -0.2)},
            "graph_variants": {"graph_bad": {"protected": _metric(0.29, 0.22, 0.28, -0.19)}},
        },
    }
    summary = db._summarize(payloads)
    assert summary["decision"] == "stop_repeating_current_context_residual_or_gated_protocols"
    assert summary["safe_positive_context_variants"] == []


def test_gate_passes_for_negative_context_rescue_decision() -> None:
    payload = {
        "source": "fresh_synthesis_from_cached_verified_context_runs",
        "input_files": {key: True for key in db.INPUTS},
        "protocol_status": [
            {"protocol": "goal_scene_gated", "loaded": True},
            {"protocol": "neighbor_interaction_gated", "loaded": True},
            {"protocol": "sequence_context", "loaded": True},
            {"protocol": "graph_context", "loaded": True},
        ],
        "summary": {
            "variant_rows": [{"safe_positive_increment": False} for _ in range(10)],
            "safe_positive_context_variants": [],
            "decision": "stop_repeating_current_context_residual_or_gated_protocols",
            "root_cause": "negative context evidence",
            "required_next_protocol_change": ["a", "b", "c"],
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = db._gate(payload)
    assert gate["verdict"] == "stage42_db_context_rescue_decision_pass"
    assert gate["passed"] == gate["total"]


def test_run_writes_isolated_outputs(tmp_path: Path, monkeypatch) -> None:
    input_paths = {
        "goal_scene_gated": tmp_path / "goal.json",
        "neighbor_interaction_gated": tmp_path / "neighbor.json",
        "sequence_context": tmp_path / "sequence.json",
        "graph_context": tmp_path / "graph.json",
        "context_forensics": tmp_path / "ci.json",
    }
    common = _payload_with_variants()
    for path in input_paths.values():
        path.write_text("{}", encoding="utf-8")
    input_paths["goal_scene_gated"].write_text(__import__("json").dumps(common), encoding="utf-8")
    input_paths["neighbor_interaction_gated"].write_text(__import__("json").dumps(common), encoding="utf-8")
    input_paths["sequence_context"].write_text(
        __import__("json").dumps(
            {
                "source": "fresh_run",
                "stage42_ar_gate": {"verdict": "partial"},
                "baseline_family_only": {"protected": _metric(0.3, 0.3, 0.3, -0.2)},
                "sequence_variants": {"sequence_bad": {"protected": _metric(0.29, 0.22, 0.28, -0.19)}},
            }
        ),
        encoding="utf-8",
    )
    input_paths["graph_context"].write_text(
        __import__("json").dumps(
            {
                "source": "fresh_run",
                "stage42_as_gate": {"verdict": "partial"},
                "baseline_family_only": {"protected": _metric(0.3, 0.3, 0.3, -0.2)},
                "graph_variants": {"graph_bad": {"protected": _metric(0.29, 0.22, 0.28, -0.19)}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(db, "INPUTS", input_paths)
    monkeypatch.setattr(db, "REPORT_JSON", tmp_path / "context_rescue_decision_stage42.json")
    monkeypatch.setattr(db, "REPORT_MD", tmp_path / "context_rescue_decision_stage42.md")
    monkeypatch.setattr(db, "GATE_MD", tmp_path / "stage42_stage_db_gate.md")
    payload = db.run_stage42_context_rescue_decision(refresh_readmes=False)
    assert payload["stage42_db_gate"]["verdict"] == "stage42_db_context_rescue_decision_pass"
    assert db.REPORT_JSON.exists()
    assert db.REPORT_MD.exists()
    assert db.GATE_MD.exists()
