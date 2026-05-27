from __future__ import annotations

from src.stage42_official_source_link_audit import _audit_rows, _gate, _summary


def _row(dataset_id: str) -> dict:
    return {
        "dataset_id": dataset_id,
        "domain": "x",
        "priority_rank": 1,
        "after_terms_potential": {"estimated_t50_windows": 10, "estimated_t100_windows": 5},
        "user_confirmation": {
            "terms_accepted_by_user": False,
            "terms_acceptance_date": "",
            "allowed_use": "",
            "local_path": "",
            "source_identity": "",
            "confirmed_by_user": "",
        },
    }


def test_stage42_em_audit_rows_preserve_manual_terms_block() -> None:
    rows = _audit_rows([_row("ucy_crowd_original"), _row("eth_biwi_original"), _row("trajnetplusplus_official")])

    assert all(row["manual_terms_required"] for row in rows)
    assert all(not row["auto_download_allowed_now"] for row in rows)
    assert all(not row["conversion_ready_now"] for row in rows)
    assert any("graphics.cs.ucy.ac.cy" in row["official_url_candidates"][0] for row in rows)
    assert any("vision.ee.ethz.ch" in row["official_url_candidates"][0] for row in rows)


def test_stage42_em_summary_never_counts_conversion() -> None:
    rows = _audit_rows([_row("ucy_crowd_original")])
    summary = _summary(rows, {"exists": True, "conversion_ready_targets": 0, "blocked_targets": 1})

    assert summary["auto_download_allowed_now"] == 0
    assert summary["conversion_ready_now"] == 0
    assert summary["converted_now"] == 0
    assert summary["evaluated_now"] == 0
    assert summary["estimated_t50_after_terms"] == 10


def test_stage42_em_gate_requires_no_download_and_no_metric_claim() -> None:
    rows = _audit_rows(
        [
            _row("ucy_crowd_original"),
            _row("eth_biwi_original"),
            _row("trajnetplusplus_official"),
            _row("opentraj_toolkit"),
            _row("aerialmpt_or_other_topdown"),
        ]
    )
    payload = {
        "summary": _summary(rows, {"exists": True, "conversion_ready_targets": 0, "blocked_targets": 5}),
        "official_source_rows": rows,
        "readiness_manifest": {"exists": True},
        "user_action_written": True,
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
