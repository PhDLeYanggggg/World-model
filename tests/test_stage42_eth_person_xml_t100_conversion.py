from pathlib import Path

from src import stage42_eth_person_xml_t100_conversion as s42bl


def test_xml_rows_parse_frame_agent_centers() -> None:
    rows = s42bl._xml_rows(Path("external_data/OpenTraj/datasets/ETH-Person/data/seq0_assc_gt.xml"))
    assert rows
    assert {"frame_id", "agent_id", "x", "y"}.issubset(rows[0])
    assert min(row["frame_id"] for row in rows) >= 0


def test_source_key_deduplicates_interp_variant() -> None:
    base = Path("external_data/OpenTraj/datasets/ETH-Person/data/seq0_assc_gt.xml")
    interp = Path("external_data/OpenTraj/datasets/ETH-Person/data/seq0_assc_gt-interp.xml")
    assert s42bl._source_key_for_path(base) == s42bl._source_key_for_path(interp)


def test_candidate_sources_are_strict_independent_and_t100_capable() -> None:
    sources = s42bl._candidate_sources()
    keys = [src["independent_key"] for src in sources]
    assert len(keys) == len(set(keys))
    assert sum(str(src["relative_path"]).startswith("ETH-Person/") for src in sources) >= 3
    assert all(src["t100_capable"] for src in sources)


def test_build_windows_contains_eval_only_errors() -> None:
    sources = s42bl._candidate_sources()
    source = next(src for src in sources if str(src["relative_path"]).startswith("ETH-Person/"))
    windows = s42bl._build_windows(source)
    assert windows
    row = windows[0]
    assert "errors_eval_only" in row
    assert "constant_velocity_causal_fd" in row["errors_eval_only"]
    assert row["metric_status"] == "unverified"


def test_gate_blocks_official_claim_even_if_technical_result_exists() -> None:
    payload = {
        "source": "fresh_technical_dry_run_terms_unverified",
        "bk_verdict": "stage42_bk_local_source_verification_pass",
        "summary": {
            "eth_person_xml_sources": 4,
            "strict_independent_sources": 4,
            "source_cv_folds": 4,
            "t100_windows_total": 100,
            "technical_t100_mean_improvement_vs_fallback": 0.1,
            "license_terms_confirmed": False,
            "official_converted_dataset_claim_allowed": False,
            "auto_download_executed": False,
            "global_t100_positive_claim_allowed": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
            "selection_uses_holdout": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42bl._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_bl_eth_person_xml_t100_dry_run_pass"
