from src import stage42_data_calibration as s42


def test_stage42_current_facts_block_overclaim() -> None:
    text = "\n".join(s42.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42_dataset_specs_cover_required_sources() -> None:
    ids = {row["id"] for row in s42.DATASET_SPECS}
    assert {"sdd", "opentraj", "eth_ucy", "trajnet", "ucy", "tgsim", "aerialmpt"}.issubset(ids)


def test_stage42_gate_renderer_preserves_metric_guard() -> None:
    payload = {
        "source": "fresh_run",
        "summary": {
            "converted_paths_found": 4,
            "stage42_b_external_validation_ready": True,
            "stage42_c_full_waypoint_prereq_ready": True,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    text = "\n".join(s42._render_stage_a_gate(payload))
    assert "Metric Overclaim Guard" in text
    assert "Seconds Overclaim Guard" in text
    assert "Stage5C Execution Gate" in text
    assert "`7 / 7`" in text
