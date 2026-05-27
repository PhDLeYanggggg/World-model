from src.stage42_source_terms_prefill import _candidate_paths, _gate, _preferred_path


def test_candidate_paths_marks_raw_source_and_cache():
    row = {
        "path_status": [
            {"path": "external_data/OpenTraj/datasets/UCY", "exists": True, "size_mb": 1.0, "sample_extensions": {".txt": 2}},
            {"path": "data/stage20_world_state/ucy_crowd", "exists": True, "size_mb": 0.1, "sample_extensions": {".json": 1}},
            {"path": "missing", "exists": False},
        ]
    }
    candidates = _candidate_paths(row)
    assert len(candidates) == 2
    assert candidates[0]["is_raw_source_candidate"] is True
    assert candidates[1]["is_derived_or_cache"] is True


def test_preferred_path_uses_raw_source_candidate_first():
    candidates = [
        {"path": "data/stage20_world_state/ucy_crowd", "is_raw_source_candidate": False},
        {"path": "external_data/OpenTraj/datasets/UCY", "is_raw_source_candidate": True},
    ]
    assert _preferred_path(candidates) == "external_data/OpenTraj/datasets/UCY"


def test_gb_gate_passes_for_safe_prefill_payload():
    payload = {
        "source": "fresh_stage42_gb_source_terms_prefill",
        "input_status": {"ga_exists": True, "terms_template_exists": True},
        "summary": {
            "datasets_prefilled": 5,
            "datasets_with_suggested_local_path": 5,
            "raw_source_candidate_rows": 4,
            "terms_accepted_by_user_count": 0,
            "conversion_ready_now": 0,
        },
        "prefill_draft_written": True,
        "user_action_required_written": True,
        "claim_boundary": {
            "download_executed": False,
            "conversion_executed": False,
            "evaluation_executed": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = _gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_gb_source_terms_prefill_pass"
