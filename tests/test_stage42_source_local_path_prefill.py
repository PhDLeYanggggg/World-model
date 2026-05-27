from pathlib import Path

from src import stage42_source_local_path_prefill as hy


def test_path_summary_detects_common_ucy_files(tmp_path: Path) -> None:
    scene = tmp_path / "zara01"
    scene.mkdir()
    (scene / "H.txt").write_text("1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
    (scene / "obsmat.txt").write_text("0 1 2 3\n", encoding="utf-8")
    (scene / "video.avi").write_bytes(b"fake")
    row = hy._path_summary(scene)
    assert row["exists"] is True
    assert row["has_homography_file"] is True
    assert row["has_obsmat"] is True
    assert row["has_video"] is True


def test_gate_preserves_legal_block() -> None:
    payload = {
        "inputs": {"gap_exists": True, "validation_exists": True},
        "prefill_rows": [
            {
                "dataset_id": "ucy_crowd_original",
                "local_path_found": True,
                "terms_accepted_by_user": False,
                "conversion_ready_now": False,
                "estimated_t50_windows_after_terms": 10,
                "path_summaries": [{"has_obsmat": True}],
            },
            {
                "dataset_id": "eth_biwi_original",
                "local_path_found": True,
                "terms_accepted_by_user": False,
                "conversion_ready_now": False,
                "estimated_t50_windows_after_terms": 1,
                "path_summaries": [{"has_obsmat": True}],
            },
            {
                "dataset_id": "trajnetplusplus_official",
                "local_path_found": False,
                "terms_accepted_by_user": False,
                "conversion_ready_now": False,
                "estimated_t50_windows_after_terms": 0,
                "path_summaries": [{"has_ndjson": False}],
            },
            {
                "dataset_id": "aerialmpt_or_other_topdown",
                "local_path_found": False,
                "terms_accepted_by_user": False,
                "conversion_ready_now": False,
                "estimated_t50_windows_after_terms": 0,
                "path_summaries": [{"has_zip": False}],
            },
            {
                "dataset_id": "opentraj_toolkit",
                "local_path_found": True,
                "terms_accepted_by_user": False,
                "conversion_ready_now": False,
                "estimated_t50_windows_after_terms": 0,
                "path_summaries": [{"has_obsmat": False}],
            },
        ],
        "actions": {"downloaded": False, "converted": False, "trained": False, "evaluated": False},
        "user_action_required": {"exists": True},
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "readme_updates": {"readmes_updated": True},
    }
    gate = hy._gate(payload)
    assert gate["verdict"] == "stage42_hy_source_local_path_prefill_pass"
    assert gate["passed"] == gate["total"]


def test_gate_rejects_conversion_claim() -> None:
    payload = {
        "inputs": {"gap_exists": True, "validation_exists": True},
        "prefill_rows": [
            {
                "dataset_id": dataset_id,
                "local_path_found": dataset_id in {"ucy_crowd_original", "eth_biwi_original"},
                "terms_accepted_by_user": False,
                "conversion_ready_now": False,
                "estimated_t50_windows_after_terms": 1,
                "path_summaries": [{"has_obsmat": True}],
            }
            for dataset_id in [
                "ucy_crowd_original",
                "eth_biwi_original",
                "trajnetplusplus_official",
                "aerialmpt_or_other_topdown",
                "opentraj_toolkit",
            ]
        ],
        "actions": {"downloaded": False, "converted": True, "trained": False, "evaluated": False},
        "user_action_required": {"exists": True},
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "readme_updates": {"readmes_updated": True},
    }
    gate = hy._gate(payload)
    assert gate["gates"]["no_conversion"] is False


def test_source_identity_hint_keeps_opentraj_as_user_confirmed_mirror() -> None:
    hint = hy._source_identity_hint("ucy_crowd_original", "external_data/OpenTraj/datasets/UCY")
    assert "OpenTraj" in hint
    assert "user must confirm" in hint
