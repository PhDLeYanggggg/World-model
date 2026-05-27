from src.stage42_source_terms_packet import _build_payload, _missing_fields, _validate_readiness


def test_missing_fields_requires_positive_confirmation():
    row = {
        "dataset_id": "ucy",
        "official_source_url_confirmed": True,
        "local_path_confirmed": True,
        "source_identity_confirmed": True,
        "terms_accepted_by_user": False,
        "terms_acceptance_date": "2026-05-27",
        "allowed_use": "research_only",
        "derived_data_allowed": True,
        "redistribution_allowed": False,
        "citation_required": True,
        "confirmed_by_user": "tester",
    }
    assert "terms_accepted_by_user" in _missing_fields(row)


def test_readiness_stays_blocked_for_blank_generated_template():
    packet_rows = [
        {
            "dataset_id": "ucy",
            "domain": "UCY",
            "official_url": "https://example.test/ucy",
            "best_local_path_candidate": "external_data/OpenTraj/datasets/UCY",
            "local_path_found": True,
        }
    ]
    template_rows = [
        {
            "dataset_id": "ucy",
            "local_path": "external_data/OpenTraj/datasets/UCY",
            "official_source_url_confirmed": False,
            "local_path_confirmed": False,
            "source_identity_confirmed": False,
            "terms_accepted_by_user": False,
            "terms_acceptance_date": "",
            "allowed_use": "",
            "derived_data_allowed": None,
            "redistribution_allowed": None,
            "citation_required": None,
            "confirmed_by_user": "",
        }
    ]
    readiness = _validate_readiness(packet_rows, template_rows)
    assert readiness["ready_count"] == 0
    assert readiness["blocked_count"] == 1
    assert readiness["rows"][0]["conversion_ready"] is False


def test_readiness_passes_only_after_all_terms_confirmed():
    packet_rows = [
        {
            "dataset_id": "ucy",
            "domain": "UCY",
            "official_url": "https://example.test/ucy",
            "best_local_path_candidate": "external_data/OpenTraj/datasets/UCY",
            "local_path_found": True,
        }
    ]
    template_rows = [
        {
            "dataset_id": "ucy",
            "local_path": "external_data/OpenTraj/datasets/UCY",
            "official_source_url_confirmed": True,
            "local_path_confirmed": True,
            "source_identity_confirmed": True,
            "terms_accepted_by_user": True,
            "terms_acceptance_date": "2026-05-27",
            "allowed_use": "research_only",
            "derived_data_allowed": True,
            "redistribution_allowed": False,
            "citation_required": True,
            "confirmed_by_user": "tester",
        }
    ]
    readiness = _validate_readiness(packet_rows, template_rows)
    assert readiness["ready_count"] == 1
    assert readiness["rows"][0]["reason"] == "ready_for_guarded_conversion_dry_run"


def test_build_payload_preserves_hy_boundaries():
    hy = {
        "prefill_rows": [
            {
                "dataset_id": "ucy_crowd_original",
                "name": "UCY",
                "domain": "UCY",
                "official_url": "https://example.test/ucy",
                "best_local_path_candidate": "external_data/OpenTraj/datasets/UCY",
                "local_path_found": True,
                "estimated_t50_windows_after_terms": 10,
                "estimated_t100_windows_after_terms": 5,
                "source_cv_after_terms": True,
                "path_summaries": [
                    {
                        "path": "external_data/OpenTraj/datasets/UCY",
                        "file_count": 2,
                        "total_size_bytes": 100,
                        "has_obsmat": True,
                        "has_homography_file": True,
                    }
                ],
            }
        ],
        "stage42_hy_gate": {"verdict": "stage42_hy_source_local_path_prefill_pass"},
    }
    payload = _build_payload(hy)
    assert payload["summary"]["targets"] == 1
    assert payload["summary"]["ready_for_guarded_conversion_now"] == 0
    assert payload["actions"]["converted"] is False
    assert payload["claim_boundary"]["metric_or_seconds_claim"] is False
