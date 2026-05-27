from src.stage42_hz_to_cg_intake_bridge import _bridge_templates, _confirmation_missing, _hz_to_user_confirmation


def test_hz_to_user_confirmation_preserves_blank_legal_block():
    hz_row = {
        "dataset_id": "ucy_crowd_original",
        "official_url": "https://example.test/ucy",
        "local_path": "external_data/OpenTraj/datasets/UCY",
        "terms_accepted_by_user": False,
        "terms_acceptance_date": "",
        "allowed_use": "",
        "derived_data_allowed": None,
        "redistribution_allowed": None,
        "source_identity": "",
        "confirmed_by_user": "",
        "official_source_url_confirmed": False,
        "local_path_confirmed": False,
        "source_identity_confirmed": False,
    }
    confirmation = _hz_to_user_confirmation(hz_row)
    missing = _confirmation_missing(confirmation)
    assert confirmation["terms_accepted_by_user"] is False
    assert "terms_accepted_by_user" in missing
    assert "official_source_url_confirmed" in missing


def test_hz_to_user_confirmation_allows_false_redistribution_as_confirmed_value():
    hz_row = {
        "dataset_id": "ucy_crowd_original",
        "official_url": "https://example.test/ucy",
        "local_path": "external_data/OpenTraj/datasets/UCY",
        "terms_accepted_by_user": True,
        "terms_acceptance_date": "2026-05-27",
        "allowed_use": "research_only",
        "derived_data_allowed": True,
        "redistribution_allowed": False,
        "source_identity": "official_source_confirmed_by_user",
        "confirmed_by_user": "tester",
        "official_source_url_confirmed": True,
        "local_path_confirmed": True,
        "source_identity_confirmed": True,
    }
    confirmation = _hz_to_user_confirmation(hz_row)
    assert confirmation["redistribution_allowed"] is False
    assert _confirmation_missing(confirmation) == []


def test_bridge_templates_maps_hz_rows_without_activating_validator_input():
    hz = {
        "confirmations": [
            {
                "dataset_id": "ucy_crowd_original",
                "official_url": "https://example.test/ucy",
                "local_path": "external_data/OpenTraj/datasets/UCY",
                "terms_accepted_by_user": False,
                "terms_acceptance_date": "",
                "allowed_use": "",
                "source_identity": "",
                "confirmed_by_user": "",
            }
        ]
    }
    cg = {
        "datasets": [
            {
                "dataset_id": "ucy_crowd_original",
                "domain": "UCY",
                "official_url_from_prior_audit": "https://example.test/ucy",
                "user_confirmation": {},
            }
        ]
    }
    bridged, rows = _bridge_templates(hz, cg)
    assert bridged["active_validator_input"] is False
    assert bridged["datasets"][0]["stage42_ia_bridge"]["active_validator_input"] is False
    assert rows[0]["ready_if_activated_now"] is False


def test_bridge_templates_can_express_ready_if_user_confirms_everything():
    hz = {
        "confirmations": [
            {
                "dataset_id": "ucy_crowd_original",
                "official_url": "https://example.test/ucy",
                "local_path": "external_data/OpenTraj/datasets/UCY",
                "terms_accepted_by_user": True,
                "terms_acceptance_date": "2026-05-27",
                "allowed_use": "research_only",
                "derived_data_allowed": True,
                "redistribution_allowed": False,
                "source_identity": "official_source_confirmed_by_user",
                "confirmed_by_user": "tester",
                "official_source_url_confirmed": True,
                "local_path_confirmed": True,
                "source_identity_confirmed": True,
            }
        ]
    }
    cg = {
        "datasets": [
            {
                "dataset_id": "ucy_crowd_original",
                "domain": "UCY",
                "official_url_from_prior_audit": "https://example.test/ucy",
                "user_confirmation": {},
            }
        ]
    }
    _, rows = _bridge_templates(hz, cg)
    assert rows[0]["ready_if_activated_now"] is True
