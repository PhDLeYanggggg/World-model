from src.stage42_local_calibrated_source_terms_prefill import (
    _gate,
    _links_from_text,
    run_stage42_local_calibrated_source_terms_prefill,
)


def test_link_extractor_keeps_unique_http_urls():
    links = _links_from_text("see https://example.org/a and http://example.org/b. and https://example.org/a")
    assert links == ["https://example.org/a", "http://example.org/b"]


def test_prefill_keeps_user_acceptance_empty():
    payload = run_stage42_local_calibrated_source_terms_prefill(refresh_readmes=False)
    for row in payload["prefill_rows"]:
        must = row["must_be_filled_by_user"]
        assert must["terms_accepted_by_user"] is False
        assert must["accepted_by_user"] == ""
        assert row["conversion_ready_now"] is False
    assert payload["summary"]["conversion_ready_now"] == 0


def test_gate_requires_official_hints_without_conversion():
    payload = run_stage42_local_calibrated_source_terms_prefill(refresh_readmes=False)
    gate = _gate(payload)
    assert gate["verdict"] == "stage42_jp_local_calibrated_source_terms_prefill_pass"
    assert payload["summary"]["official_hint_rows"] >= 2
    assert payload["summary"]["converted_now"] == 0
    assert payload["claim_boundary"]["prefill_is_permission"] is False
    wild = next(row for row in payload["prefill_rows"] if row["dataset_name"] == "Wild-Track")
    assert any("epfl.ch" in url for url in wild["official_url_candidates"])
    pets = next(row for row in payload["prefill_rows"] if row["dataset_name"] == "PETS-2009-S2L1")
    assert any("reading.ac.uk" in url for url in pets["official_url_candidates"])
