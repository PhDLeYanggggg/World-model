from pathlib import Path

from src import stage42_eth_person_terms_audit as bm


def test_opentraj_mit_license_is_toolkit_only_for_eth_person() -> None:
    text = Path("external_data/OpenTraj/LICENSE.txt").read_text(encoding="utf-8")
    row = bm._classify_opentraj_license(text)
    assert row["license_name"] == "MIT"
    assert row["scope_classification"] == "software_toolkit_only"
    assert row["can_cover_eth_person_dataset"] is False


def test_eth_person_has_no_local_terms_file() -> None:
    files = bm._find_terms_files(Path("external_data/OpenTraj/datasets/ETH-Person"))
    assert files == []


def test_official_url_is_recorded_from_opentraj_readme() -> None:
    text = Path("external_data/OpenTraj/README.md").read_text(encoding="utf-8")
    assert bm._extract_eth_person_official_url(text) == "https://data.vision.ee.ethz.ch/cvl/aess/"


def test_terms_audit_gate_blocks_official_claim_despite_positive_bl() -> None:
    payload = bm._build_payload()
    gate = payload["stage42_bm_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_bm_eth_person_terms_audit_pass_claim_blocked"
    assert payload["summary"]["bl_technical_t100_all_folds_safe_positive"] is True
    assert payload["summary"]["official_converted_dataset_claim_allowed"] is False
    assert payload["summary"]["deployable_t100_claim_allowed"] is False
    assert payload["summary"]["global_t100_positive_claim_allowed"] is False
