from pathlib import Path

from src import stage42_post_bj_local_source_verification as s42bk


def test_xml_parser_finds_t100_tracks_in_eth_person() -> None:
    path = Path("external_data/OpenTraj/datasets/ETH-Person/data/seq0_assc_gt.xml")
    row = s42bk._parse_xml(path)
    assert row["domain"] == "ETH_UCY"
    assert row["file_format"] == "xml"
    assert row["t100_capable"] is True
    assert row["estimated_t100_windows"] > 0
    assert row["independent_key"].startswith("ETH_UCY::ETH-Person/data/")


def test_trajnet_snippets_are_not_t100_capable() -> None:
    path = Path("external_data/OpenTraj/datasets/TrajNet/Train/crowds/students001.txt")
    row = s42bk._parse_file(path)
    assert row["domain"] == "TrajNet"
    assert row["parsed_rows"] > 0
    assert row["max_track_points"] <= 20
    assert row["t100_capable"] is False


def test_classify_marks_eth_person_candidates_and_trajnet_loader_gap() -> None:
    rows = [
        {
            "relative_path": "ETH-Person/data/seq0_assc_gt.xml",
            "domain": "ETH_UCY",
            "independent_key": "ETH_UCY::ETH-Person/data/seq0_assc_gt",
            "file_format": "xml",
            "max_track_points": 303,
            "t100_capable": True,
            "estimated_t100_windows": 335,
            "synthetic_or_diagnostic": False,
        },
        {
            "relative_path": "TrajNet/Train/crowds/students001.txt",
            "domain": "TrajNet",
            "independent_key": "TrajNet::TrajNet/Train",
            "file_format": "txt",
            "parsed_rows": 100,
            "max_track_points": 20,
            "t100_capable": False,
            "estimated_t100_windows": 0,
            "synthetic_or_diagnostic": False,
        },
    ]
    bj = {
        "domain_support": {
            "ETH_UCY": {"independent_sources": 1, "additional_independent_sources_needed": 2},
            "UCY": {"independent_sources": 4, "additional_independent_sources_needed": 0},
            "TrajNet": {"independent_sources": 0, "additional_independent_sources_needed": 3},
        }
    }
    classified = s42bk._classify(rows, bj)
    assert classified["by_domain"]["ETH_UCY"]["independent_t100_groups"] == 1
    assert classified["conversion_candidates"][0]["relative_path"] == "ETH-Person/data/seq0_assc_gt.xml"
    assert classified["loader_gaps"][0]["domain"] == "TrajNet"


def test_gate_requires_loader_gap_and_no_overclaim() -> None:
    payload = {
        "source": "fresh_post_bj_local_source_verification",
        "bj_verdict": "stage42_bj_post_bi_t100_source_package_pass",
        "file_count": 10,
        "summary": {
            "eth_person_xml_candidates": ["a", "b", "c"],
            "can_repair_eth_ucy_with_local_candidates_after_license_confirmation": True,
            "trajnet_t100_capable_files": 0,
            "trajnet_loader_gap_files": 5,
            "auto_download_executed": False,
            "global_t100_positive_claim_allowed": False,
        },
        "claim_boundary": {
            "converted_dataset_claim_allowed": False,
            "metric_or_seconds_claim": False,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42bk._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_bk_local_source_verification_pass"
