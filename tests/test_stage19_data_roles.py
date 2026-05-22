from __future__ import annotations

from pathlib import Path

from src.stage14_pipeline import read_json
from src.stage19_pipeline import build_wam_data_registry, build_wam_jepa_dataset, generate_simulation_data


def test_stage19_registry_has_wam_categories_and_no_unauthorized_downloads():
    registry = build_wam_data_registry()
    categories = set(registry["category_counts"])
    assert "real_topdown_trajectory" in categories
    assert "simulation data" in categories
    assert "human_egocentric_video" in categories
    assert registry["legal_download_policy"].startswith("No unauthorized")
    ego_rows = [row for row in registry["sources"] if row["category"] == "human_egocentric_video"]
    assert ego_rows
    assert all(row["auto_download_allowed"] is False for row in ego_rows)


def test_stage19_simulation_is_not_official_real_eval():
    generate_simulation_data(quick=True)
    sim_files = sorted(Path("data/stage19_simulation").glob("*.npz"))
    assert sim_files
    report = read_json("outputs/reports/stage19_simulation_report.json", {})
    assert report["official_real_success"] is False
    assert "not real-world success" in report["sim_to_real_gap"]


def test_stage19_wam_dataset_separates_data_roles():
    result = build_wam_jepa_dataset(quick=True)
    assert result["role_separation"] is True
    samples = read_json("data/stage19_wam_jepa_dataset/samples.json", [])
    assert samples
    sim_samples = [sample for sample in samples if sample["data_role"] == "simulation_curriculum"]
    assert sim_samples
    assert all(sample["official_real_eval"] is False for sample in sim_samples)

