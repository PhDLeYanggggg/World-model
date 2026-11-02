import json
from pathlib import Path

from src.evaluation.m3w_selection_forecast_report import RELATIVE, load, render

ROOT = Path(__file__).resolve().parents[1]


def test_complete_scalar_replay_retains_failed_primary_and_safety():
    _, summary, complete, verify, rows, _ = load(ROOT)
    assert len(rows) == 54 and len(complete["groups"]) == 54
    assert summary["unique_target_rows"] == 38102
    assert summary["actual_forecaster_training_seeds"] == [17, 29, 43]
    assert summary["primary"]["known_pp"]["CI95"][0] > 0
    assert summary["primary"]["completion_lower_pp"]["CI95"][1] < 0
    assert not summary["primary_lower_positive"]
    assert not summary["history_original_record_risk_pass"]
    assert summary["safety"]["history"]["supported_record_views"] == 0
    assert summary["safety"]["history"]["worst_known_easy_net_degradation_percent"] > 2
    assert verify["all28_raw_labels_reparsed"] and verify["all54_metric_groups_exact"]
    assert summary["model_selection_outcomes_opened"]
    assert not any(summary[k] for k in ("calibration_access", "confirmation_access", "deployment_changed", "submission_ready"))


def test_report_rebuild_denies_all_private_file_reads(monkeypatch):
    original = Path.open
    opened = []

    def guarded(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if "r" in mode:
            resolved = path.resolve()
            assert not resolved.is_relative_to(ROOT / "data"), "Private file read in scalar report"
            opened.append(resolved)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    receipt = render(ROOT)
    assert receipt["private_input_reads"] == 0 and opened
    prose = (ROOT / RELATIVE / "readout.md").read_text()
    assert "The primary fails" in prose
    assert "now exposed for model selection" in prose
    assert "not an\nuntouched test" in prose


def test_original_contract_files_remain_hash_bound():
    import hashlib
    registration = json.loads((ROOT / RELATIVE / "registration.json").read_text())
    for key in ("bindings", "dependencies"):
        for path, expected in registration[key].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
