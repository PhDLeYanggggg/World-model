import copy
import pytest
from scripts.build_m3w_evidence_manuscript import ROOT, SOURCES, build, check_metric, load_sources, tables


@pytest.fixture(scope="module")
def documents():
    return load_sources(ROOT)


def test_pinned_development_artifacts(documents):
    assert set(documents) == set(SOURCES)
    result = build(documents)
    assert result["new_training"] is False
    assert result["independent_confirmation"] is False
    assert result["primary_gates"]["each_scene_seed_easy"] is False
    assert result["selected_instances"] == 29668
    assert result["selected_unknown_ADE"] == 423
    assert result["selected_incomplete"] == 4194


def test_arithmetic_is_not_window_weighted(documents):
    metric = copy.deepcopy(documents["conditional_cost_v1"]["summaries"]["strict_stop"]["ADE"])
    expected = check_metric(metric)
    for i, row in enumerate(metric["by_scene"].values()):
        row["rows"] = 10 ** i
    assert check_metric(metric) == expected
    metric["equal_scene_gain_percent"] += .1
    with pytest.raises(ValueError, match="equal-site"):
        check_metric(metric)


def test_zero_reference_not_smoothed(documents):
    metric = copy.deepcopy(documents["conditional_cost_v1"]["summaries"]["strict_stop"]["ADE"])
    metric["by_scene"]["coupa"]["reference_error"] = 0
    with pytest.raises(ValueError, match="epsilon"):
        check_metric(metric)


def test_negative_controls_and_provenance_survive_export(documents):
    result = build(documents)
    for row in result["rows"]:
        value = documents[row["source"]]
        for key in row["json_pointer"].strip("/").split("/"):
            value = value[key]
        assert row["ADE_gain_pct"] == pytest.approx(value["ADE"]["equal_scene_gain_percent"])
    assert result["paired_contrasts"][1]["mean_gain_difference_pp"] < 0
    assert result["paired_contrasts"][-1]["mean_gain_difference_pp"] == 0
    assert "gate fails" in tables(result)
    assert len(result["scene_seed_easy"]) == 12
    assert len(result["site_metrics"]) == 32
    assert result["native_vs_old_loss_relative_gain_percent"] == pytest.approx(5.558311955025696)


def test_manuscript_table_matches_export(documents):
    result = build(documents)
    manuscript = (ROOT / "outputs/publication_readiness_2026_09/evidence_manuscript_v1/manuscript.md").read_text()
    for row in result["rows"][1:]:
        values = (row["ADE_gain_pct"], row["FDE_gain_pct"], row["hard_gain_pct"], row["easy_degradation_pct"])
        assert " | " + " | ".join(f"{v:.3f}" for v in values) + " |" in manuscript


def test_changed_source_is_not_silently_accepted(tmp_path):
    name = next(iter(SOURCES))
    path = tmp_path / "outputs/publication_readiness_2026_09" / name / "analysis.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="Changed paper source"):
        load_sources(tmp_path)
