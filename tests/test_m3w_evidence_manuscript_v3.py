import copy
import json
import re
from pathlib import Path

import pytest

from scripts.build_m3w_evidence_manuscript_v3 import (
    BASE, OUTPUT, ROOT, SOURCES, artifacts, build, load_sources,
)


@pytest.fixture(scope="module")
def docs():
    return load_sources()


def test_explicit_roles_and_no_scientific_promotion(docs):
    e = build(docs)
    assert e["numerical_results_source"] == "cached_verified"
    assert e["assembly_source"] == "fresh_run"
    assert e["temporal_auxiliary_readout"] == "not_run"
    assert (e["planned_training_fits"], e["trained_new_auxiliary_heads"]) == (216, 0)
    for key in ("new_training", "new_forecast_evaluation", "new_bootstrap",
                "independent_confirmation", "risk_calibrated", "deployment_changed",
                "submission_ready", "stage5c_executed", "smc_enabled"):
        assert e[key] is False


def test_selected_support_is_not_promoted_to_twelve_localities(docs):
    rows = {r["id"]: r for r in build(docs)["european_contrasts"]}
    assert rows["E9"]["localities"] == 11
    assert rows["E9"]["ci_low"] < 0 < rows["E9"]["ci_high"]
    assert rows["E6"]["localities"] == 12
    assert rows["E6"]["ci_high"] < 0
    assert rows["E6"]["unit"] != rows["E3"]["unit"]


def test_every_contrast_has_explicit_source_unit_and_role(docs):
    e = build(docs)
    assert len(e["european_contrasts"]) == 11
    assert len(e["sdd_contrasts"]) == 4
    for row in e["european_contrasts"]:
        assert row["source"] in SOURCES
        assert row["json_pointer"].startswith("/")
        assert row["result_source"] == "cached_verified"
        assert row["evidence_role"] == "exposed_development"
    assert e["sdd_contrasts"][0]["CI_low_pp"] < 0
    assert e["sdd_contrasts"][1]["CI_low_pp"] > 0


def test_policy_failure_and_unknowns_preserved(docs):
    policies = {p["arm"]: p for p in build(docs)["european_policies"]}
    assert len(policies) == 5
    p = policies["extended"]
    assert (p["known_label_violations"], p["violations"]) == (4, 7)
    assert (p["unknown_selected"], p["defined_easy_risk"]) == (926, 47)
    assert p["worst_easy_upper"] > .02
    assert policies["additive"]["worst_easy_upper"] > 1


def test_temporal_cancellation_not_improvement(docs):
    e = build(docs)
    assert e["cancellation_fraction"] == pytest.approx(.39838, abs=1e-5)
    assert e["temporal_selected_counts"]["unknown_selected"] == 918
    assert not e["deployment_changed"]


@pytest.mark.parametrize("mutation,message", [
    (lambda d: d["temporal"].update(independent_confirmation=True), "Development"),
    (lambda d: d["extension"].update(localities=13), "population"),
    (lambda d: d["extension"].update(deployment_changed=True), "Deployment"),
    (lambda d: d["training_status"]["pilot_attempt"].update(real_optimizer_updates=1), "snapshot"),
    (lambda d: d["readout_registration"].update(risk_budget=.03), "contract"),
    (lambda d: d["readout_registration"]["expected_source_heads"].pop(), "heads"),
    (lambda d: d["temporal"]["pooled_validation"].update(selected_harm=0), "identity"),
])
def test_invalid_changes_fail_closed(docs, mutation, message):
    d = copy.deepcopy(docs)
    mutation(d)
    with pytest.raises(ValueError, match=message):
        build(d)


def test_changed_statistical_arithmetic_is_rejected(docs):
    d = copy.deepcopy(docs)
    d["temporal"]["contrasts"]["validation_temporal_minus_rowmean_leaf_signed_error"]["mean"] += 1
    with pytest.raises(ValueError, match="locality average"):
        build(d)


def test_changed_hash_is_rejected(tmp_path):
    relative = SOURCES["sdd"][0]
    p = tmp_path / BASE / relative
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    with pytest.raises(ValueError, match="Changed source"):
        load_sources(tmp_path)


def test_saved_exports_match_sources_byte_for_byte(docs):
    folder = ROOT / OUTPUT
    expected = artifacts(docs, (folder / "manuscript.template.md").read_text())
    for filename, text in expected.items():
        assert (folder / filename).read_bytes() == text.encode()
        assert "\r" not in text
    assert json.loads(expected["evidence.json"]) == build(docs)


def test_manuscript_carries_main_limitations_and_new_prior_work(docs):
    text = artifacts(docs, (ROOT / OUTPUT / "manuscript.template.md").read_text())["manuscript.md"]
    for phrase in ("not a submission-ready paper", "not_run", "eleven supported localities",
                   "Conformal Policy Control", "zero optimizer updates", "not independent",
                   "not the probability of failure", "Stage5C has not been executed; SMC is off"):
        assert phrase in " ".join(text.split())
    assert "{{" not in text


def test_missing_table_marker_is_rejected(docs):
    template = (ROOT / OUTPUT / "manuscript.template.md").read_text()
    with pytest.raises(ValueError, match="table slot"):
        artifacts(docs, template.replace("{{EU_TABLE}}", ""))


def test_local_document_links_resolve():
    for file in (ROOT / OUTPUT).glob("*.md"):
        text = file.read_text()
        assert not re.search(r"\]\s+\(", text), file
        for dest in re.findall(r"\]\(([^)]+)\)", text):
            if "://" not in dest and not dest.startswith("#"):
                assert (file.parent / dest.split("#")[0]).resolve().exists(), (file, dest)
