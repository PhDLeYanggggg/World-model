import copy
import json
from pathlib import Path

import pytest
from scripts.build_m3w_evidence_manuscript_v2 import (
    ROOT, OUTPUT, SOURCES, ARMS, MANIFEST_SHA, build, load_sources,
    pinned_json, pointer, table_text, verify_local_lineage,
)


@pytest.fixture(scope="module")
def docs():
    return load_sources(ROOT)


def test_source_scope_and_false_claims(docs):
    e = build(docs)
    assert len(SOURCES) == 16
    assert (len(e["rows"]), len(e["site_metrics"]), len(e["site_seed_metrics"])) == (8, 32, 96)
    for key in ("new_training", "new_target_readout", "new_bootstrap", "independent_confirmation",
                "risk_calibrated", "deployment", "stage5c_executed", "smc_enabled", "submission_ready",
                "original_forest_primary_pass", "square_loss_primary_pass"):
        assert e[key] is False


def test_every_value_has_a_source_and_no_seed_is_dropped(docs):
    e = build(docs)
    for r in e["rows"]:
        s = pointer(docs[r["source"]], r["json_pointer"])
        assert r["predictor"] == "eqmotion_fixed_head_ramp_action"
        assert r["ADE_gain_pct"] == pytest.approx(s["ADE"]["equal_scene_gain_percent"])
        assert r["selected"] == sum(v["selected"] for v in s["seeds"].values())
        assert r["worst_site_seed_easy_degradation_pct"] == max(
            -v["gain_percent"] for k in s["seeds"].values()
            for v in k["subsets"]["positive_easy"]["by_scene"].values())


def test_primary_failures_not_replaced_by_same_count_success(docs):
    e = build(docs)
    c = {r["label"]: r for r in e["contrasts"]}
    assert c["Forest minus log neural (original primary)"]["CI_low_pp"] < 0
    assert c["Forest minus log neural (same-count risk)"]["CI_low_pp"] > 0
    assert c["Square minus log neural (strict primary)"]["CI_low_pp"] < 0
    assert c["Square minus log neural (same-count risk)"]["CI_high_pp"] < 0
    assert all(not r["observed_easy_ceiling_pass"] for r in e["rows"] if r["arm"].endswith("gain"))


def test_missing_outcomes_not_certified_safe(docs):
    rows = {r["arm"]: r for r in build(docs)["rows"]}
    assert rows["square_strict"]["selected_unknown"] == 606
    assert rows["square_strict"]["selected_incomplete"] == 5680
    assert rows["forest_ratio"]["selected_unknown"] == 190
    assert rows["log_strict"]["zero_CV_harmed"] == 1


def test_joint_opportunities_not_reported_as_outcomes(docs):
    e = build(docs)
    assert sum(r["nonadditive_opportunities"] for r in e["joint_support"]) == 266
    assert sum(r["combinations_enumerated"] for r in e["joint_support"]) == 61024
    assert all(r["predictive_lift_status"] == "not_run" for r in e["joint_support"])
    bad = copy.deepcopy(docs)
    bad["protected_joint_support_v1"]["outcomes_evaluated"] = True
    with pytest.raises(ValueError, match="scope"):
        build(bad)


def test_arithmetic_tampering_is_rejected(docs):
    bad = copy.deepcopy(docs)
    bad["fraction_square_v1"]["contrasts"]["square_strict_minus_log_strict"]["mean_gain_difference_pp"] += 1
    with pytest.raises(ValueError, match="equal-physical"):
        build(bad)


def test_hash_tampering_is_rejected(tmp_path):
    p = tmp_path / "changed.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="Changed evidence"):
        pinned_json(p, "0"*64)


def test_manuscript_main_table_exactly_matches_export(docs):
    text = (ROOT / OUTPUT / "manuscript.md").read_text()
    table = table_text(build(docs))
    for line in table.splitlines():
        if any(line.startswith("| "+label+" |") for _, label in ARMS):
            assert line in text
    assert "not an independently confirmed method" in text
    assert "Worst easy" in text
    assert "Abhin Shah" in text
    assert "Harshay Shah" not in text


def test_saved_export_matches_sources(docs):
    saved = json.loads((ROOT / OUTPUT / "evidence.json").read_text())
    assert saved == build(docs)


def test_lineage_validation_rejects_wrong_family(monkeypatch, docs):
    sites = ["coupa", "deathCircle", "gates", "hyang"]
    views = []
    archives = {r["view"]: r for r in docs["native_eqmotion_v1"]["archives"]}
    def producer(excluded, seed):
        return dict(family="eqmotion_fixed_head", excluded_sites=list(excluded),
            training_sites=sorted(set(sites)-excluded), preprocessing_fit_sites=sorted(set(sites)-excluded),
            seed=seed, checkpoint_selection_sites=[], calibration_sites=[])
    for s in sites:
        for seed in (17, 29, 43):
            views.append(dict(outer_site=s, seed=seed,
                outer_producer=dict(producer=producer({s}, seed), prediction=archives[f"{s}_seed{seed}"]),
                groups=[dict(inner_site=t, producer=producer({s, t}, seed)) for t in sites if t != s]))
    monkeypatch.setattr("scripts.build_m3w_evidence_manuscript_v2.pinned_json", lambda *a: {"views": views})
    receipt = verify_local_lineage(Path("/unused"), docs)
    assert receipt["outer_views"] == 12 and receipt["inner_groups"] == 36
    views[0]["outer_producer"]["producer"]["family"] = "transformer"
    with pytest.raises(ValueError, match="family"):
        verify_local_lineage(Path("/unused"), docs)


def test_local_metadata_receipt_does_not_claim_checkpoint_replay():
    r = json.loads((ROOT / OUTPUT / "local_lineage_verification.json").read_text())
    assert r["manifest_sha256"] == MANIFEST_SHA
    assert r["no_prediction_or_target_arrays_read"]
    assert not r["independent_confirmation"]
