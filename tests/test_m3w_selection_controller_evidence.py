import hashlib
import json
from pathlib import Path

from scripts import report_m3w_selection_controller as report
from src.world_model.m3w_causal_descriptor_head import read_checkpoint
from scripts.replay_m3w_dimensionless_training import exact

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT/"outputs/publication_readiness_2026_09/european_selection_controller_v1"


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_and_amended_protocol_and_failed_pilot_remain_bound():
    first = json.loads((HOME/"registration.json").read_text())
    current = json.loads((HOME/"registration_v2.json").read_text())
    assert current["original_registration"]["sha256"] == digest(HOME/"registration.json")
    assert current["original_failed_recovery_pilot"]["sha256"] == digest(HOME/"pilot_failure_v1.json")
    for path, value in current["bindings"].items(): assert digest(ROOT/path) == value
    for path, value in current["dependencies"].items(): assert digest(ROOT/path) == value
    assert first["fit_scopes"] == current["fit_scopes"]
    assert first["primary"] == current["primary"]
    failure = json.loads((HOME/"pilot_failure_v1.json").read_text())
    assert not failure["context_arm_recovery_scientific_state_exact"]
    assert not failure["recovery_tolerance_relaxed"]
    assert json.loads((HOME/"pilot.json").read_text())["engineering_pass"]


def test_all18_real_full_source_fits_matched_and_not_selection_trained():
    freeze = json.loads((HOME/"training_freeze.json").read_text())
    assert len(freeze["fits"]) == 18 and not freeze["selection_labels_used"]
    cells = {}
    for ref in freeze["fits"]:
        path = ROOT/ref["path"]; assert digest(path) == ref["sha256"]
        fit = json.loads(path.read_text())
        assert fit["step"] == fit["budget"] >= 2000
        assert fit["minimum_query_visits"] >= 4 and fit["never_drawn_queries"] == 0
        assert fit["unknown_supervised_rows"] == 0 and fit["parameters"] == 29253
        assert len(fit["source_sites"]) == 8 and len(fit["producer_sites"]) == 4
        assert not set(fit["source_sites"]) & set(fit["producer_sites"])
        cp = ROOT/fit["checkpoint"]["path"]; assert digest(cp) == fit["checkpoint"]["sha256"]
        state = read_checkpoint(cp)
        cells.setdefault(fit["trial"], {})[fit["arm"]] = state
    assert len(cells) == 9
    for arms in cells.values():
        for key in ("initial_model", "parameters", "budget", "sampler", "draw_hash", "fit_ids", "preprocess", "torch_rng"):
            exact(arms["row"][key],arms["context"][key])
    for arm in ("row","context"):
        assert json.loads((HOME/("first_full_refit_"+arm+".json")).read_text())["scientific_state_exact"]


def test_complete_policy_summary_replays_without_private_inputs(monkeypatch):
    original = Path.open
    def guarded(path,*args,**kwargs):
        mode = args[0] if args else kwargs.get("mode","r")
        if "r" in mode:
            assert not path.resolve().is_relative_to(ROOT/"data"), "Private input in scalar reconstruction"
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    summary,rows,_,_,_,verify = report.verified()
    assert len(rows) == 54 and summary["selection_localities"] == 6
    assert summary["unique_target_rows"] == 38102
    assert summary["policy_candidate_pass"] == (summary["primary_pass"] and summary["original_record_risk_pass"])
    assert verify["all18_cost_inference_exact"] and verify["all54_frozen_action_metric_groups_exact"]
    assert not verify["optimization_reexecuted"]
    assert not summary["selection_labels_used_for_training"]
    assert not any(summary[k] for k in ("calibration_access","confirmation_access","pair_interactions_optimized",
                                      "deployment_changed","submission_ready","stage5c_executed","smc_enabled"))
    for row in rows:
        for view in row["count_matched_record_support"].values():
            if not view["nonempty"]: assert view["status"] == "not_estimable_empty_count_matched_control"
    assert report.render()["private_input_reads"] == 0


def test_causal_action_audit_does_not_turn_predicted_gain_into_real_benefit():
    audit = json.loads((HOME/"causal_action_diagnosis.json").read_text())
    assert not any(audit[k] for k in ("selection_label_arrays_read","future_targets_read","new_training","threshold_search","policy_changed"))
    assert len(audit["cells"]) == 18
    for arm in ("row","context"):
        values = audit["aggregate"][arm]
        assert values["rows"] == 38102*9
        assert (values["predicted_neural_jointly_admitted_rows"] + values["predicted_neural_gain_positive_but_guard_rejected_rows"]
                == values["predicted_neural_positive_gain_rows"])
        independent = values["policy_choices"]["independent"]
        joint = values["policy_choices"]["joint_matched"]
        assert independent["selected"] == joint["selected"] > 0
        assert independent["selected"] == independent["neural"] + independent["physical"]
    assert audit["scope"] == "causal_admission_volume_not_actual_harm_calibration_or_irreducible_uncertainty"
