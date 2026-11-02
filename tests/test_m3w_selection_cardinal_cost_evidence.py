from pathlib import Path
import hashlib
import json

from scripts import report_m3w_selection_cardinal_cost as report
from src.world_model.m3w_causal_descriptor_head import read_checkpoint
from scripts.replay_m3w_dimensionless_training import exact


def test_registered_scope_retains_failed_parent_and_closed_roles():
    reg=json.loads((report.HOME/"registration.json").read_text())
    for key in ("bindings","dependencies"):
        for path,digest in reg[key].items():assert report.sha(report.ROOT/path)==digest
    assert reg["new_fits"]==27 and reg["cached_control_fits"]==9
    assert reg["selection_already_exposed"]
    assert not any(reg[k] for k in ("new_joint_solver","model_selection_labels_for_training","threshold_search",
        "calibration_access","confirmation_access","stage5c_executed","smc_enabled"))
    pilot=json.loads((report.HOME/"pilot.json").read_text())
    assert pilot["engineering_pass"] and pilot["optimizer_updates"]==800
    assert all(v["exact_recovery"] for v in pilot["cells"])
    assert pilot["cells"][0]["original_control_100_step_exact"]


def test_all27_fits_full_and_same_control_draws_initialization():
    training=json.loads((report.HOME/"training_freeze.json").read_text())
    parent=report.ROOT/"outputs/publication_readiness_2026_09/european_selection_controller_v1/fits"
    assert len(training["fits"])==27 and not training["selection_labels_used"]
    for ref in training["fits"]:
        fit=report.public(ref,"fits");cp=report.ROOT/fit["checkpoint"]["path"]
        assert report.sha(cp)==fit["checkpoint"]["sha256"]
        old=json.loads((parent/(fit["trial"]+"_context.json")).read_text())
        assert fit["step"]==fit["budget"]==old["budget"]
        assert fit["parameters"]==29253 and fit["minimum_query_visits"]>=4
        assert fit["draw_hash"]==old["draw_hash"] and fit["fit_ids_sha256"]==old["fit_ids_sha256"]
        assert fit["unknown_supervised_rows"]==0 and not set(fit["source_sites"])&set(fit["producer_sites"])
        a=read_checkpoint(cp);b=read_checkpoint(report.ROOT/old["checkpoint"]["path"])
        for k in ("initial_model","sampler","preprocess","torch_rng"):exact(a[k],b[k])
    for arm in ("mean_centered","weighted_absolute","mean_absolute"):
        assert json.loads((report.HOME/("first_full_refit_"+arm+".json")).read_text())["scientific_state_exact"]


def test_complete_report_only_reads_public_scalar_evidence(monkeypatch):
    original=Path.open
    def guarded(path,*args,**kwargs):
        mode=args[0] if args else kwargs.get("mode","r")
        if "r" in mode:assert not path.resolve().is_relative_to(report.ROOT/"data")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    s,rows,fits,t,c,v=report.verified()
    assert len(rows)==54 and len(fits)==27 and s["unique_target_rows"]==38102
    assert s["policy_candidate_pass"]==(s["primary_pass"] and s["original_record_risk_pass"])
    assert report.render()["private_input_reads"]==0
    assert not any(s[k] for k in ("selection_labels_used_for_training","threshold_search","calibration_access",
        "confirmation_access","new_joint_solver","deployment_changed","submission_ready","stage5c_executed","smc_enabled"))


def test_all_scalar_traces_and_reviewed_figures_are_bound():
    t=json.loads((report.HOME/"training_traces.json").read_text())
    assert t["training_freeze_sha256"]==report.sha(report.HOME/"training_freeze.json")
    assert len(t["models"])==27 and not t["labels_or_weights_exported"]
    assert sum(len(v["trace"]) for v in t["models"])==2151
    for model in t["models"]:
        fit=json.loads((report.HOME/"fits"/(model["trial"]+"_"+model["arm"]+".json")).read_text())
        assert model["trace"][0]==fit["first_trace"] and model["trace"][-1]==fit["final_trace"]
        assert model["trace"][-1]["step"]==fit["budget"]
    review=json.loads((report.HOME/"figure_review.json").read_text())
    assert review["summary_sha256"]==report.sha(report.HOME/"summary.json")
    for name,digest in review["svg_sha256"].items():assert report.sha(report.HOME/"figures"/name)==digest
    assert review["final_both_figures_visually_reviewed"] and not review["metric_changed"]
