import json
from pathlib import Path
import numpy as np

from scripts import report_m3w_selection_uncertainty as report
from src.world_model import m3w_selection_uncertainty as api


def test_registered_ensemble_ancestry_and_feature_replay():
    reg=json.loads((report.HOME/"registration.json").read_text())
    for field in ("bindings","dependencies"):
        for path,digest in reg[field].items():assert report.sha(report.ROOT/path)==digest
    assert reg["fits"]==27 and len(reg["fields"])==16 and len(reg["folds"])==3
    for fold in reg["folds"]:
        assert len(fold["members"])==3
        assert not set(fold["producer_sites"])&set(fold["cost_sites"])
    replay=json.loads((report.HOME/"feature_verification.json").read_text())
    assert replay["all6_feature_arrays_exact"]
    frozen=json.loads((report.HOME/"feature_freeze.json").read_text())
    assert len(frozen["features"])==6 and not frozen["future_outcomes_used"]
    source_rows=[]
    for ref in frozen["features"]:
        f=report.public(ref,"features");path=report.ROOT/f["array"]["path"]
        assert report.sha(path)==f["array"]["sha256"]
        x=np.load(path,mmap_mode="r",allow_pickle=False)
        assert x.shape==(f["rows"],16) and np.isfinite(x).all() and (x>=0).all()
        if f["domain"]=="source":
            source_rows.append(f["rows"])
            assert not set(f["sites"])&set(f["producer_sites"])
        else:assert f["rows"]==38102
    assert sorted(source_rows)==[62036,91562,145422]
    assert not any(reg[k] for k in ("threshold_search","new_joint_solver","calibration_access","confirmation_access",
        "selection_labels_for_training","stage5c_executed","smc_enabled"))


def test_all27_complete_fits_match_and_disabled_models_equal_old_controls():
    frozen=json.loads((report.HOME/"training_freeze.json").read_text());assert len(frozen["fits"])==27
    assert frozen["all9_disabled_original_controls_exact"] and not frozen["selection_labels_used"]
    original=report.HOME.parent/"european_selection_cardinal_cost_v1/fits"
    grouped={}
    for ref in frozen["fits"]:
        fit=report.public(ref,"fits");path=report.ROOT/fit["checkpoint"]["path"]
        assert report.sha(path)==fit["checkpoint"]["sha256"]
        cp=api.base.read_checkpoint(path)
        assert fit["step"]==fit["budget"] and fit["minimum_query_visits"]>=4 and fit["parameters"]==30277
        assert fit["unknown_supervised_rows"]==0 and not set(fit["source_sites"])&set(fit["producer_sites"])
        assert set(cp["uncertainty_preprocess"]["catalog_source_ids"])<=set(cp["fit_ids"])
        grouped.setdefault(fit["trial"],[]).append(cp)
        if fit["arm"]=="disabled":
            old=json.loads((original/(fit["trial"]+"_mean_absolute.json")).read_text())
            other=api.base.read_checkpoint(report.ROOT/old["checkpoint"]["path"])
            api.compare_disabled(cp,other)
    for cells in grouped.values():
        for cp in cells[1:]:
            for key in ("initial_model","sampler","draw_hash","fit_ids","preprocess","uncertainty_preprocess","torch_rng"):
                api.base.exact(cells[0][key],cp[key])
    for arm in api.ARMS:assert json.loads((report.HOME/("first_full_refit_"+arm+".json")).read_text())["scientific_state_exact"]


def test_scalar_report_never_reads_private_inputs(monkeypatch):
    original=Path.open
    def guarded(path,*args,**kwargs):
        mode=args[0] if args else kwargs.get("mode","r")
        if "r" in mode:assert not path.resolve().is_relative_to(report.ROOT/"data")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    s,rows,fits,t,c,v=report.verified()
    assert len(rows)==54 and len(fits)==27 and s["unique_target_rows"]==38102
    assert s["policy_candidate_pass"]==(s["primary_pass"] and s["negative_control_pass"] and s["original_record_risk_pass"])
    assert not any(s[k] for k in ("threshold_search","new_joint_solver","selection_labels_used_for_training",
        "calibration_access","confirmation_access","deployment_changed","submission_ready","stage5c_executed","smc_enabled"))
    assert report.render()["private_input_reads"]==0


def test_public_traces_and_reviewed_figures_match_frozen_values():
    traces=json.loads((report.HOME/"training_traces.json").read_text())
    assert len(traces["models"])==27 and not traces["labels_or_weights_exported"]
    assert traces["training_freeze_sha256"]==report.sha(report.HOME/"training_freeze.json")
    assert sum(len(r["trace"]) for r in traces["models"])==2151
    for row in traces["models"]:
        fit=json.loads((report.HOME/"fits"/(row["trial"]+"_"+row["arm"]+".json")).read_text())
        assert row["trace"][0]==fit["first_trace"] and row["trace"][-1]==fit["final_trace"]
    review=json.loads((report.HOME/"figure_review.json").read_text())
    assert review["summary_sha256"]==report.sha(report.HOME/"summary.json")
    for name,digest in review["svg_sha256"].items():assert report.sha(report.HOME/"figures"/name)==digest
    assert review["both_final_figures_visually_reviewed"] and not review["metric_changed"]
