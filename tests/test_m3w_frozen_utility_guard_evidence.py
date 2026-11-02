import json
from pathlib import Path
from scripts import report_m3w_frozen_utility_guard as report


def test_registered_freeze_keeps_models_and_original_two_percent_rules():
    reg=json.loads((report.HOME/"registration.json").read_text())
    for name,digest in reg["bindings"].items():assert report.sha(report.ROOT/name)==digest
    assert not any(reg[k] for k in ("new_training","threshold_search","new_joint_solver","calibration_access",
        "confirmation_access","stage5c_executed","smc_enabled"))
    f=json.loads((report.HOME/"policy_freeze.json").read_text());assert len(f["policies"])==9
    assert not f["labels_read_for_this_comparison"] and not f["new_cost_inference"]
    a=f["causal_diagnosis"]
    assert a["row_views"]==38102*9
    assert a["original_positive_real_guard_but_new_gain_nonpositive_action_views"]==30135
    assert a["switch_counts"]["fixed_utility_real_guard"]==52715


def test_complete_public_report_rebuild_never_opens_private_data(monkeypatch):
    original=Path.open
    def guarded(path,*args,**kwargs):
        mode=args[0] if args else kwargs.get("mode","r")
        if "r" in mode:assert not path.resolve().is_relative_to(report.ROOT/"data")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    s,rows,c,v,f=report.verified()
    assert len(rows)==54 and v["all9_choice_matrices_exact"] and v["all54_metric_groups_exact"]
    assert s["policy_candidate_pass"]==(s["primary_pass"] and s["advancement_pass"] and s["original_record_risk_pass"])
    assert not any(s[k] for k in ("new_training","new_cost_inference","threshold_search","new_joint_solver",
        "calibration_access","confirmation_access","deployment_changed","submission_ready","stage5c_executed","smc_enabled"))
    assert report.render()["private_input_reads"]==0
