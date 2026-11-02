import json
from pathlib import Path
from scripts import report_m3w_selection_cost_transport as report


def test_scalar_diagnosis_rebuild_never_reads_private_arrays(monkeypatch):
    original=Path.open
    def guarded(path,*args,**kwargs):
        mode=args[0] if args else kwargs.get("mode","r")
        if "r" in mode:assert not path.resolve().is_relative_to(report.ROOT/"data")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    s,c,v=report.verified()
    assert len(c["groups"])==252 and s["source_resubstitution_not_validation"]
    assert v["selection_predictions_cached_verified"]
    for domain in ("source_fit","selection"):
        for arm in ("row","context"):
            a=s["pooled_descriptive_action_view_diagnostics"][domain][arm]
            assert sum(x["rows"] for x in a["neural_funnel"].values())==a["useful_known_neural_rows"]
            assert a["subsets"]["selected"]["observed_easy_risk_on_known"]>.02
    assert report.render()["private_input_reads"]==0
    assert not any(s[k] for k in ("policy_changed","new_policy_evidence","calibration_access","confirmation_access",
        "deployment_changed","submission_ready","stage5c_executed","smc_enabled"))
