"""Explain free-count failures without opening calibration or confirmation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_recipient_release_capacity as base
from src.evaluation.m3w_dual_reference_capacity import action_masses
from src.evaluation.m3w_selection_forecast_readout import errors


run, np = base.run, base.np


def main():
    binding, identity, cut = base.register()
    path = base.OUT/"analysis.json"
    result = run.read(path)
    if result["registration"] != binding or result["calibration_access"] or result["independent_confirmation"]:
        raise ValueError("Require fitting-only solver results")
    rows = []
    for rec, g, cv, ols, bank, y, valid, inv, counts, w in base.fitting_blocks(identity, lambda **_: None):
        r = next(x for x in result["recordings"] if x["recording"] == rec)["menus"]["curved_free"]
        if r["status"] == "verified_both":
            continue
        cv_error = errors(cv, y, valid)[0]
        easy = valid.any(1) & (cv_error > 0.) & (cv_error <= cut["easy"])
        m = action_masses(cv, ols, bank, y, valid, w, cut["easy"])
        singly_feasible = m["easy"] & (m["risk"][1] <= 0.)
        if r["status"] == "risk_infeasible" and singly_feasible.any():
            raise ValueError("A feasible single easy action contradicts free-count infeasibility")
        reason = ("no_easy_reference_in_roster" if not easy.any() else
                  "no_legal_easy_candidate" if not m["easy"].any() else
                  "no_easy_candidate_under_two_percent" if not singly_feasible.any() else
                  "negative_dual_reference_upper" if r["status"] == "negative_upper" else "unresolved")
        ref = float(w[valid.any(1)] @ cv_error[valid.any(1)])
        rows.append(dict(recording=rec, source=g, reason=reason, easy_rows=int(easy.sum()),
                         singly_feasible_easy_choices=int(singly_feasible.sum()),
                         maximin_upper_pp=(100*r["upper_gain_mass"]/ref
                                           if r["upper_gain_mass"] is not None and ref > 0 else None)))
    run.once(base.OUT/"support.json", run.stable(dict(parent=run.artifact(path), rows=rows,
        no_easy_reference_recordings=sum(r["easy_rows"] == 0 for r in rows),
        calibration_access=False, confirmation_access=False, risk_rule_changed=False)))


if __name__ == "__main__":
    main()
