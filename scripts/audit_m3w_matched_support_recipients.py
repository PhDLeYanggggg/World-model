"""Exact hindsight risk check with each neural arm's recipients frozen."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_matched_support_moments as job
from src.evaluation.m3w_recipient_release_capacity import evidence


run, np = job.run, job.np
ARMS = ("marginal_record", "supervised_record", "supervised_query")


def optimize_paths(cv, bank, y, valid, recipients):
    """Minimize all original CV-relative risk numerators at a fixed set."""
    floor, candidates, y, valid = cv[recipients], bank[recipients], y[recipients], valid[recipients]
    if not len(recipients):
        return np.empty(0, int)
    radius = np.linalg.norm(candidates-floor[:, None], axis=-1).max(2)
    costs = np.stack([job.errors(p, y, valid)[0] for p in candidates.swapaxes(0, 1)], axis=1)
    value = np.where(valid.any(1)[:, None], costs, radius)
    # Do not replace an action with identity: that would change the budget.
    ranked = np.where(radius > 0., value, np.inf)
    if not np.isfinite(ranked.min(1)).all():
        raise ValueError("A frozen recipient has no legal path")
    return ranked.argmin(1)


def main():
    binding, identity, cut = job.register()
    freeze = run.read(job.OUT/"freeze.json")
    result = run.read(job.OUT/"analysis.json")
    if freeze["registration"] != binding or result["registration"] != binding or freeze["confirmation_access"]:
        raise ValueError("Require same frozen predictions")
    _, _, blocks = job.prepare(identity, lambda **_: None)
    block = {b["rec"]: b for b in blocks}
    rows = []
    for mark in freeze["predictions"]:
        d = run.read(run.checked(mark))
        b = block[d["recording"]]
        cells = {}
        for arm in ARMS:
            action, _ = d["decisions"][job.ARMS.index(arm)]
            on = np.flatnonzero(np.asarray(action) >= 0)
            k = optimize_paths(b["cv"], b["bank"], b["target"], b["valid"], on)
            result = evidence(b["cv"], b["ols"], b["bank"], b["target"], b["valid"],
                b["inverse"], None, b["weight"], b["rec"], cut, on, k)
            if result["switched"] != b["counts"].sum():
                raise ValueError("Oracle must preserve the original recipients")
            cells[arm] = dict(supported=result["risk"]["supported"],
                missing_easy=result["risk"]["completion_easy_risk_upper"] is None,
                also_positive_both=bool(result["risk"]["supported"] and min(result["gain_mass"]) > 0.),
                optimal_all_risk=result["risk"]["completion_all_risk_upper"],
                optimal_easy_risk=result["risk"]["completion_easy_risk_upper"])
        rows.append(dict(recording=b["rec"], source=b["source"], seed=d["seed"], cells=cells))
    summary = {}
    for a in ARMS:
        summary[a] = {k: sum(r["cells"][a][k] for r in rows)
                      for k in ("supported", "missing_easy", "also_positive_both")}
    run.once(job.OUT/"recipient_audit.json", run.stable(dict(
        parent=run.artifact(job.OUT/"freeze.json"), summary=summary, rows=rows,
        exact_for_risk_not_for_dual_reference_utility=True, fitting_only=True,
        outcome_aware_only=True, confirmation_access=False, deployment=False)))


if __name__ == "__main__":
    main()
