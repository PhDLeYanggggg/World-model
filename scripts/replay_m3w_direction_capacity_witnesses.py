"""Use verified quarter-path witnesses to resolve nested circle timeouts."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_direction_recipient_capacity as prior

job, run, np = prior.job, prior.run, prior.np


def main():
    _, identity, cut, _ = job.register()
    p = job.OUT / "recipient_capacity.json"
    old = run.read(p)
    need = {r["recording"] for r in old["recordings"] if r["menus"]["quarter"]["status"] == "verified_both"
            and r["menus"]["quarter_circle"]["status"] == "unresolved"}
    result = []
    if need:
        for rec, group, _, cv, ols, bank, target, valid, q, counts, w in job.job.prior.prior.old.blocks(
                identity, lambda **_: None):
            if rec not in need:
                continue
            f = job.PRIVATE / f"recipient_{rec}.json"
            original = run.read(f)
            if original["key"] != old["registration"] or original["source"] != group:
                raise ValueError("The recovered witness must come from the registered quarter run")
            base = original["outcomes"]["quarter"]
            rows, arms = (np.asarray(v, int) for v in base["witness"])
            d = .25*(bank[rows, arms]-cv[rows])
            known = valid[rows].any(1)
            z = np.where(known[:, None, None], target[rows], ols[rows])-cv[rows]
            candidate = cv[rows]+job.model.circle_oracle(d, z, valid[rows] | (~known)[:, None])
            bank = cv[:, None].copy()
            bank[rows, 0] = candidate
            val = prior.evidence(cv, ols, bank, target, valid, q, counts, w, rec, cut, rows, np.zeros_like(arms))
            if not val["risk"]["supported"] or min(val["gain_mass"]) <= 0.:
                raise ValueError("Exact projection did not preserve the witnessed condition")
            result.append(dict(recording=rec, source=group, original=run.artifact(f), risk_and_gain=val))
    if {r["recording"] for r in result} != need:
        raise ValueError("Missing inherited witness")
    summary = old["summary"]["quarter_circle"].copy()
    summary["verified_both"] += len(result)
    summary["unresolved"] -= len(result)
    run.once(job.OUT / "recipient_capacity_replay.json", run.stable(dict(parent=run.artifact(p),
        code=run.artifact(Path(__file__)), inherited=result, circle_summary=summary, label_aware_only=True,
        confirmation_access=False)))


if __name__ == "__main__":
    main()
