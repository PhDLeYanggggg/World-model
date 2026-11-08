"""Measure fixed-budget recipient capacity after releasing direction."""

import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_matched_direction as job
from src.evaluation.m3w_recipient_release_capacity import STATES, assess, evidence

run, np = job.run, job.np
LIMITS = (1., 4.)


def main():
    job.torch.set_num_threads(4)
    job.torch.set_num_interop_threads(1)
    binding, identity, cut, _ = job.register()
    p = job.OUT / "analysis.json"
    if run.read(p)["registration"] != binding:
        raise ValueError("Require the completed fixed-recipient direction experiment")
    reg = job.OUT / "recipient_capacity_registration.json"
    run.once(reg, run.stable(dict(direction=run.artifact(p), code=run.artifact(Path(__file__)),
        optimizer=run.artifact(Path(assess.__code__.co_filename)),
        per_query_counts_unchanged=True, only_original_fitting_groups=True,
        menus=["quarter", "quarter_circle"], solve_limits=LIMITS,
        original_2pct=True, future_labels_for_diagnostic_only=True, confirmation_access=False)))
    key = run.sha(reg)
    start = time.monotonic()
    results = []
    with (job.PRIVATE / "recipient_capacity.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for rec, g, _, cv, ols, bank, target, valid, q, counts, w in job.job.prior.prior.old.blocks(
                identity, lambda **_: None):
            d = .25*(bank-cv[:, None])
            quarter = cv[:, None]+d
            known = valid.any(1)
            z = np.where(known[:, None, None], target, ols)-cv
            m = valid | (~known)[:, None]
            circle = np.stack([cv+job.model.circle_oracle(a, z, m) for a in d.swapaxes(0, 1)], axis=1)
            np.testing.assert_allclose(np.linalg.norm(circle-cv[:, None], axis=-1),
                                       np.linalg.norm(d, axis=-1), rtol=1e-9, atol=1e-9)
            f = job.PRIVATE / f"recipient_{rec}.json"
            if not f.exists():
                outcomes = {name: assess(cv, ols, options, target, valid, q, counts, w, rec, cut, budgets=LIMITS)
                            for name, options in (("quarter", quarter), ("quarter_circle", circle))}
                run.once(f, run.stable(dict(key=key, recording=rec, source=g, switches=int(counts.sum()),
                                            outcomes=outcomes)))
            receipt = run.read(f)
            if (receipt["key"], receipt["recording"], receipt["source"], receipt["switches"]) != (
                    key, rec, g, int(counts.sum())):
                raise ValueError("Recipient capacity checkpoint mismatch")
            public = {}
            for name, options in (("quarter", quarter), ("quarter_circle", circle)):
                result = receipt["outcomes"][name]
                if result["status"] == "verified_both":
                    rows, arms = (np.asarray(v, dtype=int) for v in result["witness"])
                    check = evidence(cv, ols, options, target, valid, q, counts, w, rec, cut, rows, arms)
                    if check != result["evidence"] or not check["risk"]["supported"]:
                        raise ValueError("Witness must replay under the original rule")
                public[name] = {k: v for k, v in result.items() if k != "witness"}
            a, b = public["quarter"]["status"], public["quarter_circle"]["status"]
            if a == "verified_both" and b not in ("verified_both", "unresolved"):
                raise ValueError("Circle bound cannot exclude a verified flat witness")
            results.append(dict(recording=rec, source=g, switches=receipt["switches"], menus=public))
            heartbeat = dict(pid=os.getpid(), elapsed_seconds=round(time.monotonic()-start, 1),
                             processed=len(results), rec=rec, quarter=a, circle=b)
            run.kin.prior.source.atomic(job.PRIVATE / "recipient_heartbeat.json", heartbeat)
            print(heartbeat, flush=True)
    if len(results) != 24 or sum(r["switches"] for r in results) != 31923:
        raise ValueError("The fixed budget or fitting roster changed")
    summary = {menu: {status: sum(r["menus"][menu]["status"] == status for r in results)
                      for status in STATES} for menu in ("quarter", "quarter_circle")}
    run.once(job.OUT / "recipient_capacity.json", run.stable(dict(registration=key, summary=summary,
        recordings=results, same_query_edit_counts=True, label_aware_not_a_policy=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
