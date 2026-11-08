"""Check whether three held-source learned banks expand dual-reference capacity."""

import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_weighted_velocity as study
from scripts import audit_m3w_recipient_release_capacity as earlier
from src.evaluation.m3w_recipient_release_capacity import STATES, assess, evidence

run, job, np, torch = study.run, study.job, study.np, study.torch
LIMITS = (1., 3.)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, identity, cut, _ = study.register()
    fresh = study.OUT / "analysis.json"
    old = earlier.OUT / "analysis.json"
    current, original = run.read(fresh), run.read(old)
    old_key = earlier.OUT / "registration.json"
    if (current["registration"] != key or current["confirmation_access"]
            or original["registration"] != run.sha(old_key) or run.read(old_key)["confirmation_access"]):
        raise ValueError("Require source-only banks and verified original witnesses")
    r = study.OUT / "capacity_registration.json"
    run.once(r, run.stable(dict(analysis=run.artifact(fresh), original=run.artifact(old),
        original_registration=run.artifact(old_key),
        code=run.artifact(Path(__file__)), solver=run.artifact(Path(assess.__code__.co_filename)),
        original_exact_query_counts=True, candidates_added=["fixed_robust", "ade", "dual"],
        seeds=job.SEEDS, limits=LIMITS, hindsight_only=True, confirmation_access=False)))
    binding = run.sha(r)
    roster = {r["recording"]: r for r in original["recordings"]}
    heads = {(r["held"], r["seed"]): r for r in current["fits"]}
    start = time.monotonic()
    rows = []
    with (study.PRIVATE / "capacity.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for rec, g, h, cv, ols, bank, y, valid, q, counts, w in job.prior.prior.old.blocks(
                identity, lambda **_: None):
            before = roster[rec]["menus"]["curved_exact"]["status"]
            old_path = earlier.PRIVATE / f"capacity_{rec}.json"
            prior = run.read(old_path)["menus"]["curved_exact"]
            if prior["status"] != before or roster[rec]["source"] != g or roster[rec]["switches"] != int(counts.sum()):
                raise ValueError("Original geometry or witness changed")
            ht = torch.from_numpy(h.copy())
            robust = study.model.robust_path(ht.to(torch.float64)).numpy()
            for seed in job.SEEDS:
                output = study.PRIVATE / f"capacity_{rec}_{seed}.json"
                doc = heads[g, seed]
                if g in (*doc["fit"], *doc["calibration"]):
                    raise ValueError("Forecast used its own held source")
                paths = [robust]
                for k in study.HEADS:
                    saved = torch.load(run.checked(doc["checkpoints"][k]), weights_only=True, map_location="cpu")
                    if saved["identity"]["held"] != g or saved["identity"]["key"] != key:
                        raise ValueError("Forecast checkpoint identity changed")
                    net = study.model.WeightedVelocity()
                    net.load_state_dict(saved["model"])
                    with torch.inference_mode():
                        paths.append(net.eval()(ht)[0].numpy().astype(float))
                p = np.concatenate((bank, np.stack(paths, axis=1)), axis=1)
                if not output.exists():
                    result = prior if before == "verified_both" else assess(
                        cv, ols, p, y, valid, q, counts, w, rec, cut, budgets=LIMITS)
                    run.once(output, run.stable(dict(binding=binding, recording=rec, source=g,
                        seed=seed, original_status=before, outcome=result)))
                value = run.read(output)
                if (value["binding"], value["recording"], value["source"], value["seed"]) != (binding, rec, g, seed):
                    raise ValueError("Stale capacity checkpoint")
                result = value["outcome"]
                if result["status"] == "verified_both":
                    ids, arms = (np.asarray(v, int) for v in result["witness"])
                    expected = evidence(cv, ols, p, y, valid, q, counts, w, rec, cut, ids, arms)
                    if expected != result["evidence"]:
                        raise ValueError("Witness does not replay on the actual bank")
                rows.append(dict(value, outcome={k: v for k, v in result.items() if k != "witness"}))
            state = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 2), recording=rec,
                         completed=len(rows), results=[r["outcome"]["status"] for r in rows[-3:]])
            run.kin.prior.source.atomic(study.PRIVATE / "capacity_heartbeat.json", state)
            print(state, flush=True)
    if len(rows) != 72:
        raise ValueError("Require all 24 recordings and three training seeds")
    totals = {s: sum(r["outcome"]["status"] == s for r in rows) for s in STATES}
    added = [r for r in rows if r["original_status"] != "verified_both" and r["outcome"]["status"] == "verified_both"]
    run.once(study.OUT / "capacity.json", run.stable(dict(registration=binding, summary=totals, rows=rows,
        original_verified_views=sum(r["original_status"] == "verified_both" for r in rows),
        new_verified_views=len(added), new_recordings=len({r["recording"] for r in added}),
        labels_used_for_hindsight_allocation_only=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
