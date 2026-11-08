"""Relax the fixed half blend bank to arbitrary convex combinations."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_past_sampling_audit as job
from src.evaluation import m3w_sampling_convex_bound as convex
from src.evaluation.m3w_sampling_risk_bound import lower_bound
from src.evaluation.m3w_selection_forecast_readout import errors


run, np = job.run, job.np


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--verify", action="store_true")
    a = cli.parse_args()
    binding, identity, cut = job.register()
    cap = job.OUT/"fitting_capacity.json"
    discrete = run.read(cap)
    meta = dict(parent=run.artifact(cap), code=run.artifact(Path(__file__)),
                method=run.artifact(Path(convex.__file__)), registration=binding,
                target_role="original_fitting_only", scope="hindsight_convex_relaxation",
                confirmation_access=False, deployment=False)
    cache = job.PRIVATE/"convex_capacity.json"
    if not cache.exists():
        if a.verify:
            raise ValueError("Verification may not open fitting labels")
        train = job.prior.base.train_source
        data = train.fitting_only(identity, lambda **_: None)
        admission = train.prep.load_admission(job.ROOT, job.ROOT/run.read(train.source.CONFIG)["admission"])
        support = run.read(train.bank.old.HOME/"movingdronecrowd_intake_v1/analysis.json")
        sizes = {r["clip"]: r["window_support"]["obs8_pred12_native_stride1"]["history_complete_agent_instances"]
                 for r in support["recordings"]}
        cells, lo = [], 0
        for rec, previous in zip(data["recordings"], discrete["cells"], strict=True):
            hi = lo+sizes[admission["recordings"][rec]["clip"]]
            h = data["geometry"][lo:hi, :16].reshape(-1, 8, 2)
            cv = data["geometry"][lo:hi, 332:356].reshape(-1, 12, 2).astype(float)
            ols = job.prior.base.motion.slope(h[:, -4:])[:, None]*job.previous.LEAD
            legal = np.any(cv != ols, axis=(1, 2))
            on = job.prior.method.menu(job.prior.method.backtests(h), legal)["three_origins_2"] > 0
            y, m = data["target"][lo:hi], data["valid"][lo:hi]
            # Use the exact native path vertices, including any fitting float32
            # rounding; each future step is relaxed separately.
            bank = np.stack(list(job.method.candidates(h, cv, ols, on).values()), axis=2)
            delta = np.linalg.norm(bank-cv[:, :, None], axis=-1)
            # Every true path's maximum deviation is at least its final-step
            # deviation. Allowing a separate hull choice at each step can only
            # reduce the known error and preserves a lower bound.
            d = convex.hull_distance(cv[:, -1:], bank[:, -1])[:, 0]
            observed = np.where(m[..., None], y, 0.)
            distance = np.column_stack([convex.hull_distance(observed[:, i:i+1], bank[:, i])[:, 0]
                                        for i in range(y.shape[1])])
            count = m.sum(1)
            e = np.divide(np.where(m, distance, 0.).sum(1), count,
                          out=np.full(len(count), np.nan), where=count > 0)
            e0 = np.stack([errors(bank[:, :, k], y, m)[0] for k in range(bank.shape[2])], axis=1)
            if ((e[m.any(1)] > e0[m.any(1)].min(1)+1e-9).any() or (d > delta.max(1).min(1)+1e-9).any()
                    or rec != previous["recording"]):
                raise ValueError("Convex relaxation must dominate the discrete bank")
            _, bound = lower_bound(e[:, None], d[:, None], errors(cv, y, m)[0], m.any(1), on, cut["easy"])
            cells.append(dict(recording=rec, source=previous["source"], bound=bound))
            lo = hi
        if lo != len(data["ids"]):
            raise ValueError("Fitting roster changed")
        run.once(cache, run.stable(dict(meta=meta, cells=cells)))
    stored = run.read(cache)
    if stored["meta"] != meta:
        raise ValueError("Stale capacity cache")
    rows = stored["cells"]
    excluded = sum(any(v is None or v > .02 for v in r["bound"].values()) for r in rows)
    run.once(job.OUT/"convex_capacity.json", run.stable(dict(meta, cells=rows,
        recordings=len(rows), excluded_by_optimistic_bound=excluded, allowed_by_bound=len(rows)-excluded)))


if __name__ == "__main__":
    main()
