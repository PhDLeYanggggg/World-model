"""Fitting-only hindsight bound: can any schedule assignment fix risk?"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_past_sampling_audit as job
from src.evaluation.m3w_sampling_risk_bound import lower_bound
from src.evaluation.m3w_selection_forecast_readout import errors, policy


run, cal, np = job.run, job.cal, job.np


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--verify", action="store_true")
    args = cli.parse_args()
    binding, identity, cut = job.register()
    analysis_path = job.OUT/"analysis.json"
    analysis = run.read(analysis_path)
    if analysis["registration"] != binding or analysis["confirmation_opened"]:
        raise ValueError("Only the registered fitting cohort may be audited")
    meta = dict(parent=run.artifact(analysis_path), code=run.artifact(Path(__file__)),
                scope="hindsight_diagnostic_fitting_only", independent_confirmation=False)
    checkpoint = job.PRIVATE/"capacity_cells.json"
    if not checkpoint.exists():
        if args.verify:
            raise ValueError("Verification may not reopen labels")
        train = job.prior.base.train_source
        data = train.fitting_only(identity, lambda **_: None)
        cfg = run.read(train.source.CONFIG)
        admission = train.prep.load_admission(job.ROOT, job.ROOT/cfg["admission"])
        support = run.read(train.bank.old.HOME/"movingdronecrowd_intake_v1/analysis.json")
        size = {r["clip"]: r["window_support"]["obs8_pred12_native_stride1"]["history_complete_agent_instances"]
                for r in support["recordings"]}
        cells, lo = [], 0
        for rec in data["recordings"]:
            cell = admission["recordings"][rec]
            hi = lo+size[cell["clip"]]
            h = data["geometry"][lo:hi, :16].reshape(-1, 8, 2)
            cv = data["geometry"][lo:hi, 332:356].reshape(-1, 12, 2).astype(float)
            ols = job.prior.base.motion.slope(h[:, -4:])[:, None]*job.previous.LEAD
            legal = np.any(cv != ols, axis=(1, 2))
            active = job.prior.method.menu(job.prior.method.backtests(h), legal)["three_origins_2"] > 0
            target, valid = data["target"][lo:hi], data["valid"][lo:hi]
            bank = list(job.method.candidates(h, cv, ols, active).values())
            b = errors(cv, target, valid)[0]
            e = np.stack([errors(p, target, valid)[0] for p in bank], axis=1)
            delta = np.stack([np.linalg.norm(p-cv, axis=-1).max(1) for p in bank], axis=1)
            choice, bound = lower_bound(e, delta, b, valid.any(1), active, cut["easy"])
            p = np.stack(bank, axis=1)[np.arange(len(h)), choice]
            actual = policy(p, cv, cv, target, valid, np.full(len(h), rec),
                            easy_cut=cut["easy"], hard_cut=cut["hard"])["recordings"][rec]
            for a, k in (("completion_all_risk_upper", "all_lower"), ("completion_easy_risk_upper", "easy_lower"),
                         ("known_easy_net_degradation", "signed_easy_lower")):
                if actual[a] is not None and bound[k] is not None and actual[a] < bound[k]-1e-10:
                    raise ValueError("Constructive witness below the lower bound")
            cells.append(dict(recording=rec, source=cell["exclusion_group"], bound=bound, oracle=actual))
            lo = hi
        if lo != len(data["ids"]):
            raise ValueError("Fitting population mismatch")
        run.once(checkpoint, run.stable(dict(meta=meta, cells=cells)))
    stored = run.read(checkpoint)
    if stored["meta"] != meta:
        raise ValueError("Fitting risk bound cache changed")
    cells = stored["cells"]
    bad = lambda c: any(v is None or v > .02 for v in c["bound"].values())
    run.once(job.OUT/"fitting_capacity.json", run.stable(dict(meta, cells=cells,
        recordings=len(cells), excluded_by_optimistic_bound=sum(bad(c) for c in cells),
        allowed_by_bound=sum(not bad(c) for c in cells),
        constructed_oracle_passes=sum(c["oracle"]["supported"] for c in cells))))


if __name__ == "__main__":
    main()
