"""Reallocate the original Hannover displacement budget without changing risk."""

import hashlib
from pathlib import Path
import platform
import sys
from zipfile import ZipFile

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_optional_reference_bridge as parent
from src.evaluation import m3w_shared_radius_capacity as capacity

run, np, torch, transfer = parent.run, parent.np, parent.torch, parent.transfer
OUT = parent.OUT.parent / "hannover_shared_radius_v1"
TOL = parent.POSITIVE_PP


def score(cv, ols, bank, y, valid, name, cuts, eligible, radius):
    m = capacity.coefficients(cv, ols, bank, y, valid, cuts[0])
    budget = float(radius.sum())
    if budget <= 0. or m["cv_mass"] <= 0.:
        raise ValueError("The registered comparison needs positive radius and CV mass")
    pp = 100. / m["cv_mass"]
    frontiers = {k: capacity.radius_frontier(cv, ols, y, valid, e) for k, e in
                 (("old_recipients", eligible), ("all_recipients", np.ones(len(cv), bool)))}
    result = dict(rows=len(cv), original_recipients=int(eligible.sum()), original_peak_sum=budget,
                  cv_lower_vs_ols4_pp=float(m["offset"][1] * pp), finite={}, relaxed={})
    for k, f in frontiers.items():
        minimum = capacity.break_even_radius(f)
        result["relaxed"][k] = dict(
            upper_ols4_pp=[capacity.upper_gain(f, budget * scale) * pp for scale in (1., 2.)],
            minimum_budget_multiple=None if minimum is None else minimum / budget)
    for scale in (1., 2.):
        take, result_solver = capacity.solve(m, budget * scale)
        result["finite"][str(int(scale))] = result_solver
        if take is None:
            continue
        row, arm = m["row"][take], m["arm"][take]
        p = cv.copy()
        if len(np.unique(row)) != len(row):
            raise ValueError("More than one action on a row")
        p[row] = bank[row, arm]
        cost = float(np.linalg.norm(p - cv, axis=-1).max(1).sum())
        risk = parent.policy(p, cv, cv, y, valid, np.full(len(y), name),
                             easy_cut=cuts[0], hard_cut=cuts[1])["recordings"][name]
        gains = np.array([transfer.paired(ref, p, y, valid)[1].sum() * pp for ref in (cv, ols)])
        np.testing.assert_allclose(gains, pp * (m["offset"] + m["gain"][:, take].sum(1)), atol=1e-9)
        np.testing.assert_allclose(cost, m["cost"][take].sum(), atol=1e-10)
        bound = result_solver["upper_pp"]
        relaxed = result["relaxed"]["old_recipients"]["upper_ols4_pp"][int(scale) - 1]
        if (cost > budget * scale * (1. + 1e-8) or
                (bound is not None and gains.min() > bound + TOL) or gains[1] > relaxed + TOL):
            raise ValueError("Witness violates budget or exceeds a verified upper bound")
        result_solver.update(witness_gains_pp=gains.tolist(), original_risk_supported=risk["supported"],
                             positive_both=bool(risk["supported"] and gains.min() > TOL),
                             used_budget_fraction=cost / (budget * scale), edits=len(row),
                             action_counts=np.bincount(arm, minlength=bank.shape[1]).tolist())
    return result


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, frozen, intake, scale = parent.register()
    parent_path = parent.OUT / "analysis.json"
    previous = run.read(parent_path)
    if previous["registration"] != key or previous["confirmation_access"]:
        raise ValueError("Missing verified exposed-source bridge analysis")
    OUT.mkdir(parents=True, exist_ok=True)
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(
        previous=run.artifact(parent_path), code=run.artifact(Path(__file__)),
        solver=run.artifact(Path(capacity.__file__)), source="already exposed Hannover D1/D2/D3",
        original_budget="sum of per-row peak half-incumbent displacement over recording",
        budget_multiples=[1, 2], objective="maximin full-grid lower gain versus CV and OLS4",
        finite_menu="same eligible roster; optional frozen half-radius and untrimmed OLS4 bridges",
        relaxed="arbitrary per-step directions and radii, risk ignored, OLS4 lower gain only",
        positive_pp=TOL, split_changed=False, original_2pct=True, confirmation_access=False)))
    easy = []
    for site in parent.source.SITES:
        with np.load(parent.source.prior.PRIVATE / f"{site}_cohort.npz", allow_pickle=False) as z:
            h, y = z["history"], z["future"]
        cv = transfer.motion.rollout(h, 12)[0]
        complete = np.isfinite(y).all((1, 2))
        loss = np.linalg.norm(cv[complete] - y[complete], axis=-1).mean(1)
        easy.append(loss[loss > 0.])
    cuts = np.quantile(np.concatenate(easy), [.25, .75])
    nets = transfer.frozen_models(*frozen[:5])
    study = parent.previous
    rows = []
    with ZipFile(study.ZIP) as z:
        for clip in study.CLIPS:
            raw = z.read(study.PREFIX + f"{clip}_trajectory.csv")
            if hashlib.sha256(raw).hexdigest() != intake["clips"][clip]["sha256"]:
                raise ValueError("Hannover clip hash changed")
            h, y, ids = study.windows(study.parse_csv(raw)[0], scale)
            valid = np.isfinite(y).all(2)
            for seed in study.choice_study.SEEDS:
                cv, ols, inc, _, delta = transfer.forecast(h, nets[seed])
                paths, on = transfer.matched_paths(cv, inc, delta, .5)
                radius = .5 * np.linalg.norm(inc - cv, axis=-1).max(1)
                bank = np.stack([paths[k] for k in transfer.ARMS], axis=1)
                union = parent.menus(cv, ols, bank, on)["union"]
                rows.append(dict(clip=clip, seed=seed, queries=len(ids),
                                 **score(cv, ols, union, y, valid, clip, cuts, on, radius)))
                print(f"{clip} {seed} {rows[-1]['finite']}", flush=True)
    if len(rows) != 9:
        raise ValueError("Incomplete Hannover clip/seed roster")
    run.once(OUT / "analysis.json", run.stable(dict(
        registration=run.sha(reg), result_source="fresh_run", frozen_forecast_source="cached_verified",
        rows=rows, actions=(*transfer.ARMS, "ols25", "ols50", "ols100"),
        hindsight_only=True, original_2pct=True, deployment_changed=False, confirmation_access=False)))


if __name__ == "__main__":
    main()
