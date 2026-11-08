"""Fit a causal cost selector on eight source groups and score the ninth."""

import argparse
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_recipient_release_capacity as cap
from src.world_model import m3w_record_joint_actions as model
from src.evaluation.m3w_selection_forecast_readout import errors


run, np, prior = cap.run, cap.np, cap.prior.prior
OUT = cap.OUT.parent/"recipient_release_crossfit_v2"
PRIVATE = cap.PRIVATE.parent/OUT.name
ARMS = ("recent_hindcast", "independent_gain", "predicted_joint")
PRICES = (0., 1., 4., 16., 64., 256.)
PAIRS = {f"{a}_vs_{'cv' if b == prior.cal.BASE else b}": (b, a, prior.cal.BASE)
         for a, b in ((prior.cal.OLS, prior.cal.BASE),
                      *((a, b) for a in ARMS for b in (prior.cal.BASE, prior.cal.OLS)),
                      ("predicted_joint", "independent_gain"), ("predicted_joint", "recent_hindcast"))}


def register():
    parent, identity, cut = cap.register()
    p = OUT/"registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), fit=run.artifact(Path(model.__file__)),
        path_registration=parent, bank=run.artifact(cap.OUT/"analysis.json"), folds="leave_one_fitting_source_group_out",
        price_grid=PRICES, equal_fitting_group_weights=True, fixed_ridge=.002, causal_inputs=True,
        training_easy_threshold="positive known CV error 25th percentile in eight fitting groups only",
        evaluation_easy_threshold="unchanged frozen protocol",
        selector_fits=9, deterministic_no_seed_replication=True, inference_risk_budget=.02,
        calibration_access=False, confirmation_access=False, neural_training=False, no_threshold_tuning=True)))
    return run.sha(p), identity, cut


def blocks(identity, beat):
    train = prior.base.train_source
    data = train.fitting_only(identity, beat)
    lo = 0
    for rec, g, cv, ols, bank, y, valid, inv, counts, w in cap.fitting_blocks(identity, lambda **_: None):
        hi = lo+len(cv)
        h = data["geometry"][lo:hi, :16].reshape(-1, 8, 2)
        if not np.all(data["sites"][lo:hi] == g):
            raise ValueError("Fitting geometry and source order mismatch")
        np.testing.assert_array_equal(cv.reshape(-1, 24), data["geometry"][lo:hi, 332:356])
        np.testing.assert_array_equal(y, data["target"][lo:hi])
        np.testing.assert_array_equal(valid, data["valid"][lo:hi])
        lo = hi
        yield rec, g, h, cv, ols, bank, y, valid, inv, counts, w
    if lo != len(data["ids"]):
        raise ValueError("Incomplete original geometry")


def features(h, bank, cv, inverse):
    crowd = np.bincount(inverse)[inverse]-1
    return model.descriptors(h, bank, cv, crowd)


def choose(gain, all_excess, easy_excess, legal, weight):
    """Use predicted original inequalities; actual certification is separate."""
    g = weight[:, None]*gain
    rows = np.arange(len(gain))
    eligible = legal & (gain > 0.)

    def ranked(score, ok):
        score = np.where(ok, score, -np.inf)
        k = score.argmax(1)
        return np.where(score[rows, k] > 0., k, -1)

    independent = ranked(g, eligible)
    singleton = ranked(g, eligible & (all_excess < 0.) & (easy_excess < 0.))
    choices = [singleton]
    choices.extend(ranked(g-price*(all_excess+easy_excess), eligible) for price in PRICES)
    best = np.full(len(rows), -1)
    score = 0.
    for c in choices:
        on = c >= 0
        ix = rows[on], c[on]
        total = float(g[ix].sum())
        if total > score and all_excess[ix].sum() < 0. and easy_excess[ix].sum() < 0.:
            best, score = c, total
    return independent, best


def moments(binding, identity, beat):
    p = PRIVATE/"coefficients.json"
    if p.exists():
        fit = run.read(p)
        if fit["registration"] != binding or len(fit["folds"]) != 9:
            raise ValueError("Stale crossfit receipt")
        return fit
    data = prior.base.train_source.fitting_only(identity, lambda **_: None)
    groups = sorted(set(data["sites"]))
    if len(groups) != 9:
        raise ValueError("Require all fitting groups")
    base_error = errors(data["geometry"][:, 332:356].reshape(-1, 12, 2), data["target"], data["valid"])[0]
    known = np.isfinite(base_error) & (base_error > 0.)
    threshold = {g: float(np.quantile(base_error[known & (data["sites"] != g)], .25)) for g in groups}
    del data
    totals = {g: {} for g in groups}
    masses = {}
    for rec, g, h, cv, _, bank, y, valid, inv, _, w in blocks(identity, beat):
        x, scale = features(h, bank, cv, inv)
        mass = float(w.sum())
        for held in groups:
            if held == g:
                continue
            moment = tuple(z*mass for z in model.fitting_moments(x, bank, cv, y, valid, scale, w, threshold[held]))
            cell = totals[held]
            cell[g] = tuple(a+b for a, b in zip(cell[g], moment, strict=True)) if g in cell else moment
        masses[g] = masses.get(g, 0.)+mass
        beat(stage="moment", recording=rec)
    folds = {}
    for g in groups:
        train = [k for k in groups if k != g]
        if set(totals[g]) != set(train):
            raise ValueError("Unexpected group used for fitted moments")
        group_stats = [tuple(z/masses[k] for z in totals[g][k]) for k in train]
        folds[g] = dict(training_sources=train, training_easy_cut=threshold[g],
                        coefficients=model.coefficients(group_stats).tolist())
    fit = run.stable(dict(registration=binding, folds=folds))
    run.once(p, fit)
    return fit


def choices(held, fold, h, cv, ols, bank, inv, w):
    if held in fold["training_sources"] or len(fold["training_sources"]) != 8:
        raise ValueError("Held source entered its own model")
    x, scale = features(h, bank, cv, inv)
    ranked = model.decision_costs(x, np.asarray(fold["coefficients"]), bank, cv, scale)
    ind, joint = choose(*ranked, w)
    legal = np.any(cv != ols, axis=(1, 2))
    action = prior.method.menu(prior.method.backtests(h), legal)["three_origins_2"]
    return np.stack((np.where(action > 0, 0, -1), ind, joint))


def execute(binding, identity, cut, beat, replay):
    fit_path = PRIVATE/"coefficients.json"
    if replay and not fit_path.exists():
        raise ValueError("Replay may not fit")
    fit = moments(binding, identity, beat)
    marks = []
    for rec, g, h, cv, ols, bank, _, _, inv, _, w in blocks(identity, beat):
        p = PRIVATE/f"choices_{rec}.json"
        if not p.exists():
            if replay:
                raise ValueError("Replay may not infer")
            acts = choices(g, fit["folds"][g], h, cv, ols, bank, inv, w)
            run.once(p, run.stable(dict(registration=binding, recording=rec, source=g,
                fit=run.artifact(fit_path), choices=acts.tolist())))
        doc = run.read(p)
        if (doc["registration"] != binding or doc["fit"] != run.artifact(fit_path)
                or doc["source"] != g or doc["recording"] != rec):
            raise ValueError("Held source predictions changed")
        marks.append(run.artifact(p))
    run.once(OUT/"freeze.json", run.stable(dict(registration=binding, fits=9, predictions=marks,
        held_group_excluded_from_own_moments=True, calibration_access=False, confirmation_access=False)))
    rows = []
    for (rec, g, _, cv, ols, bank, y, valid, inv, _, _), mark in zip(blocks(identity, beat), marks, strict=True):
        p = PRIVATE/f"score_{rec}.json"
        if not p.exists():
            if replay:
                raise ValueError("Replay may not score missing predictions")
            select = np.asarray(run.read(run.checked(mark))["choices"])
            paths = {prior.cal.BASE: cv, prior.cal.OLS: ols}
            for arm, k in zip(ARMS, select, strict=True):
                pred = cv.copy()
                on = k >= 0
                pred[on] = bank[np.flatnonzero(on), k[on]]
                paths[arm] = pred
            result = prior.base.score_recording(paths, PAIRS, y, valid, inv, rec,
                                               easy_cut=cut["easy"], hard_cut=cut["hard"])
            run.once(p, run.stable(dict(predictions=mark, row=dict(result, source=g, seed=0))))
        block = run.read(p)
        if block["predictions"] != mark or block["row"]["source"] != g:
            raise ValueError("Fitting score cache mismatch")
        rows.append(block["row"])
        beat(stage="score", completed=len(rows))
    summary = prior.base.grouped_readout(rows, {r["recording"]: r["source"] for r in rows}, (0,))
    orig = run.read(cap.prior.OUT/"analysis.json")["fitting"]
    if (summary["effects"]["recent_hindcast_vs_cv"] != orig["effects"]["recent_vs_cv"]
            or summary["risk"]["recent_hindcast"] != orig["risk"]["recent"]):
        raise ValueError("Frozen fitting baseline did not reproduce")
    run.once(OUT/"analysis.json", run.stable(dict(registration=binding, summary=summary,
        neural_training=False, diagnostics_only=True, independent_confirmation=False, deployment=False)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", action="store_true")
    a = cli.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding, identity, cut = register()

        def beat(**v):
            v.update(pid=os.getpid())
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", v)
            if "record" not in v:
                print(v, flush=True)

        execute(binding, identity, cut, beat, a.replay)


if __name__ == "__main__":
    main()
