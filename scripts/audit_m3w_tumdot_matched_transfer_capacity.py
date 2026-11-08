"""Separate missing-label debit from matched-menu headroom on TUMDOT."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_tumdot_matched_finite_transfer as study
from scripts.audit_m3w_opendd_finite_guard_capacity import best_risk

run, np, torch = study.run, study.np, study.torch


def best_utility(cv, paths, target, valid):
    """Exact per-row choice for the fixed-recipient, fixed-radius lower score."""
    bank = np.stack(paths, axis=1)
    u = np.column_stack([study.paired(cv, p, target, valid)[1].sum(1) for p in paths])
    return bank[np.arange(len(bank)), u.argmax(1)]


def split_debit(ref, p, y, valid, den):
    known, conservative = study.paired(ref, p, y, valid)
    debit = known-conservative
    available = valid.any(1)
    return dict(observed_pp=float(100*known.sum()/den),
                partial_unknown_pp=float(100*debit[available].sum()/den),
                wholly_unknown_pp=float(100*debit[~available].sum()/den),
                conservative_pp=float(100*conservative.sum()/den))


def one(rec, cv, ols, base, deltas, y, valid, cuts, fraction):
    paths, on = study.matched_paths(cv, base, deltas, fraction)
    ref = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    attribution = {k: split_debit(cv, p, y, valid, ref) for k, p in paths.items()}
    attribution["tangent_vs_hindcast"] = split_debit(paths["hindcast"], paths["tangent"], y, valid, ref)
    menus = {k: [paths[n] for n in names] for k, names in
             (("original", study.ARMS[:3]), ("expanded", study.ARMS))}
    capacity = {}
    for k, options in menus.items():
        lo = best_risk(cv, options, y, valid)
        hi = best_utility(cv, options, y, valid)
        for a in (lo, hi):
            np.testing.assert_array_equal((a != cv).any((1, 2)), on)
            np.testing.assert_allclose(np.linalg.norm(a-cv, axis=-1).max(1),
                fraction*np.linalg.norm(base-cv, axis=-1).max(1), atol=1e-13, rtol=1e-7)
        def read(a):
            v = study.source.score(a, cv, ols, y, valid, rec, cuts, paths["hindcast"])
            return dict(lower=v["lower_pp"], passed=v["risk"]["supported"],
                undefined_easy=v["risk"]["completion_easy_risk_upper"] is None)
        capacity[k] = dict(risk_optimum=read(lo), utility_optimum=read(hi))
    if (capacity["original"]["risk_optimum"]["passed"] and not capacity["expanded"]["risk_optimum"]["passed"]
            or capacity["expanded"]["utility_optimum"]["lower"]["cv"]
            < capacity["original"]["utility_optimum"]["lower"]["cv"]-1e-8):
        raise ValueError("Expanding the action menu lost certified capacity")
    return dict(attribution=attribution, capacity=capacity)


def summarize(records, part):
    rows = [r[part] for r in records]
    a = rows[0]["attribution"]
    out = {k: {m: float(np.mean([r["attribution"][k][m] for r in rows])) for m in a[k]} for k in a}
    cap = {}
    for k in ("original", "expanded"):
        cap[k] = {}
        for o in ("risk_optimum", "utility_optimum"):
            cell = [r["capacity"][k][o] for r in rows]
            cap[k][o] = dict(mean_lower={n: float(np.mean([r["lower"][n] for r in cell]))
                                        for n in cell[0]["lower"]},
                positive_cv=sum(r["lower"]["cv"] > 0. for r in cell),
                passed=sum(r["passed"] for r in cell),
                joint_positive_cv=sum(r["lower"]["cv"] > 0. and r["passed"] for r in cell),
                undefined_easy=sum(r["undefined_easy"] for r in cell))
    return dict(attribution=out, capacity=cap, views=len(rows))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, fit, tangent, prior, warm, recent, before = study.register()
    p = study.OUT / "analysis.json"
    original = run.read(p)
    if original["registration"] != key or not original["source_is_previously_exposed"]:
        raise ValueError("This diagnostic requires the matched exposed transfer")
    reg = study.OUT / "capacity_registration.json"
    run.once(reg, run.stable(dict(parent=run.artifact(p), code=run.artifact(Path(__file__)),
        labels_only_for_diagnostic=True, matched_recipients_and_peak=True, original_2pct=True,
        no_training_or_calibration=True, confirmation_access=False)))
    models = study.frozen_models(fit, tangent, prior, warm, recent)
    cuts = np.asarray(before["open_dd_positive_complete_cv_cuts"])
    rows = []
    for rec, (name, _) in study.source.ROSTER.items():
        tracks, _, _ = study.source.read_tracks(study.source.SOURCE / name)
        h, y, _ = study.source.windows(tracks)
        valid = np.isfinite(y).all(2)
        for seed in study.parent.job.SEEDS:
            cv, ols, base, _, deltas = study.forecast(h, models[seed])
            rows.append(dict(recording=rec, seed=seed, **{k: one(rec, cv, ols, base, deltas, y, valid, cuts, f)
                                                         for k, f in study.FRACTIONS.items()}))
    if len(rows) != 27:
        raise ValueError("Require every previously exposed recording-seed view")
    summary = {k: summarize(rows, k) for k in study.FRACTIONS}
    for k, value in summary.items():
        for arm in study.ARMS:
            np.testing.assert_allclose(value["attribution"][arm]["conservative_pp"],
                original["summary"][k]["policies"][arm]["lower_vs_cv"], atol=1e-12)
    run.once(study.OUT / "capacity.json", run.stable(dict(registration=run.sha(reg), summary=summary,
        rows=rows, uses_future_for_hindsight_only=True, no_target_tuning=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
