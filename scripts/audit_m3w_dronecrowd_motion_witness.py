"""Compare future-informed zero-harm witnesses within frozen full-scene banks."""

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_dronecrowd_motion_floor as main_study
from src.evaluation.m3w_recording_forecast_comparison import comparison_masses, merge_comparisons
from src.evaluation.m3w_selection_forecast_readout import errors, policy


cal, run, np, model = main_study.cal, main_study.run, main_study.np, main_study.model
PUBLIC, PRIVATE = main_study.PUBLIC, main_study.PRIVATE
MODES = ("ols4_only", "all_frozen")


def witness(cv, ols, alternatives, y, v):
    """Keep CV for unlabelled actors; only accept strictly smaller observed ADE."""
    p = cv.copy()
    best, _ = errors(cv, y, v)
    kind = np.zeros(len(p), dtype=int)
    for i, candidate in enumerate((ols, *alternatives), 1):
        e, _ = errors(candidate, y, v)
        use = v.any(1) & (e < best)
        p[use], best[use], kind[use] = candidate[use], e[use], i
    np.testing.assert_array_equal(p[~v.any(1)], cv[~v.any(1)])
    return p, kind


def recording(manifest, cell, receipt, meta, provenance, cut):
    rec = cell["sequence"]
    source = cal.reader(manifest, rec, meta, provenance)
    a = main_study.old_arrays(cell)
    y, v = cal.label_arrays(source, a)
    c, o = a["cv"], a["ols"]
    weights = np.repeat(1./np.diff(a["offsets"]), np.diff(a["offsets"]))
    coeff = np.load(run.checked(receipt["coefficients"]), mmap_mode="r", allow_pickle=False)
    # Past-best and guarded forecasts are already in the analytic menu.
    more = [model.paths(coeff[:, i]) for i in range(3)]
    out = {}
    for k, bank in zip(MODES, ([], more), strict=True):
        p, types = witness(c, o, bank, y, v)
        risk = policy(p, c, c, y, v, np.zeros(len(y), dtype=int),
                      easy_cut=cut["easy"], hard_cut=cut["hard"])["recordings"]["0"]
        fields = ("completion_all_risk_upper", "completion_easy_risk_upper", "known_easy_net_degradation")
        if any(risk[f] is not None and risk[f] > 1e-12 for f in fields):
            raise ValueError("Known-ADE monotonicity violated the original risk scorer")
        out[k] = dict(risk=risk, picks=np.bincount(types, minlength=5).tolist(),
            cv=comparison_masses(c, p, c, y, v, weights),
            ols4=comparison_masses(o, p, c, y, v, weights))
    return out


def summarize(rows, roster):
    keys = sorted(set(roster.values()))
    by_group = {}
    for k in keys:
        group = [r for n, r in rows.items() if roster[n] == k]
        by_group[k] = {m: {f: merge_comparisons([r[m][f] for r in group])["completion_lower_pp"]
                          for f in ("cv", "ols4")} for m in MODES}
    draws = np.random.default_rng(main_study.SEED).integers(len(keys), size=(main_study.DRAWS, len(keys)))
    metrics = {}
    for m in MODES:
        per = [r[m] for r in rows.values()]
        counts = Counter()
        for p in per:
            ok = p["risk"]["supported"]
            counts.update(risk_pass=int(ok), undefined_easy=int(p["risk"]["completion_easy_risk_upper"] is None),
                          dual_pass=int(ok and all(merge_comparisons([p[k]])["completion_lower_pp"] > 0.
                                                   for k in ("cv", "ols4"))))
        metric = dict(counts=counts, picks=np.sum([p["picks"] for p in per], axis=0).tolist())
        for f in ("cv", "ols4"):
            v = np.asarray([by_group[k][m][f] for k in keys])
            metric[f"lower_vs_{f}_pp"] = dict(point=float(v.mean()),
                ci95=np.quantile(v[draws].mean(1), [.025, .975]).tolist(), positive_groups=int((v > 0.).sum()))
        metrics[m] = metric
    return dict(metrics=metrics, by_group=by_group)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true", help="Recompute group summaries without opening labels")
    args = p.parse_args()
    with (PRIVATE/"witness.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, manifest, old, roster, _ = main_study.register()
        rpath = PUBLIC/"analysis.json"
        parent = run.read(rpath)
        if parent["registration_sha256"] != key or parent["confirmation_access"] or not parent["original_ols4_replayed"]:
            raise ValueError("Expected the completed motion controls")
        seal = run.read(run.checked(parent["freeze"]))
        if seal["registration_sha256"] != key or seal["labels_used"] or seal["confirmation_access"]:
            raise ValueError("Require the original coefficients frozen before scoring")
        reg = PUBLIC/"witness_registration.json"
        run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), parent=run.artifact(rpath),
            mode="outcome-aware zero-known-harm constructive witness; no matched budget or deployable upper bound",
            comparison="CV and OLS4 with the frozen full-scene estimates; unchanged per-recording risk",
            confirmation_access=False)))
        token = run.sha(reg)
        meta, provenance, cut = main_study.context(roster)
        rows = {}
        for i, (cell, ref) in enumerate(zip(old["clips"], seal["recordings"], strict=True), 1):
            rec = cell["sequence"]
            if rec != ref["recording"] or rec not in roster:
                raise ValueError("Mismatched calibration roster")
            path = PRIVATE/f"{rec}.witness.json"
            if not path.exists():
                if args.replay:
                    raise ValueError("Replay cannot reopen labels")
                run.once(path, run.stable(dict(registration_sha256=token, coefficients=ref["coefficients"],
                    row=recording(manifest, cell, ref, meta, provenance, cut))))
            v = run.read(path)
            if v["registration_sha256"] != token or v["coefficients"] != ref["coefficients"]:
                raise ValueError("Witness checkpoint mismatch")
            rows[rec] = v["row"]
            main_study.heartbeat("witness_replay" if args.replay else "witness", rec, i, len(roster))
        out = dict(registration_sha256=token, **summarize(rows, roster), recordings=len(rows), source_groups=15,
            pick_order=["cv", "ols4", "median4", "fitted_origin", "pooled_origin"], result_source="fresh_run",
            hindsight_only=True, risk_rule_changed=False, confirmation_access=False, deployment_changed=False)
        run.once(PUBLIC/"witness.json", run.stable(out))
        print(out["metrics"], flush=True)


if __name__ == "__main__":
    main()
