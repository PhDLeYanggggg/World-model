"""Diagnose scalar full-horizon cost versus partial-label ADE, without policy edits."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def step_errors(baseline, candidate, target, valid, scale):
    b, p, y = (np.asarray(v, float) for v in (baseline, candidate, target))
    valid, scale = np.asarray(valid), np.asarray(scale, float)
    if (b.ndim != 3 or b.shape[-1] != 2 or p.shape != b.shape or y.shape != b.shape
            or valid.shape != b.shape[:2] or valid.dtype != bool or scale.shape != (len(b),)):
        raise ValueError("Aligned forecasts, future labels, boolean mask and per-row scale required")
    if (not np.isfinite(b).all() or not np.isfinite(p).all() or not np.isfinite(y[valid]).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Finite predictions, supported targets and positive scales required")
    safe = np.where(valid[..., None], y, 0)
    errors = [np.linalg.norm(v-safe, axis=-1) * scale[:, None] for v in (b, p)]
    return tuple(np.where(valid, e, np.nan) for e in errors)


def prefix_summary(cv_error, neural_error, selected):
    cv, neural, selected = np.asarray(cv_error, float), np.asarray(neural_error, float), np.asarray(selected)
    if cv.ndim != 2 or neural.shape != cv.shape or selected.shape != (len(cv),) or selected.dtype != bool:
        raise ValueError("Aligned step errors and boolean selection required")
    if not np.array_equal(np.isfinite(cv), np.isfinite(neural)):
        raise ValueError("Error support mismatch")
    if np.isinf(cv).any() or np.isinf(neural).any() or np.any(cv[np.isfinite(cv)] < 0) or np.any(neural[np.isfinite(neural)] < 0):
        raise ValueError("Nonnegative finite observed errors required")
    full = np.isfinite(cv).all(1)
    use = selected & full
    g = np.cumsum(cv[use]-neural[use], axis=1)/np.arange(1, cv.shape[1]+1)
    positive_full = g[:, -1] >= 0
    early_harm = g < 0
    rows = int(use.sum())
    return dict(selected=int(selected.sum()), complete=rows,
        full_horizon_nonharmful=int(positive_full.sum()),
        full_nonharmful_but_any_prefix_harmful=int((positive_full & early_harm.any(1)).sum()),
        prefix=[dict(steps=k+1, harmful=int(early_harm[:, k].sum()),
            full_nonharmful_but_prefix_harmful=int((positive_full & early_harm[:, k]).sum()),
            harm_mean=float(np.maximum(-g[:, k], 0).mean()) if rows else None,
            benefit_mean=float(np.maximum(g[:, k], 0).mean()) if rows else None)
            for k in range(cv.shape[1])],
        full_horizon_harm_mean=float(np.maximum(-g[:, -1], 0).mean()) if rows else None,
        max_prefix_harm_mean=float(np.maximum(-g, 0).max(1).mean()) if rows else None)


def observed_support_summary(cv_error, neural_error, selected):
    cv, neural, selected = np.asarray(cv_error, float), np.asarray(neural_error, float), np.asarray(selected)
    prefix_summary(cv, neural, selected)  # Validate shared shapes and support contract.
    valid = np.isfinite(cv); count = valid.sum(1)
    step_gain = np.where(valid, cv-neural, 0)
    gain = np.divide(step_gain.sum(1), count, out=np.full(len(cv), np.nan), where=count > 0)
    prefix = np.all(valid == (np.arange(cv.shape[1])[None, :] < count[:, None]), axis=1)
    rows = []
    for k in range(cv.shape[1]+1):
        use = selected & (count == k)
        value = gain[use]
        rows.append(dict(valid_steps=k, rows=int(use.sum()), prefix_mask_rows=int((use & prefix).sum()),
            harmful=int((value < 0).sum()) if k else None,
            gross_harm_sum=float(np.maximum(-value, 0).sum()) if k else None,
            gross_benefit_sum=float(np.maximum(value, 0).sum()) if k else None))
    return rows


def main():
    import platform
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        raise RuntimeError("Native arm64 required before Torch import")
    from scripts.run_m3w_log_cost import load, load_states
    from scripts.run_m3w_bounded_cost import read_arrays
    from scripts.run_m3w_native_forecast import assert_current, immutable_json
    from src.evaluation.m3w_experiment_contract import file_digest
    import torch

    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, predictions, prior, refs, identity = load()
    _, states = load_states(cfg, views, identity)
    public = ROOT/cfg["reports"]
    analysis = json.loads((public/"analysis.json").read_text())
    digest = file_digest(public/"analysis.json")
    sources = {}
    for name in ("replay.json", "independent_verification.json"):
        r = json.loads((public/name).read_text())
        assert r["all_checks_passed"] and r["analysis_sha256"] == digest
        sources[name] = file_digest(public/name)
    conc = json.loads((public/"harm_concentration.json").read_text())
    assert conc["analysis_sha256"] == digest
    sources["harm_concentration.json"] = file_digest(public/"harm_concentration.json")
    assert analysis["identity"] == identity
    n = len(data["sites"])
    target = read_arrays(data, np.arange(n), "target")
    valid = read_arrays(data, np.arange(n), "valid")
    baseline = data["geometry"][:, 332:356].reshape(n, 12, 2)
    old_archives = {a["view"]: a for a in prior["archives"]}
    fitting, held, support = [], [], []
    for a in analysis["archives"]:
        key = a["view"]; meta = views[key]
        for path, expected in ((a["path"], a["sha256"]), (old_archives[key]["path"], old_archives[key]["sha256"])):
            assert file_digest(ROOT/path) == expected
        with np.load(ROOT/a["path"], allow_pickle=False) as z:
            ids, take = z["ids"].copy(), z["strict_stop"].copy()
        with np.load(ROOT/old_archives[key]["path"], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z["ids"]); retained = take & z["strict_stop"]
        with np.load(ROOT/predictions[key]["path"], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z["ids"]); pred = z["prediction"].copy()
        cv, neural = step_errors(baseline[ids], pred, target[ids], valid[ids], data["scale"][ids])
        den = valid[ids].sum(1)
        cv_ade = np.divide(np.nansum(cv, axis=1), den, out=np.full(len(ids), np.nan), where=den>0)
        easy = (cv_ade > 0) & (cv_ade <= states[key]["preprocess"]["positive_easy_cut"])
        for region, bits in (("retained", retained), ("log_strict", take)):
            for subset, mask in (("all", np.ones(len(ids), bool)), ("positive_easy", easy)):
                tag = dict(view=key, site=meta["outer_site"], seed=meta["seed"], region=region, subset=subset)
                held.append(dict(**tag, **prefix_summary(cv, neural, bits & mask)))
                s = observed_support_summary(cv, neural, bits & mask)
                support.append(dict(**tag, support=s))
                c = next(v for v in conc["rows"] if all(v[k]==tag[k] for k in tag))
                assert sum(v["rows"] for v in s) == c["selected"] and s[0]["rows"] == c["unknown"]
                np.testing.assert_allclose(sum(v["gross_harm_sum"] for v in s if v["valid_steps"]>0), c["gross_harm_sum"], rtol=1e-10, atol=1e-8)
        with np.load(ROOT/meta["inputs_path"], allow_pickle=False) as z:
            ti, tp = z["ids"].copy(), z["prediction"].copy()
        np.testing.assert_array_equal(ti, np.flatnonzero(data["sites"] != meta["outer_site"]))
        tb, tn = step_errors(baseline[ti], tp, target[ti], valid[ti], data["scale"][ti])
        with np.load(ROOT/meta["targets_path"], allow_pickle=False) as z:
            np.testing.assert_array_equal(ti, z["ids"])
            full = valid[ti].all(1); gain = (tb[full]-tn[full]).mean(1)
            np.testing.assert_allclose(np.maximum(-gain, 0), z["harm"][full], rtol=1e-10, atol=1e-8)
        for site in sorted(set(data["sites"][ti])):
            fitting.append(dict(view=key, fitting_site=site,
                **prefix_summary(tb, tn, data["sites"][ti] == site)))
    out = dict(result_source="fresh_post_readout_target_support_diagnosis_cached_verified_forecasts",
        analysis_sha256=digest, verification_sources=sources,
        code_sha256=file_digest(Path(__file__)), tests_sha256=file_digest(ROOT/"tests/test_m3w_horizon_cost_support.py"),
        fitting_prefix_costs=fitting, held_complete_prefix_costs=held, held_observed_support=support,
        future_mask_used_for_labels_and_audit_only=True, changed_decisions=False,
        new_training=False, changed_primary_metric=False, changed_data_roles=False,
        closed_role_readout=False, threshold_search=False, independent_confirmation=False,
        risk_calibrated=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity)
    immutable_json(public/"horizon_cost_support.json", out)
    print(json.dumps(dict(fitting_records=len(fitting), held_prefix_records=len(held), support_records=len(support), all_reductions_match=True)))


if __name__ == "__main__":
    main()
