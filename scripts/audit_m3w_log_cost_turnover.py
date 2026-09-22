"""Decompose frozen log/square-loss decisions without proposing a new policy."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_review_veto import region_stats


def turnover_masks(old, new):
    old, new = np.asarray(old), np.asarray(new)
    if old.dtype != bool or new.dtype != bool or old.ndim != 1 or new.shape != old.shape:
        raise ValueError("Aligned one-dimensional boolean decisions required")
    return dict(retained=old & new, dropped=old & ~new, added=new & ~old,
                neither=~old & ~new)


def gain_contributions(reference, candidate, groups, subset):
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    subset = np.asarray(subset)
    if reference.ndim != 1 or candidate.shape != reference.shape or subset.shape != reference.shape:
        raise ValueError("Aligned errors and subset required")
    if subset.dtype != bool or set(groups) != {"retained", "dropped", "added", "neither"}:
        raise ValueError("Boolean subset and complete turnover partition required")
    for mask in groups.values():
        if np.asarray(mask).dtype != bool or np.asarray(mask).shape != reference.shape:
            raise ValueError("Aligned boolean group masks required")
    if not np.all(np.stack(list(groups.values())).sum(0) == 1):
        raise ValueError("Turnover groups must partition every row exactly once")
    if np.any(np.isfinite(reference) != np.isfinite(candidate)):
        raise ValueError("Candidate/reference label support must match")
    if np.any(reference[np.isfinite(reference)] < 0) or np.any(candidate[np.isfinite(candidate)] < 0):
        raise ValueError("Errors cannot be negative")
    supported = subset & np.isfinite(reference)
    denominator = float(reference[supported].sum())
    numerator = {name: float((reference[take & supported] - candidate[take & supported]).sum())
                 for name, take in groups.items()}
    # All groups use the same subset denominator, so gain changes add exactly.
    parts = {name: 100 * value / denominator if denominator > 0 else None
             for name, value in numerator.items()}
    old_net = numerator["retained"] + numerator["dropped"]
    new_net = numerator["retained"] + numerator["added"]
    return dict(supported_rows=int(supported.sum()), reference_error_sum=denominator,
                group_net_error_reduction_sum=numerator, group_gain_contribution_pp=parts,
                old_gain_percent=100 * old_net / denominator if denominator > 0 else None,
                new_gain_percent=100 * new_net / denominator if denominator > 0 else None,
                difference_pp=100 * (new_net - old_net) / denominator if denominator > 0 else None)


def main():
    import platform
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        raise RuntimeError("Native arm64 required before importing Torch")
    from scripts.run_m3w_log_cost import load
    from scripts.run_m3w_bounded_cost import read_arrays
    from scripts.run_m3w_native_forecast import assert_current, immutable_json
    from scripts.verify_m3w_native_joint_controls import distances
    from src.evaluation.m3w_experiment_contract import file_digest
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg, data, views, predictions, prior, refs, identity = load()
    public = ROOT / cfg["reports"]
    result = json.loads((public / "analysis.json").read_text())
    digest = file_digest(public / "analysis.json")
    checks = {}
    for name in ("replay.json", "independent_verification.json"):
        check = json.loads((public / name).read_text())
        assert check["all_checks_passed"] and check["analysis_sha256"] == digest
        checks[name] = file_digest(public / name)
    assert result["identity"] == identity
    n = len(data["sites"])
    y = read_arrays(data, np.arange(n), "target")
    valid = read_arrays(data, np.arange(n), "valid")
    cv = data["geometry"][:, 332:356].reshape(n, 12, 2)
    reference, _ = distances(cv, y, valid, data["scale"])
    old_archives = {a["view"]: a for a in prior["archives"]}
    rows, decomposition = [], []
    for archive in result["archives"]:
        key = archive["view"]
        old_archive = old_archives[key]
        for a in (archive, old_archive):
            assert file_digest(ROOT / a["path"]) == a["sha256"]
        with np.load(ROOT / archive["path"], allow_pickle=False) as z:
            ids, new, score = (z[k].copy() for k in ("ids", "strict_stop", "score"))
        with np.load(ROOT / old_archive["path"], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z["ids"])
            old, old_score = z["strict_stop"].copy(), z["score"].copy()
        with np.load(ROOT / predictions[key]["path"], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z["ids"])
            error, _ = distances(z["prediction"], y[ids], valid[ids], data["scale"][ids])
        groups = turnover_masks(old, new)
        delta = reference[ids] - error
        full, observed = valid[ids].all(1), valid[ids].any(1)
        pr = refs[key, "frozen_region"]["preprocess"]
        subsets = dict(all=np.ones(len(ids), bool),
                       positive_easy=(reference[ids] > 0) & (reference[ids] <= pr["positive_easy_cut"]),
                       hard=reference[ids] >= pr["hard_cut"])
        meta = dict(view=key, site=views[key]["outer_site"], seed=views[key]["seed"])
        for subset, mask in subsets.items():
            effect = gain_contributions(reference[ids], error, groups, mask)
            recorded_new = result["summaries"]["strict_stop"]["seeds"][str(meta["seed"])]
            recorded_old = prior["summaries"]["frozen_region"]["strict_stop"]["seeds"][str(meta["seed"])]
            for label, recorded in (("new", recorded_new), ("old", recorded_old)):
                expected = (recorded["ADE"] if subset == "all" else recorded["subsets"][subset])["by_scene"][meta["site"]]["gain_percent"]
                np.testing.assert_allclose(effect[f"{label}_gain_percent"], expected, rtol=1e-10, atol=1e-9)
            decomposition.append(dict(**meta, subset=subset, **effect))
            for group, take in groups.items():
                v = region_stats(take & mask, full, observed, delta,
                                 dict(square=old_score[:, 1], log=score[:, 1]))
                rows.append(dict(**meta, subset=subset, group=group, **v))
    aggregate = {}
    for group in ("retained", "dropped", "added", "neither"):
        group_rows = [r for r in rows if r["subset"] == "all" and r["group"] == group]
        aggregate[group] = {k: sum(r[k] for r in group_rows) for k in ("rows", "complete", "incomplete", "unknown")}
        aggregate[group].update({k: sum(r["costs"][k] for r in group_rows if r["costs"])
                                 for k in ("beneficial", "harmful", "tied")})
    parts = [r["difference_pp"] for r in decomposition if r["subset"] == "all"]
    np.testing.assert_allclose(np.mean(parts), result["contrasts"]["strict_stop"]["frozen_region"]["mean_gain_difference_pp"],
                               rtol=1e-10, atol=1e-9)
    out = dict(result_source="fresh_post_readout_turnover_diagnosis_from_cached_verified_decisions",
               analysis_sha256=digest, verification_sources=checks,
               code_sha256=file_digest(Path(__file__)),
               tests_sha256=file_digest(ROOT / "tests/test_m3w_log_cost_turnover.py"),
               rows=rows, decomposition=decomposition,
               repeated_seed_counts_not_independent_samples=aggregate,
               effect_unit="annotation_pixel_ADE_within_scene_then_equal_scene_gain_percent",
               primary_difference_reconstructed_pp=float(np.mean(parts)),
               physical_sites=cfg["sites"], seeds=cfg["seeds"],
               conditional_costs_complete_only=True, gain_decomposition_uses_original_supported_ADE=True,
               unknown_outcomes_not_counted_safe=True, outcome_subsets_never_inference_inputs=True,
               changed_decisions=False, new_training=False, threshold_search=False,
               independent_confirmation=False, risk_calibrated=False, deployment=False,
               stage5c_executed=False, smc_enabled=False)
    assert_current(identity)
    assert file_digest(public / "analysis.json") == digest
    immutable_json(public / "turnover_diagnosis.json", out)
    print(json.dumps(dict(counts=aggregate, primary_difference_pp=out["primary_difference_reconstructed_pp"]), indent=2))


if __name__ == "__main__":
    main()
