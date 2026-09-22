"""Post-readout diagnosis of frozen decisions, never a deployable policy search."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rank_auc(scores, harmful):
    scores, harmful = np.asarray(scores, float), np.asarray(harmful)
    if scores.ndim != 1 or harmful.shape != scores.shape or harmful.dtype != bool:
        raise ValueError("Aligned one-dimensional scores and boolean labels required")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite risk score")
    positive = int(harmful.sum())
    negative = len(scores) - positive
    if not positive or not negative:
        return None
    _, inverse, count = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (count.cumsum() - (count - 1) / 2)[inverse]
    return float((ranks[harmful].sum() - positive * (positive + 1) / 2) / (positive * negative))


def region_stats(use, complete, observed, delta, predicted_harms):
    use, complete, observed = (np.asarray(x, bool) for x in (use, complete, observed))
    delta = np.asarray(delta, float)
    if any(x.shape != delta.shape for x in (use, complete, observed)) or delta.ndim != 1:
        raise ValueError("Aligned row masks required")
    if np.any(complete & ~observed):
        raise ValueError("Complete labels cannot be unobserved")
    take = use & complete
    if not np.isfinite(delta[take]).all():
        raise ValueError("Complete outcomes must be finite")
    values = delta[take]
    out = dict(rows=int(use.sum()), complete=int(take.sum()),
               unknown=int((use & ~observed).sum()), incomplete=int((use & ~complete).sum()))
    if not len(values):
        return dict(**out, costs=None)
    gain, harm = np.maximum(values, 0), np.maximum(-values, 0)
    predicted = {}
    for name, score in predicted_harms.items():
        score = np.asarray(score, float)
        if score.shape != delta.shape or not np.isfinite(score[take]).all():
            raise ValueError("Aligned finite predicted costs required")
        predicted[name] = float(score[take].mean())
    return dict(**out, costs=dict(benefit_mean=float(gain.mean()), harm_mean=float(harm.mean()),
        net_mean=float(values.mean()), net_sum=float(values.sum()),
        beneficial=int((values > 0).sum()), harmful=int((values < 0).sum()),
        tied=int((values == 0).sum()), predicted_harm_mean=predicted))


def main():
    import platform
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        raise RuntimeError("Native arm64 required before importing Torch")
    from scripts.run_m3w_cross_objective_review import load
    from scripts.run_m3w_bounded_cost import read_arrays
    from scripts.run_m3w_native_forecast import assert_current, immutable_json
    from scripts.verify_m3w_native_joint_controls import distances
    from src.evaluation.m3w_experiment_contract import file_digest
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg, _, data, views, predictions, _, _, _, nomination, identity = load()
    public = ROOT / cfg["reports"]
    result = json.loads((public / "analysis.json").read_text())
    digest = file_digest(public / "analysis.json")
    evidence = {}
    for name in ("replay.json", "independent_verification.json"):
        check = json.loads((public / name).read_text())
        assert check["all_checks_passed"] and check["analysis_sha256"] == digest
        evidence[name] = file_digest(public / name)
    assert result["identity"] == identity
    n = len(data["sites"])
    y = read_arrays(data, np.arange(n), "target")
    valid = read_arrays(data, np.arange(n), "valid")
    baseline = data["geometry"][:, 332:356].reshape(n, 12, 2)
    reference, _ = distances(baseline, y, valid, data["scale"])
    rows, rankings = [], []
    for archive in result["archives"]:
        key = archive["view"]
        assert file_digest(ROOT / archive["path"]) == archive["sha256"]
        with np.load(ROOT / archive["path"], allow_pickle=False) as z:
            ids = z["ids"].copy()
            a, b, c = (z[k].copy() for k in ("nomination_score", "native_score", "fraction_score"))
            chosen = {k: z[k].copy() for k in ("nomination", "reviewed", "matched_nomination")}
        with np.load(ROOT / predictions[key]["path"], allow_pickle=False) as z:
            np.testing.assert_array_equal(z["ids"], ids)
            error, _ = distances(z["prediction"], y[ids], valid[ids], data["scale"][ids])
        delta = reference[ids] - error
        full, observed = valid[ids].all(1), valid[ids].any(1)
        rejected = chosen["nomination"] & ~chosen["reviewed"]
        native_veto = chosen["nomination"] & (b[:, 1] > .1 * a[:, 0])
        fraction_veto = chosen["nomination"] & (c[:, 1] > .1 * a[:, 0])
        np.testing.assert_array_equal(rejected, native_veto | fraction_veto)
        groups = dict(**chosen, rejected=rejected, native_only_veto=native_veto & ~fraction_veto,
                      fraction_only_veto=fraction_veto & ~native_veto, both_veto=native_veto & fraction_veto)
        pr = nomination[key]["preprocess"]
        subsets = dict(all=np.ones(len(ids), bool),
            positive_easy=(reference[ids] > 0) & (reference[ids] <= pr["positive_easy_cut"]),
            hard=reference[ids] >= pr["hard_cut"])
        harms = dict(nomination=a[:, 1], native=b[:, 1], fraction=c[:, 1],
                     review=np.maximum.reduce((a[:, 1], b[:, 1], c[:, 1])))
        for subset, mask in subsets.items():
            stats = {g: region_stats(bits & mask, full, observed, delta, harms) for g, bits in groups.items()}
            for field in ("rows", "complete", "unknown", "incomplete"):
                assert stats["nomination"][field] == stats["reviewed"][field] + stats["rejected"][field]
                assert stats["rejected"][field] == sum(stats[g][field] for g in
                    ("native_only_veto", "fraction_only_veto", "both_veto"))
            net = lambda g: 0 if stats[g]["costs"] is None else stats[g]["costs"]["net_sum"]
            np.testing.assert_allclose(net("nomination"), net("reviewed") + net("rejected"), rtol=1e-10, atol=1e-8)
            for group, v in stats.items():
                rows.append(dict(view=key, site=views[key]["outer_site"], seed=views[key]["seed"],
                                 subset=subset, group=group, **v))
        # Outcome-defined subsets below are diagnostic labels, never selector inputs.
        use = chosen["nomination"] & full
        assert (a[use, 0] > 0).all()
        rankings.append(dict(view=key, complete_nominations=int(use.sum()),
            harmful_nominations=int((delta[use] < 0).sum()),
            harm_ratio_auc={name: rank_auc(score[use] / a[use, 0], delta[use] < 0)
                            for name, score in harms.items()}))
        for g in ("nomination", "reviewed", "matched_nomination", "rejected"):
            old = next(v for v in result["conditional_quality"] if v["view"] == key and v["group"] == g)
            fresh = next(v for v in rows if v["view"] == key and v["group"] == g and v["subset"] == "all")
            assert fresh["rows"] == old["rows"] and fresh["complete"] == old["complete"]
            if old["costs"] is not None:
                for measured, expected in ((fresh["costs"]["harm_mean"], old["costs"]["realized_harm"]),
                    (fresh["costs"]["net_mean"], old["costs"]["realized_net_gain"]),
                    (fresh["costs"]["predicted_harm_mean"]["review"], old["costs"]["review_harm"])):
                    np.testing.assert_allclose(measured, expected, rtol=1e-10, atol=1e-10)
    aggregate = {}
    for group in ("nomination", "reviewed", "rejected", "matched_nomination"):
        selected = [v for v in rows if v["group"] == group and v["subset"] == "all"]
        aggregate[group] = {k: sum(v[k] for v in selected) for k in ("rows", "complete", "unknown", "incomplete")}
        aggregate[group].update({k: sum(v["costs"][k] for v in selected if v["costs"]) for k in ("beneficial", "harmful", "tied")})
        aggregate[group]["positive_net_views"] = sum(v["costs"]["net_mean"] > 0 for v in selected if v["costs"])
        aggregate[group]["review_underpredicts_mean_harm_views"] = sum(
            v["costs"]["predicted_harm_mean"]["review"] < v["costs"]["harm_mean"] for v in selected if v["costs"])
    out = dict(result_source="fresh_post_readout_diagnosis_from_frozen_verified_decisions",
        source_analysis_sha256=digest, verification_sources=evidence,
        code_sha256=file_digest(Path(__file__)), test_sha256=file_digest(ROOT / "tests/test_m3w_review_veto_audit.py"),
        rows=rows, rankings=rankings, counts_across_repeated_seeds_not_independent_samples=aggregate,
        physical_sites=len(cfg["sites"]), seeds=cfg["seeds"],
        effect_unit="annotation_pixel_per_complete_12_step_trajectory",
        partial_or_unknown_outcomes_excluded_from_conditional_costs=True,
        changed_decisions=False, threshold_search=False, new_training=False,
        calibrated_risk=False, independent_confirmation=False, deployment=False,
        stage5c_executed=False, smc_enabled=False)
    assert_current(identity)
    assert file_digest(public / "analysis.json") == digest
    immutable_json(public / "veto_diagnosis.json", out)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
