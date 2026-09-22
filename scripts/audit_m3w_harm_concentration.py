"""Read-only track/recording support audit of verified frozen risk decisions."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def mass_concentration(mass, keys):
    mass, keys = np.asarray(mass, float), np.asarray(keys)
    if mass.ndim != 1 or keys.shape != mass.shape or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("Aligned finite nonnegative masses and group keys required")
    names, inverse = np.unique(keys, return_inverse=True)
    values = np.bincount(inverse, weights=mass, minlength=len(names))
    order = np.lexsort((names, -values))
    total = float(values.sum())
    positive = int((values > 0).sum())
    top = [dict(key=str(names[i]), mass=float(values[i]), fraction=float(values[i] / total))
           for i in order[:5] if values[i] > 0] if total > 0 else []
    return dict(groups=len(names), positive_mass_groups=positive, total_mass=total,
                top1_share=float(values[order[:1]].sum() / total) if total > 0 else None,
                top5_share=float(values[order[:5]].sum() / total) if total > 0 else None,
                top10_share=float(values[order[:10]].sum() / total) if total > 0 else None,
                groups_for_half_mass=int(np.searchsorted(np.cumsum(values[order]), total / 2) + 1) if total > 0 else 0,
                inverse_herfindahl=float(total ** 2 / np.dot(values, values)) if total > 0 else None,
                top5=top)


def overlap_components(tracks, frames, selected, *, raw_span):
    tracks, frames, selected = np.asarray(tracks), np.asarray(frames), np.asarray(selected)
    if (tracks.ndim != 1 or frames.shape != tracks.shape or selected.shape != tracks.shape
            or selected.dtype != bool or frames.dtype.kind not in "iu" or raw_span < 0):
        raise ValueError("Aligned track IDs, integer frames, boolean mask and nonnegative span required")
    ids = np.flatnonzero(selected)
    ids = ids[np.lexsort((frames[ids], tracks[ids]))]
    if len(ids) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    start = np.ones(len(ids), bool)
    same = tracks[ids[1:]] == tracks[ids[:-1]]
    gaps = frames[ids[1:]] - frames[ids[:-1]]
    if np.any(same & (gaps == 0)):
        raise ValueError("Duplicate query frame within a recording-qualified track")
    start[1:] = ~same | (gaps > raw_span)
    return ids, np.cumsum(start) - 1


def region_summary(reference, candidate, chosen, complete, recordings, tracks, frames, raw_span):
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    chosen, complete = np.asarray(chosen), np.asarray(complete)
    n = len(reference)
    if (reference.ndim != 1 or any(np.shape(v) != (n,) for v in
            (candidate, chosen, complete, recordings, tracks, frames))
            or chosen.dtype != bool or complete.dtype != bool):
        raise ValueError("Aligned row arrays and boolean masks required")
    if np.isinf(reference).any() or np.isinf(candidate).any():
        raise ValueError("Infinite errors are not missing outcomes")
    observed = np.isfinite(reference)
    if not np.array_equal(observed, np.isfinite(candidate)) or np.any(complete & ~observed):
        raise ValueError("Inconsistent future-label support")
    if np.any(reference[observed] < 0) or np.any(candidate[observed] < 0):
        raise ValueError("Errors must be nonnegative")
    use = chosen & observed
    benefit = np.zeros(n); harm = np.zeros(n)
    benefit[use] = np.maximum(reference[use] - candidate[use], 0)
    harm[use] = np.maximum(candidate[use] - reference[use], 0)
    harmed = use & (harm > 0)
    ids, components = overlap_components(tracks, frames, harmed, raw_span=raw_span)
    return dict(selected=int(chosen.sum()), supported=int(use.sum()),
                complete=int((chosen & complete).sum()),
                partial_supported=int((use & ~complete).sum()), unknown=int((chosen & ~observed).sum()),
                beneficial=int((benefit > 0).sum()), harmful=int(harmed.sum()),
                tied=int((use & (benefit == 0) & (harm == 0)).sum()),
                gross_benefit_sum=float(benefit.sum()), gross_harm_sum=float(harm.sum()),
                net_error_reduction_sum=float(benefit.sum() - harm.sum()),
                complete_harm_sum=float(harm[complete].sum()), partial_supported_harm_sum=float(harm[~complete].sum()),
                track_harm=mass_concentration(harm, tracks),
                recording_harm=mass_concentration(harm, recordings),
                overlap_component_harm=mass_concentration(harm[ids], components))


def main():
    import platform
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        raise RuntimeError("Native arm64 required before Torch import")
    from scripts.run_m3w_log_cost import load, load_states
    from scripts.run_m3w_bounded_cost import read_arrays
    from scripts.run_m3w_native_forecast import assert_current, immutable_json
    from scripts.verify_m3w_native_joint_controls import distances
    from src.evaluation.m3w_experiment_contract import file_digest
    import torch

    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, predictions, prior, refs, identity = load()
    _, states = load_states(cfg, views, identity)
    public = ROOT / cfg["reports"]
    result = json.loads((public / "analysis.json").read_text())
    digest = file_digest(public / "analysis.json")
    assert result["identity"] == identity
    checks = {}
    for name in ("replay.json", "independent_verification.json"):
        r = json.loads((public / name).read_text())
        assert r["all_checks_passed"] and r["analysis_sha256"] == digest
        checks[name] = file_digest(public / name)
    turnover = json.loads((public / "turnover_diagnosis.json").read_text())
    assert turnover["analysis_sha256"] == digest and not turnover["changed_decisions"]
    checks["turnover_diagnosis.json"] = file_digest(public / "turnover_diagnosis.json")
    setup_path = ROOT / "configs/m3w_sdd_auxiliary_v1.json"
    setup = json.loads(setup_path.read_text())
    assert setup["observed_steps"] == 8 and setup["predicted_steps"] == 12 and setup["raw_frame_stride"] == 12
    # Connected overlapping input/target supports are descriptive, not IID events.
    raw_span = (setup["observed_steps"] - 1 + setup["predicted_steps"]) * setup["raw_frame_stride"]
    n = len(data["sites"])
    y = read_arrays(data, np.arange(n), "target")
    valid = read_arrays(data, np.arange(n), "valid")
    baseline = data["geometry"][:, 332:356].reshape(n, 12, 2)
    cv, _ = distances(baseline, y, valid, data["scale"])
    full = valid.all(1)
    old_archives = {a["view"]: a for a in prior["archives"]}
    rows, sampling, by_site, feature_width = [], [], {}, set()
    for a in result["archives"]:
        key = a["view"]; meta = views[key]; old = old_archives[key]
        for source in (a, old):
            assert file_digest(ROOT / source["path"]) == source["sha256"]
        with np.load(ROOT / a["path"], allow_pickle=False) as z:
            ids, take = z["ids"].copy(), z["strict_stop"].copy()
        with np.load(ROOT / old["path"], allow_pickle=False) as z:
            np.testing.assert_array_equal(z["ids"], ids)
            retained = take & z["strict_stop"]
        with np.load(ROOT / predictions[key]["path"], allow_pickle=False) as z:
            np.testing.assert_array_equal(z["ids"], ids)
            error, _ = distances(z["prediction"], y[ids], valid[ids], data["scale"][ids])
        cp = states[key]; pr = cp["preprocess"]
        feature_width.add(len(pr["mean"]))
        easy = (cv[ids] > 0) & (cv[ids] <= pr["positive_easy_cut"])
        site = meta["outer_site"]
        for region, chosen in (("retained", retained), ("log_strict", take)):
            for subset, mask in (("all", np.ones(len(ids), bool)), ("positive_easy", easy)):
                stats = region_summary(cv[ids], error, chosen & mask, full[ids],
                    data["recordings"][ids], data["tracks"][ids], data["frames"][ids], raw_span)
                rows.append(dict(view=key, site=site, seed=meta["seed"], region=region, subset=subset, **stats))
                if region == "retained":
                    original = next(v for v in turnover["rows"] if v["view"] == key and v["subset"] == subset and v["group"] == "retained")
                    assert stats["selected"] == original["rows"] and stats["complete"] == original["complete"]
        known = np.isfinite(error)
        harmful = retained & easy & known & (error > cv[ids])
        by_site.setdefault(site, []).append(dict(seed=meta["seed"], ids=ids, retained=retained,
            harmed=harmful, harm=np.where(harmful, np.nan_to_num(error-cv[ids], nan=0), 0)))
        with np.load(ROOT / meta["inputs_path"], allow_pickle=False) as z:
            ti = z["ids"].copy()
        np.testing.assert_array_equal(ti, np.flatnonzero(data["sites"] != site))
        assert cp["draws"].sum() == 3072000 and not cp["draws"][~pr["known"]].any()
        for training_site in sorted(set(data["sites"][ti])):
            t = data["sites"][ti] == training_site
            expected = pr["weights"][t]
            influence = expected * cp["loss_weights"][t]
            sampling.append(dict(view=key, fitting_site=training_site,
                complete_training_rows=int((t & pr["known"]).sum()),
                expected_draw_mass=mass_concentration(expected, data["tracks"][ti][t]),
                actual_draw_mass=mass_concentration(cp["draws"][t], data["tracks"][ti][t]),
                expected_weighted_loss_mass=mass_concentration(influence, data["tracks"][ti][t])))
    consensus = []
    for site, records in by_site.items():
        assert sorted(r["seed"] for r in records) == cfg["seeds"]
        ids = records[0]["ids"]
        for r in records: np.testing.assert_array_equal(ids, r["ids"])
        count = np.sum([r["harmed"] for r in records], axis=0)
        mean_harm = np.mean([r["harm"] for r in records], axis=0)
        consensus.append(dict(site=site, unique_queries_harmed_in_any_seed=int((count > 0).sum()),
            harmed_in_exactly_n_seeds={str(k):int((count == k).sum()) for k in (1, 2, 3)},
            share_of_repeated_seed_harm_from_all_three=float(mean_harm[count == 3].sum() / mean_harm.sum()) if mean_harm.sum() else None,
            mean_seed_harm_by_track=mass_concentration(mean_harm, data["tracks"][ids]),
            mean_seed_harm_by_recording=mass_concentration(mean_harm, data["recordings"][ids])))
    assert feature_width == {356}
    out = dict(result_source="fresh_post_readout_support_diagnosis_cached_verified_decisions",
        analysis_sha256=digest, verification_sources=checks,
        code_sha256=file_digest(Path(__file__)), tests_sha256=file_digest(ROOT / "tests/test_m3w_harm_concentration.py"),
        setup_sha256=file_digest(setup_path), raw_window_support_span=raw_span,
        feature_width=356, base_cost_features=355, extra_feature="log1p_past_only_forecast_disagreement",
        feature_width_note="registration_prose_says_355_but_matched_code_and_checkpoints_have_always_used_356",
        rows=rows, training_sampling=sampling, cross_seed_consensus=consensus,
        aggregation_unit="within_physical_site_only_no_cross_site_native_error_pooling",
        overlap_components_are_not_independent_events=True, track_ids_recording_qualified_not_unique_people=True,
        unknown_outcomes_not_counted_safe=True, new_training=False, changed_decisions=False,
        threshold_search=False, closed_role_readout=False, independent_confirmation=False,
        deployment=False, risk_calibration=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity)
    assert file_digest(public / "analysis.json") == digest
    immutable_json(public / "harm_concentration.json", out)
    print(json.dumps(dict(records=len(rows), training_groups=len(sampling),
        consensus=[{k:v for k,v in c.items() if not k.startswith("mean_seed")} for c in consensus], feature_width=356), indent=2))


if __name__ == "__main__":
    main()
