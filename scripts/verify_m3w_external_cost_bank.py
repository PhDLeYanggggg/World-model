"""Independently recompute cost targets and sampling, then replay source probes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys

if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Native arm64 Python required")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch

from scripts.run_m3w_native_forecast import load as source_load, array_hash, assert_current, immutable_json
from src.evaluation.m3w_recording_lineage import sha256
from src.world_model.m3w_native_gain_harm import cost_features, standardized
from src.world_model.m3w_bounded_cost_head import build, predict


def direct_labels(baseline, candidate, target, valid, scale):
    b, p, y, s = map(lambda a: np.asarray(a, float), (baseline, candidate, target, scale))
    mask = np.asarray(valid)
    if (p.shape != b.shape or y.shape != p.shape or p.ndim != 3 or p.shape[1:] != (12, 2)
            or mask.dtype != bool or mask.shape != p.shape[:2] or s.shape != (len(p),)
            or not np.isfinite(p).all() or not np.isfinite(b).all()
            or not np.isfinite(y[mask]).all() or not np.isfinite(s).all() or (s <= 0).any()):
        raise ValueError("Finite aligned trajectories and explicit label support required")
    d = np.sqrt(((p-b)**2).sum(2)).mean(1)*s
    full = mask.all(1)
    gain = (np.sqrt(((b[full]-y[full])**2).sum(2)).mean(1)
            - np.sqrt(((p[full]-y[full])**2).sum(2)).mean(1))*s[full]
    labels = np.full((len(p), 2), np.nan)
    labels[full, 0], labels[full, 1] = np.clip(gain, 0, None), np.clip(-gain, 0, None)
    if (labels[full].sum(1) > d[full]+1e-9*(1+d[full])).any():
        raise ValueError("Independent triangle bound failed")
    return labels, d, full


def replay_draws(sites, complete, seed, *, updates=3000, batch_size=256):
    groups = [np.flatnonzero(complete & (sites == s)) for s in sorted(set(sites))]
    if any(not len(g) for g in groups):
        raise ValueError("Each fitting scene requires supported targets")
    rng = torch.Generator().manual_seed(seed+7919)
    draws = np.zeros(len(sites), np.int64)
    for _ in range(updates):
        choices = torch.randint(len(groups), (batch_size,), generator=rng).numpy()
        for k, group in enumerate(groups):
            offsets = torch.randint(len(group), (int((choices == k).sum()),), generator=rng).numpy()
            np.add.at(draws, group[offsets], 1)
    return draws


def serialized_costs(baseline, candidate, target, valid, scale):
    """Preserve the registered per-step native-scale arithmetic for file hashes."""
    b, p, y, s = map(lambda a: np.asarray(a, float), (baseline, candidate, target, scale))
    labels, independent_distance, full = direct_labels(b, p, y, valid, s)
    distance = (np.linalg.norm(p-b, axis=2)*s[:, None]).mean(1)
    wire = np.full_like(labels, np.nan)
    cv = (np.linalg.norm(b[full]-y[full], axis=2)*s[full, None]).mean(1)
    error = (np.linalg.norm(p[full]-y[full], axis=2)*s[full, None]).mean(1)
    wire[full] = np.column_stack((np.maximum(cv-error, 0), np.maximum(error-cv, 0)))
    np.testing.assert_allclose(wire, labels, rtol=1e-10, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(distance, independent_distance, rtol=1e-12, atol=1e-12)
    return wire, distance, full


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    public = ROOT/"outputs/publication_readiness_2026_09/external_cost_bank_v1"
    path = public/"analysis.json"
    analysis = json.loads(path.read_text())
    replay = json.loads((public/"replay.json").read_text())
    if not replay["all_checks_passed"] or replay["analysis_sha256"] != sha256(path):
        raise ValueError("Completed exact replay required before independent verification")
    identity = analysis["identity"]
    assert_current(identity)
    _, data, _, population = source_load()
    if population["population_sha256"] != identity["population_sha256"]:
        raise ValueError("Source population changed")
    b = data["geometry"][:, 332:356].reshape(-1, 12, 2)
    checks = []
    for key, view in identity["views"].items():
        p = np.empty(b.shape, dtype=np.float32)
        seen = np.zeros(len(p), np.int8)
        for producer in view["producers"]:
            with np.load(ROOT/producer["prediction"]["path"], allow_pickle=False) as z:
                ids = z["ids"]
                np.testing.assert_array_equal(ids, np.flatnonzero(data["sites"] == producer["row_site"]))
                if producer["row_site"] in producer["training_sites"]:
                    raise ValueError("OOF producer exposure")
                p[ids], seen[ids] = z["prediction"], seen[ids]+1
        if not np.all(seen == 1):
            raise ValueError("Incomplete or duplicate source alignment")
        labels, d, full = serialized_costs(b, p, data["target"], data["valid"], data["scale"])
        # Reconstruct the causal feature input separately from labels.
        x, _ = cost_features(data["geometry"], p, data["scale"])
        x = np.column_stack((x, np.log1p(d))).astype(np.float32)
        expected_draws = replay_draws(data["sites"], full, view["seed"])
        ids = np.concatenate([g[np.linspace(0, len(g)-1, 16, dtype=int)]
            for g in [np.flatnonzero(data["sites"] == s) for s in identity["config"]["sites"]]])
        for record in [r for r in analysis["models"] if r["view"] == key]:
            if sha256(ROOT/record["checkpoint"]) != record["checkpoint_sha256"]:
                raise ValueError("Changed cost checkpoint")
            forest = record["head"] == "matched_fraction_forest"
            cp = joblib.load(ROOT/record["checkpoint"]) if forest else torch.load(
                ROOT/record["checkpoint"], map_location="cpu", weights_only=False)
            np.testing.assert_array_equal(cp["draws"], expected_draws)
            np.testing.assert_array_equal(cp["preprocess"]["known"], full)
            if array_hash(labels) != record["labels_sha256"] or array_hash(x, d) != record["inputs_sha256"]:
                raise ValueError("Training target or causal input identity changed")
            pr = cp["preprocess"]
            if forest:
                np.testing.assert_array_equal(cp["sample_weight"], expected_draws*(d > 0))
                cp["model"].set_params(n_jobs=1)
                score = cp["model"].predict(standardized(x[ids], pr))*d[ids, None]
            else:
                model = build(x.shape[1], 64, view["seed"])
                model.load_state_dict(cp["model"])
                score = predict(model, x[ids], d[ids], pr, "bounded_fraction")
                with torch.no_grad():
                    positive = torch.nn.functional.softplus(model.network(
                        torch.from_numpy(standardized(x[ids], pr)))).numpy().astype(float)
                expected = d[ids, None]*positive/(1+positive.sum(1, keepdims=True))
                np.testing.assert_allclose(score, expected, rtol=2e-6, atol=2e-6)
            if array_hash(score) != record["score_sha256"]:
                raise ValueError("Fixed source score replay differs")
            if (not np.isfinite(score).all() or (score < 0).any()
                    or np.any(score.sum(1) > d[ids]+2e-6*(1+d[ids]))):
                raise ValueError("Invalid bounded score")
            checks.append(dict(view=key, head=record["head"], matched_sampling=True,
                recomputed_complete_targets=int(full.sum()), probe_score_hash_exact=True))
    if len(checks) != 12:
        raise ValueError("Incomplete independent endpoint check")
    record = dict(analysis_sha256=sha256(path), verifier_sha256=sha256(Path(__file__)),
        verifier_tests_sha256=sha256(ROOT/"tests/test_m3w_external_cost_verification.py"),
        checks=checks, all_checks_passed=True, result_source="fresh_verification_of_cached_fits",
        independent_implementation_same_agent=True, independent_research_confirmation=False,
        new_training=False, reserved_source_rows=0)
    immutable_json(public/"independent_verification.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
