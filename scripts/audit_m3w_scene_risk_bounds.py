"""Analytical and synthetic bound checks only; no forecasting/calibration data read."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_scene_risk_bounds import (
    hb_p_value, hb_upper_mean, joint_monotonicity_witness, zero_loss_required_scenes,
)


OUTPUT = ROOT / "outputs/publication_readiness_2026_09/scene_risk_bound_feasibility_v1"
SOURCES = {
    "ltt": {"url": "https://arxiv.org/abs/2110.01052v5", "version": "v5",
            "title": "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control",
            "authors": ["Anastasios N. Angelopoulos", "Stephen Bates", "Emmanuel J. Candes",
                        "Michael I. Jordan", "Lihua Lei"],
            "read_scope": "sections 1.1, 2.1-2.3; Proposition 1 equation 1 and Bonferroni Proposition 2",
            "use": "existing bounded-mean p-value and family-wise error method; not new M3W theory"},
    "crc": {"url": "https://arxiv.org/abs/2208.02814v4", "version": "v4",
            "title": "Conformal Risk Control",
            "authors": ["Anastasios N. Angelopoulos", "Stephen Bates", "Adam Fisch", "Lihua Lei", "Tal Schuster"],
            "read_scope": "sections 1.1 and 2.1; Theorem 1 assumptions and expectation guarantee",
            "use": "distinguish monotone expectation control from high-probability fixed-family screening"},
}
CODE = ("src/evaluation/m3w_scene_risk_bounds.py", "scripts/audit_m3w_scene_risk_bounds.py",
        "tests/test_m3w_scene_risk_bounds.py", "src/world_model/m3w_joint_intervention.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_checks():
    requirements, upper_bounds = [], []
    for m, k in ((1, 1), (5, 2), (10, 2), (20, 2)):
        for tolerance in (.02, .1, .2):
            level = .05/(m*k)
            requirements.append({
                "family_size": m, "risk_count": k, "illustrative_tolerance": tolerance,
                "delta": .05, "observed_loss": 0.,
                "hoeffding_required_independent_scenes": math.ceil(math.log(1/level)/(2*tolerance**2)),
                "hb_required_independent_scenes": zero_loss_required_scenes(
                    tolerance, .05, family_size=m, risk_count=k),
            })
        for n in (4, 6, 30, 70, 112, 149):
            upper_bounds.append({
                "assumed_independent_scenes": n, "family_size": m, "risk_count": k,
                "observed_loss": 0., "delta": .05,
                "hoeffding_upper": min(1., math.sqrt(math.log(m*k/.05)/(2*n))),
                "hb_upper": hb_upper_mean(0., n, .05/(m*k)),
            })

    exact_tests = []
    for n in (5, 20, 80):
        for low, high, probability in ((0., 1., .1), (.02, .4, .25), (.1, .9, .65)):
            null_mean = low+(high-low)*probability
            for level in (.01, .05, .1):
                rejected = [j for j in range(n+1)
                            if hb_p_value(low+(high-low)*j/n, n, null_mean) <= level]
                size = sum(math.comb(n, j)*probability**j*(1-probability)**(n-j) for j in rejected)
                if size > level+1e-13:
                    raise AssertionError("Enumerated null rejection rate exceeds its error level")
                exact_tests.append({"n": n, "loss_support": [low, high], "high_probability": probability,
                    "null_mean": null_mean, "error_level": level, "exact_rejection_probability": size,
                    "rejected_high_counts": rejected, "pass": True})

    # The ten policy-risk entries may be correlated. Bonferroni does not require
    # their independence; these two extremes are numerical fixtures only.
    n, family, tolerance, true_risk, delta = 300, 10, .02, .02001, .05
    rejected = np.array([hb_p_value(j/n, n, tolerance) <= delta/family for j in range(n+1)])
    one_size = float(binom.pmf(np.arange(n+1), n, true_risk)[rejected].sum())
    rng = np.random.default_rng(273)
    replicates = 20000
    counts = rng.binomial(n, true_risk, size=(replicates, family))
    independent_rate = float(rejected[counts].any(axis=1).mean())
    correlated_rate = float(rejected[counts[:, 0]].mean())
    family_check = {"assumed_iid_scenes": n, "frozen_policy_risk_pairs": family,
        "illustrative_tolerance": tolerance, "true_risk": true_risk, "delta": delta,
        "seed": 273, "synthetic_replicates": replicates,
        "exact_single_pair_size": one_size,
        "exact_independent_pairs_fwer": -math.expm1(family*math.log1p(-one_size)),
        "exact_identical_pairs_fwer": one_size,
        "arbitrary_pair_dependence_union_upper": min(1., family*one_size),
        "simulated_independent_pairs_fwer": independent_rate,
        "simulated_identical_pairs_fwer": correlated_rate,
        "simulation_is_not_proof_or_real_risk_estimate": True}
    if family*one_size > delta+1e-13:
        raise AssertionError("Exact family-wise fixture bound exceeds delta")

    cluster_n, copies, p, alpha = 6, 1000, .2, .1
    rejected_cluster = [j for j in range(cluster_n+1) if hb_p_value(j/cluster_n, cluster_n, alpha) <= .05]
    rejected_copies = [j for j in range(cluster_n+1) if hb_p_value(j/cluster_n, cluster_n*copies, alpha) <= .05]
    size = lambda choices: sum(math.comb(cluster_n, j)*p**j*(1-p)**(cluster_n-j) for j in choices)
    dependence = {"real_independent_synthetic_units": cluster_n, "copies_per_unit": copies,
        "true_risk": p, "illustrative_tolerance": alpha, "delta": .05,
        "valid_cluster_size": size(rejected_cluster), "invalid_copied_window_size": size(rejected_copies),
        "exact_enumeration_not_observed_m3w_failure": True}

    return {
        "schema_version": 1, "result_source": "fresh_run_analytical_and_synthetic_only",
        "sources": SOURCES, "implementation_sha256": {p: sha(ROOT/p) for p in CODE},
        "zero_loss_scene_requirements": requirements, "zero_loss_upper_bounds": upper_bounds,
        "exact_two_point_null_checks": exact_tests, "familywise_fixture": family_check,
        "dependence_witness": dependence, "joint_nonmonotonicity_witness": joint_monotonicity_witness(),
        "all_exact_checks_pass": True,
        "real_source_arrays_read": False, "real_losses_read": False,
        "scientific_roles_assigned": False, "risk_tolerance_changed": False,
        "approved_easy_relative_error_ceiling_percent": 2,
        "illustrative_unit_risk_is_relative_easy_error": False,
        "legacy_calibration_implementation_changed": False,
        "new_bound_wired_into_calibration": False,
        "independent_sites_verified": False,
        "training_or_forecasting": "not_run", "real_calibration": "not_run",
        "independent_confirmation": "not_run", "deployment": False,
        "method_novelty_established": False, "stage5c_executed": False, "smc_enabled": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    result = run_checks()
    path = OUTPUT / "analysis.json"
    if args.verify:
        if result != json.loads(path.read_text()):
            raise SystemExit("Frozen numerical diagnostic differs; do not overwrite")
        status = "cached_verified"
    else:
        if OUTPUT.exists():
            raise SystemExit("Output exists; use --verify")
        OUTPUT.mkdir(parents=True)
        with path.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        receipt = {"finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "seconds": round(time.monotonic()-start, 3), "analysis_sha256": sha(path),
            "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__}
        with (OUTPUT/"execution.json").open("x") as stream:
            stream.write(json.dumps(receipt, indent=2) + "\n")
        status = "fresh_run"
    print(json.dumps({"status": status, "analysis_sha256": sha(path),
        "exact_cases": len(result["exact_two_point_null_checks"]),
        "familywise_fixture": result["familywise_fixture"],
        "zero_loss_scene_requirements": result["zero_loss_scene_requirements"],
        "dependence_witness": result["dependence_witness"],
        "real_calibration": result["real_calibration"]}, indent=2))


if __name__ == "__main__":
    main()
