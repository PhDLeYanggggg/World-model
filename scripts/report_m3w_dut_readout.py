"""Reduce frozen DUT receipts without selecting a deployable model or policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.evaluation.m3w_recording_lineage import sha256
from scripts.run_m3w_native_forecast import immutable_json

PUBLIC = ROOT/"outputs/publication_readiness_2026_09/dut_frozen_readout_v1"


def merge(values):
    keys = values[0].keys()
    if any(v.keys() != keys for v in values):
        raise ValueError("Incompatible aggregate fields")
    return {k: (merge([v[k] for v in values]) if isinstance(values[0][k], dict)
                else sum(v[k] for v in values)) for k in keys}


def ratio_change(new, baseline):
    if baseline == 0:
        return 0. if new == 0 else None
    return 100*(new/baseline-1)


def arm_metrics(site_stats, arm, *, bucket="arms", bootstrap=3000, seed=917):
    sites = sorted(site_stats)
    if len(sites) != 2:
        raise ValueError("This registered diagnostic has exactly two physical sites")
    output = dict(sites={}, subsets={})
    for site in sites:
        stats, item = site_stats[site], site_stats[site][bucket][arm]
        out = dict(targets=stats["targets"], complete=stats["complete"], queries=stats["queries"],
            switches=item["switches"], switch_rate=item["switches"]/max(1, stats["targets"]),
            gain_lower=item["gain_lower_sum"]/max(1, stats["targets"]),
            gain_upper=item["gain_upper_sum"]/max(1, stats["targets"]), subsets={})
        for name, part in item["slices"].items():
            n = part["n"]
            out["subsets"][name] = dict(n=n,
                **{k.removesuffix("_sum"): part[k]/n if n else None
                   for k in ("ade_sum", "fde_sum", "baseline_ade_sum", "baseline_fde_sum", "normalized_ade_sum")},
                harmed=part["harm_count"])
        output["sites"][site] = out
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 2, size=(bootstrap, 2))
    for subset in site_stats[sites[0]][bucket][arm]["slices"]:
        parts = [output["sites"][site]["subsets"][subset] for site in sites]
        if any(p["n"] == 0 for p in parts):
            output["subsets"][subset] = dict(status="insufficient_support_in_at_least_one_site",
                                             counts=[p["n"] for p in parts])
            continue
        base = np.array([p["baseline_ade"] for p in parts])
        pred = np.array([p["ade"] for p in parts])
        change = ratio_change(float(pred.mean()), float(base.mean()))
        bmean = base[draws].mean(1)
        delta = pred[draws].mean(1)-bmean
        gain = np.divide(-100*delta, bmean, out=np.full(len(delta), np.nan), where=bmean > 0)
        output["subsets"][subset] = dict(status="descriptive_two_site_only",
            native_ADE=float(pred.mean()), native_FDE=float(np.mean([p["fde"] for p in parts])),
            normalized_ADE=float(np.mean([p["normalized_ade"] for p in parts])),
            baseline_ADE=float(base.mean()), improvement_percent=None if change is None else -change,
            degradation_percent=change,
            per_site_degradation_percent=[ratio_change(float(p), float(b)) for p,b in zip(pred,base)],
            mean_site_relative_improvement_percent=(float(np.mean(100*(1-pred/base)))
                                                  if (base > 0).all() else None),
            paired_two_site_bootstrap_interval=(np.quantile(gain, [.025,.975]).tolist()
                                              if np.isfinite(gain).all() else None),
            interval_is_population_certificate=False)
    output["full_population_gain_interval"] = [float(np.mean([output["sites"][s][k] for s in sites]))
                                                for k in ("gain_lower", "gain_upper")]
    return output


def analyze():
    analysis = json.loads((PUBLIC/"analysis.json").read_text())
    manifest = json.loads((PUBLIC/"manifest.json").read_text())
    if analysis["manifest_sha256"] != sha256(PUBLIC/"manifest.json"):
        raise ValueError("Changed readout manifest")
    records = []
    for ref in analysis["records"]:
        path = (ROOT/ref["path"]).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != ref["sha256"]:
            raise ValueError("Changed or escaping completed readout receipt")
        row = json.loads(path.read_text())
        if (row["manifest_sha256"] != analysis["manifest_sha256"] or row["next_query"] != row["queries"]
                or row["recording"] != ref["recording"] or set(row["stats"]) != set(analysis["views"])):
            raise ValueError("Incomplete or wrong readout identity")
        records.append(row)
    if set(r["recording"] for r in records) != set(manifest["records"]) or len(records) != 27:
        raise ValueError("Incomplete registered population")
    first_view = analysis["views"][0]
    coverage = {k:sum(r["stats"][first_view][k] for r in records)
                for k in ("queries", "targets", "visible", "unknown_cv_context", "complete", "known_steps")}
    views, baseline_controls = {}, {}
    for view in analysis["views"]:
        grouped = {s:merge([r["stats"][view] for r in records if r["site"] == s])
                   for s in manifest["config"]["physical_sites"]}
        views[view] = dict(arms={arm:arm_metrics(grouped, arm) for arm in grouped[next(iter(grouped))]["arms"]},
            rejected_outputs=sum(s["rejected_outputs"] for s in grouped.values()),
            half_unmatched=sum(s["half_unmatched"] for s in grouped.values()),
            joint_different_queries=sum(s["half_joint_diff_queries"] for s in grouped.values()))
        for r in records:
            if any(r["stats"][view][k] != r["stats"][first_view][k] for k in coverage):
                raise ValueError("Models evaluated on different populations")
        if view == first_view:
            baseline_controls = {arm:arm_metrics(grouped, arm, bucket="baseline_controls")
                                 for arm in grouped[next(iter(grouped))]["baseline_controls"]}
    seed_groups = {}
    for family in ("transformer", "eqmotion"):
        for head in ("bounded_fraction", "matched_fraction_forest"):
            members = [views[f"{family}_seed{seed}_{head}"] for seed in (17,29,43)]
            seed_groups[family+"_"+head] = {}
            for arm in members[0]["arms"]:
                x = [m["arms"][arm] for m in members]
                all_gain = [v["subsets"]["all"]["improvement_percent"] for v in x]
                easy = [z for v in x for z in v["subsets"]["easy"].get("per_site_degradation_percent", [])]
                seed_groups[family+"_"+head][arm] = dict(
                    fixed_seed_improvements_percent=all_gain,
                    mean=float(np.mean(all_gain)), sample_std=float(np.std(all_gain, ddof=1)),
                    worst_site_seed_easy_degradation_percent=(max(easy)
                        if len(easy) == 6 and all(z is not None for z in easy) else None),
                    seed_variance_does_not_increase_independent_sites=True)
    return dict(result_source="fresh_run_reduction_of_hash_verified_full_readout", coverage=coverage,
        analysis_sha256=sha256(PUBLIC/"analysis.json"), manifest_sha256=analysis["manifest_sha256"],
        physical_sites=2, recordings=27, bootstrap_resamples=3000,
        independent_units_not_window_count=True, bootstrap_scope="descriptive_two_site_not_risk_certificate",
        views=views, fixed_three_seed_groups=seed_groups, baseline_controls=baseline_controls, selected_model=None,
        risk_certificate=False, confirmation=False, deployment=False)


def number(value):
    return "not estimable" if value is None else f"{value:.4f}"


def markdown(result):
    c = result["coverage"]
    lines = ["# DUT Frozen-Chain Readout", "", "## Material Passport", "",
        "Fresh frozen-model inference on the full registered DUT population; existing source-only",
        "weights and source audits were hash-verified. No new fitting, threshold selection,",
        "independent confirmation, safety certificate or deployment. DroneCrowd stays closed.", "",
        f"27 recordings / 2 physical sites; {c['queries']:,} query frames, {c['targets']:,} past-eligible target windows,",
        f"{c['complete']:,} complete future paths; {c['targets']-c['complete']:,} incomplete/unknown paths retained at inference.",
        f"{c['visible']:,} visible-agent instances and {c['unknown_cv_context']:,} unknown-CV context instances.",
        "Counts are unique query/agent instances before replication across12 model views, not IID samples.", "",
        "Eight observed / twelve predicted native annotation frames, stride1. Source models used",
        "SDD stride12: these are not matched seconds. Coordinates remain dataset-local unverified.", "",
        "This newly registered descriptive readout uses complete-path ADE; the prior SDD",
        "manuscript also reports available-point ADE. Do not pool them or compare their headline",
        "percentages as an improvement. Here improvement is the ratio of equal-site mean errors;",
        "mean site-relative gain is separately exported, not silently substituted.", "",
        "## Every Fixed View", "",
        "Native complete-path ADE, site-equal aggregation. Improvement is relative to causal CV.",
        "Easy degradation is the maximum of the two site-specific relative ADE changes.",
        "Intervals resample only two sites: descriptive, coarse, and not population-risk evidence.", "",
        "| View | Arm | ADE | Improvement % | Easy worst-site degradation % | Hard improvement % | Two-site interval % |",
        "|---|---|---:|---:|---:|---:|---|" ]
    for view, detail in result["views"].items():
        for name, a in detail["arms"].items():
            all_, easy, hard = [a["subsets"][s] for s in ("all", "easy", "hard")]
            easy_values = easy.get("per_site_degradation_percent", [])
            worst = max(easy_values) if easy_values and all(x is not None for x in easy_values) else None
            interval = all_.get("paired_two_site_bootstrap_interval")
            lines.append(f"| {view} | {name} | {number(all_.get('native_ADE'))} | {number(all_.get('improvement_percent'))} | {number(worst)} | {number(hard.get('improvement_percent'))} | {interval} |")
    lines += ["", "## Fixed Baseline Controls", "", "No strongest-DUT winner is selected for later testing.", "",
              "| Baseline | Native ADE | Native FDE | Improvement over CV % |", "|---|---:|---:|---:|"]
    for name, item in result["baseline_controls"].items():
        a = item["subsets"]["all"]
        lines.append(f"| {name} | {number(a.get('native_ADE'))} | {number(a.get('native_FDE'))} | {number(a.get('improvement_percent'))} |")
    lines += ["", "## Interpretation Boundary", "",
        "Per-site/subset metrics, full-population missing-label gain bounds, rejected outputs and",
        "unmatched solver counts are in metrics.json. Complete-case scores do not identify the",
        "unknown absolute error of missing future trajectories. Report their support and bounds",
        "alongside the main table. Do not use a zero-switch empirical arm as a learned safety proof.", "",
        "No best seed/head/policy is selected here. Two sites cannot establish a tight2% population",
        "risk guarantee. The source-excluded fitted chain is verified, but universal historical",
        "exposure and cross-dataset exchangeability are not certified. Offline annotation prefixes",
        "are not sensor-time observations. Tail/physical-validity and final independent confirmation",
        "remain separate evidence requirements. Stage5C and SMC remain off.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--verify", action="store_true"); args=p.parse_args()
    result = analyze()
    text = markdown(result)
    if args.verify:
        if json.loads((PUBLIC/"metrics.json").read_text()) != result or (PUBLIC/"results.md").read_text() != text:
            raise ValueError("Aggregate replay changed")
        print("Aggregate reduction reproduced exactly")
    else:
        immutable_json(PUBLIC/"metrics.json", result)
        path = PUBLIC/"results.md"
        if path.exists() and path.read_text() != text:
            raise ValueError("Existing report differs")
        path.write_text(text)
        print(json.dumps(result["coverage"]))


if __name__ == "__main__":
    main()
