"""Standalone aggregate reproduction; standard library only, no model fitting."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

SITES = ("coupa", "deathCircle", "gates", "hyang")
SEEDS = ("17", "29", "43")
ARMS = ("square_strict", "log_strict", "square_ratio", "log_ratio",
        "forest_ratio", "square_gain", "log_gain", "forest_gain")
FALSE_CLAIMS = ("new_training", "new_bootstrap", "independent_confirmation",
                "risk_calibrated", "deployment", "submission_ready",
                "stage5c_executed", "smc_enabled")
SCOPE = "8_observed_12_predicted_stride12_four_design_exposed_SDD_sites"


def close(left, right, label):
    if not (math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)):
        raise ValueError(f"Inconsistent {label}")


def metric(value):
    if value["coordinate_unit"] != "annotation_pixel":
        raise ValueError("Unverified coordinate unit")
    if tuple(value["expected_scenes"]) != SITES or set(value["by_scene"]) != set(SITES):
        raise ValueError("Changed scene population")
    gains = []
    for row in value["by_scene"].values():
        ref, model = row["reference_error"], row["model_error"]
        if ref <= 0 or model < 0 or row["rows"] <= 0:
            raise ValueError("Undefined relative error or empty scene")
        gain = 100 * (ref - model) / ref
        close(gain, row["gain_percent"], "scene gain")
        gains.append(gain)
    mean = sum(gains) / len(gains)
    close(mean, value["equal_scene_gain_percent"], "equal-scene mean")
    if value["bootstrap_resamples"] != 3000 or value["bootstrap_unit"] != "physical_scene":
        raise ValueError("Changed uncertainty unit")
    low, high = value["scene_bootstrap_ci95"]
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("Invalid archived interval")
    return mean


def reconstruct(evidence):
    """Recompute scalar reductions from scene aggregates, not raw predictions."""
    if evidence["schema_version"] != 1 or evidence["scope"] != SCOPE:
        raise ValueError("Changed evidence scope")
    if any(evidence[k] is not False for k in FALSE_CLAIMS):
        raise ValueError("Unsupported scientific completion claim")
    if tuple(p["arm"] for p in evidence["policies"]) != ARMS:
        raise ValueError("Changed policy population")
    rows, sites, seeds = [], [], []
    for policy in evidence["policies"]:
        arm, label, summary = policy["arm"], policy["label"], policy["summary"]
        if set(summary["seeds"]) != set(SEEDS):
            raise ValueError("Changed seed population")
        ade, fde = summary["ADE"], summary["FDE"]
        easy, hard = summary["subsets"]["positive_easy"], summary["subsets"]["hard"]
        row = dict(arm=arm, model=label, ADE_gain_pct=metric(ade),
                   ADE_CI_low=ade["scene_bootstrap_ci95"][0],
                   ADE_CI_high=ade["scene_bootstrap_ci95"][1],
                   FDE_gain_pct=metric(fde), hard_gain_pct=metric(hard),
                   easy_degradation_pct=-metric(easy))
        worst = []
        for seed in SEEDS:
            values = summary["seeds"][seed]
            metric(values["ADE"])
            metric(values["subsets"]["positive_easy"])
            for site in SITES:
                harm = -values["subsets"]["positive_easy"]["by_scene"][site]["gain_percent"]
                worst.append(harm)
                seeds.append(dict(arm=arm, seed=int(seed), scene=site,
                    easy_degradation_pct=harm,
                    ADE_gain_pct=values["ADE"]["by_scene"][site]["gain_percent"]))
        row["worst_site_seed_easy_degradation_pct"] = max(worst)
        row["observed_easy_ceiling_pass"] = max(worst) <= 2
        for key in ("selected", "selected_unknown", "selected_incomplete", "zero_CV_harmed"):
            values = [s[key] for s in summary["seeds"].values()]
            if any(type(v) is not int or v < 0 for v in values):
                raise ValueError("Invalid diagnostic count")
            row[key] = sum(values)
        rows.append(row)
        for site in SITES:
            a, f = ade["by_scene"][site], fde["by_scene"][site]
            sites.append(dict(arm=arm, scene=site, ADE=a["model_error"],
                CV_ADE=a["reference_error"], FDE=f["model_error"], CV_FDE=f["reference_error"],
                ADE_p95=a["model_p95"], ADE_p99=a["model_p99"], CV_ADE_p95=a["reference_p95"],
                ADE_rows=a["rows"], FDE_rows=f["rows"], coordinate_unit="annotation_pixel"))
    contrasts = []
    if len(evidence["contrasts"]) != 14:
        raise ValueError("Incomplete contrast population")
    for row in evidence["contrasts"]:
        values = row["scene_differences_pp"]
        if len(values) != len(SITES) or row["resamples"] != 3000 or row["unit"] != "physical_scene":
            raise ValueError("Changed paired uncertainty population")
        close(sum(values) / len(values), row["mean_gain_difference_pp"], "paired mean")
        low, high = row["ci95_pp"]
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError("Invalid paired interval")
        contrasts.append(dict(label=row["label"], difference_pp=row["mean_gain_difference_pp"],
                              CI_low_pp=low, CI_high_pp=high))
    # Keep registered primary failures separate from the positive diagnostic.
    for label in ("Forest minus log neural (original primary)",
                  "Square minus log neural (strict primary)"):
        c = next(r for r in contrasts if r["label"] == label)
        if c["CI_low_pp"] > 0 or c["CI_high_pp"] < 0:
            raise ValueError("Frozen primary evidence changed")
    support = evidence["joint_support"]
    if len(support) != 3 or any(r["predictive_lift_status"] != "not_run" for r in support):
        raise ValueError("Joint proxy audit is not predictive evidence")
    return dict(rows=rows, site_metrics=sites, site_seed_metrics=seeds,
        contrasts=contrasts, joint_support=support, scope=SCOPE,
        result_source="recomputed_reductions_of_archived_aggregate_evidence",
        uncertainty="archived_3000_site_bootstrap_not_new_or_independent_confirmation",
        **{k: False for k in FALSE_CLAIMS})


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def outputs(result):
    table = ["# Reconstructed Development Evidence", "",
        "Four explored SDD sites; 8 observed / 12 predicted annotation steps.",
        "Not independent confirmation, calibration, new training or deployment.", "",
        "| Policy | ADE gain % | FDE gain % | Hard gain % | Worst easy degradation % | Switches |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in result["rows"]:
        table.append(f"| {r['model']} | {r['ADE_gain_pct']:.3f} | {r['FDE_gain_pct']:.3f} | "
                     f"{r['hard_gain_pct']:.3f} | {r['worst_site_seed_easy_degradation_pct']:.3f} | {r['selected']:,} |")
    table += ["", "## Preserved Paired Contrasts", "",
        "| Comparison | ADE difference (pp) | Archived 95% site-bootstrap CI (pp) |",
        "|---|---:|---:|"]
    for r in result["contrasts"]:
        table.append(f"| {r['label']} | {r['difference_pp']:+.5f} | [{r['CI_low_pp']:+.5f}, {r['CI_high_pp']:+.5f}] |")
    table += ["", "Both original primary superiority comparisons fail. The positive same-count",
              "diagnostic does not replace them. Seeds/windows are not independent sites.",
              "Missing-label easy safety remains unknown; joint-support counts are not accuracy.", ""]
    return {"results.json": json.dumps(result, indent=2, allow_nan=False)+"\n",
        "main_table.csv": csv_text(result["rows"]), "site_metrics.csv": csv_text(result["site_metrics"]),
        "site_seed_metrics.csv": csv_text(result["site_seed_metrics"]),
        "contrasts.csv": csv_text(result["contrasts"]), "joint_support.csv": csv_text(result["joint_support"]),
        "tables.md": "\n".join(table)}


def package_inputs(root):
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Symlink in package")
    manifest = json.loads(manifest_path.read_text())
    if manifest["package_scope"] != "deidentified_aggregate_reproduction_draft":
        raise ValueError("Changed package scope")
    for name, info in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe package path")
        path = root / relative
        if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
            raise ValueError("Symlink in package")
        data = path.read_bytes()
        if len(data) != info["bytes"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
            raise ValueError(f"Changed package file: {name}")
    if "evidence.json" not in manifest["files"] or "reproduce.py" not in manifest["files"]:
        raise ValueError("Missing mandatory package files")
    return json.loads((root / "evidence.json").read_text()), manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reproduced"))
    parser.add_argument("--verify", action="store_true", help="Compare with bundled expected outputs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    evidence, manifest = package_inputs(root)
    generated = outputs(reconstruct(evidence))
    for name, value in generated.items():
        expected = manifest["expected_outputs"][name]
        if hashlib.sha256(value.encode()).hexdigest() != expected:
            raise ValueError(f"Reconstruction differs from frozen output: {name}")
        if args.verify and (root / "expected" / name).read_text() != value:
            raise ValueError(f"Changed expected content: {name}")
    target = args.output.resolve()
    if target == root or root in target.parents and target.parts[len(root.parts)] in ("expected", "figures"):
        raise ValueError("Do not overwrite package evidence")
    target.mkdir(parents=True, exist_ok=True)
    for name, value in generated.items():
        path = target / name
        if path.exists() or path.is_symlink():
            raise ValueError(f"Refusing overwrite: {path.name}")
    for name, value in generated.items():
        (target / name).write_text(value)
    print(json.dumps(dict(reconstructed_files=len(generated), verified=args.verify,
        policies=len(evidence["policies"]), physical_sites=4, seeds=3,
        result_source="aggregate_reproduction_only", new_training=False,
        independent_confirmation=False, submission_ready=False)))


if __name__ == "__main__":
    main()
