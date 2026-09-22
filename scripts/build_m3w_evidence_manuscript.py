"""Rebuild paper tables from pinned, already-read development aggregates only."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = Path("outputs/publication_readiness_2026_09")
OUTPUT = REPORTS / "evidence_manuscript_v1"
SOURCES = {
    "native_forecast_v1": "e685f0afc94149a5bc2fe231cabacea6ca108a0e52843546a1519dbc6a4198b3",
    "native_eqmotion_v1": "4e5c3a274fa73b3e63d704732d4e474e7b738c859b5bf198a3f510fb78db5d21",
    "native_joint_controls_v1": "a3d2087c0415d13baceb9a6ae5eea87df96619277847cfe2ca86b4560d1d6113",
    "cost_budget_matched_v1": "956d3fb8b3c346b59d33f159fdb5d78d17ab6dd9945fe5c7fcf2a83dffabc057",
    "conditional_cost_v1": "52bef548c000a24c1c99b0e33081371bd5caf78be42ea60525383619e285788e",
    "calibration_support_v1": "7905e86a27e61d42b9b024b1868c97b4569dcbe452c6c9164951daeff7a20fd0",
}
SITES = ["coupa", "deathCircle", "gates", "hyang"]


def load_sources(root):
    documents = {}
    for name, expected in SOURCES.items():
        raw = (root / REPORTS / name / "analysis.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Changed paper source: {name}; review, do not silently refresh")
        documents[name] = json.loads(raw)
        if documents[name]["independent_confirmation"]:
            raise ValueError("This export is development-only")
    return documents


def check_metric(metric):
    """Recalculate the reported equal-site mean; never weight by window count."""
    if metric["coordinate_unit"] != "annotation_pixel":
        raise ValueError("Incompatible coordinate unit")
    if metric["expected_scenes"] != SITES or set(metric["by_scene"]) != set(SITES):
        raise ValueError("Changed scene population")
    gains = []
    for row in metric["by_scene"].values():
        ref = row["reference_error"]
        if ref <= 0:
            raise ValueError("Undefined relative error; do not add an epsilon")
        gain = 100 * (ref - row["model_error"]) / ref
        if not math.isclose(gain, row["gain_percent"], abs_tol=1e-9):
            raise ValueError("Inconsistent scene arithmetic")
        gains.append(gain)
    result = sum(gains) / len(gains)
    if not math.isclose(result, metric["equal_scene_gain_percent"], abs_tol=1e-9):
        raise ValueError("Inconsistent equal-site arithmetic")
    return result


def table_row(label, predictor, summary, source, pointer):
    ade, fde = summary["ADE"], summary["FDE"]
    easy = summary["subsets"]["positive_easy"]
    hard = summary["subsets"]["hard"]
    return dict(model=label, predictor=predictor, ADE_gain_pct=check_metric(ade),
                ADE_CI_low=ade["scene_bootstrap_ci95"][0],
                ADE_CI_high=ade["scene_bootstrap_ci95"][1],
                FDE_gain_pct=check_metric(fde), hard_gain_pct=check_metric(hard),
                easy_degradation_pct=-check_metric(easy),
                worst_site_easy_degradation_pct=max(-x["gain_percent"] for x in easy["by_scene"].values()),
                source=source, json_pointer=pointer)


def build(documents):
    eq = documents["native_eqmotion_v1"]
    matched = documents["cost_budget_matched_v1"]
    region = documents["conditional_cost_v1"]
    rows = []
    for arm, label in (("constant_velocity", "Causal CV"),
                       ("transformer", "Transformer, uncontrolled"),
                       ("eqmotion", "EqMotion K=1, uncontrolled")):
        summary = eq["summaries"][arm]
        ptr = f"/summaries/{arm}"
        if arm != "constant_velocity":
            summary = summary["mean_seed"]
            ptr += "/mean_seed"
        rows.append(table_row(label, arm, summary, "native_eqmotion_v1", ptr))
    for arm, label in (("native", "Native-cost strict"),
                       ("fraction", "Fraction-cost strict"),
                       ("tempered", "Intermediate-cost strict")):
        rows.append(table_row(label, "eqmotion", matched["summaries"][arm]["strict_stop"],
                              "cost_budget_matched_v1", f"/summaries/{arm}/strict_stop"))
    for arm, label in (("strict_stop", "Region-weighted strict"),
                       ("net_stop", "Region-weighted net-gain")):
        rows.append(table_row(label, "eqmotion", region["summaries"][arm],
                              "conditional_cost_v1", f"/summaries/{arm}"))
    strict = region["summaries"]["strict_stop"]
    site_metrics = []
    for row in rows:
        summary = documents[row["source"]]
        for key in row["json_pointer"].strip("/").split("/"):
            summary = summary[key]
        for site in SITES:
            ade, fde = summary["ADE"]["by_scene"][site], summary["FDE"]["by_scene"][site]
            site_metrics.append(dict(model=row["model"], scene=site,
                ADE=ade["model_error"], CV_ADE=ade["reference_error"],
                FDE=fde["model_error"], CV_FDE=fde["reference_error"],
                ADE_p95=ade["model_p95"], ADE_p99=ade["model_p99"],
                CV_ADE_p95=ade["reference_p95"], ADE_rows=ade["rows"], FDE_rows=fde["rows"],
                coordinate_unit="annotation_pixel", source=row["source"], json_pointer=row["json_pointer"]))
    slices = []
    for seed, summary in strict["seeds"].items():
        check_metric(summary["subsets"]["positive_easy"])
        for site in SITES:
            slices.append(dict(seed=int(seed), scene=site,
                               easy_degradation_pct=-summary["subsets"]["positive_easy"]["by_scene"][site]["gain_percent"]))
    contrast_sources = [
        ("Region weighting minus intermediate; strict rule", "conditional_cost_v1", "/contrasts/strict_stop/tempered"),
        ("Region weighting minus intermediate; matched count", "conditional_cost_v1", "/contrasts/matched_count/tempered"),
        ("Intermediate minus native; strict rule", "cost_budget_matched_v1", "/contrasts/strict_stop/tempered_minus_native"),
        ("Intermediate minus fraction; strict rule", "cost_budget_matched_v1", "/contrasts/strict_stop/tempered_minus_fraction"),
        ("Joint minus unary geometry; matched half-count", "native_joint_controls_v1", "/contrasts/half_joint_minus_half_unary"),
    ]
    contrasts = []
    for label, source, pointer in contrast_sources:
        value = documents[source]
        for key in pointer.strip("/").split("/"):
            value = value[key]
        contrasts.append(dict(label=label, source=source, json_pointer=pointer, **value))
    quality = region["conditional_quality"]
    underprediction = []
    for population in ("fitting", "held_source"):
        for group in ("all", "old_region", "new_region"):
            subset = [r for r in quality if r["population"] == population and r["group"] == group]
            underprediction.append(dict(population=population, region=group, views=len(subset),
                harm_underpredicted=sum(r["costs"]["new_predicted_harm"] < r["costs"]["realized_harm"] for r in subset)))
    return dict(result_source="fresh_paper_export_of_cached_verified_aggregate_reports",
                source_hashes=SOURCES, input_scope="six_public_aggregate_reports_only_no_new_label_readout",
                uncertainty="cached_3000_physical_site_bootstrap_four_design_exposed_sites_not_confirmation",
                rows=rows, site_metrics=site_metrics, scene_seed_easy=slices, paired_contrasts=contrasts,
                native_vs_old_loss_relative_gain_percent=check_metric(documents["native_forecast_v1"]["native_vs_matched_old_loss"]),
                native_vs_old_loss_CI=documents["native_forecast_v1"]["native_vs_matched_old_loss"]["scene_bootstrap_ci95"],
                conditional_harm=underprediction, primary_gates=region["primary_gates"],
                selected_instances=sum(v["selected"] for v in strict["seeds"].values()),
                selected_unknown_ADE=sum(v["selected_unknown"] for v in strict["seeds"].values()),
                selected_incomplete=sum(v["selected_incomplete"] for v in strict["seeds"].values()),
                new_training=False, new_bootstrap=False, new_policy_selection=False,
                independent_confirmation=False, risk_calibration=False, deployment=False,
                stage5c_executed=False, smc_enabled=False)


def csv_text(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def tables(evidence):
    lines = ["# Reconstructed Development Tables", "",
             "Cached, SHA256-verified aggregate reports; no new training or bootstrap.",
             "Four design-exposed sites; intervals are conditional, not confirmation.", "",
             "| Model / policy | ADE gain % [95% site CI] | FDE gain % | Hard gain % | Easy degradation % | Worst-site easy degradation % |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in evidence["rows"]:
        lines.append(f"| {row['model']} | {row['ADE_gain_pct']:.3f} [{row['ADE_CI_low']:.3f}, {row['ADE_CI_high']:.3f}] | {row['FDE_gain_pct']:.3f} | {row['hard_gain_pct']:.3f} | {row['easy_degradation_pct']:.3f} | {row['worst_site_easy_degradation_pct']:.3f} |")
    lines += ["", "Positive easy degradation is harm. Negative is improvement.",
              "These are two predictor families; comparisons across families are not isolated loss ablations.", "",
              "| Fixed paired contrast | Difference (pp) | Conditional 95% CI (pp) |", "|---|---:|---:|"]
    for row in evidence["paired_contrasts"]:
        low, high = row["ci95_pp"]
        lines.append(f"| {row['label']} | {row['mean_gain_difference_pp']:.5f} | [{low:.5f}, {high:.5f}] |")
    lines += ["", "Matched counts are diagnostic; equal realized risk is not established.", "",
              "| Scene | Seed 17 easy degradation % | Seed 29 | Seed 43 |", "|---|---:|---:|---:|"]
    for site in SITES:
        values = [r["easy_degradation_pct"] for r in sorted(evidence["scene_seed_easy"], key=lambda r:r["seed"]) if r["scene"] == site]
        lines.append("| " + site + " | " + " | ".join(f"{v:.5f}" for v in values) + " |")
    lines += ["", "The unchanged protection criterion applies to every scene/seed, not this table's mean.",
              "The latest combined gate fails. No new model is deployed.", ""]
    return "\n".join(lines)


def plot(evidence, path, preview=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "m3w_evidence_v1"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.35, 1]}, layout="constrained")
    chosen = evidence["rows"][1:7]
    for i, r in enumerate(chosen):
        y = len(chosen)-i-1
        axes[0].errorbar(r["ADE_gain_pct"], y,
            xerr=[[r["ADE_gain_pct"]-r["ADE_CI_low"]], [r["ADE_CI_high"]-r["ADE_gain_pct"]]],
            fmt="o", capsize=4, color="#17776e" if i < 2 else "#3d5990")
    axes[0].set_yticks(range(len(chosen)), [r["model"] for r in chosen][::-1])
    axes[0].axvline(0, color="#999999", lw=1)
    axes[0].set_xlabel("ADE improvement over CV (%)")
    axes[0].set_title("A. Average improvement is not protection", loc="left", fontsize=11)
    for j, seed in enumerate((17, 29, 43)):
        y = [next(r["easy_degradation_pct"] for r in evidence["scene_seed_easy"] if r["seed"] == seed and r["scene"] == s) for s in SITES]
        axes[1].scatter(np.arange(4)+(j-1)*.12, y, label=f"Seed {seed}", s=42, marker=("o", "s", "^")[j], color=("#17776e", "#3d5990", "#aa4c55")[j])
    axes[1].axhline(2, color="#aa4c55", linestyle="--", label="2% limit")
    axes[1].axhline(0, color="#999999", lw=.8)
    axes[1].set_xticks(range(4), SITES, rotation=20)
    axes[1].set_ylabel("Easy ADE degradation (%)\npositive = harm")
    axes[1].set_title("B. Latest strict policy, every seed retained", loc="left", fontsize=11)
    axes[1].legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x" if ax is axes[0] else "y", alpha=.15)
    fig.suptitle("SDD development only: 4 explored sites, 8 observed / 12 predicted annotation steps", fontsize=12)
    fig.savefig(path, metadata={"Date": None})
    if preview:
        fig.savefig(preview, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    evidence = build(load_sources(ROOT))
    files = {"evidence.json": json.dumps(evidence, indent=2, allow_nan=False)+"\n",
             "main_table.csv": csv_text(evidence["rows"]),
             "site_metrics.csv": csv_text(evidence["site_metrics"]), "tables.md": tables(evidence)}
    out = ROOT / OUTPUT
    if args.verify:
        for name, content in files.items():
            if (out / name).read_text() != content:
                raise ValueError(f"Paper export mismatch: {name}")
        print("Verified six pinned aggregates, arithmetic, source pointers and four paper exports; no new fit/readout")
    else:
        out.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (out / name).write_text(content)
        plot(evidence, out / "protection.svg", args.preview)
        print("Built development tables and figure; no independent confirmation or new training")


if __name__ == "__main__":
    main()
