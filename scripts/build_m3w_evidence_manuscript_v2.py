"""Export a development evidence revision; never load forecasts or target arrays."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_m3w_evidence_manuscript import (
    REPORTS, SOURCES as OLD_SOURCES, SITES, check_metric, csv_text, table_row,
)

OUTPUT = REPORTS / "evidence_manuscript_v2"
SOURCES = {**OLD_SOURCES,
    "eqmotion_nested_v1": "8367e7bd5628e01fa04d1c5451ec2c8816ddbee60dac07b1cd36a16b0eb05f77",
    "adaptive_region_cost_v1": "d1c06439e16a12fa339a72158820a8df9be4a1dfdb06da3a9d2523f61aca620b",
    "cross_objective_review_v1": "1883a8501170b81b42b32dee1b4820a0396bb7e80f1f84d680e46fad864ef15a",
    "log_cost_v1": "e344fa115e4ea0a5e73b7a367e691a2a6f2a7527df5ca50e5438f9d78860cf46",
    "prefix_cost_v1": "732d6726d011a0f46101a0d8424a9b10baefbfa3c6c77f790a078cf263f561c4",
    "temporal_intervention_v1": "d89afb790c9846ed6519b970a89e5f4b5eb8fc5325b8d082b90fccdde0e3d4b8",
    "forest_cost_v1": "7e03c0e49beea6ad526d10e1b542c0967b6287435f3c9c04d05d59437e32bb90",
    "risk_ranking_v1": "54575f34fc6c0d0ea79a78b89ff98658ccb66433f7b7f05c6f12ac9e3605fcfc",
    "fraction_square_v1": "fb56409613d2aba1950bb6d7c8f2db64a5bb4430181b4c691e0693bee45dc5f4",
    "protected_joint_support_v1": "d4165112eab6f47bfc77fdabed6b2054051ccfa01026eaaf6ee844c32b7c92b8",
}
CODE = {
    "scripts/run_m3w_cost_head_transfer.py": "fa209495b19d6df85d4922bdf9b7df3b9cb62f4ebb602cb7a55972216c49eedc",
    "scripts/run_m3w_eqmotion_cost_refit.py": "12d874bf59c033d93b45f92585e20d4e40b4cedeceb085e3eb468752a93664c5",
    "scripts/run_m3w_tempered_cost.py": "dfeeb2c161ca4f8ff54e73a14b34e9834a54effe1b7bf644831e373309d13a0b",
    "src/world_model/m3w_temporal_intervention.py": "bb5953065fcfd7a7bb78e2e7cf1b3a2db6a1757fb522d7e7d4575c5e43663c5b",
}
MANIFEST = "data/stage_cvpr2027_experiments/eqmotion_nested_v1/cost_views.json"
MANIFEST_SHA = "4d5318b334810db6ccc0921a9bf1fee6e0b745af7712bb5d489b3dbe03451d2b"
ARMS = (
    ("square_strict", "Square neural, strict"),
    ("log_strict", "Log neural, strict"),
    ("square_ratio", "Square neural, risk rank"),
    ("log_ratio", "Log neural, risk rank"),
    ("forest_ratio", "Forest, risk rank"),
    ("square_gain", "Square neural, gain rank"),
    ("log_gain", "Log neural, gain rank"),
    ("forest_gain", "Forest, gain rank"),
)
CONTRASTS = (
    ("Region minus intermediate (strict)", "conditional_cost_v1", "/contrasts/strict_stop/tempered"),
    ("Region minus intermediate (matched count)", "conditional_cost_v1", "/contrasts/matched_count/tempered"),
    ("Adaptive minus frozen region", "adaptive_region_cost_v1", "/contrasts/strict_stop/frozen_region"),
    ("Review minus matched nomination", "cross_objective_review_v1", "/contrasts/matched_nomination"),
    ("Log minus region", "log_cost_v1", "/contrasts/strict_stop/frozen_region"),
    ("Prefix guard minus terminal control", "prefix_cost_v1", "/contrasts/control_terminal"),
    ("Prefix guard minus matched terminal", "prefix_cost_v1", "/contrasts/control_matched"),
    ("Ramp minus uniform action", "temporal_intervention_v1", "/contrasts/uniform_strict"),
    ("Forest minus log neural (original primary)", "forest_cost_v1", "/contrasts/ramp/neural"),
    ("Forest minus log neural (same-count risk)", "risk_ranking_v1", "/contrasts/forest_strict_minus_neural_ratio"),
    ("Square minus log neural (strict primary)", "fraction_square_v1", "/contrasts/square_strict_minus_log_strict"),
    ("Square minus log neural (same-count risk)", "fraction_square_v1", "/contrasts/square_ratio_minus_log_ratio"),
    ("Square neural minus forest (same-count risk)", "fraction_square_v1", "/contrasts/square_ratio_minus_forest_ratio"),
    ("Earlier joint minus unary geometry", "native_joint_controls_v1", "/contrasts/half_joint_minus_half_unary"),
)


def pinned_json(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Changed evidence source: {path}")
    return json.loads(raw)


def load_sources(root):
    docs = {name: pinned_json(root / REPORTS / name / "analysis.json", sha)
            for name, sha in SOURCES.items()}
    for name, doc in docs.items():
        if doc["independent_confirmation"]:
            raise ValueError(f"Cannot relabel development evidence: {name}")
    for path, sha in CODE.items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Changed predictor-producing code: {path}")
    latest = docs["fraction_square_v1"]["identity"]["source_bindings"]
    if latest[MANIFEST] != MANIFEST_SHA:
        raise ValueError("Wrong nested EqMotion producers")
    for path, sha in CODE.items():
        if latest[path] != sha:
            raise ValueError("Latest experiment is not bound to this predictor chain")
    if docs["eqmotion_nested_v1"]["views_manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("Nested manifest/report mismatch")
    return docs


def pointer(document, path):
    for key in path.strip("/").split("/"):
        document = document[key]
    return document


def verify_local_lineage(root, docs):
    """Optional private metadata check, not checkpoint or numerical replay."""
    manifest = pinned_json(root / MANIFEST, MANIFEST_SHA)
    archives = {r["view"]: r for r in docs["native_eqmotion_v1"]["archives"]}
    seen, groups = set(), 0
    for view in manifest["views"]:
        site, seed = view["outer_site"], view["seed"]
        key = f"{site}_seed{seed}"
        if key in seen:
            raise ValueError("Duplicated outer view")
        seen.add(key)
        outer = view["outer_producer"]
        if outer["prediction"] != archives[key]:
            raise ValueError("Outer prediction is not the registered EqMotion forecast")
        producers = [(outer["producer"], {site})]
        if {g["inner_site"] for g in view["groups"]} != set(SITES) - {site}:
            raise ValueError("Incomplete inner-site coverage")
        for group in view["groups"]:
            producers.append((group["producer"], {site, group["inner_site"]}))
            groups += 1
        for producer, excluded in producers:
            if (producer["family"] != "eqmotion_fixed_head"
                    or set(producer["excluded_sites"]) != excluded
                    or set(producer["training_sites"]) != set(SITES) - excluded
                    or set(producer["preprocessing_fit_sites"]) != set(SITES) - excluded
                    or producer["seed"] != seed
                    or producer["checkpoint_selection_sites"]
                    or producer["calibration_sites"]):
                raise ValueError("Predictor-family or producer-exclusion violation")
    if seen != {f"{s}_seed{k}" for s in SITES for k in (17, 29, 43)}:
        raise ValueError("Incomplete outer-view coverage")
    return dict(result_source="fresh_metadata_check_of_cached_verified_producer_manifest",
                manifest_sha256=MANIFEST_SHA, outer_views=len(seen), inner_groups=groups,
                family="eqmotion_fixed_head", no_prediction_or_target_arrays_read=True,
                independent_confirmation=False, all_checks_passed=True)


def build(docs):
    rows, site_seed, site_metrics = [], [], []
    source = "fraction_square_v1"
    for arm, label in ARMS:
        path = f"/summaries/{arm}"
        summary = pointer(docs[source], path)
        row = table_row(label, "eqmotion_fixed_head_ramp_action", summary, source, path)
        row["arm"] = arm
        worst = []
        for seed, value in summary["seeds"].items():
            check_metric(value["subsets"]["positive_easy"])
            for site in SITES:
                easy = -value["subsets"]["positive_easy"]["by_scene"][site]["gain_percent"]
                worst.append(easy)
                site_seed.append(dict(arm=arm, seed=int(seed), scene=site,
                    easy_degradation_pct=easy, ADE_gain_pct=value["ADE"]["by_scene"][site]["gain_percent"],
                    source=source, json_pointer=f"{path}/seeds/{seed}"))
        if set(summary["seeds"]) != {"17", "29", "43"}:
            raise ValueError("Changed seed population")
        row["worst_site_seed_easy_degradation_pct"] = max(worst)
        row["observed_easy_ceiling_pass"] = max(worst) <= 2
        for field in ("selected", "selected_unknown", "selected_incomplete", "zero_CV_harmed"):
            row[field] = sum(v[field] for v in summary["seeds"].values())
        rows.append(row)
        for site in SITES:
            ade, fde = summary["ADE"]["by_scene"][site], summary["FDE"]["by_scene"][site]
            site_metrics.append(dict(arm=arm, scene=site, ADE=ade["model_error"],
                CV_ADE=ade["reference_error"], FDE=fde["model_error"], CV_FDE=fde["reference_error"],
                ADE_p95=ade["model_p95"], ADE_p99=ade["model_p99"],
                CV_ADE_p95=ade["reference_p95"],
                ADE_rows=ade["rows"], FDE_rows=fde["rows"], coordinate_unit="annotation_pixel",
                source=source, json_pointer=path))
    contrasts = []
    for label, name, path in CONTRASTS:
        c = pointer(docs[name], path)
        values = c["scene_differences_pp"]
        if len(values) != 4 or not math.isclose(sum(values) / 4, c["mean_gain_difference_pp"], abs_tol=1e-9):
            raise ValueError("Contrast is not an equal-physical-site difference")
        if c["resamples"] != 3000 or c["unit"] != "physical_scene":
            raise ValueError("Changed uncertainty unit")
        contrasts.append(dict(label=label, difference_pp=c["mean_gain_difference_pp"],
            CI_low_pp=c["ci95_pp"][0], CI_high_pp=c["ci95_pp"][1],
            source=name, json_pointer=path))
    joint = docs["protected_joint_support_v1"]
    if any(joint[k] for k in ("outcomes_evaluated", "future_target_arrays_loaded", "future_mask_arrays_loaded")):
        raise ValueError("Joint-support diagnostic changed scope")
    support = []
    for pool, values in joint["summaries"].items():
        a = values["all"]
        support.append(dict(pool=pool, **{k: a[k] for k in (
            "queries", "nonadditive_opportunities", "unique_recording_frame_opportunities",
            "combinations_enumerated", "feasible_combinations", "enumeration_blockers",
            "product_range_below_unary_gap", "changed_identity_queries")},
            predictive_lift_status="not_run", source="protected_joint_support_v1",
            json_pointer=f"/summaries/{pool}/all"))
    return dict(result_source="fresh_export_of_cached_verified_aggregate_evidence",
        source_hashes=SOURCES, predictor_code_hashes=CODE, nested_manifest_sha256=MANIFEST_SHA,
        rows=rows, site_metrics=site_metrics, site_seed_metrics=site_seed,
        contrasts=contrasts, joint_support=support,
        original_forest_primary_pass=docs["forest_cost_v1"]["primary_joint_empirical_pass"],
        square_loss_primary_pass=docs[source]["primary_joint_empirical_pass"],
        scope="8_observed_12_predicted_stride12_four_design_exposed_SDD_sites",
        uncertainty="reused_3000_physical_site_bootstrap_not_new_confirmation_or_adaptive_adjustment",
        new_training=False, new_target_readout=False, new_bootstrap=False,
        independent_confirmation=False, risk_calibrated=False, deployment=False,
        stage5c_executed=False, smc_enabled=False, submission_ready=False)


def table_text(e):
    lines = ["# Current Controlled Results", "",
        "Rebuilt from pinned public aggregates. All policies below use EqMotion ramp candidates.",
        "Four explored SDD sites; three seeds; 8 observed / 12 predicted annotation steps.",
        "Negative easy degradation means improvement. Worst means maximum over all 12 site/seed views.", "",
        "| Policy | ADE gain % | FDE gain % | Hard gain % | Worst easy degradation % | Switches |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in e["rows"]:
        lines.append(f"| {r['model']} | {r['ADE_gain_pct']:.3f} | {r['FDE_gain_pct']:.3f} | {r['hard_gain_pct']:.3f} | {r['worst_site_seed_easy_degradation_pct']:.3f} | {r['selected']:,} |")
    lines += ["", "Counts include repeated query/seed instances, not independent people/events.",
        "Same counts do not imply the same displacement mass or realized harm budget.",
        "Both original primary superiority gates fail; no secondary winner replaces them.", "",
        "## Paired Development Contrasts", "",
        "| Preserved comparison | ADE difference (pp) | 95% site-bootstrap CI (pp) |",
        "|---|---:|---:|"]
    for r in e["contrasts"]:
        lines.append(f"| {r['label']} | {r['difference_pp']:+.5f} | [{r['CI_low_pp']:+.5f}, {r['CI_high_pp']:+.5f}] |")
    lines += ["", "The last contrast concerns earlier full-forecast joint controls, not the new ramp support audit.",
        "Bootstrap intervals condition on four development-exposed sites; they are not independent confirmation.", "",
        "## Joint Support, Not Forecasting Accuracy", "",
        "| Pool | Queries including seeds | Nonadditive opportunities | Unique opportunity frames | Changed queries including seeds |",
        "|---|---:|---:|---:|---:|"]
    for r in e["joint_support"]:
        lines.append(f"| {r['pool']} | {r['queries']:,} | {r['nonadditive_opportunities']} | {r['unique_recording_frame_opportunities']} | {r['changed_identity_queries']} |")
    lines += ["", "No future-target readout or predictive-lift estimate was made in this support audit.", ""]
    return "\n".join(lines)


def figures(e):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "m3w_evidence_v2"})
    outputs = {}
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.1), layout="constrained")
    for i, r in enumerate(e["rows"]):
        y = 7-i
        color = "#167868" if r["observed_easy_ceiling_pass"] else "#b74750"
        ax[0].errorbar(r["ADE_gain_pct"], y, fmt="o", color=color, capsize=3,
            xerr=[[r["ADE_gain_pct"]-r["ADE_CI_low"]], [r["ADE_CI_high"]-r["ADE_gain_pct"]]])
        ax[1].scatter(r["worst_site_seed_easy_degradation_pct"], y, color=color, s=40)
    ax[0].set_yticks(range(8), [r["model"] for r in e["rows"]][::-1])
    ax[1].set_yticks(range(8), [""]*8)
    ax[0].set_xlabel("ADE improvement over CV (%)")
    ax[1].set_xlabel("Worst site/seed easy degradation (%)")
    ax[0].set_title("A. Mean utility with site-bootstrap intervals", fontsize=11)
    ax[1].set_title("B. Observed easy-case protection", fontsize=11)
    ax[1].axvline(2, color="#b74750", ls="--", label="2% observed ceiling")
    ax[1].legend(fontsize=9, loc="lower left")
    for a in ax:
        a.axvline(0, color="#777777", lw=.7)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="x", alpha=.15)
    fig.suptitle("EqMotion ramp candidates: four explored SDD sites, three seeds", fontsize=12)
    outputs["risk_tradeoff"] = fig
    fig, ax = plt.subplots(figsize=(12.8, 5.2), layout="constrained")
    selected = [e["contrasts"][i] for i in (8, 9, 10, 11, 12)]
    for i, r in enumerate(selected):
        ax.errorbar(r["difference_pp"], len(selected)-i-1, fmt="o", capsize=4, color="#355d9a",
            xerr=[[r["difference_pp"]-r["CI_low_pp"]], [r["CI_high_pp"]-r["difference_pp"]]])
    ax.set_yticks(range(len(selected)), [r["label"] for r in selected][::-1])
    ax.axvline(0, color="#b74750", ls="--")
    ax.set_xlabel("Paired ADE improvement difference (percentage points)")
    ax.set_title("Primary failures retained alongside same-count diagnostics", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    outputs["paired_contrasts"] = fig
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--verify-local-lineage", action="store_true")
    p.add_argument("--preview-dir", type=Path)
    args = p.parse_args()
    docs = load_sources(ROOT)
    e = build(docs)
    files = {"evidence.json": json.dumps(e, indent=2, allow_nan=False)+"\n",
             "main_table.csv": csv_text(e["rows"]), "site_metrics.csv": csv_text(e["site_metrics"]),
             "site_seed_metrics.csv": csv_text(e["site_seed_metrics"]),
             "contrasts.csv": csv_text(e["contrasts"]), "joint_support.csv": csv_text(e["joint_support"]),
             "tables.md": table_text(e)}
    out = ROOT / OUTPUT
    if args.verify_local_lineage:
        receipt = verify_local_lineage(ROOT, docs)
        files["local_lineage_verification.json"] = json.dumps(receipt, indent=2)+"\n"
    figs = figures(e)
    import matplotlib.pyplot as plt
    for name, fig in figs.items():
        buffer = io.StringIO()
        fig.savefig(buffer, format="svg", metadata={"Date": None})
        files[name+".svg"] = "\n".join(s.rstrip() for s in buffer.getvalue().splitlines())+"\n"
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.preview_dir / (name+".png"), dpi=140)
        plt.close(fig)
    if args.verify:
        for name, expected in files.items():
            if (out / name).read_text() != expected:
                raise ValueError(f"Changed manuscript export: {name}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (out / name).write_text(text)
    print(json.dumps(dict(verified=args.verify, sources=len(docs), outputs=len(files),
        local_lineage_checked=args.verify_local_lineage, new_training=False,
        new_target_readout=False, independent_confirmation=False, submission_ready=False)))


if __name__ == "__main__":
    main()
