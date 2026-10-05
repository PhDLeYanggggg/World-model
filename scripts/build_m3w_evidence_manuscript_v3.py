"""Reconcile pinned public aggregates without reading rows or fitting a model."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = Path("outputs/publication_readiness_2026_09")
OUTPUT = BASE / "evidence_manuscript_v3"
SOURCES = {
    "sdd": ("evidence_manuscript_v2/evidence.json", "3938ca9e97ccf76bb207c625771e32db41b79439f586be2f57294decd9ff9e41"),
    "support": ("european_cost_support_diagnostic_v1/summary.json", "52174104b9a95ccedc510ec36b834e3080484320d52ab46c60c5dad38b7e8309"),
    "extension": ("european_leaf_quality_extension_v1/summary.json", "9aca9427df575a509910873c2f8aeecd4c505a7d87daa7bed51da82b7e44d2ce"),
    "temporal": ("european_temporal_target_audit_v1/summary.json", "4c33f59b0d0860671795defc2605e3be823536f2b3939709c35d7fb173d3817f"),
    "training_status": ("european_temporal_auxiliary_v1/run_status.json", "e46bcff055357cbb88c9d3ee4bd2a21a73bd5da9f4f4f97b9a5aada02b02dddb"),
    "readout_registration": ("european_temporal_auxiliary_v1/readout/registration.json", "a200b10430cd4be39606f025f01eeeeeda0899b992a69bd1f73be0d988cd0fde"),
    "training_protocol": ("european_temporal_auxiliary_v1/protocol.md", "aa2962f5058eb4270027dd4adbd2088cac6bae152604cbfb17b22825cd47ba7e"),
    "data_card": ("european_fixed_producer_roles_v1/data_and_asset_card.md", "f37f11c8df3d887e7cc0be2dcd78c6b1f45d1780ecf7779f8b9189d5bc8e4dce"),
}
CONTRASTS = (
    ("E1", "Cost minus original, TRAIN", "support", "/training_projected_error_change/global_weighted_MSE_change", "normalized signed-score MSE"),
    ("E2", "Cost minus original, validation", "support", "/cohorts/all/global_weighted_MSE_change", "normalized signed-score MSE"),
    ("E3", "Extension minus cost, validation", "extension", "/extended_minus_cost_signed_MSE", "normalized signed-score MSE"),
    ("E4", "Extension minus original, validation", "extension", "/extended_minus_original_signed_MSE", "normalized signed-score MSE"),
    ("E5", "Extension minus additive, validation", "extension", "/extended_minus_additive_signed_MSE", "normalized signed-score MSE"),
    ("E6", "Temporal minus row-mean leaf, validation", "temporal", "/contrasts/validation_temporal_minus_rowmean_leaf_signed_error", "normalized step-error MSE"),
    ("E7", "Temporal minus global temporal, validation", "temporal", "/contrasts/validation_temporal_minus_global_temporal_signed_error", "normalized step-error MSE"),
    ("E8", "Temporal minus row-mean leaf, complete labels", "temporal", "/contrasts/complete_validation_temporal_minus_rowmean_leaf_signed_error", "normalized step-error MSE"),
    ("E9", "Temporal minus row-mean leaf, original selected", "temporal", "/contrasts/selected_validation_temporal_minus_rowmean_leaf_signed_error", "normalized step-error MSE"),
    ("E10", "Extension minus cost, full utility", "extension", "/extended_minus_cost_full_utility_percent", "percent of full known reference cost"),
    ("E11", "Extension minus cost, matched utility", "extension", "/extended_minus_cost_matched_utility_percent", "percent of full known reference cost"),
)
SDD_CONTRASTS = (
    "Forest minus log neural (original primary)",
    "Forest minus log neural (same-count risk)",
    "Square minus log neural (strict primary)",
    "Square minus log neural (same-count risk)",
)


def load_sources(root=ROOT):
    docs = {}
    for key, (relative, digest) in SOURCES.items():
        raw = (root / BASE / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Changed source: {relative}")
        docs[key] = json.loads(raw) if relative.endswith(".json") else raw.decode()
    return docs


def pointer(doc, path):
    for key in path.strip("/").split("/"):
        doc = doc[key]
    return doc


def require(value, message):
    if not value:
        raise ValueError(message)


def build(docs):
    for name in ("sdd", "support", "extension", "temporal"):
        require(docs[name]["independent_confirmation"] is False, "Development role changed")
    for name in ("support", "extension", "temporal"):
        require(docs[name]["groups"] == 72 and docs[name]["localities"] == 12,
                "Changed European source population")
        require(docs[name]["deployment_changed"] is False, "Deployment claim changed")
    status, registration = docs["training_status"], docs["readout_registration"]
    require(status["real_input_check"]["optimizer_updates"] == 0
            and status["pilot_attempt"]["real_optimizer_updates"] == 0
            and status["pilot_attempt"]["new_real_checkpoints"] == 0,
            "Training snapshot changed; revise the manuscript")
    require(registration["risk_budget"] == .02 and not registration["threshold_search"]
            and not registration["checkpoint_selection"] and not registration["independent_roles_read"],
            "Registered scientific contract changed")
    heads = registration["expected_source_heads"]
    require(len(heads) == len({(h["group"], h["head_seed"]) for h in heads}) == 72,
            "Missing or duplicated source heads")
    require({h["head_seed"] for h in heads} == {17, 29, 43}, "Head seeds changed")
    require(len({h["source"] for h in heads}) == 12, "Locality coverage changed")
    rows = []
    for key, label, source, path, unit in CONTRASTS:
        item = pointer(docs[source], path)
        mean, ci = item["mean"], item["CI95"]
        require(len(ci) == 2 and all(math.isfinite(x) for x in [mean, *ci])
                and ci[0] <= ci[1], "Invalid interval")
        sites = item["localities"]
        if isinstance(sites, dict):
            require(math.isclose(sum(sites.values()) / len(sites), mean, abs_tol=1e-12),
                    "Incorrect locality average")
            require(item["bootstrap_draws"] == 3000, "Changed bootstrap size")
            sites = len(sites)
        require(sites == (11 if key == "E9" else 12), "Changed contrast support")
        rows.append(dict(id=key, contrast=label, estimate=mean, ci_low=ci[0], ci_high=ci[1],
                         localities=sites, unit=unit, source=source, json_pointer=path,
                         evidence_role="exposed_development", result_source="cached_verified"))
    values = {r["id"]: r["estimate"] for r in rows}
    require(math.isclose(values["E2"] + values["E3"], values["E4"], abs_tol=1e-12),
            "Cross-study MSE contrast arithmetic changed")
    pooled = docs["temporal"]["pooled_validation"]
    require(math.isclose(pooled["selected_step_harm"] - pooled["selected_cancellation"],
                         pooled["selected_harm"], abs_tol=1e-9), "Cancellation identity failed")
    policies = []
    for arm in ("original", "additive", "poisson", "cost", "extended"):
        item = docs["extension"][arm]
        require(0 <= item["complete_support"] <= item["defined_easy_risk"] <= 72,
                "Invalid risk support")
        require(0 <= item["unknown_selected"] <= item["selected"], "Invalid unknown counts")
        policies.append(dict(arm=arm, **item, source="extension", json_pointer=f"/{arm}"))
    sdd = {x["label"]: x for x in docs["sdd"]["contrasts"]}
    require(not docs["sdd"]["original_forest_primary_pass"]
            and not docs["sdd"]["square_loss_primary_pass"], "Do not replace failed primary tests")
    return dict(assembly_source="fresh_run", numerical_results_source="cached_verified",
                source_hashes={str(BASE / p): h for p, h in SOURCES.values()},
                european_contrasts=rows, european_policies=policies,
                sdd_contrasts=[sdd[k] for k in SDD_CONTRASTS],
                temporal_selected_counts=pooled,
                cancellation_fraction=pooled["selected_cancellation"] / pooled["selected_step_harm"],
                planned_source_heads=len(heads), planned_training_fits=3 * len(heads),
                trained_new_auxiliary_heads=0, temporal_auxiliary_readout="not_run",
                new_training=False, new_forecast_evaluation=False, new_bootstrap=False,
                independent_confirmation=False, risk_calibrated=False, deployment_changed=False,
                submission_ready=False, stage5c_executed=False, smc_enabled=False)


def table_text(evidence):
    sdd = ["| SDD comparison | ADE gain difference (pp) | Nominal 95% CI |",
           "|---|---:|---:|"]
    for row in evidence["sdd_contrasts"]:
        sdd.append(f"| {row['label']} | {row['difference_pp']:+.6f} | "
                   f"[{row['CI_low_pp']:+.6f}, {row['CI_high_pp']:+.6f}] |")
    eu = ["| ID | EuropeanSquares comparison | Change | Nominal 95% CI | Localities |",
          "|---|---|---:|---:|---:|"]
    for row in evidence["european_contrasts"]:
        eu.append(f"| {row['id']} | {row['contrast']} | {row['estimate']:+.6f} | "
                  f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | {row['localities']} |")
    policy = ["| European policy | Selected occurrences | Unknown selected | Complete support / 72 | Defined easy risk / 72 | Known violations | Upper-bound violations | Worst easy upper % |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in evidence["european_policies"]:
        policy.append(f"| {row['arm']} | {row['selected']:,} | {row['unknown_selected']:,} | "
                      f"{row['complete_support']} | {row['defined_easy_risk']} | "
                      f"{row['known_label_violations']} | {row['violations']} | "
                      f"{100 * row['worst_easy_upper']:.4f} |")
    return {"SDD_TABLE": "\n".join(sdd), "EU_TABLE": "\n".join(eu),
            "POLICY_TABLE": "\n".join(policy)}


def artifacts(docs, template):
    evidence = build(docs)
    tables = table_text(evidence)
    manuscript = template
    for key, value in tables.items():
        require(manuscript.count("{{" + key + "}}") == 1, "Missing or duplicate table slot")
        manuscript = manuscript.replace("{{" + key + "}}", value)
    require("{{" not in manuscript, "Unresolved template marker")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(evidence["european_contrasts"][0]),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(evidence["european_contrasts"])
    return {"evidence.json": json.dumps(evidence, indent=2, allow_nan=False) + "\n",
            "european_contrasts.csv": stream.getvalue(),
            "tables.md": "# Separate Development Studies\n\n" + "\n\n".join(tables.values()) + "\n",
            "manuscript.md": manuscript}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify committed exports without writing")
    args = parser.parse_args()
    folder = ROOT / OUTPUT
    content = artifacts(load_sources(), (folder / "manuscript.template.md").read_text())
    for name, text in content.items():
        path = folder / name
        if args.check:
            require(path.exists() and path.read_bytes() == text.encode(), f"Export mismatch: {name}")
        else:
            path.write_bytes(text.encode())
    print(json.dumps(dict(status="checked" if args.check else "exported",
                          pinned_sources=len(SOURCES), new_training=False,
                          new_forecast_evaluation=False, submission_ready=False)))


if __name__ == "__main__":
    main()
