"""Audit publication evidence without retraining or overwriting historical results."""
from __future__ import annotations

import ast
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publication_readiness_2026_09"
SOURCE = ROOT / "src/stage44_worldcore.py"
REPORT = ROOT / "outputs/stage44_worldcore/stage44_worldcore_metrics.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def literal(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def synthetic_checks(tree: ast.Module) -> dict[str, Any]:
    # Import torch only after the architecture guard, without importing the pipeline.
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        return {"status": "not_run", "reason": "Apple runtime is not arm64"}
    import torch
    from torch import nn

    torch.set_num_threads(4)
    torch.manual_seed(20260916)
    names = {"WorldCoreModel", "_best_variant"}
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    scope = {
        "torch": torch, "nn": nn, "Mapping": Mapping, "Any": Any,
        "TOKEN_TYPES": literal(tree, "TOKEN_TYPES"),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), scope)
    dims = {t: 3 for t in scope["TOKEN_TYPES"]}
    model = scope["WorldCoreModel"](
        dims, hidden_dim=16, latent_dim=8, include_baseline=True,
        include_scene=True, include_interaction=True, use_jepa=True, use_transformer=True,
    )
    before = {n: p.detach().clone() for n, p in model.future_world_encoder.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    output = model({t: torch.randn(8, 3) for t in dims}, torch.randn(8, 14))
    loss = nn.functional.mse_loss(output["z_next"], output["future_world_latent"])
    loss.backward()
    target_grads = sum(p.grad is not None for p in model.future_world_encoder.parameters())
    context_grads = sum(p.grad is not None for p in model.encoders.parameters())
    optimizer.step()
    target_unchanged = all(torch.equal(before[n], p.detach()) for n, p in model.future_world_encoder.named_parameters())

    def row(value: float) -> dict[str, Any]:
        return {"validation_policy": {"objective": 1.0}, "test_eval": {"protected": {
            "full_waypoint_ade_improvement_vs_floor": value,
            "t50_full_waypoint_ade_improvement_vs_floor": value,
            "hard_failure_full_waypoint_ade_improvement_vs_floor": value,
            "easy_degradation_vs_floor": 0.0,
        }}}

    results = {"a": row(0.1), "b": row(0.2)}
    chosen_before = scope["_best_variant"](results)[0]
    results["a"] = row(0.3)
    chosen_after = scope["_best_variant"](results)[0]
    return {
        "status": "fresh_run", "data": "synthetic_only_not_prediction_evidence",
        "runtime": {"machine": platform.machine(), "torch": torch.__version__, "threads": 4},
        "target_encoder_parameters_with_gradient": target_grads,
        "context_encoder_parameters_with_gradient": context_grads,
        "target_encoder_unchanged_after_optimizer_step": target_unchanged,
        "best_variant_changes_when_only_test_metrics_change": chosen_before != chosen_after,
    }


def cache_inventory() -> dict[str, Any]:
    import numpy as np

    patterns = [
        "data/stage43_full_waypoint_supervision_cache/stage43_full_waypoint_supervision_{split}.npz",
        "data/stage43_scene_raster_proxy_cache/stage43_scene_proxy_features_{split}.npz",
        "data/stage43_all_agent_current_graph_cache/stage43_all_agent_current_graph_{split}.npz",
        "data/stage43_all_agent_history_graph_cache/stage43_all_agent_history_graph_{split}.npz",
    ]
    # Resolve the scene path from its source declaration, rather than assuming its name.
    scene_tree = ast.parse((ROOT / "src/stage43_scene_raster_proxy_tokens.py").read_text())
    for n in scene_tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DATA_DIR" for t in n.targets):
            scene_dir = ast.literal_eval(n.value.args[0])
            patterns[1] = scene_dir + "/stage43_scene_proxy_features_{split}.npz"
    files = []
    split_rows = {}
    source_sets = {}
    scene_sets = {}
    for pattern in patterns:
        for split in ["train", "val", "test"]:
            relative = pattern.format(split=split)
            path = ROOT / relative
            entry = {"path": relative, "exists": path.exists()}
            if path.exists():
                entry.update(bytes=path.stat().st_size, sha256=sha256(path))
                if "full_waypoint_supervision_cache" in relative:
                    with np.load(path, allow_pickle=False) as data:
                        counts = lambda key: {str(k): int(v) for k, v in zip(*np.unique(data[key], return_counts=True))}
                        split_rows[split] = {
                            "rows": len(data["horizon"]), "horizons": counts("horizon"),
                            "datasets": counts("dataset"), "sources": counts("source_file"),
                            "scenes": counts("scene_id"),
                        }
                        source_sets[split] = set(data["source_file"].astype(str))
                        scene_sets[split] = set(data["scene_id"].astype(str))
            files.append(entry)
    # Match Stage44's ordered combined hash; all these caches are below 256 MiB.
    parts = []
    for entry in files:
        if entry.get("bytes", 0) > 256 * 1024 * 1024:
            raise ValueError("Legacy metadata hash requires separate review for large caches")
        parts.append(f"{entry['path']}:{entry.get('sha256', 'missing')}")
    overlaps = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if a in source_sets and b in source_sets:
            overlaps[f"{a}_{b}"] = {"source_overlap": sorted(source_sets[a] & source_sets[b]), "scene_overlap": sorted(scene_sets[a] & scene_sets[b])}
    return {
        "status": "fresh_run", "files": files, "current_cache_rows": split_rows,
        "source_and_scene_overlap": overlaps,
        "current_combined_hash": hashlib.sha256("\n".join(parts).encode()).hexdigest(),
        "not_proven": ["cross_format_duplicate_recordings", "teacher_training_lineage", "complete_feature_causality"],
    }


def main() -> None:
    text = SOURCE.read_text()
    tree = ast.parse(text)
    report = json.loads(REPORT.read_text())
    inventory = cache_inventory()
    checkpoints = []
    for name, result in report["variants"].items():
        path = ROOT / result["checkpoint"]
        digest = sha256(path) if path.exists() else None
        checkpoints.append({"variant": name, "exists": path.exists(), "sha256": digest, "matches_recorded": digest == result["checkpoint_sha256"]})
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    findings = [
        {"id": "test_selected_variant", "line": functions["_best_variant"].lineno,
         "finding": "Final architecture ranking reads test_eval; it is not confirmatory model selection."},
        {"id": "test_driven_repair", "line": functions["_needs_repair"].lineno,
         "finding": "Repair trigger reads test_eval; the current recorded run contains seven base variants and no repair variants."},
        {"id": "proxy_semantics", "line": functions["_loss"].lineno,
         "finding": "Interaction label is hard OR failure; validity label is waypoint completeness; density label is historical density."},
        {"id": "gain_harm_target_mismatch", "source": "src/stage43_full_waypoint_latent_dynamics.py:304",
         "finding": "Gain uses candidate oracle vs strongest; harm uses easy OR small oracle margin, not realized neural-vs-floor harm."},
        {"id": "nonstandard_comparator", "source": "src/stage43_full_waypoint_latent_dynamics.py:296",
         "finding": "Floor waypoints are linear interpolation of a selected endpoint; errors are per-row scale-normalized over four waypoints."},
        {"id": "asserted_leakage_gate", "line": functions["_run"].lineno,
         "finding": "The Stage44 no-leakage gate reads declared flags, not a complete data-lineage or test-selection audit."},
        {"id": "target_encoder_not_updated", "source": "src/stage44_worldcore.py:217",
         "finding": "The randomly initialized future target encoder is detached and has no EMA update in this module."},
    ]
    historical_best = report["best_variant"]
    validation_best = max(report["variants"], key=lambda n: report["variants"][n]["validation_policy"]["objective"])
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fresh_run", "scope": "source_artifact_and_synthetic_audit_only",
        "training_status": "not_run", "real_data_prediction_reproduction": "not_run",
        "source_sha256": sha256(SOURCE), "historical_metrics_sha256": sha256(REPORT),
        "historical_run_date": report["generated_at_utc"], "historical_mode": report["mode"],
        "historical_rows": report["rows"], "historical_epochs": {k: len(v["training_history"]) for k, v in report["variants"].items()},
        "historical_best_variant": historical_best, "retrospective_validation_best": validation_best,
        "validation_best_coincides_with_historical_best": historical_best == validation_best,
        "selection_caveat": "Coinciding rankings do not undo test access or restore an untouched holdout.",
        "historical_selected_metrics": report["variants"][historical_best]["test_eval"]["protected"],
        "recorded_input_hash": report["input_hash"],
        "input_hash_matches_current_cache": inventory["current_combined_hash"] == report["input_hash"],
        "checkpoints": checkpoints, "cache_inventory": inventory,
        "synthetic_checks": synthetic_checks(tree), "findings": findings,
        "stage44_bootstrap_recorded": any("bootstrap" in k for k in report),
        "claim_status": "exploratory_evidence_not_submission_ready",
        "stage5c_executed": False, "smc_enabled": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# M3W Submission Evidence Audit", "",
        "Fresh source/artifact audit and synthetic diagnostics. No model retraining or real-data prediction replay.", "",
        f"- Historical run: {payload['historical_run_date']}; mode: {payload['historical_mode']}; rows: {payload['historical_rows']}",
        f"- Recorded input hash matches present caches: {payload['input_hash_matches_current_cache']}",
        f"- Checkpoints matching recorded SHA256: {sum(x['matches_recorded'] for x in checkpoints)}/{len(checkpoints)}",
        f"- Retrospective validation best: {validation_best}; recorded test-selected best: {historical_best}",
        "- Equal validation/test winners do not restore independent confirmation.",
        "- Reported gains are scale-normalized four-waypoint ADE vs an endpoint-interpolated floor, not directly Stage37 FDE or community-standard ADE.",
        "", "## Findings", "",
    ]
    lines += [f"- {f['id']}: {f['finding']}" for f in findings]
    lines += ["", "## Synthetic Checks", "", "```json", json.dumps(payload["synthetic_checks"], indent=2), "```", "", "## Scope", "", "No training, new benchmark gain, complete no-leakage certification, or deployment promotion is claimed.", "Current caches must not be called cached_verified replay unless input identities and complete lineage match.", "Dataset-local/raw-frame only; no metric, seconds-level, true-3D or foundation claims. Stage5C and SMC remain disabled."]
    (OUT / "evidence_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: payload[k] for k in ["claim_status", "input_hash_matches_current_cache", "validation_best_coincides_with_historical_best", "synthetic_checks"]}, indent=2))


if __name__ == "__main__":
    main()
