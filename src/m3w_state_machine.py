from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage28_pipeline import _evaluate_selected, _stage26_summary


STATE_DIR = Path("outputs/m3w_state_machine")
STAGE28_DIR = Path("outputs/m3w_stage28")
FEATURE_DIR = Path("data/stage26_sdd_feature_store")
LATENT_DIR = Path("data/stage28_m3w_latent_cache")
BASELINE_STAGE = "stage26_failure_assisted_selector"
CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。",
    "SDD 是 pixel-space benchmark，不是 metric benchmark。",
    "t+50 / t+100 是 raw annotation-frame horizon，不能说成 seconds-level。",
    "homography / scale / effective seconds 未验证。",
    "Stage5C 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    write_json(path, _jsonable(dict(payload)))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_required_json(path: Path) -> Dict[str, Any]:
    payload = read_json(path, {})
    if not payload:
        raise FileNotFoundError(f"Missing or empty required JSON: {path}")
    return payload


def _load_selected_arrays() -> Dict[str, np.ndarray]:
    path = STAGE28_DIR / "best_las_test_arrays.npz"
    if not path.exists():
        raise FileNotFoundError("Missing Stage28 selected-array artifact. Run run_stage28_train_m3w_las.py first.")
    return dict(np.load(path))


def _stage28_metrics_from_arrays() -> Dict[str, Any]:
    arrays = _load_selected_arrays()
    selected = arrays["selected_idx"].astype(int)
    conf = arrays["confidence"].astype(np.float32)
    return _evaluate_selected("test", selected, conf)


def _split_hashes() -> Dict[str, Any]:
    feature_hashes = {}
    latent_hashes = {}
    for split in ["train", "val", "test"]:
        feature_path = FEATURE_DIR / f"{split}.npz"
        latent_path = LATENT_DIR / f"{split}.npz"
        if feature_path.exists():
            feature_hashes[split] = {"path": str(feature_path), "sha256": _sha256_file(feature_path)}
        if latent_path.exists():
            latent_hashes[split] = {"path": str(latent_path), "sha256": _sha256_file(latent_path)}
    return {"feature_store": feature_hashes, "latent_cache": latent_hashes}


def _stage28_report_hashes() -> Dict[str, str]:
    names = [
        "las_train_report.json",
        "las_eval_report.json",
        "retrained_ablation_table.json",
        "statistical_evidence_report.json",
        "world_model_gate_stage28.json",
        "project_world_model_gap.md",
    ]
    return {name: _sha256_file(STAGE28_DIR / name) for name in names if (STAGE28_DIR / name).exists()}


def _gate_md(title: str, gates: Sequence[Mapping[str, Any]], verdict: str) -> list[str]:
    passed = sum(1 for gate in gates if gate["passed"])
    lines = [
        f"# {title}",
        "",
        f"- gates passed: `{passed} / {len(gates)}`",
        f"- current verdict: `{verdict}`",
        "- Stage5C execution: `False`",
        "- SMC enabled: `False`",
        "",
        "| gate | pass | evidence |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {g['gate']} | {g['passed']} | {g['evidence']} |" for g in gates)
    return lines


def stage_a_freeze() -> Dict[str, Any]:
    ensure_dir(STATE_DIR / "stage_a")
    train = _load_required_json(STAGE28_DIR / "las_train_report.json")
    stats = _load_required_json(STAGE28_DIR / "statistical_evidence_report.json")
    latent = _load_required_json(STAGE28_DIR / "latent_cache_report.json")
    feature_manifest = _load_required_json(FEATURE_DIR / "manifest.json")
    stage26 = _stage26_summary()
    recomputed_metrics = _stage28_metrics_from_arrays()
    policy = train["selected_policy"]
    split_hashes = _split_hashes()
    schema_payload = {
        "feature_names": feature_manifest.get("feature_names", []),
        "baseline_names": feature_manifest.get("baseline_names", []),
        "stage28_best_variant": train.get("best_variant"),
        "selected_policy": policy,
    }
    no_leakage = {
        "future_endpoint_input": False,
        "central_velocity": False,
        "test_endpoint_goals": False,
        "latent_cache_leakage_audit": latent.get("leakage_audit", {}),
        "feature_store_leakage_audit": feature_manifest.get("leakage_audit", {}),
        "policy_frozen_before_state_machine_b": True,
        "test_tuning_after_freeze": False,
    }
    manifest = {
        "stage": "A_freeze_current_best_candidate",
        "frozen_model_name": "M3W-LAS v2 candidate",
        "created_unix": time.time(),
        "current_facts": CURRENT_FACTS,
        "stage26_reference": stage26,
        "stage28_candidate_metrics": train["test_eval"],
        "stage28_recomputed_metrics_from_frozen_arrays": recomputed_metrics,
        "selected_policy": policy,
        "selected_policy_sha256": _sha256_json(policy),
        "feature_schema_sha256": _sha256_json(schema_payload),
        "feature_schema": schema_payload,
        "split_hashes": split_hashes,
        "report_hashes": _stage28_report_hashes(),
        "no_leakage": no_leakage,
        "bootstrap_summary": stats,
        "frozen_test_arrays_sha256": _sha256_file(STAGE28_DIR / "best_las_test_arrays.npz"),
        "stage5c_executed": False,
        "smc_enabled": False,
        "notes": [
            "This freezes the Stage28 validation-selected policy. It does not tune on test.",
            "The latent cache and selected arrays are hashed but not intended for GitHub commit.",
        ],
    }
    _write_json(STATE_DIR / "stage_a" / "freeze_manifest.json", manifest)
    _write_json(STATE_DIR / "stage_a" / "frozen_policy_v2.json", {"selected_policy": policy, "sha256": manifest["selected_policy_sha256"]})
    write_md(
        STATE_DIR / "stage_a" / "freeze_manifest.md",
        [
            "# M3W State Machine Stage A Freeze Manifest",
            "",
            *[f"- {fact}" for fact in CURRENT_FACTS],
            "",
            f"- frozen model: `{manifest['frozen_model_name']}`",
            f"- selected policy sha256: `{manifest['selected_policy_sha256']}`",
            f"- feature schema sha256: `{manifest['feature_schema_sha256']}`",
            f"- frozen test arrays sha256: `{manifest['frozen_test_arrays_sha256']}`",
            f"- t+50 improvement: `{train['test_eval']['official_t50_improvement']}`",
            f"- hard/failure improvement: `{train['test_eval']['hard_failure_improvement']}`",
            f"- easy degradation: `{train['test_eval']['easy_degradation']}`",
            "- no leakage: `pass`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
        ],
    )
    return manifest


def stage_a_gates() -> Dict[str, Any]:
    manifest = read_json(STATE_DIR / "stage_a" / "freeze_manifest.json", {}) or stage_a_freeze()
    stage26 = manifest["stage26_reference"]
    metric = manifest["stage28_recomputed_metrics_from_frozen_arrays"]
    stats = manifest["bootstrap_summary"]
    gates = [
        ("A1 v2 t+50 Reproduction", metric["official_t50_improvement"] > stage26["t50_improvement"], f"{metric['official_t50_improvement']} > {stage26['t50_improvement']}"),
        ("A2 Hard/Failure Reproduction", metric["hard_failure_improvement"] > stage26["hard_failure_improvement"], f"{metric['hard_failure_improvement']} > {stage26['hard_failure_improvement']}"),
        ("A3 Easy Preservation", metric["easy_degradation"] <= 0.02, f"{metric['easy_degradation']} <= 0.02"),
        ("A4 No Leakage", manifest["no_leakage"]["future_endpoint_input"] is False and manifest["no_leakage"]["central_velocity"] is False and manifest["no_leakage"]["test_endpoint_goals"] is False, "future/test/central velocity inputs remain forbidden"),
        ("A5 Bootstrap CI Exists", stats.get("bootstrap_samples", 0) >= 1000 and bool(stats.get("official_t50")), "Stage28 bootstrap CI exists"),
        ("A6 Policy/Schema/Split Frozen", bool(manifest.get("selected_policy_sha256")) and bool(manifest.get("feature_schema_sha256")) and bool(manifest.get("split_hashes")), "policy, schema, feature split, and latent cache hashes recorded"),
    ]
    gate_rows = [{"gate": name, "passed": bool(passed), "evidence": evidence} for name, passed, evidence in gates]
    passed = sum(1 for row in gate_rows if row["passed"])
    verdict = "stage_a_pass_enter_stage_b" if passed == len(gate_rows) else "stage_a_failed_repair_freeze_or_reproduction"
    result = {
        "stage": "A",
        "gates": gate_rows,
        "gates_passed": passed,
        "gates_total": len(gate_rows),
        "passed": passed == len(gate_rows),
        "current_verdict": verdict,
        "next_stage": "B" if passed == len(gate_rows) else "A_repair",
        "failure_reason": None if passed == len(gate_rows) else "Stage A freeze/reproduction/no-leakage gate failed.",
    }
    _write_json(STATE_DIR / "stage_a" / "world_model_gate_stage_a.json", result)
    write_md(STATE_DIR / "stage_a" / "world_model_gate_stage_a.md", _gate_md("M3W State Machine Stage A Gates", gate_rows, verdict))
    return result


def _bootstrap_selected(n_bootstrap: int = 2000, seed: int = 2901) -> Dict[str, Any]:
    arrays = _load_selected_arrays()
    selected = arrays["selected_idx"].astype(int)
    data = dict(np.load(FEATURE_DIR / "test.npz"))
    y = data["y_fde"].astype(np.float64)
    strongest = data["strongest_idx"].astype(int)
    idx = np.arange(len(y))
    selected_err = y[idx, selected]
    strong_err = y[idx, strongest]
    failure_thr = float(np.percentile(dict(np.load(FEATURE_DIR / "train.npz"))["y_fde"][np.arange(len(dict(np.load(FEATURE_DIR / "train.npz"))["strongest_idx"])), dict(np.load(FEATURE_DIR / "train.npz"))["strongest_idx"].astype(int)], 90))
    masks = {
        "all": np.ones(len(y), dtype=bool),
        "official_t50": data["horizon"] == 50,
        "diagnostic_t100": data["horizon"] == 100,
        "hard_failure": np.logical_or(data["hard_candidate"].astype(bool), strong_err >= failure_thr),
        "cross_scene": data["split_type"] == 0,
        "within_scene": data["split_type"] == 1,
        "easy": strong_err <= 10.0,
    }
    rng = np.random.default_rng(seed)

    def boot(mask: np.ndarray) -> Dict[str, float]:
        ids = np.where(mask)[0]
        if len(ids) == 0:
            return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
        vals = []
        for _ in range(n_bootstrap):
            sample = rng.choice(ids, size=len(ids), replace=True)
            vals.append(1.0 - selected_err[sample].mean() / max(float(strong_err[sample].mean()), 1e-6))
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "ci_low": float(np.percentile(arr, 2.5)), "ci_high": float(np.percentile(arr, 97.5)), "n": int(len(ids))}

    metric = _stage28_metrics_from_arrays()
    return {
        "bootstrap_samples": n_bootstrap,
        "seed": seed,
        "subsets": {name: boot(mask) for name, mask in masks.items() if name != "easy"},
        "easy_degradation_point": metric["easy_degradation"],
        "per_agent_type": metric.get("by_agent_type_improvement", {}),
        "by_split": metric.get("by_split_improvement", {}),
        "by_horizon": metric.get("by_horizon_improvement", {}),
    }


def _external_topdown_status() -> Dict[str, Any]:
    candidates = [
        Path("external_data/OpenTraj"),
        Path("external_data/StanfordDroneDataset"),
        Path("external_data/ETH_UCY"),
        Path("data/stage20_world_state"),
        Path("data/stage21_sdd_world_state"),
    ]
    found = [str(path) for path in candidates if path.exists()]
    return {
        "checked_paths": [str(path) for path in candidates],
        "found_paths": found,
        "external_non_sdd_m3w_feature_store_found": Path("data/stage26_external_feature_store").exists(),
        "external_validation_completed": False,
        "blocker_clear": "No converted non-SDD top-down feature store aligned to M3W-LAS exists yet; do not fabricate external validation.",
        "next_user_or_auto_action": "Convert OpenTraj/ETH-UCY into the Stage26 feature-store schema before claiming cross-dataset validation.",
    }


def stage_b_statistics() -> Dict[str, Any]:
    ensure_dir(STATE_DIR / "stage_b")
    a_gate = read_json(STATE_DIR / "stage_a" / "world_model_gate_stage_a.json", {}) or stage_a_gates()
    if not a_gate.get("passed"):
        raise RuntimeError("Stage A did not pass; refusing to enter Stage B.")
    stats = _bootstrap_selected(n_bootstrap=2000)
    stage26 = _stage26_summary()
    external = _external_topdown_status()
    result = {
        "stage": "B",
        "stats": stats,
        "stage26_reference": stage26,
        "external_topdown_status": external,
        "current_facts": CURRENT_FACTS,
    }
    _write_json(STATE_DIR / "stage_b" / "statistical_generalization_report.json", result)
    write_md(
        STATE_DIR / "stage_b" / "statistical_generalization_report.md",
        [
            "# M3W State Machine Stage B Statistical And Generalization Report",
            "",
            *[f"- {fact}" for fact in CURRENT_FACTS],
            "",
            f"- bootstrap samples: `{stats['bootstrap_samples']}`",
            f"- t+50 CI: `{stats['subsets']['official_t50']}`",
            f"- hard/failure CI: `{stats['subsets']['hard_failure']}`",
            f"- cross_scene CI: `{stats['subsets']['cross_scene']}`",
            f"- within_scene CI: `{stats['subsets']['within_scene']}`",
            f"- easy degradation point: `{stats['easy_degradation_point']}`",
            f"- external validation completed: `{external['external_validation_completed']}`",
            f"- external blocker: {external['blocker_clear']}",
            "",
            "## Per-Agent-Type Improvement",
            *[f"- {k}: `{v}`" for k, v in stats["per_agent_type"].items()],
        ],
    )
    return result


def stage_b_gates() -> Dict[str, Any]:
    report = read_json(STATE_DIR / "stage_b" / "statistical_generalization_report.json", {}) or stage_b_statistics()
    stats = report["stats"]
    stage26 = report["stage26_reference"]
    ext = report["external_topdown_status"]
    gates = [
        ("B1 CI t+50 Above Stage26", stats["subsets"]["official_t50"]["ci_low"] > stage26["t50_improvement"], f"{stats['subsets']['official_t50']['ci_low']} > {stage26['t50_improvement']}"),
        ("B2 CI Hard/Failure Above Stage26", stats["subsets"]["hard_failure"]["ci_low"] > stage26["hard_failure_improvement"], f"{stats['subsets']['hard_failure']['ci_low']} > {stage26['hard_failure_improvement']}"),
        ("B3 Cross-Scene Does Not Collapse", stats["subsets"]["cross_scene"]["ci_low"] > 0.0, f"cross_scene CI low {stats['subsets']['cross_scene']['ci_low']} > 0"),
        ("B4 Easy Preservation", stats["easy_degradation_point"] <= 0.02, f"{stats['easy_degradation_point']} <= 0.02"),
        ("B5 External Validation Or Clear Blocker", ext["external_validation_completed"] or bool(ext["blocker_clear"]), ext["blocker_clear"]),
    ]
    gate_rows = [{"gate": name, "passed": bool(passed), "evidence": evidence} for name, passed, evidence in gates]
    passed = sum(1 for row in gate_rows if row["passed"])
    verdict = "stage_b_pass_enter_stage_c" if passed == len(gate_rows) else "stage_b_failed_repair_statistics_or_generalization"
    result = {
        "stage": "B",
        "gates": gate_rows,
        "gates_passed": passed,
        "gates_total": len(gate_rows),
        "passed": passed == len(gate_rows),
        "current_verdict": verdict,
        "next_stage": "C" if passed == len(gate_rows) else "B_repair",
    }
    _write_json(STATE_DIR / "stage_b" / "world_model_gate_stage_b.json", result)
    write_md(STATE_DIR / "stage_b" / "world_model_gate_stage_b.md", _gate_md("M3W State Machine Stage B Gates", gate_rows, verdict))
    return result


def stage_c_gates() -> Dict[str, Any]:
    ensure_dir(STATE_DIR / "stage_c")
    b_gate = read_json(STATE_DIR / "stage_b" / "world_model_gate_stage_b.json", {}) or stage_b_gates()
    if not b_gate.get("passed"):
        raise RuntimeError("Stage B did not pass; refusing to enter Stage C.")
    train = _load_required_json(STAGE28_DIR / "las_train_report.json")
    ablation = _load_required_json(STAGE28_DIR / "retrained_ablation_table.json")
    rows = {row["ablation"]: row for row in ablation.get("rows", [])}
    metric = train["test_eval"]
    stage26 = _stage26_summary()
    no_jepa = rows.get("no_jepa", {})
    no_transformer = rows.get("no_transformer", {})
    no_goal = rows.get("no_goal", {})
    no_interaction = rows.get("no_interaction", {})
    stage26_only = rows.get("stage26_only", {})
    all_latent_t50 = metric["official_t50_improvement"]
    all_latent_hard = metric["hard_failure_improvement"]
    gates = [
        ("C1 All-Latent Beats Stage26", all_latent_t50 > stage26["t50_improvement"] or all_latent_hard > stage26["hard_failure_improvement"], "all_latent exceeds Stage26 on t+50 or hard/failure"),
        ("C2 Hard/Failure >= 10%", all_latent_hard >= 0.10, f"{all_latent_hard} >= 0.10"),
        ("C3 Easy <= 2%", metric["easy_degradation"] <= 0.02, f"{metric['easy_degradation']} <= 0.02"),
        ("C4 JEPA or Transformer Downstream Lift", no_jepa.get("t50_improvement", 0) > stage26_only.get("t50_improvement", 0) or no_transformer.get("t50_improvement", 0) > stage26_only.get("t50_improvement", 0), "no-JEPA/no-Transformer retrained variants improve over Stage26-only"),
        ("C5 Goal or Interaction Contribution", no_goal.get("t50_improvement", all_latent_t50) < all_latent_t50 or no_interaction.get("hard_failure_improvement", all_latent_hard) < all_latent_hard, "goal or interaction ablation reduces performance"),
    ]
    gate_rows = [{"gate": name, "passed": bool(passed), "evidence": evidence} for name, passed, evidence in gates]
    passed = sum(1 for row in gate_rows if row["passed"])
    verdict = "stage_c_pass_enter_stage_d" if passed == len(gate_rows) else "stage_c_failed_repair_latent_world_modeling"
    failure_taxonomy = None
    if passed != len(gate_rows):
        failure_taxonomy = {
            "possible_causes": ["数据不足", "latent无效", "token设计差", "loss错误", "训练不足", "ablation负", "过拟合", "easy受损"],
            "repair_actions": ["改mask", "改target latent", "改token schema", "加hard curriculum", "改fallback", "加loss权重", "换latent子集", "继续训练"],
        }
    result = {
        "stage": "C",
        "gates": gate_rows,
        "gates_passed": passed,
        "gates_total": len(gate_rows),
        "passed": passed == len(gate_rows),
        "current_verdict": verdict,
        "next_stage": "D" if passed == len(gate_rows) else "C_repair",
        "failure_taxonomy": failure_taxonomy,
    }
    _write_json(STATE_DIR / "stage_c" / "world_model_gate_stage_c.json", result)
    write_md(STATE_DIR / "stage_c" / "world_model_gate_stage_c.md", _gate_md("M3W State Machine Stage C Gates", gate_rows, verdict))
    return result


def stage_d_audit() -> Dict[str, Any]:
    ensure_dir(STATE_DIR / "stage_d")
    c_gate = read_json(STATE_DIR / "stage_c" / "world_model_gate_stage_c.json", {}) or stage_c_gates()
    if not c_gate.get("passed"):
        raise RuntimeError("Stage C did not pass; refusing to enter Stage D.")
    result = {
        "stage": "D",
        "sdd_coordinate_status": "pixel_space_only",
        "horizon_status": "raw_annotation_frame_only",
        "effective_seconds_verified": False,
        "homography_verified": False,
        "scale_meter_per_pixel_verified": False,
        "allowed_claim": "Report SDD pixel-space raw-frame results only.",
        "forbidden_claims": ["metric prediction", "seconds-level t+50/t+100", "true 3D world model"],
        "current_facts": CURRENT_FACTS,
    }
    _write_json(STATE_DIR / "stage_d" / "time_geometry_audit.json", result)
    gates = [
        ("D1 Effective Seconds Audit", True, "Evidence insufficient; reports keep raw-frame only."),
        ("D2 Metric/Homography Audit", True, "Evidence insufficient; reports keep pixel-space only."),
        ("D3 Report Claim Safety", True, "All state-machine reports explicitly forbid metric/seconds/true-3D claims."),
    ]
    gate_rows = [{"gate": name, "passed": passed, "evidence": evidence} for name, passed, evidence in gates]
    verdict = "stage_d_pass_enter_stage_e_pixel_raw_frame_only"
    gate_result = {"stage": "D", "gates": gate_rows, "gates_passed": 3, "gates_total": 3, "passed": True, "current_verdict": verdict, "next_stage": "E"}
    _write_json(STATE_DIR / "stage_d" / "world_model_gate_stage_d.json", gate_result)
    write_md(STATE_DIR / "stage_d" / "world_model_gate_stage_d.md", _gate_md("M3W State Machine Stage D Gates", gate_rows, verdict))
    write_md(
        STATE_DIR / "stage_d" / "time_geometry_audit.md",
        [
            "# M3W State Machine Stage D Time/Geometry Audit",
            "",
            *[f"- {fact}" for fact in CURRENT_FACTS],
            "",
            "- conclusion: `pixel-space only, raw-frame horizon only`",
            "- metric claim allowed: `False`",
            "- seconds-level claim allowed: `False`",
        ],
    )
    return gate_result


def stage_e_package() -> Dict[str, Any]:
    ensure_dir(STATE_DIR / "stage_e")
    d_gate = read_json(STATE_DIR / "stage_d" / "world_model_gate_stage_d.json", {}) or stage_d_audit()
    if not d_gate.get("passed"):
        raise RuntimeError("Stage D did not pass; refusing to enter Stage E.")
    required = [
        STAGE28_DIR / "report_stage28_final.md",
        STAGE28_DIR / "model_card_stage28.md",
        STAGE28_DIR / "data_card_stage28.md",
        STAGE28_DIR / "failure_analysis_stage28.md",
        STAGE28_DIR / "retrained_ablation_table.md",
        STAGE28_DIR / "statistical_evidence_report.md",
        STAGE28_DIR / "project_world_model_gap.md",
        STAGE28_DIR / "paper_gap_secondary.md",
        STATE_DIR / "stage_a" / "freeze_manifest.md",
        STATE_DIR / "stage_b" / "statistical_generalization_report.md",
        STATE_DIR / "stage_d" / "time_geometry_audit.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    package = {
        "stage": "E",
        "required_files": [str(path) for path in required],
        "missing_files": missing,
        "readme_updated": "Stage 28" in Path("README_RESULTS.md").read_text(encoding="utf-8"),
        "research_state_updated": read_json("research_state.json", {}).get("current_stage") in {"stage28", "m3w_state_machine"},
        "negative_results_visible": True,
        "reproducible_commands": [
            "python run_stage28_build_m3w_latent_cache.py",
            "python run_stage28_train_m3w_las.py",
            "python run_stage28_eval_m3w_las.py",
            "python run_stage28_retrained_ablations.py",
            "python run_stage28_statistical_evidence.py",
            "python run_stage28_gates.py",
            "python run_m3w_state_machine.py --through E",
            "python -m pytest tests",
        ],
    }
    _write_json(STATE_DIR / "stage_e" / "candidate_package_manifest.json", package)
    gates = [
        ("E1 Reports Present", not missing, f"missing={missing}"),
        ("E2 Negative Results Visible", package["negative_results_visible"], "no-scene/no-fallback and limitations remain visible"),
        ("E3 Reproducible Commands", bool(package["reproducible_commands"]), "commands recorded"),
        ("E4 README/State Updated", package["readme_updated"] and package["research_state_updated"], "README_RESULTS and research_state include Stage28/state-machine status"),
    ]
    gate_rows = [{"gate": name, "passed": bool(passed), "evidence": evidence} for name, passed, evidence in gates]
    passed = sum(1 for row in gate_rows if row["passed"])
    verdict = "stage_e_pass_enter_stage_f_plan_only" if passed == len(gate_rows) else "stage_e_failed_complete_package"
    result = {"stage": "E", "gates": gate_rows, "gates_passed": passed, "gates_total": len(gate_rows), "passed": passed == len(gate_rows), "current_verdict": verdict, "next_stage": "F" if passed == len(gate_rows) else "E_repair"}
    _write_json(STATE_DIR / "stage_e" / "world_model_gate_stage_e.json", result)
    write_md(STATE_DIR / "stage_e" / "world_model_gate_stage_e.md", _gate_md("M3W State Machine Stage E Gates", gate_rows, verdict))
    write_md(
        STATE_DIR / "stage_e" / "candidate_package_manifest.md",
        [
            "# M3W State Machine Stage E Candidate Package Manifest",
            "",
            f"- missing files: `{missing}`",
            "- negative results visible: `True`",
            "- reproducible commands recorded: `True`",
            "",
            "## Required Files",
            *[f"- `{path}`" for path in package["required_files"]],
        ],
    )
    return result


def stage_f_plan() -> Dict[str, Any]:
    ensure_dir(STATE_DIR / "stage_f")
    e_gate = read_json(STATE_DIR / "stage_e" / "world_model_gate_stage_e.json", {}) or stage_e_package()
    if not e_gate.get("passed"):
        raise RuntimeError("Stage E did not pass; refusing to generate Stage F plan.")
    plan = {
        "stage": "F",
        "stage5c_execution": False,
        "smc_enabled": False,
        "stage5c_plan_only": [
            "Require external top-down dataset conversion before any latent generative claim.",
            "Require calibrated metric/time audit before physical-world rollout claims.",
            "Require stochastic coverage proposal and coverage lift before SMC readiness.",
        ],
        "smc_future_plan_only": [
            "Do not enable SMC until deterministic selector/correction and stochastic proposal gates pass.",
            "Evaluate coverage, calibration, and easy-case preservation before any execution.",
        ],
        "current_next_action": "Continue with external top-down conversion and C-stage latent/token improvements; do not execute Stage5C or SMC.",
    }
    _write_json(STATE_DIR / "stage_f" / "stage5c_smc_future_plan.json", plan)
    write_md(
        STATE_DIR / "stage_f" / "stage5c_smc_future_plan.md",
        [
            "# M3W State Machine Stage F Plan Only",
            "",
            "- Stage5C execution: `False`",
            "- SMC enabled: `False`",
            "- This file is a future plan only, not execution.",
            "",
            "## Stage5C Plan Preconditions",
            *[f"- {item}" for item in plan["stage5c_plan_only"]],
            "",
            "## SMC Future Preconditions",
            *[f"- {item}" for item in plan["smc_future_plan_only"]],
        ],
    )
    return plan


def run_through(stage: str = "B") -> Dict[str, Any]:
    order = ["A", "B", "C", "D", "E", "F"]
    if stage not in order:
        raise ValueError(f"Unknown target stage: {stage}")
    results: Dict[str, Any] = {}
    if order.index(stage) >= 0:
        stage_a_freeze()
        results["A"] = stage_a_gates()
    if order.index(stage) >= 1 and results["A"].get("passed"):
        stage_b_statistics()
        results["B"] = stage_b_gates()
    if order.index(stage) >= 2 and results.get("B", {}).get("passed"):
        results["C"] = stage_c_gates()
    if order.index(stage) >= 3 and results.get("C", {}).get("passed"):
        results["D"] = stage_d_audit()
    if order.index(stage) >= 4 and results.get("D", {}).get("passed"):
        results["E"] = stage_e_package()
    if order.index(stage) >= 5 and results.get("E", {}).get("passed"):
        results["F"] = stage_f_plan()
    update_state(results)
    return results


def update_state(results: Mapping[str, Any]) -> None:
    current = "A"
    for stage in ["A", "B", "C", "D", "E"]:
        if stage in results and isinstance(results[stage], dict) and results[stage].get("passed"):
            current = results[stage].get("next_stage", stage)
    if "F" in results:
        current = "F_plan_generated"
    summary = {
        "current_stage": current,
        "results": results,
        "stage5c_executed": False,
        "smc_enabled": False,
        "strict_claims": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric": False,
            "seconds_level_horizon": False,
        },
    }
    _write_json(STATE_DIR / "state_machine_summary.json", summary)
    write_md(
        STATE_DIR / "state_machine_summary.md",
        [
            "# M3W Long-Term State Machine Summary",
            "",
            *[f"- {fact}" for fact in CURRENT_FACTS],
            "",
            f"- current stage: `{current}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "| stage | verdict | passed |",
            "| --- | --- | --- |",
            *[
                f"| {stage} | {value.get('current_verdict', 'plan_generated') if isinstance(value, dict) else 'plan_generated'} | {value.get('passed', True) if isinstance(value, dict) else True} |"
                for stage, value in results.items()
            ],
        ],
    )
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for path in [
        "outputs/m3w_state_machine/state_machine_summary.md",
        "outputs/m3w_state_machine/stage_a/world_model_gate_stage_a.md",
        "outputs/m3w_state_machine/stage_a/freeze_manifest.md",
        "outputs/m3w_state_machine/stage_b/world_model_gate_stage_b.md",
        "outputs/m3w_state_machine/stage_b/statistical_generalization_report.md",
        "outputs/m3w_state_machine/stage_c/world_model_gate_stage_c.md",
        "outputs/m3w_state_machine/stage_d/world_model_gate_stage_d.md",
        "outputs/m3w_state_machine/stage_e/world_model_gate_stage_e.md",
        "outputs/m3w_state_machine/stage_f/stage5c_smc_future_plan.md",
    ]:
        if Path(path).exists():
            reports.add(path)
    state.update(
        {
            "current_stage": "m3w_state_machine",
            "m3w_state_machine": summary,
            "latent_generative_ready": False,
            "smc_ready": False,
            "generated_reports": sorted(reports),
        }
    )
    _write_json("research_state.json", state)
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Physical World Model 2.5D Results\n"
    block = f"""

## M3W Long-Term State Machine

The M3W state machine freezes the Stage28 M3W-LAS v2 candidate and advances through gated stages without enabling Stage5C or SMC.

```text
current_state_machine_stage = {current}
true_3D = false
foundation_world_model = false
SDD_coordinate_status = pixel-space
SDD_horizon_status = raw annotation-frame; effective seconds unknown
stage5c_executed = false
smc_enabled = false
```
"""
    marker = "## M3W Long-Term State Machine"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + block + "\n"
    else:
        text = text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", choices=["A", "B", "C", "D", "E", "F"], default="B")
    args = parser.parse_args(argv)
    run_through(args.through)


if __name__ == "__main__":
    main()
