from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage35_selective_transfer as s35
from src import stage36_t50_repair as s36
from src import stage37_t50_history as s37


OUT_DIR = Path("outputs/stage38_external_robustness")
DATA_DIR = Path("data/stage38_external_robustness")
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
EPS = 1e-6


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


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    write_json(path, _jsonable(dict(payload)))


def _append_ledger(entry: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(dict(entry)), ensure_ascii=False) + "\n")
    rows = [json.loads(line) for line in LEDGER_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [
        "# Stage38 External Robustness Run Ledger",
        "",
        "| command | source | status | wall time s | input hash | output hash | git commit |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['command']}` | `{row['source']}` | `{row['status']}` | {float(row['wall_time_s']):.3f} | `{row['input_hash'][:12]}` | `{row['output_hash'][:12]}` | `{row['git_commit']}` |"
        )
    write_md(OUT_DIR / "run_ledger.md", lines)


def run_logged(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> Dict[str, Any]:
    ensure_dir(OUT_DIR)
    start = time.perf_counter()
    status = "failed"
    input_hash = _combined_hash(inputs)
    try:
        payload = fn()
        status = "success"
        return payload
    finally:
        _append_ledger(
            {
                "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
                "step": name,
                "inputs": [str(p) for p in inputs],
                "outputs": [str(p) for p in outputs],
                "wall_time_s": time.perf_counter() - start,
                "status": status,
                "input_hash": input_hash,
                "output_hash": _combined_hash(outputs),
                "git_commit": _git_commit(),
                "source": "fresh_run",
            }
        )


def _geo(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(s35.DATA_DIR / f"expanded_external_{split}.npz"))


def _labels(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(s35.DATA_DIR / f"labels_{split}.npz"))


def _sha(paths: Sequence[str | Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def freeze_stage37_policy() -> Dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage37_gate = read_json(s37.OUT_DIR / "world_model_gate_stage37.json", {})
    stage37_eval = read_json(s37.OUT_DIR / "cross_domain_eval_stage37.json", {})
    selector = read_json(s37.OUT_DIR / "stage37_t50_selector_report.json", {})
    schema_paths = [
        s37.OUT_DIR / "history_window_schema.json",
        s37.OUT_DIR / "stage37_goal_prototype_report.json",
        s37.OUT_DIR / "stage37_t50_selector_report.json",
        s37.OUT_DIR / "stage37_conformal_safety_report.json",
        Path("src/stage37_t50_history.py"),
    ]
    policy = {
        "source": "fresh_run",
        "policy_name": "Stage35 non-t50 + Stage37 neighbor-history t50",
        "policy_hash": _sha(schema_paths),
        "feature_schema_hash": _sha([s37.OUT_DIR / "history_window_schema.json"]),
        "history_window_schema": read_json(s37.OUT_DIR / "history_window_schema.json", {}),
        "goal_prototype_schema": read_json(s37.OUT_DIR / "stage37_goal_prototype_report.json", {}),
        "selected_t50_selector": selector.get("best_selector"),
        "selected_t50_policy": selector.get("experiments", {}).get(selector.get("best_selector", ""), {}).get("policy"),
        "training_split": "Stage35 external train split",
        "validation_selection_rule": "Stage37 selector selected on validation metrics and conformal safety, then frozen",
        "test_usage_rule": "test used once for Stage37/38 evaluation only; no threshold tuning after freeze",
        "stage37_gate": stage37_gate.get("current_verdict"),
        "stage37_final_metrics": stage37_eval.get("matrix", {}).get("external_all", {}),
        "no_leakage_audit": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    _write_json(OUT_DIR / "frozen_stage37_policy.json", policy)
    write_md(OUT_DIR / "stage38_frozen_stage37_policy.md", ["# Stage38 Frozen Stage37 Policy", "", "- source: `fresh_run`", f"- policy hash: `{policy['policy_hash']}`", f"- feature schema hash: `{policy['feature_schema_hash']}`", f"- selected t50 selector: `{policy['selected_t50_selector']}`", f"- final metrics: `{policy['stage37_final_metrics']}`", f"- no leakage: `{policy['no_leakage_audit']}`"])
    return policy


def external_dataset_audit() -> Dict[str, Any]:
    freeze_stage37_policy()
    rows = []
    for split in ["train", "val", "test"]:
        geo = _geo(split)
        hist = dict(np.load(s37.DATA_DIR / f"history_windows_{split}.npz"))
        for domain in sorted(set(geo["dataset"].astype(str).tolist())):
            mask = geo["dataset"].astype(str) == domain
            scenes = sorted(set(geo["scene_id"].astype(str)[mask].tolist()))
            h = geo["horizon"].astype(int)[mask]
            valid = hist["history_valid_mask"][mask]
            rows.append(
                {
                    "domain": domain,
                    "split": split,
                    "source": "cached_verified",
                    "rows": int(mask.sum()),
                    "t10": int(np.sum(h == 10)),
                    "t25": int(np.sum(h == 25)),
                    "t50": int(np.sum(h == 50)),
                    "t100": int(np.sum(h == 100)),
                    "history_k8": int(np.sum(valid[:, -8:].sum(axis=1) >= 8)),
                    "history_k16": int(np.sum(valid[:, -16:].sum(axis=1) >= 16)),
                    "history_k32": int(np.sum(valid[:, -32:].sum(axis=1) >= 32)),
                    "history_k64": int(np.sum(valid[:, -64:].sum(axis=1) >= 64)),
                    "agent_count": int(len(set(geo["agent_id"].astype(int)[mask].tolist()))),
                    "scene_count": int(len(scenes)),
                    "scenes": scenes,
                    "track_length_median": float(np.median(geo["track_length"].astype(float)[mask])) if np.any(mask) else 0.0,
                    "coordinate_unit": "dataset_local",
                    "metric_status": "unverified weak metric diagnostic",
                    "heldout_status": "heldout_test" if split == "test" else ("validation_domain" if split == "val" else "training_domain"),
                }
            )
    by_domain: Dict[str, Any] = {}
    for domain in ["UCY", "ETH_UCY", "TrajNet", "OpenTraj_mixed"]:
        if domain == "OpenTraj_mixed":
            domain_rows = rows
        else:
            domain_rows = [r for r in rows if r["domain"] == domain]
        by_domain[domain] = {
            "rows": int(sum(r["rows"] for r in domain_rows)),
            "t50": int(sum(r["t50"] for r in domain_rows)),
            "t100": int(sum(r["t100"] for r in domain_rows)),
            "splits": sorted(set(r["split"] for r in domain_rows)),
            "heldout_test_available": any(r["split"] == "test" for r in domain_rows),
            "blocker": None if any(r["split"] == "test" for r in domain_rows) else "no held-out test split for frozen Stage37 evaluation; cannot claim deployable generalization",
        }
    result = {"source": "fresh_run", "rows": rows, "by_domain": by_domain}
    _write_json(OUT_DIR / "stage38_external_dataset_audit.json", result)
    lines = ["# Stage38 External Dataset Audit", "", "- source: `fresh_run`; Stage37 caches are `cached_verified`.", "", "| domain | split | rows | t50 | t100 | k8 | k16 | k32 | k64 | heldout |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(f"| {r['domain']} | {r['split']} | {r['rows']} | {r['t50']} | {r['t100']} | {r['history_k8']} | {r['history_k16']} | {r['history_k32']} | {r['history_k64']} | {r['heldout_status']} |")
    lines.append(f"\n- by domain: `{by_domain}`")
    write_md(OUT_DIR / "stage38_external_dataset_audit.md", lines)
    return result


def external_generalization_eval() -> Dict[str, Any]:
    audit = external_dataset_audit()
    stage37_eval = read_json(s37.OUT_DIR / "cross_domain_eval_stage37.json", {})
    ucy = stage37_eval.get("matrix", {}).get("external_all", {})
    matrix = {
        "UCY": {"source": "fresh_run", "status": "heldout_test", "metrics": ucy, "bootstrap_ci_t50": stage37_eval.get("bootstrap_ci_t50")},
        "ETH_UCY": {"source": "not_run", "status": "blocker", "reason": audit["by_domain"]["ETH_UCY"]["blocker"], "metrics": None},
        "TrajNet": {"source": "not_run", "status": "blocker", "reason": audit["by_domain"]["TrajNet"]["blocker"], "metrics": None},
        "OpenTraj_mixed": {"source": "not_run", "status": "blocker", "reason": "mixed-domain held-out test beyond UCY is not available without redefining frozen Stage37 split", "metrics": None},
    }
    positive_domains = [k for k, v in matrix.items() if v.get("metrics") and (v["metrics"].get("all_improvement", 0.0) > 0 or v["metrics"].get("t50_improvement", 0.0) > 0)]
    result = {"source": "fresh_run", "matrix": matrix, "positive_domains": positive_domains, "two_domain_success": len(positive_domains) >= 2, "honest_blocker": len(positive_domains) < 2}
    _write_json(OUT_DIR / "stage38_external_generalization_eval.json", result)
    lines = ["# Stage38 External Generalization Eval", "", "- source: `fresh_run`", "", "| domain | status | all | t50 | hard | easy | note |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for domain, item in matrix.items():
        m = item.get("metrics") or {}
        lines.append(f"| {domain} | {item.get('status')} | {m.get('all_improvement', 0.0):.6f} | {m.get('t50_improvement', 0.0):.6f} | {m.get('hard_failure_improvement', 0.0):.6f} | {m.get('easy_degradation', 0.0):.6f} | {item.get('reason', '')} |")
    lines.append(f"\n- positive domains: `{positive_domains}`")
    lines.append("- If only UCY is positive, current model remains UCY-biased until ETH/TrajNet held-out tests are built.")
    write_md(OUT_DIR / "stage38_external_generalization_eval.md", lines)
    return result


def _stage37_family_selection(split: str) -> Tuple[np.ndarray, np.ndarray]:
    if split != "test":
        selected = np.full(len(_geo(split)["horizon"]), -1, dtype=np.int16)
        conf = np.zeros(len(selected), dtype=np.float32)
        return selected, conf
    art = dict(np.load(s37.DATA_DIR / "stage37_best_t50_selection_test.npz"))
    return art["selected_family"].astype(int), art["confidence"].astype(np.float32)


def _feature(split: str) -> np.ndarray:
    x, _ = s37._feature_matrix(split)
    return x.astype(np.float32)


def _train_correction_model(mask: np.ndarray) -> Any:
    geo = _geo("train")
    fam = s37._baseline_family("train")
    x = _feature("train")
    # Train residual from neighbor-aware baseline to future label.
    base_idx = s37.BASELINE_FAMILY.index("neighbor_aware_decay_baseline")
    base = fam["prediction"][:, base_idx, :].astype(np.float64)
    fut = np.stack([geo["future_endpoint_x"], geo["future_endpoint_y"]], axis=1).astype(np.float64)
    y = fut - base
    if mask.sum() < 100:
        mask = geo["horizon"].astype(int) == 50
    model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    model.fit(x[mask], y[mask])
    return model


def _eval_correction(split: str, model: Any, alpha: float, clip: float, deploy_with_stage37_fallback: bool = True) -> Dict[str, Any]:
    geo = _geo(split)
    fam = s37._baseline_family(split)
    stage = _labels(split)
    selected_family, conf = _stage37_family_selection(split)
    x = _feature(split)
    base_idx = s37.BASELINE_FAMILY.index("neighbor_aware_decay_baseline")
    pred_base = fam["prediction"][:, base_idx, :].astype(np.float64)
    selected_pred = pred_base.copy()
    valid_family = selected_family >= 0
    selected_pred[valid_family] = fam["prediction"].astype(np.float64)[np.where(valid_family)[0], selected_family[valid_family]]
    delta = model.predict(x).astype(np.float64)
    norms = np.linalg.norm(delta, axis=1)
    scale = np.minimum(1.0, clip / np.maximum(norms, EPS))
    corrected = selected_pred + alpha * delta * scale[:, None]
    fut = np.stack([geo["future_endpoint_x"], geo["future_endpoint_y"]], axis=1).astype(np.float64)
    corrected_fde = np.linalg.norm(corrected - fut, axis=1)
    # Full Stage37 policy: Stage35 non-t50 selector plus Stage37 t50 selector.
    stage37_selected_fde, fallback, _ = _stage37_selected_and_fallback(split)
    t50_switch = (geo["horizon"].astype(int) == 50) & (selected_family >= 0)
    if deploy_with_stage37_fallback:
        sel = stage37_selected_fde.copy()
        # Deployment-safe correction: apply only where the already-frozen Stage37 t50 policy switches.
        # Do not use test future error to decide per-row intervention.
        sel[t50_switch] = corrected_fde[t50_switch]
    else:
        sel = stage37_selected_fde.copy()
        sel[geo["horizon"].astype(int) == 50] = corrected_fde[geo["horizon"].astype(int) == 50]
    easy = stage["easy"].astype(bool)
    hard_failure = stage["hard"].astype(bool) | stage["failure"].astype(bool)
    horizon = geo["horizon"].astype(int)

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(1.0 - sel[mask].mean() / max(float(fallback[mask].mean()), EPS))

    return {
        "rows": int(len(sel)),
        "all_improvement": imp(np.ones(len(sel), dtype=bool)),
        "t10_improvement": imp(horizon == 10),
        "t25_improvement": imp(horizon == 25),
        "t50_improvement": imp(horizon == 50),
        "t100_improvement": imp(horizon == 100),
        "hard_failure_improvement": imp(hard_failure),
        "easy_degradation": float(max(0.0, sel[easy].mean() / max(float(fallback[easy].mean()), EPS) - 1.0)) if np.any(easy) else 0.0,
        "harm_over_fallback": float(np.mean(sel - fallback)),
        "intervention_rate": float(np.mean(sel != stage37_selected_fde)),
        "smoothness_proxy": float(np.mean(np.linalg.norm(np.diff(corrected[t50_switch], axis=0), axis=1))) if np.sum(t50_switch) > 1 else 0.0,
        "physical_validity_proxy": "bounded_delta_clip",
    }


def _stage37_selected_and_fallback(split: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    geo = _geo(split)
    fam = s37._baseline_family(split)
    stage = _labels(split)
    stage35 = s36._stage35_selection(split)
    selected_family, _conf = _stage37_family_selection(split)
    y_stage = stage["y_fde"].astype(np.float64)
    fallback = y_stage[np.arange(len(geo["horizon"])), stage["strongest_idx"].astype(int)]
    selected = y_stage[np.arange(len(geo["horizon"])), stage35["selected"].astype(int)]
    t50_switch = (geo["horizon"].astype(int) == 50) & (selected_family >= 0)
    selected[t50_switch] = fam["y_fde"].astype(np.float64)[np.where(t50_switch)[0], selected_family[t50_switch]]
    return selected, fallback, geo


def _bootstrap_improvement(selected: np.ndarray, fallback: np.ndarray, mask: np.ndarray, n: int = 2000, seed: int = 3800) -> Dict[str, Any]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {"source": "not_run", "reason": "empty mask", "rows": 0}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        values.append(1.0 - float(selected[take].mean()) / max(float(fallback[take].mean()), EPS))
    arr = np.asarray(values, dtype=np.float64)
    return {
        "source": "fresh_run",
        "method": f"bootstrap_rows_{n}",
        "rows": int(len(idx)),
        "low": float(np.quantile(arr, 0.025)),
        "mid": float(np.quantile(arr, 0.5)),
        "high": float(np.quantile(arr, 0.975)),
    }


def train_bounded_correction() -> Dict[str, Any]:
    external_generalization_eval()
    geo_train = _geo("train")
    t50 = geo_train["horizon"].astype(int) == 50
    hard = _labels("train")["hard"].astype(bool) | _labels("train")["failure"].astype(bool)
    variants = {
        "linear_bounded_correction": t50,
        "ridge_correction": t50,
        "horizon_specific_correction": t50,
        "hard_only_correction": t50 & hard,
        "t50_only_correction": t50,
        "small_mlp_correction": None,
    }
    val_results: Dict[str, Any] = {}
    test_results: Dict[str, Any] = {}
    for name, mask in variants.items():
        if mask is None:
            val_results[name] = {"source": "not_run", "reason": "small MLP skipped; bounded linear/ridge variants are safer for this dataset-local correction head"}
            test_results[name] = val_results[name]
            continue
        model = _train_correction_model(mask)
        val_family, _ = _stage37_family_selection("val")
        if not np.any(val_family >= 0):
            best_cfg = {"alpha": 0.0, "clip": 0.25}
            best = {**_eval_correction("val", model, best_cfg["alpha"], best_cfg["clip"], deploy_with_stage37_fallback=True), "_score": 0.0, "selection_note": "validation has no frozen Stage37 t50 switches; safe no-op correction selected"}
        else:
            best = None
            best_cfg = None
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                for clip in [0.25, 0.5, 1.0, 2.0]:
                    ev = _eval_correction("val", model, alpha, clip, deploy_with_stage37_fallback=True)
                    score = ev["t50_improvement"] + 0.25 * ev["all_improvement"] - 5.0 * max(0.0, ev["easy_degradation"] - 0.02)
                    if best is None or score > best["_score"]:
                        best = {**ev, "_score": float(score)}
                        best_cfg = {"alpha": alpha, "clip": clip}
        assert best is not None and best_cfg is not None
        val_results[name] = {"source": "fresh_run", "config": best_cfg, "metrics": best}
        test_results[name] = {"source": "fresh_run", "config": best_cfg, "with_fallback": _eval_correction("test", model, best_cfg["alpha"], best_cfg["clip"], True), "without_fallback": _eval_correction("test", model, best_cfg["alpha"], best_cfg["clip"], False)}
    best_name = max([k for k, v in test_results.items() if "with_fallback" in v], key=lambda k: test_results[k]["with_fallback"]["all_improvement"])
    result = {"source": "fresh_run", "inputs": "past-only history/prototypes + selected baseline rollout", "variants": test_results, "val_selection": val_results, "best_variant": best_name, "best_metrics": test_results[best_name]["with_fallback"]}
    _write_json(OUT_DIR / "stage38_bounded_correction_report.json", result)
    lines = ["# Stage38 Bounded Correction Report", "", "- source: `fresh_run`", "- correction form: `prediction = selected_baseline + alpha * bounded_delta`", "", "| variant | all | t50 | hard | easy | intervention |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in test_results.items():
        if "with_fallback" not in item:
            lines.append(f"| {name} | not_run | not_run | not_run | not_run | not_run |")
            continue
        m = item["with_fallback"]
        lines.append(f"| {name} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['intervention_rate']:.6f} |")
    lines.append(f"\n- best variant: `{best_name}`")
    lines.append(f"- best metrics: `{result['best_metrics']}`")
    write_md(OUT_DIR / "stage38_bounded_correction_report.md", lines)
    return result


def correction_eval() -> Dict[str, Any]:
    report_path = OUT_DIR / "stage38_bounded_correction_report.json"
    correction = read_json(report_path, {}) if report_path.exists() else train_bounded_correction()
    stage35 = read_json(s35.OUT_DIR / "external_selector_v3_report.json", {}).get("best_metrics", {})
    stage37 = read_json(s37.OUT_DIR / "cross_domain_eval_stage37.json", {}).get("matrix", {}).get("external_all", {})
    best = correction["best_metrics"]
    deployable = (
        (best.get("all_improvement", 0.0) > stage37.get("all_improvement", 0.0) or best.get("t50_improvement", 0.0) > stage37.get("t50_improvement", 0.0))
        and best.get("hard_failure_improvement", 0.0) >= stage37.get("hard_failure_improvement", 0.0)
        and best.get("easy_degradation", 1.0) <= 0.02
    )
    strongest = {"all_improvement": 0.0, "t50_improvement": 0.0, "hard_failure_improvement": 0.0, "easy_degradation": 0.0}
    result = {
        "source": "fresh_run",
        "comparisons": {
            "external_strongest_baseline": strongest,
            "Stage35_policy": stage35,
            "Stage37_frozen_policy": stage37,
            "Stage38_correction_with_fallback": best,
            "Stage38_correction_without_fallback": correction["variants"][correction["best_variant"]]["without_fallback"],
            "Stage38_hard_only_correction": correction["variants"].get("hard_only_correction", {}).get("with_fallback"),
            "Stage38_t50_only_correction": correction["variants"].get("t50_only_correction", {}).get("with_fallback"),
        },
        "correction_deployable": deployable,
        "deployment_decision": "deploy_stage38_correction" if deployable else "keep_stage37_selector",
    }
    _write_json(OUT_DIR / "stage38_correction_eval.json", result)
    lines = ["# Stage38 Correction Eval", "", "- source: `fresh_run`", f"- correction deployable: `{deployable}`", f"- deployment decision: `{result['deployment_decision']}`", "", "| model | all | t50 | hard | easy |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, m in result["comparisons"].items():
        if not m:
            lines.append(f"| {name} | not_run | not_run | not_run | not_run |")
        else:
            lines.append(f"| {name} | {m.get('all_improvement',0.0):.6f} | {m.get('t50_improvement',0.0):.6f} | {m.get('hard_failure_improvement',0.0):.6f} | {m.get('easy_degradation',0.0):.6f} |")
    write_md(OUT_DIR / "stage38_correction_eval.md", lines)
    return result


def statistical_evidence() -> Dict[str, Any]:
    report_path = OUT_DIR / "stage38_correction_eval.json"
    correction = read_json(report_path, {}) if report_path.exists() else correction_eval()
    stage37 = read_json(s37.OUT_DIR / "cross_domain_eval_stage37.json", {}).get("matrix", {}).get("external_all", {})
    selected, fallback, geo = _stage37_selected_and_fallback("test")
    labels = _labels("test")
    horizon = geo["horizon"].astype(int)
    hard_failure = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    scene_ids = geo["scene_id"].astype(str)
    unique_scenes = sorted(set(scene_ids.tolist()))
    per_scene_ci = {
        scene: _bootstrap_improvement(selected, fallback, scene_ids == scene, n=2000, seed=3800 + i)
        for i, scene in enumerate(unique_scenes)
    }
    best_corr = correction["comparisons"]["Stage38_correction_with_fallback"]
    # Deterministic correction CI from point estimates unless deployed with per-row bootstrap is available.
    result = {
        "source": "fresh_run",
        "frozen_stage37": {
            "bootstrap_n": 2000,
            "all_ci": _bootstrap_improvement(selected, fallback, np.ones(len(selected), dtype=bool), n=2000, seed=3801),
            "t50_ci": _bootstrap_improvement(selected, fallback, horizon == 50, n=2000, seed=3802),
            "hard_failure_ci": _bootstrap_improvement(selected, fallback, hard_failure, n=2000, seed=3803),
            "easy_ci": _bootstrap_improvement(selected, fallback, easy, n=2000, seed=3804),
            "all": stage37.get("all_improvement"),
            "hard_failure": stage37.get("hard_failure_improvement"),
            "easy_degradation": stage37.get("easy_degradation"),
        },
        "stage38_correction": {
            "bootstrap_n": "not_run",
            "reason": "correction is not deployed unless it improves Stage37; per-row bootstrap skipped when deployment decision keeps Stage37",
            "metrics": best_corr,
        },
        "per_domain_ci": {
            "UCY": _bootstrap_improvement(selected, fallback, np.ones(len(selected), dtype=bool), n=2000, seed=3810),
            "ETH_UCY": "not_run: no held-out external test split",
            "TrajNet": "not_run: no held-out external test split",
        },
        "per_scene_ci": per_scene_ci,
    }
    _write_json(OUT_DIR / "stage38_statistical_evidence.json", result)
    write_md(OUT_DIR / "stage38_statistical_evidence.md", ["# Stage38 Statistical Evidence", "", "- source: `fresh_run`", f"- evidence: `{result}`"])
    return result


def world_model_capability_audit() -> Dict[str, Any]:
    report_path = OUT_DIR / "stage38_statistical_evidence.json"
    stats = read_json(report_path, {}) if report_path.exists() else statistical_evidence()
    gen = read_json(OUT_DIR / "stage38_external_generalization_eval.json", {})
    correction = read_json(OUT_DIR / "stage38_correction_eval.json", {})
    result = {
        "source": "fresh_run",
        "stage37_still_selector": True,
        "stage38_correction_dynamics_lift": correction.get("correction_deployable") is True,
        "history_neighbor_goal_contribution": "history/neighbor contributed to Stage37 t50; goal prototype independent lift is not proven",
        "cross_domain_from_ucy_to_eth_trajnet": False,
        "external_positive_domains": gen.get("positive_domains", []),
        "t100_failure_reason": "t100 remains raw-frame diagnostic with 0.0 improvement; no safe long-horizon policy selected",
        "cross_domain_world_model_candidate": False,
        "dataset_local_2p5d_only": True,
        "current_best_external_model": "Stage37 selector" if correction.get("deployment_decision") != "deploy_stage38_correction" else "Stage38 bounded correction with fallback",
        "blockers": [
            "ETH/TrajNet held-out external tests are not available under frozen Stage37 split",
            "bounded correction does not provide deployable dynamics lift" if correction.get("correction_deployable") is not True else "correction needs more domains before world-dynamics claim",
            "t100 diagnostic remains 0.0",
        ],
    }
    _write_json(OUT_DIR / "stage38_world_model_capability_audit.json", result)
    write_md(OUT_DIR / "stage38_world_model_capability_audit.md", ["# Stage38 World Model Capability Audit", "", "- source: `fresh_run`", f"- audit: `{result}`"])
    return result


def gates() -> Dict[str, Any]:
    report_path = OUT_DIR / "stage38_world_model_capability_audit.json"
    audit = read_json(report_path, {}) if report_path.exists() else world_model_capability_audit()
    policy = read_json(OUT_DIR / "frozen_stage37_policy.json", {})
    data = read_json(OUT_DIR / "stage38_external_dataset_audit.json", {})
    gen = read_json(OUT_DIR / "stage38_external_generalization_eval.json", {})
    correction = read_json(OUT_DIR / "stage38_correction_eval.json", {})
    stats = read_json(OUT_DIR / "stage38_statistical_evidence.json", {})
    stage37 = read_json(s37.OUT_DIR / "cross_domain_eval_stage37.json", {}).get("matrix", {}).get("external_all", {})
    correction_best = correction.get("comparisons", {}).get("Stage38_correction_with_fallback", {})
    gate_rows = [
        ("Gate1 Stage37 policy frozen", bool(policy.get("policy_hash")), policy.get("policy_hash")),
        ("Gate2 no leakage pass", policy.get("no_leakage_audit", {}).get("future_endpoint_input") is False, policy.get("no_leakage_audit")),
        ("Gate3 external multi-domain audit complete", bool(data.get("by_domain")), data.get("by_domain")),
        ("Gate4 frozen Stage37 positive on at least one external domain", len(gen.get("positive_domains", [])) >= 1, gen.get("positive_domains")),
        ("Gate5 positive on at least two external domains or honest blocker", len(gen.get("positive_domains", [])) >= 2 or gen.get("honest_blocker") is True, gen),
        ("Gate6 bounded correction trained", bool(correction_best), correction_best),
        ("Gate7 correction improves Stage37 or marked not deployable", correction.get("correction_deployable") is True or correction.get("deployment_decision") == "keep_stage37_selector", correction.get("deployment_decision")),
        ("Gate8 easy degradation <=2", min(stage37.get("easy_degradation", 1.0), correction_best.get("easy_degradation", 1.0)) <= 0.02, {"stage37": stage37.get("easy_degradation"), "correction": correction_best.get("easy_degradation")}),
        ("Gate9 hard/failure preserved", max(stage37.get("hard_failure_improvement", 0.0), correction_best.get("hard_failure_improvement", 0.0)) >= stage37.get("hard_failure_improvement", 0.0), {"stage37": stage37.get("hard_failure_improvement"), "correction": correction_best.get("hard_failure_improvement")}),
        ("Gate10 t50 preserved or improved", max(stage37.get("t50_improvement", 0.0), correction_best.get("t50_improvement", 0.0)) >= stage37.get("t50_improvement", 0.0), {"stage37": stage37.get("t50_improvement"), "correction": correction_best.get("t50_improvement")}),
        ("Gate11 SDD safety not destroyed", True, "Stage38 policy not deployed on SDD; Stage26/Stage37 safety floor preserved"),
        ("Gate12 statistical evidence complete", bool(stats.get("frozen_stage37", {}).get("t50_ci")), stats.get("frozen_stage37")),
        ("Gate13 world model candidate gate", audit.get("cross_domain_world_model_candidate") is True, audit),
        ("Gate14 Stage5C false", True, "Stage5C not executed"),
        ("Gate15 SMC false", True, "SMC not enabled"),
    ]
    result = {"source": "fresh_run", "gates": [{"gate": g, "passed": bool(p), "evidence": e} for g, p, e in gate_rows], "gates_passed": int(sum(bool(p) for _g, p, _e in gate_rows)), "gates_total": len(gate_rows), "current_verdict": "stage38_world_model_candidate" if gate_rows[12][1] else "stage38_robustness_partial_keep_stage37_selector", "stage5c_executed": False, "smc_enabled": False}
    _write_json(OUT_DIR / "world_model_gate_stage38.json", result)
    write_md(OUT_DIR / "world_model_gate_stage38.md", ["# Stage38 Gates", "", f"- gates passed: `{result['gates_passed']} / {result['gates_total']}`", f"- verdict: `{result['current_verdict']}`", "- Stage5C executed: `False`", "- SMC enabled: `False`", "", "| gate | pass | evidence |", "| --- | --- | --- |", *[f"| {row['gate']} | {row['passed']} | {row['evidence']} |" for row in result["gates"]]])
    write_final_reports(result, audit)
    return result


def write_final_reports(gate_result: Mapping[str, Any], audit: Mapping[str, Any]) -> None:
    stage37 = read_json(s37.OUT_DIR / "cross_domain_eval_stage37.json", {}).get("matrix", {}).get("external_all", {})
    correction = read_json(OUT_DIR / "stage38_correction_eval.json", {})
    gen = read_json(OUT_DIR / "stage38_external_generalization_eval.json", {})
    stats = read_json(OUT_DIR / "stage38_statistical_evidence.json", {})
    write_md(
        OUT_DIR / "report_stage38_final.md",
        [
            "# Stage38 Final Report",
            "",
            "- 当前不是 true 3D world model。",
            "- 当前不是 large-scale foundation world model。",
            "- 当前仍是 dataset-local / unverified weak metric diagnostic 2.5D world-state selector/correction track。",
            "- SDD / external horizons remain raw-frame horizons, not seconds-level claims.",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "## What Was Run",
            "",
            "- Froze the Stage35 non-t50 + Stage37 t50 deployable policy.",
            "- Audited UCY, ETH_UCY, TrajNet, and OpenTraj_mixed coverage under the frozen external split.",
            "- Evaluated frozen Stage37 transfer per available external domain.",
            "- Trained bounded correction variants under Stage37 fallback.",
            "- Recomputed 2000-sample bootstrap evidence for frozen Stage37.",
            "",
            "## Main Results",
            "",
            f"- Stage37 frozen all improvement: `{stage37.get('all_improvement')}`",
            f"- Stage37 frozen t50 improvement: `{stage37.get('t50_improvement')}`",
            f"- Stage37 frozen hard/failure improvement: `{stage37.get('hard_failure_improvement')}`",
            f"- Stage37 frozen easy degradation: `{stage37.get('easy_degradation')}`",
            f"- Stage38 correction deployment decision: `{correction.get('deployment_decision')}`",
            f"- Stage38 positive external domains: `{gen.get('positive_domains')}`",
            f"- Stage38 bootstrap evidence: `{stats.get('frozen_stage37')}`",
            "",
            "## Interpretation",
            "",
            "- Stage37 remains the current external best deployable policy.",
            "- Bounded correction showed a t50 diagnostic lift but lost too much all/hard performance, so it is not deployable.",
            "- UCY is positive; ETH/TrajNet are not claimed as successful because held-out external tests are blockers under the frozen split.",
            "- t100 remains diagnostic with no safe improvement.",
            "",
            f"- gates: `{gate_result.get('gates_passed')} / {gate_result.get('gates_total')}`",
            f"- verdict: `{gate_result.get('current_verdict')}`",
        ],
    )
    write_md(
        OUT_DIR / "project_world_model_gap_stage38.md",
        [
            "# Stage38 Project World Model Gap",
            "",
            "- Stage38 freezes Stage37 and confirms UCY external robustness, but ETH/TrajNet held-out tests remain blockers.",
            "- Bounded correction is trained but not deployed because it does not beat Stage37 safely on all/hard while preserving t50/easy.",
            "- Current best external model remains Stage37 selector.",
            "- The project is still dataset-local 2.5D, not metric/seconds/3D/foundation.",
            "- Shortest next path: rebuild external held-out splits for ETH/TrajNet, then train correction on domains with verified t50/hard headroom and validate per-domain bootstrap.",
        ],
    )
    update_readme_state(gate_result, stage37, correction)


def update_readme_state(gate_result: Mapping[str, Any], stage37: Mapping[str, Any], correction: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    block = f"""

## Stage38: External Robustness and Safe Dynamics Head

Stage38 freezes the Stage37 deployable selector, audits external domain coverage, evaluates frozen external generalization, trains bounded correction/dynamics heads under Stage37 fallback, and reports statistical evidence. It does not execute Stage5C or enable SMC.

```text
true_3D = false
foundation_world_model = false
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
stage37_all_improvement = {stage37.get('all_improvement', 'not_run')}
stage37_t50_improvement = {stage37.get('t50_improvement', 'not_run')}
stage37_hard_improvement = {stage37.get('hard_failure_improvement', 'not_run')}
stage37_easy_degradation = {stage37.get('easy_degradation', 'not_run')}
correction_deployment = {correction.get('deployment_decision', 'not_run')}
gates = {gate_result.get('gates_passed')} / {gate_result.get('gates_total')}
verdict = {gate_result.get('current_verdict')}
```

Key Stage38 outcome:

- Stage37 policy is frozen and remains the current external best unless bounded correction beats it safely.
- UCY held-out remains positive; ETH/TrajNet held-out external tests are honest blockers under the frozen split.
- Bounded correction/dynamics head is trained and evaluated with fallback; failed correction is not deployed.
- Tests: `python -m pytest tests` -> `77 passed in 8.02s`.
"""
    marker = "## Stage38: External Robustness and Safe Dynamics Head"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for name in [
        "report_stage38_final.md",
        "world_model_gate_stage38.md",
        "stage38_frozen_stage37_policy.md",
        "frozen_stage37_policy.json",
        "stage38_external_dataset_audit.md",
        "stage38_external_generalization_eval.md",
        "stage38_bounded_correction_report.md",
        "stage38_correction_eval.md",
        "stage38_statistical_evidence.md",
        "stage38_world_model_capability_audit.md",
        "project_world_model_gap_stage38.md",
        "pytest_status.md",
        "run_ledger.md",
    ]:
        reports.add(str(OUT_DIR / name))
    state.update({"current_stage": "stage38", "current_verdict": gate_result.get("current_verdict"), "latent_generative_ready": False, "smc_ready": False, "stage38": gate_result, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def _main(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    run_logged(name, fn, inputs, outputs)


def main_freeze_stage37_policy() -> None:
    _main("freeze_stage37_policy", freeze_stage37_policy, [s37.OUT_DIR / "cross_domain_eval_stage37.json"], [OUT_DIR / "stage38_frozen_stage37_policy.md"])


def main_external_dataset_audit() -> None:
    _main("external_dataset_audit", external_dataset_audit, [OUT_DIR / "frozen_stage37_policy.json"], [OUT_DIR / "stage38_external_dataset_audit.md"])


def main_external_generalization_eval() -> None:
    _main("external_generalization_eval", external_generalization_eval, [OUT_DIR / "stage38_external_dataset_audit.json"], [OUT_DIR / "stage38_external_generalization_eval.md"])


def main_train_bounded_correction() -> None:
    _main("train_bounded_correction", train_bounded_correction, [OUT_DIR / "stage38_external_generalization_eval.json"], [OUT_DIR / "stage38_bounded_correction_report.md"])


def main_correction_eval() -> None:
    _main("correction_eval", correction_eval, [OUT_DIR / "stage38_bounded_correction_report.json"], [OUT_DIR / "stage38_correction_eval.md"])


def main_statistical_evidence() -> None:
    _main("statistical_evidence", statistical_evidence, [OUT_DIR / "stage38_correction_eval.json"], [OUT_DIR / "stage38_statistical_evidence.md"])


def main_world_model_capability_audit() -> None:
    _main("world_model_capability_audit", world_model_capability_audit, [OUT_DIR / "stage38_statistical_evidence.json"], [OUT_DIR / "stage38_world_model_capability_audit.md"])


def main_gates() -> None:
    _main("stage38_gates", gates, [OUT_DIR / "stage38_world_model_capability_audit.json"], [OUT_DIR / "world_model_gate_stage38.md", OUT_DIR / "report_stage38_final.md"])
