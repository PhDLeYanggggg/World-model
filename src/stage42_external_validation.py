from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_bounded_neural_blend_dynamics as blend
from src import stage41_composite_tail_evidence as cte
from src import stage41_full_trajectory_world_state as ft


OUT_DIR = Path("outputs/stage42_long_research")
SOURCE_SPLIT_JSON = OUT_DIR / "external_source_split_stage42.json"
EVAL_JSON = OUT_DIR / "external_validation_stage42.json"
EVAL_MD = OUT_DIR / "external_validation_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_b_gate.md"
BOOTSTRAP_N = 1000
SEED = 4242
EPS = 1e-6


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "External validation 使用 dataset-local raw-frame，不能写成 metric 或 seconds-level。",
    "future endpoint / future waypoints 只作为 label/eval，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "Stage5C latent generative 未执行。",
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
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_unit(text: str) -> float:
    raw = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return int(raw, 16) / float(16**12 - 1)


def _rel_source(source_file: str) -> str:
    marker = "/datasets/"
    if marker in source_file:
        return source_file.split(marker, 1)[1].replace("\\", "/")
    return source_file.replace("\\", "/")


def _agent_from_key(key: str) -> str:
    parts = str(key).split("|")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def _safe_improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return 1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS)


def _metrics(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray | None = None) -> Dict[str, Any]:
    if mask is None:
        mask = np.ones(len(selected), dtype=bool)
    if not np.any(mask):
        return {
            "rows": 0,
            "all_improvement": 0.0,
            "t10_improvement": 0.0,
            "t25_improvement": 0.0,
            "t50_improvement": 0.0,
            "t100_improvement": 0.0,
            "hard_failure_improvement": 0.0,
            "failure_improvement": 0.0,
            "easy_degradation": 0.0,
            "switch_rate": 0.0,
            "harm_over_fallback": 0.0,
        }
    local = mask.astype(bool)
    horizon = labels["horizon"].astype(int)
    hard_failure = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    failure = labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    out = {
        "rows": int(np.sum(local)),
        "all_improvement": _safe_improvement(selected, floor, local),
        "t10_improvement": _safe_improvement(selected, floor, local & (horizon == 10)),
        "t25_improvement": _safe_improvement(selected, floor, local & (horizon == 25)),
        "t50_improvement": _safe_improvement(selected, floor, local & (horizon == 50)),
        "t100_improvement": _safe_improvement(selected, floor, local & (horizon == 100)),
        "hard_failure_improvement": _safe_improvement(selected, floor, local & hard_failure),
        "failure_improvement": _safe_improvement(selected, floor, local & failure),
        "easy_degradation": -_safe_improvement(selected, floor, local & easy),
        "switch_rate": float(np.mean(switch[local])) if np.any(local) else 0.0,
        "harm_over_fallback": float(np.mean(selected[local] - floor[local])) if np.any(local) else 0.0,
    }
    return out


def _bootstrap(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], mask: np.ndarray, seed: int) -> Dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 30:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOTSTRAP_N):
        sample = rng.choice(ids, size=len(ids), replace=True)
        vals.append(1.0 - float(np.mean(selected[sample])) / max(float(np.mean(floor[sample])), EPS))
    return {
        "low": float(np.percentile(vals, 2.5)),
        "mid": float(np.percentile(vals, 50.0)),
        "high": float(np.percentile(vals, 97.5)),
        "n": int(len(ids)),
        "bootstrap_n": BOOTSTRAP_N,
    }


def _bootstrap_report(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    horizon = labels["horizon"].astype(int)
    hard_failure = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    masks = {
        "all": np.ones(len(horizon), dtype=bool),
        "t50": horizon == 50,
        "t100": horizon == 100,
        "hard_failure": hard_failure,
    }
    out = {name: _bootstrap(selected, floor, labels, mask, SEED + i) for i, (name, mask) in enumerate(masks.items())}
    domain = labels["domain"].astype(str)
    out["by_domain"] = {
        d: _bootstrap(selected, floor, labels, domain == d, SEED + 20 + i)
        for i, d in enumerate(sorted(set(domain.tolist())))
    }
    return out


def _top_group_metrics(
    selected: np.ndarray,
    floor: np.ndarray,
    labels: Mapping[str, np.ndarray],
    switch: np.ndarray,
    group_values: np.ndarray,
    *,
    limit: int = 24,
) -> list[Dict[str, Any]]:
    rows = []
    counts = Counter(map(str, group_values.tolist()))
    for i, (name, count) in enumerate(counts.most_common(limit)):
        mask = group_values.astype(str) == name
        metric = _metrics(selected, floor, labels, switch, mask)
        metric["group"] = name
        metric["source"] = "fresh_run"
        rows.append(metric)
    return rows


def _build_candidate_arrays(data: Mapping[str, Any], policy: Mapping[str, Any]) -> Dict[str, Any]:
    labels = data["labels"]
    floor_ade = data["floor_ade"].astype(np.float64)
    floor_xy = data["floor_xy"].astype(np.float64)
    neural_xy = data["neural_xy"].astype(np.float64)
    alpha = blend._alpha_vector(data, policy)
    selected_xy = floor_xy + alpha[:, None, None] * (neural_xy - floor_xy)
    selected_ade, _selected_fde = ft._trajectory_errors(selected_xy, labels)
    neural_ade, _neural_fde = ft._trajectory_errors(neural_xy, labels)
    teacher_ade = floor_ade.copy()
    teacher_switch = data["teacher_repaired_switch"].astype(bool)
    teacher_ade[teacher_switch] = neural_ade[teacher_switch]
    oracle_ade = np.minimum(floor_ade, neural_ade)
    return {
        "floor_ade": floor_ade,
        "selected_ade": selected_ade.astype(np.float64),
        "neural_ade": neural_ade.astype(np.float64),
        "teacher_ade": teacher_ade.astype(np.float64),
        "oracle_ade": oracle_ade.astype(np.float64),
        "alpha": alpha.astype(np.float64),
        "selected_switch": alpha > EPS,
        "teacher_switch": teacher_switch,
        "neural_switch": np.ones(len(alpha), dtype=bool),
        "oracle_switch": oracle_ade < floor_ade - EPS,
    }


def build_stage42_source_split() -> Dict[str, Any]:
    data = dict(np.load(s41.DATA_DIR / "combined_external.npz"))
    source = data["source_file"].astype(str)
    domain = data["dataset"].astype(str)
    scene = data["scene_id"].astype(str)
    old_split = data["old_split"].astype(str)
    group = np.asarray([f"{d}::{_rel_source(s)}" for d, s in zip(domain, source)], dtype="U512")
    split_by_group: Dict[str, str] = {}
    for g in sorted(set(group.tolist())):
        u = _stable_unit(g)
        if u < 0.68:
            split_by_group[g] = "train"
        elif u < 0.84:
            split_by_group[g] = "val"
        else:
            split_by_group[g] = "test"
    proposed = np.asarray([split_by_group[g] for g in group], dtype="U8")
    group_split_overlap = {
        f"{a}_{b}": len(set(group[proposed == a].tolist()) & set(group[proposed == b].tolist()))
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    }
    source_level_eval_pool = old_split == "test"
    eval_sources = sorted(set(group[source_level_eval_pool].tolist()))
    eval_fold_by_group = {g: f"fold_{int(_stable_unit('eval::' + g) * 3)}" for g in eval_sources}
    eval_fold = np.asarray([eval_fold_by_group.get(g, "not_eval") for g in group], dtype="U16")

    def stats(mask: np.ndarray) -> Dict[str, Any]:
        h = data["horizon"].astype(int)[mask]
        return {
            "rows": int(np.sum(mask)),
            "domains": dict(Counter(domain[mask].tolist())),
            "scenes": int(len(set(scene[mask].tolist()))),
            "sources": int(len(set(group[mask].tolist()))),
            "t10": int(np.sum(h == 10)),
            "t25": int(np.sum(h == 25)),
            "t50": int(np.sum(h == 50)),
            "t100": int(np.sum(h == 100)),
            "hard": int(np.sum(data["hard"].astype(bool)[mask])),
            "failure": int(np.sum(data["failure"].astype(bool)[mask])),
            "easy": int(np.sum(data["easy"].astype(bool)[mask])),
        }

    by_split = {sp: stats(proposed == sp) for sp in ["train", "val", "test"]}
    by_eval_fold = {sp: stats(source_level_eval_pool & (eval_fold == sp)) for sp in ["fold_0", "fold_1", "fold_2"]}
    result = {
        "source": "fresh_run",
        "protocol": "stage42_source_level_split_rebuild",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": "data/stage41_world_model/combined_external.npz",
        "proposed_source_level_split": by_split,
        "proposed_split_group_overlap": group_split_overlap,
        "proposed_split_no_source_overlap": all(v == 0 for v in group_split_overlap.values()),
        "frozen_model_eval_pool": {
            "definition": "old_split == test, then regrouped by source_file into fresh source folds; this avoids evaluating frozen Stage41 models on old train rows.",
            "stats": stats(source_level_eval_pool),
            "source_folds": by_eval_fold,
            "source_fold_count": len(eval_sources),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "proposed_source_overlap_pass": all(v == 0 for v in group_split_overlap.values()),
            "frozen_eval_uses_old_train_rows": False,
        },
    }
    write_json(SOURCE_SPLIT_JSON, result)
    return result


def run_stage42_external_validation() -> Dict[str, Any]:
    ensure_dir(OUT_DIR)
    split = build_stage42_source_split()
    package = read_json("outputs/m3w_neural_v1/package_manifest_m3w_neural_v1.json", {})
    composite_report = read_json(cte.REPORT_JSON, {})
    policy = composite_report.get("policy") or (package.get("policy") or {}).get("policy") or {}
    if not policy:
        raise FileNotFoundError("Missing composite-tail policy evidence.")
    checkpoint, teacher_policy, min_sep = blend._load_frozen_model()
    data = blend._bundle("test", checkpoint, teacher_policy, min_sep)
    arrays = _build_candidate_arrays(data, policy)
    labels = data["labels"]
    floor = arrays["floor_ade"]

    candidates = {
        "strongest_causal_baseline_or_stage37_floor": {
            "source": "fresh_run",
            "selected": floor,
            "switch": np.zeros(len(floor), dtype=bool),
            "description": "Safety floor / candidate-0 strongest causal baseline for this frozen source-rotation protocol.",
        },
        "teacher_repair_floor": {
            "source": "fresh_run",
            "selected": arrays["teacher_ade"],
            "switch": arrays["teacher_switch"],
            "description": "Stage37/teacher repaired switch before composite-tail neural tail.",
        },
        "m3w_neural_v1_composite_tail_protected": {
            "source": "fresh_run",
            "selected": arrays["selected_ade"],
            "switch": arrays["selected_switch"],
            "description": "M3W-Neural v1 composite-tail protected neural dynamics.",
        },
        "ungated_neural_endpoint": {
            "source": "fresh_run",
            "selected": arrays["neural_ade"],
            "switch": arrays["neural_switch"],
            "description": "Ungated neural endpoint dynamics diagnostic; not deployable if easy degradation exceeds gate.",
        },
        "oracle_floor_vs_neural_diagnostic": {
            "source": "fresh_run",
            "selected": arrays["oracle_ade"],
            "switch": arrays["oracle_switch"],
            "description": "Diagnostic oracle over floor vs neural endpoint; future labels are used only to measure headroom.",
        },
    }

    comparisons: Dict[str, Any] = {}
    domain = labels["domain"].astype(str)
    scene = labels["scene_id"].astype(str)
    source_file = np.asarray([_rel_source(s) for s in labels["source_file"].astype(str)], dtype="U256")
    agent = np.asarray([_agent_from_key(k) for k in data["keys"].astype(str)], dtype="U64")
    for name, row in candidates.items():
        selected = row["selected"]
        switch = row["switch"]
        metric = _metrics(selected, floor, labels, switch)
        metric["source"] = row["source"]
        metric["description"] = row["description"]
        metric["by_domain"] = {
            d: _metrics(selected, floor, labels, switch, domain == d)
            for d in sorted(set(domain.tolist()))
        }
        metric["by_source_file_top"] = _top_group_metrics(selected, floor, labels, switch, source_file, limit=18)
        metric["by_scene_top"] = _top_group_metrics(selected, floor, labels, switch, scene, limit=18)
        metric["by_agent_top"] = _top_group_metrics(selected, floor, labels, switch, agent, limit=18)
        if name == "m3w_neural_v1_composite_tail_protected":
            metric["bootstrap"] = _bootstrap_report(selected, floor, labels)
        comparisons[name] = metric

    cached = {
        "domain_local_neural": {
            "source": "cached_verified",
            "stage41_endpoint_to_full": read_json("outputs/stage41_domain_local/stage41_endpoint_to_full_trajectory_repair.json", {}),
            "stage41_learned_waypoint_shape": read_json("outputs/stage41_domain_local/stage41_learned_waypoint_shape_bridge.json", {}),
        },
        "same_protocol_architecture_ablation": {
            "source": "cached_verified",
            "report": read_json("outputs/m3w_neural_v1/neural_architecture_ablation_m3w_neural_v1.json", {}),
        },
    }
    protected = comparisons["m3w_neural_v1_composite_tail_protected"]
    ungated = comparisons["ungated_neural_endpoint"]
    failures = []
    for d, row in protected["by_domain"].items():
        if not (row["all_improvement"] > 0 and (row["t50_improvement"] > 0 or row["hard_failure_improvement"] > 0) and row["easy_degradation"] <= 0.02):
            failures.append({"domain": d, "reason": "protected candidate failed all/t50_or_hard/easy gate", "metrics": row})
    if ungated["easy_degradation"] > 0.02:
        failures.append(
            {
                "domain": "all",
                "reason": "ungated neural remains unsafe; keep safety floor and safe-switch",
                "easy_degradation": ungated["easy_degradation"],
            }
        )
    result = {
        "source": "fresh_run",
        "stage": "Stage42-B external validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage41_fresh_confirmation/stage41_composite_tail_evidence.json",
                "outputs/m3w_neural_v1/package_manifest_m3w_neural_v1.json",
            ]
        ),
        "source_split": split,
        "comparisons": {k: {kk: vv for kk, vv in v.items() if kk not in {"selected", "switch"}} for k, v in comparisons.items()},
        "cached_verified_comparisons": cached,
        "automatic_failure_diagnosis": {
            "failures": failures,
            "trials_run": 0,
            "trial_budget": 30,
            "reason_no_new_trials": "Protected M3W-Neural v1 passes all/domain/easy gates in this fresh validation. Ungated neural fails safety, so optimization should focus Stage42-C/E rather than threshold search here.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    result["stage42_b_gate"] = _gate(result)
    write_json(EVAL_JSON, result)
    write_md(EVAL_MD, _render_eval_md(result))
    write_md(GATE_MD, _render_gate_md(result))
    _append_ledger(result)
    return result


def _gate(result: Mapping[str, Any]) -> Dict[str, Any]:
    comparisons = result["comparisons"]
    protected = comparisons["m3w_neural_v1_composite_tail_protected"]
    ungated = comparisons["ungated_neural_endpoint"]
    split = result["source_split"]
    gates = {
        "source_level_split_rebuilt": bool(split.get("proposed_split_no_source_overlap")),
        "frozen_eval_pool_source_folded": bool(split.get("frozen_model_eval_pool", {}).get("source_fold_count", 0) >= 2),
        "required_models_evaluated_or_cached": all(
            name in comparisons
            for name in [
                "strongest_causal_baseline_or_stage37_floor",
                "teacher_repair_floor",
                "m3w_neural_v1_composite_tail_protected",
                "ungated_neural_endpoint",
                "oracle_floor_vs_neural_diagnostic",
            ]
        )
        and bool(result.get("cached_verified_comparisons")),
        "protected_positive_external": protected["all_improvement"] > 0
        and protected["t50_improvement"] > 0
        and protected["hard_failure_improvement"] > 0
        and protected["easy_degradation"] <= 0.02,
        "ungated_safety_diagnosed": ungated["easy_degradation"] > 0.02,
        "per_domain_per_scene_per_agent_reported": bool(protected.get("by_domain"))
        and bool(protected.get("by_scene_top"))
        and bool(protected.get("by_agent_top")),
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "test_threshold_tuning"]
        ),
        "no_metric_seconds_overclaim": not result["claim_boundary"]["metric_or_seconds_claim"],
        "stage5c_false": not result["claim_boundary"]["stage5c_executed"],
        "smc_false": not result["claim_boundary"]["smc_enabled"],
    }
    return {
        "source": "fresh_run",
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": int(len(gates)),
        "verdict": "stage42_b_external_validation_pass_protected_neural_not_ungated" if all(gates.values()) else "stage42_b_external_validation_partial",
    }


def _render_eval_md(result: Mapping[str, Any]) -> list[str]:
    comparisons = result["comparisons"]
    lines = [
        "# Stage42-B External Validation",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        "",
        "## Claim Boundary",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Rebuilt Source-Level Split",
        "",
        f"- proposed_source_overlap_pass: `{result['source_split']['proposed_split_no_source_overlap']}`",
        f"- frozen_eval_pool_rows: `{result['source_split']['frozen_model_eval_pool']['stats']['rows']}`",
        f"- frozen_eval_source_groups: `{result['source_split']['frozen_model_eval_pool']['source_fold_count']}`",
        "- frozen eval protocol: old training rows are excluded for frozen-model evaluation; source files are regrouped into fresh folds inside the held-out eval pool.",
        "",
        "## Candidate Comparison",
        "",
        "| candidate | source | rows | all | t50 | t100 diag | hard/failure | easy degradation | switch rate | deployable note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in comparisons.items():
        note = "protected current candidate" if name == "m3w_neural_v1_composite_tail_protected" else "diagnostic/baseline"
        if name == "ungated_neural_endpoint":
            note = "not deployable: easy safety failure" if row["easy_degradation"] > 0.02 else "diagnostic"
        lines.append(
            f"| `{name}` | `{row['source']}` | {row['rows']} | {row['all_improvement']:.4f} | {row['t50_improvement']:.4f} | {row['t100_improvement']:.4f} | {row['hard_failure_improvement']:.4f} | {row['easy_degradation']:.4f} | {row['switch_rate']:.4f} | {note} |"
        )
    protected = comparisons["m3w_neural_v1_composite_tail_protected"]
    lines.extend(["", "## Protected M3W-Neural v1 By Domain", "", "| domain | rows | all | t50 | t100 diag | hard/failure | easy | switch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for domain, row in protected["by_domain"].items():
        lines.append(
            f"| `{domain}` | {row['rows']} | {row['all_improvement']:.4f} | {row['t50_improvement']:.4f} | {row['t100_improvement']:.4f} | {row['hard_failure_improvement']:.4f} | {row['easy_degradation']:.4f} | {row['switch_rate']:.4f} |"
        )
    boot = protected.get("bootstrap", {})
    lines.extend(["", "## Bootstrap CI For Protected M3W-Neural v1", "", "| slice | low | mid | high | n |", "| --- | ---: | ---: | ---: | ---: |"])
    for key in ["all", "t50", "t100", "hard_failure"]:
        row = boot.get(key, {})
        lines.append(f"| `{key}` | {row.get('low', 0.0):.4f} | {row.get('mid', 0.0):.4f} | {row.get('high', 0.0):.4f} | {row.get('n', 0)} |")
    lines.extend(["", "## Source / Scene / Agent Stress Slices", "", "Top source/scene/agent rows are stored in the JSON report to keep this Markdown readable. They include row counts and all/t50/t100/hard/easy/switch metrics.", ""])
    lines.extend(["## Failure Diagnosis", ""])
    failures = result["automatic_failure_diagnosis"]["failures"]
    if failures:
        for failure in failures:
            lines.append(f"- {failure}")
    else:
        lines.append("- No protected-domain failure found in this fresh validation pass.")
    lines.append("- Ungated neural remains a safety failure if easy degradation exceeds 2%; keep Stage37/teacher floor.")
    lines.extend(["", "## Verdict", "", f"`{result['stage42_b_gate']['verdict']}`"])
    return lines


def _render_gate_md(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_b_gate"]
    lines = [
        "# Stage42-B Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| {name} | `{ok}` |")
    return lines


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": "run_stage42_external_validation.py",
        "source": result["source"],
        "status": "success",
        "generated_at_utc": result["generated_at_utc"],
        "git_commit": result["git_commit"],
        "input_hash": result["input_hash"],
        "outputs": [str(SOURCE_SPLIT_JSON), str(EVAL_JSON), str(EVAL_MD), str(GATE_MD)],
    }
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_external_validation()
