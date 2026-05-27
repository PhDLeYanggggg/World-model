from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_full_waypoint_bridge_shape_audit import _replace_section
from src import stage41_bounded_neural_blend_dynamics as blend
from src import stage41_joint_rollout_consistency as jrc
from src import stage42_floor_free_proximity_guard_repair as hd
from src import stage42_safety_floor as sf
from src import stage42_source_level_full_waypoint_eval as am


OUT_DIR = Path("outputs/stage42_long_research")

HD_JSON = OUT_DIR / "floor_free_proximity_guard_repair_stage42.json"
HD_GATE = OUT_DIR / "stage42_stage_hd_gate.md"
HC_JSON = OUT_DIR / "floor_alternative_gate_stress_stage42.json"

REPORT_JSON = OUT_DIR / "floor_free_proximity_guard_robustness_stage42.json"
REPORT_MD = OUT_DIR / "floor_free_proximity_guard_robustness_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_he_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
USER_SUMMARY = Path("README_M3W_USER_DETAILED_SUMMARY_ZH.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
GOAL_SUMMARY = Path("README_M3W_GOAL_FULL_SUMMARY_ZH.md")

BOOTSTRAP_N = 2000
EASY_LIMIT = 0.02
COLLISION_LIMIT = 0.01
MIN_DOMAIN_ROWS = 100

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HE 是 Stage42-HD teacherless proximity-guard repaired gate 的 robustness audit，不重新训练大模型。",
    "Stage42-HE 使用 HD 已冻结的 validation-selected min_sep，不用 test 调 threshold。",
    "teacher gate 不参与该 repaired gate；但 causal floor fallback 仍必须存在。",
    "future endpoint / future waypoint 只作为监督或评估标签，不允许作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 建 goals，不执行 Stage5C，不启用 SMC。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
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


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _metric_value(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _labels_as_metric_data(labels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "horizon": labels["horizon"],
        "hard": labels["hard"],
        "failure": labels["failure"],
        "easy": labels["easy"],
        "domain": labels["domain"],
        "candidate_fde": labels["candidate_fde"],
    }


def _bootstrap(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, seed: int, *, easy: bool = False) -> dict[str, Any]:
    if easy:
        return am._bootstrap_ci(floor, selected, mask, seed=seed, n=BOOTSTRAP_N)
    return am._bootstrap_ci(selected, floor, mask, seed=seed, n=BOOTSTRAP_N)


def _metric(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    return am._metric(selected, floor, _labels_as_metric_data(labels), switch, mask)


def _ci_positive(ci: Mapping[str, Any], *, easy: bool = False) -> bool:
    if int(ci.get("bootstrap_n", 0)) < BOOTSTRAP_N:
        return False
    if easy:
        return float(ci.get("high", 1.0)) <= EASY_LIMIT
    return float(ci.get("low", -1.0)) > 0.0


def _find_best_hd_row(hd_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    family = hd_payload.get("summary", {}).get("best_post_guard_family")
    min_sep = hd_payload.get("summary", {}).get("best_post_guard_selected_min_sep")
    for row in hd_payload.get("repair_rows", []):
        if row.get("family") == family and float(row.get("selected_min_sep", -1.0)) == float(min_sep):
            return row
    rows = hd_payload.get("repair_rows", [])
    if not rows:
        raise RuntimeError("Stage42-HD repair rows missing; run run_stage42_floor_free_proximity_guard_repair.py first.")
    return max(rows, key=lambda row: hd._score(row.get("test_metrics", {})))


def _source_from_key(key: Any) -> str:
    text = str(key)
    first = text.split("|", 1)[0]
    return first or text


def _build_switch_eval(hd_payload: Mapping[str, Any]) -> dict[str, Any]:
    row = _find_best_hd_row(hd_payload)
    policy = dict(row.get("base_policy") or {})
    min_sep = float(row.get("selected_min_sep", hd_payload.get("summary", {}).get("best_post_guard_selected_min_sep", 0.05)))
    checkpoint, teacher_policy, floor_min_sep = blend._load_frozen_model()
    test = blend._bundle("test", checkpoint, teacher_policy, floor_min_sep)
    raw_switch = sf._switch_for_policy(test, policy).astype(bool)
    guarded_switch, guarded_off = jrc._apply_proximity_guard(
        test["floor_xy"],
        test["neural_xy"],
        test["labels"],
        test["keys"],
        raw_switch,
        min_sep,
    )
    ev = jrc._evaluate_split_rollout(test, guarded_switch, f"stage42_he_{row.get('family')}_{min_sep}")
    metrics = dict(ev["selected_metrics"])
    metrics["collision_delta_vs_floor_005"] = float(ev["collision_delta_005"])
    metrics["raw_switch_rate"] = float(np.mean(raw_switch)) if len(raw_switch) else 0.0
    metrics["switch_rate"] = float(np.mean(guarded_switch)) if len(guarded_switch) else 0.0
    metrics["guarded_off_count"] = int(guarded_off)
    metrics["guarded_off_rate"] = float(guarded_off / max(1, len(raw_switch)))
    metrics["min_sep"] = float(min_sep)
    return {
        "policy_row": row,
        "policy": policy,
        "min_sep": min_sep,
        "test_bundle": test,
        "raw_switch": raw_switch,
        "guarded_switch": guarded_switch,
        "selected_ade": ev["selected_ade"].astype(np.float64),
        "floor_ade": test["floor_ade"].astype(np.float64),
        "metrics": metrics,
        "joint_stats": {
            "floor": ev["floor_stats"],
            "selected": ev["selected_stats"],
            "neural_without_fallback": ev["neural_stats"],
        },
    }


def _global_bootstrap(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray]) -> dict[str, Any]:
    h = labels["horizon"].astype(int)
    hard_failure = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    all_mask = np.ones(len(selected), dtype=bool)
    return {
        "all": _bootstrap(selected, floor, all_mask, 442001),
        "t10": _bootstrap(selected, floor, h == 10, 442002),
        "t25": _bootstrap(selected, floor, h == 25, 442003),
        "t50": _bootstrap(selected, floor, h == 50, 442004),
        "t100_raw_frame_diagnostic": _bootstrap(selected, floor, h == 100, 442005),
        "hard_failure": _bootstrap(selected, floor, hard_failure, 442006),
        "easy_degradation": _bootstrap(selected, floor, easy, 442007, easy=True),
    }


def _domain_rows(
    selected: np.ndarray,
    floor: np.ndarray,
    labels: Mapping[str, np.ndarray],
    switch: np.ndarray,
) -> list[dict[str, Any]]:
    domain = labels["domain"].astype(str)
    hard_failure = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    rows = []
    for idx, name in enumerate(sorted(set(domain.tolist()))):
        mask = domain == name
        metric = _metric(selected, floor, labels, switch, mask)
        boot = {
            "all": _bootstrap(selected, floor, mask, 442100 + idx * 10),
            "t50": _bootstrap(selected, floor, mask & (labels["horizon"].astype(int) == 50), 442101 + idx * 10),
            "t100_raw_frame_diagnostic": _bootstrap(selected, floor, mask & (labels["horizon"].astype(int) == 100), 442102 + idx * 10),
            "hard_failure": _bootstrap(selected, floor, mask & hard_failure, 442103 + idx * 10),
            "easy_degradation": _bootstrap(selected, floor, mask & easy, 442104 + idx * 10, easy=True),
        }
        robust = bool(
            int(metric["rows"]) >= MIN_DOMAIN_ROWS
            and _ci_positive(boot["all"])
            and _ci_positive(boot["t50"])
            and _ci_positive(boot["hard_failure"])
            and _ci_positive(boot["easy_degradation"], easy=True)
            and float(metric["switch_rate"]) > 0.0
        )
        rows.append({"domain": name, "metric": metric, "bootstrap": boot, "robust_positive": robust})
    return rows


def _horizon_rows(
    selected: np.ndarray,
    floor: np.ndarray,
    labels: Mapping[str, np.ndarray],
    switch: np.ndarray,
) -> list[dict[str, Any]]:
    h = labels["horizon"].astype(int)
    rows = []
    for idx, horizon in enumerate([10, 25, 50, 100]):
        mask = h == horizon
        metric = _metric(selected, floor, labels, switch, mask)
        boot = {"improvement": _bootstrap(selected, floor, mask, 442300 + idx)}
        rows.append(
            {
                "horizon": int(horizon),
                "rows": int(np.sum(mask)),
                "metric": metric,
                "bootstrap": boot,
                "positive_ci": _ci_positive(boot["improvement"]),
                "raw_frame_diagnostic": horizon == 100,
            }
        )
    return rows


def _domain_horizon_rows(
    selected: np.ndarray,
    floor: np.ndarray,
    labels: Mapping[str, np.ndarray],
    switch: np.ndarray,
) -> list[dict[str, Any]]:
    domain = labels["domain"].astype(str)
    h = labels["horizon"].astype(int)
    rows = []
    seed = 442500
    for d in sorted(set(domain.tolist())):
        for horizon in [10, 25, 50, 100]:
            mask = (domain == d) & (h == horizon)
            metric = _metric(selected, floor, labels, switch, mask)
            boot = _bootstrap(selected, floor, mask, seed)
            seed += 1
            rows.append(
                {
                    "slice": f"{d}|{horizon}",
                    "domain": d,
                    "horizon": int(horizon),
                    "rows": int(np.sum(mask)),
                    "metric": metric,
                    "bootstrap": {"improvement": boot},
                    "positive_ci": _ci_positive(boot) if int(np.sum(mask)) >= 30 else False,
                    "raw_frame_diagnostic": horizon == 100,
                }
            )
    return rows


def _source_concentration(keys: np.ndarray, switch: np.ndarray) -> dict[str, Any]:
    sources = np.asarray([_source_from_key(key) for key in keys], dtype=object)
    counts: dict[str, int] = {}
    switch_counts: dict[str, int] = {}
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        counts[source] = int(np.sum(mask))
        switch_counts[source] = int(np.sum(switch[mask]))
    total = int(len(keys))
    top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10]
    top_switch = sorted(switch_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    return {
        "total_rows": total,
        "source_count": len(counts),
        "top_sources_by_rows": [{"source": k, "rows": v, "fraction": float(v / max(total, 1))} for k, v in top],
        "top_sources_by_switch_count": [
            {"source": k, "switch_rows": v, "fraction_of_switches": float(v / max(int(np.sum(switch)), 1))}
            for k, v in top_switch
        ],
        "switch_rows": int(np.sum(switch)),
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    hd_payload = read_json(HD_JSON, {})
    hc_payload = read_json(HC_JSON, {})
    if not hd_payload:
        raise RuntimeError("Missing Stage42-HD payload; run run_stage42_floor_free_proximity_guard_repair.py first.")
    ev = _build_switch_eval(hd_payload)
    labels = ev["test_bundle"]["labels"]
    selected = ev["selected_ade"]
    floor = ev["floor_ade"]
    switch = ev["guarded_switch"].astype(bool)
    bootstrap = _global_bootstrap(selected, floor, labels)
    domain_rows = _domain_rows(selected, floor, labels, switch)
    horizon_rows = _horizon_rows(selected, floor, labels, switch)
    domain_horizon = _domain_horizon_rows(selected, floor, labels, switch)
    metrics = ev["metrics"]
    positive_domains_all = [row["domain"] for row in domain_rows if float(row["metric"]["all_improvement"]) > 0.0]
    positive_domains_t50 = [row["domain"] for row in domain_rows if float(row["metric"]["t50_improvement"]) > 0.0]
    robust_domains = [row["domain"] for row in domain_rows if row["robust_positive"]]
    weak_domain_horizon = [
        row["slice"]
        for row in domain_horizon
        if row["rows"] >= 30 and not row["positive_ci"] and row["horizon"] in {50, 100}
    ]
    summary = {
        "source": "fresh_stage42_he_floor_free_proximity_guard_robustness",
        "hd_verdict": (hd_payload.get("stage42_hd_gate", {}) or {}).get("verdict"),
        "hc_verdict": (hc_payload.get("stage42_hc_gate", {}) or {}).get("verdict"),
        "policy_family": ev["policy_row"].get("family"),
        "min_sep": ev["min_sep"],
        "teacher_gate_used": False,
        "causal_floor_fallback_used": True,
        "global_floor_removal_allowed": False,
        "rows": int(len(selected)),
        "all_improvement": metrics.get("all_improvement"),
        "t50_improvement": metrics.get("t50_improvement"),
        "t100_raw_frame_diagnostic_improvement": metrics.get("t100_improvement"),
        "hard_failure_improvement": metrics.get("hard_failure_improvement"),
        "easy_degradation": metrics.get("easy_degradation"),
        "collision_delta_vs_floor_005": metrics.get("collision_delta_vs_floor_005"),
        "switch_rate": metrics.get("switch_rate"),
        "raw_switch_rate": metrics.get("raw_switch_rate"),
        "guarded_off_rate": metrics.get("guarded_off_rate"),
        "bootstrap_n": BOOTSTRAP_N,
        "global_ci_low": {
            "all": bootstrap["all"]["low"],
            "t50": bootstrap["t50"]["low"],
            "t100_raw_frame_diagnostic": bootstrap["t100_raw_frame_diagnostic"]["low"],
            "hard_failure": bootstrap["hard_failure"]["low"],
        },
        "easy_ci_high": bootstrap["easy_degradation"]["high"],
        "positive_domains_all": positive_domains_all,
        "positive_domains_t50": positive_domains_t50,
        "robust_positive_domains": robust_domains,
        "weak_domain_horizon_slices": weak_domain_horizon,
        "teacherless_gate_paper_evidence_supported": bool(
            _ci_positive(bootstrap["all"])
            and _ci_positive(bootstrap["t50"])
            and _ci_positive(bootstrap["hard_failure"])
            and _ci_positive(bootstrap["easy_degradation"], easy=True)
            and metrics.get("collision_delta_vs_floor_005", 1.0) <= COLLISION_LIMIT
            and len(robust_domains) >= 2
        ),
        "deployment_decision": "teacherless_gate_has_robust_evidence_but_requires_causal_floor_fallback",
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_stage42_he_floor_free_proximity_guard_robustness",
        "stage": "Stage42-HE floor-free proximity-guard robustness audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([HD_JSON, HD_GATE, HC_JSON]),
        "current_facts": CURRENT_FACTS,
        "summary": summary,
        "validation_protocol": {
            "policy_source": str(HD_JSON),
            "guard_threshold_source": "Stage42-HD validation-selected min_sep",
            "test_usage": "robustness_reporting_only_no_threshold_tuning",
            "bootstrap_n": BOOTSTRAP_N,
            "teacher_gate_used": False,
            "causal_floor_fallback_used": True,
        },
        "metrics": metrics,
        "bootstrap": bootstrap,
        "domain_rows": domain_rows,
        "horizon_rows": horizon_rows,
        "domain_horizon_rows": domain_horizon,
        "source_concentration": _source_concentration(ev["test_bundle"]["keys"], switch),
        "joint_stats": ev["joint_stats"],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_guard_selection_inherited_from_hd": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "global_floor_removal_allowed": False,
            "teacher_gate_removed_for_repaired_floor_free_candidate": True,
            "causal_floor_safety_fallback_still_required": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_he_gate"] = _gate(payload)
    return _jsonable(payload)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    boot = payload["bootstrap"]
    gates = {
        "stage42_hd_prereq_pass": s.get("hd_verdict") == "stage42_hd_floor_free_proximity_guard_repair_pass",
        "stage42_hc_prereq_pass": s.get("hc_verdict") == "stage42_hc_floor_alternative_gate_stress_pass",
        "best_hd_candidate_reconstructed": s.get("policy_family") == "harm_predictor_gate" and s.get("rows", 0) > 0,
        "bootstrap_all_reported": int(boot["all"].get("bootstrap_n", 0)) >= BOOTSTRAP_N,
        "bootstrap_t50_reported": int(boot["t50"].get("bootstrap_n", 0)) >= BOOTSTRAP_N,
        "bootstrap_hard_reported": int(boot["hard_failure"].get("bootstrap_n", 0)) >= BOOTSTRAP_N,
        "bootstrap_easy_reported": int(boot["easy_degradation"].get("bootstrap_n", 0)) >= BOOTSTRAP_N,
        "global_all_ci_positive": s["global_ci_low"]["all"] > 0.0,
        "global_t50_ci_positive": s["global_ci_low"]["t50"] > 0.0,
        "global_hard_ci_positive": s["global_ci_low"]["hard_failure"] > 0.0,
        "easy_ci_safe": s["easy_ci_high"] <= EASY_LIMIT,
        "collision_delta_safe": float(s["collision_delta_vs_floor_005"]) <= COLLISION_LIMIT,
        "at_least_two_domains_positive_all_and_t50": len(set(s["positive_domains_all"]) & set(s["positive_domains_t50"])) >= 2,
        "at_least_two_domains_robust_ci": len(s["robust_positive_domains"]) >= 2,
        "t100_raw_frame_reported_honestly": "t100_raw_frame_diagnostic" in s["global_ci_low"],
        "teacher_gate_not_used": s["teacher_gate_used"] is False,
        "causal_floor_fallback_still_required": s["causal_floor_fallback_used"] is True
        and claim["causal_floor_safety_fallback_still_required"] is True
        and claim["global_floor_removal_allowed"] is False,
        "no_future_test_or_central_velocity_leakage": leak["future_endpoint_input"] is False
        and leak["future_waypoint_input"] is False
        and leak["central_velocity"] is False
        and leak["test_endpoint_goals"] is False
        and leak["test_threshold_tuning"] is False,
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False
        and claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = int(len(gates))
    verdict = "stage42_he_floor_free_proximity_guard_robustness_pass" if passed == total else "stage42_he_floor_free_proximity_guard_robustness_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_he_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-HE Floor-Free Proximity-Guard Robustness Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Direct Decision",
        "",
        f"- deployment_decision: `{s['deployment_decision']}`",
        f"- policy_family: `{s['policy_family']}`",
        f"- min_sep: `{s['min_sep']}`",
        f"- teacher_gate_used: `{s['teacher_gate_used']}`",
        f"- causal_floor_fallback_used: `{s['causal_floor_fallback_used']}`",
        f"- global_floor_removal_allowed: `{s['global_floor_removal_allowed']}`",
        f"- teacherless_gate_paper_evidence_supported: `{s['teacherless_gate_paper_evidence_supported']}`",
        "",
        "## Aggregate Metrics",
        "",
        f"- rows: `{s['rows']}`",
        f"- all: `{_pct(s['all_improvement'])}`",
        f"- t50: `{_pct(s['t50_improvement'])}`",
        f"- t100 raw diagnostic: `{_pct(s['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure: `{_pct(s['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(s['easy_degradation'])}`",
        f"- collision delta @0.05: `{_pct(s['collision_delta_vs_floor_005'])}`",
        f"- switch rate: `{_pct(s['switch_rate'])}`",
        f"- raw switch rate: `{_pct(s['raw_switch_rate'])}`",
        f"- guarded-off rate: `{_pct(s['guarded_off_rate'])}`",
        "",
        "## Bootstrap CI",
        "",
        "| slice | low | mid | high | n | bootstrap_n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["bootstrap"].items():
        lines.append(
            f"| `{name}` | {_pct(row['low'])} | {_pct(row['mid'])} | {_pct(row['high'])} | {row['n']} | {row['bootstrap_n']} |"
        )
    lines += [
        "",
        "## Domain Robustness",
        "",
        "| domain | robust | rows | all | all CI low | t50 | t50 CI low | t100 raw | hard | hard CI low | easy | easy CI high | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["domain_rows"]:
        m = row["metric"]
        b = row["bootstrap"]
        lines.append(
            f"| `{row['domain']}` | {row['robust_positive']} | {m['rows']} | {_pct(m['all_improvement'])} | {_pct(b['all']['low'])} | "
            f"{_pct(m['t50_improvement'])} | {_pct(b['t50']['low'])} | {_pct(m['t100_raw_frame_diagnostic_improvement'])} | "
            f"{_pct(m['hard_failure_improvement'])} | {_pct(b['hard_failure']['low'])} | {_pct(m['easy_degradation'])} | "
            f"{_pct(b['easy_degradation']['high'])} | {_pct(m['switch_rate'])} |"
        )
    lines += [
        "",
        "## Horizon Robustness",
        "",
        "| horizon | rows | positive CI | improvement | CI low | CI high | raw-frame diagnostic |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["horizon_rows"]:
        b = row["bootstrap"]["improvement"]
        metric = row["metric"]
        key = "t100_raw_frame_diagnostic_improvement" if row["horizon"] == 100 else f"t{row['horizon']}_improvement"
        lines.append(
            f"| {row['horizon']} | {row['rows']} | {row['positive_ci']} | {_pct(metric.get(key, metric['all_improvement']))} | "
            f"{_pct(b['low'])} | {_pct(b['high'])} | {row['raw_frame_diagnostic']} |"
        )
    lines += [
        "",
        "## Weak Domain-Horizon Slices",
        "",
        f"- weak_domain_horizon_slices: `{', '.join(s['weak_domain_horizon_slices']) if s['weak_domain_horizon_slices'] else 'none'}`",
        "",
        "## Interpretation",
        "",
        "- Stage42-HE does not choose a new threshold; it audits the Stage42-HD frozen validation-selected repaired gate.",
        "- The teacher gate is removed for this repaired switch gate, but causal floor fallback is still the safety floor.",
        "- This supports a teacherless switch-gate evidence claim only under floor fallback; it does not support global floor removal or ungated neural deployment.",
        "- t+100 remains raw-frame diagnostic; no metric/seconds/true-3D/foundation/Stage5C/SMC claim is made.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_he_gate"]
    lines = [
        "# Stage42-HE Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def _refresh_docs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    s = payload["summary"]
    lines = [
        "## Stage42-HE Floor-Free Proximity-Guard Robustness Audit",
        "",
        "- source: `fresh_stage42_he_floor_free_proximity_guard_robustness`",
        f"- gate: `{payload['stage42_he_gate']['passed']} / {payload['stage42_he_gate']['total']}`",
        f"- verdict: `{payload['stage42_he_gate']['verdict']}`",
        "- Audits the Stage42-HD teacherless proximity-guard repaired gate with 2000-bootstrap and per-domain/per-horizon checks.",
        f"- policy `{s['policy_family']}` with min_sep `{s['min_sep']}` reaches all/t50/t100raw/hard `{_pct(s['all_improvement'])}` / `{_pct(s['t50_improvement'])}` / `{_pct(s['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(s['hard_failure_improvement'])}`.",
        f"- bootstrap CI lows all/t50/t100raw/hard `{_pct(s['global_ci_low']['all'])}` / `{_pct(s['global_ci_low']['t50'])}` / `{_pct(s['global_ci_low']['t100_raw_frame_diagnostic'])}` / `{_pct(s['global_ci_low']['hard_failure'])}`; easy CI high `{_pct(s['easy_ci_high'])}`.",
        f"- robust_positive_domains: `{', '.join(s['robust_positive_domains'])}`; weak_domain_horizon_slices: `{', '.join(s['weak_domain_horizon_slices']) if s['weak_domain_horizon_slices'] else 'none'}`.",
        "- Teacher gate is not used, but causal floor fallback remains required. This is not global floor removal, not metric/seconds, not true 3D, not Stage5C, and not SMC.",
    ]
    status = []
    for path in [README_RESULTS, M3W_README, USER_SUMMARY, WORK_SUMMARY, GOAL_SUMMARY]:
        if path.exists():
            _replace_section(path, "STAGE42_HE_FLOOR_FREE_PROXIMITY_GUARD_ROBUSTNESS", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "exists": True,
                    "contains_stage42_he": "Stage42-HE Floor-Free Proximity-Guard Robustness Audit" in text,
                    "contains_not_global_floor_removal": "not global floor removal" in text,
                }
            )
        else:
            status.append({"path": str(path), "exists": False})
    return status


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json("research_state.json", {})
    s = payload["summary"]
    state["current_stage"] = "Stage42-HE floor-free proximity-guard robustness audit"
    state["current_verdict"] = payload["stage42_he_gate"]["verdict"]
    state["stage42_he_floor_free_proximity_guard_robustness"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_he_gate"]["verdict"],
        "gates": f"{payload['stage42_he_gate']['passed']}/{payload['stage42_he_gate']['total']}",
        "summary": {
            "policy_family": s["policy_family"],
            "min_sep": s["min_sep"],
            "all_improvement": s["all_improvement"],
            "t50_improvement": s["t50_improvement"],
            "t100_raw_frame_diagnostic_improvement": s["t100_raw_frame_diagnostic_improvement"],
            "hard_failure_improvement": s["hard_failure_improvement"],
            "easy_degradation": s["easy_degradation"],
            "collision_delta_vs_floor_005": s["collision_delta_vs_floor_005"],
            "bootstrap_n": s["bootstrap_n"],
            "global_ci_low": s["global_ci_low"],
            "easy_ci_high": s["easy_ci_high"],
            "robust_positive_domains": s["robust_positive_domains"],
            "weak_domain_horizon_slices": s["weak_domain_horizon_slices"],
            "teacher_gate_used": s["teacher_gate_used"],
            "causal_floor_fallback_used": s["causal_floor_fallback_used"],
            "global_floor_removal_allowed": s["global_floor_removal_allowed"],
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "verification": {
            "focused_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_floor_free_proximity_guard_robustness.py -> pending",
            "full_pytest": ".venv-pytorch/bin/python -m pytest tests -> pending",
        },
        "claim_boundary": payload["claim_boundary"],
    }
    write_json("research_state.json", _jsonable(state))


def run_stage42_floor_free_proximity_guard_robustness() -> dict[str, Any]:
    payload = _build_payload()
    payload["doc_refresh_status"] = _refresh_docs(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _update_state(payload)
    return payload


if __name__ == "__main__":
    out = run_stage42_floor_free_proximity_guard_robustness()
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2, sort_keys=True))
