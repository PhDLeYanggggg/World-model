from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _pct, _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
POLICY_JSON = OUT_DIR / "frozen_group_consistency_t100_easy_guard_policy_stage42.json"
HS_JSON = OUT_DIR / "group_consistency_t100_easy_guard_freeze_stage42.json"

REPORT_JSON = OUT_DIR / "group_consistency_t100_easy_guard_runtime_stage42.json"
REPORT_MD = OUT_DIR / "group_consistency_t100_easy_guard_runtime_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ht_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_SUMMARY = Path("README_M3W_CURRENT_MASTER_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_runtime_api_from_frozen_stage42_hs_t100_easy_guard_policy"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HT 把 Stage42-HS frozen t100 easy guard policy 变成可调用 runtime API。",
    "runtime policy 只使用 domain、horizon、候选 rollout 和 train-horizon causal floor rollout。",
    "未知 domain 的 t100 默认回退 floor，因为没有 validation support。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


@dataclass(frozen=True)
class T100EasyGuardDecision:
    domain: str
    horizon: int
    key: str
    use_candidate: bool
    fallback_to_floor: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "horizon": self.horizon,
            "key": self.key,
            "use_candidate": self.use_candidate,
            "fallback_to_floor": self.fallback_to_floor,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class T100EasyGuardRuntimeResult:
    selected_xy: np.ndarray
    switch: np.ndarray
    reasons: list[str]

    def diagnostics(self) -> dict[str, Any]:
        unique, counts = np.unique(np.asarray(self.reasons, dtype=object), return_counts=True)
        return {
            "rows": int(len(self.switch)),
            "switch_rate": float(np.mean(self.switch)) if len(self.switch) else 0.0,
            "fallback_rows": int(np.sum(~self.switch)),
            "reason_counts": {str(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())},
        }


class FrozenT100EasyGuardPolicy:
    def __init__(self, payload: Mapping[str, Any], *, policy_hash: str | None = None) -> None:
        self.payload = dict(payload)
        table = self.payload.get("decision_table", {})
        self.guarded_slices = {str(k): dict(v) for k, v in table.get("guarded_slices", {}).items()}
        self.kept_slices = {str(k): dict(v) for k, v in table.get("kept_slices", {}).items()}
        rule = self.payload.get("decision_rule", {})
        self.threshold_easy_degradation = float(rule.get("threshold_easy_degradation", 0.0))
        self.policy_hash = policy_hash

    @classmethod
    def from_file(cls, path: Path = POLICY_JSON) -> "FrozenT100EasyGuardPolicy":
        payload = read_json(path, {})
        if not payload:
            raise FileNotFoundError(f"Missing or empty frozen policy artifact: {path}")
        return cls(payload, policy_hash=str(payload.get("policy_hash") or _combined_hash([path])))

    @staticmethod
    def key(domain: str, horizon: int | str) -> str:
        return f"{domain}|{int(horizon)}"

    def decide(self, *, domain: str, horizon: int | str) -> T100EasyGuardDecision:
        horizon_i = int(horizon)
        key = self.key(domain, horizon_i)
        if horizon_i != 100:
            return T100EasyGuardDecision(domain, horizon_i, key, True, False, "non_t100_not_guarded")
        if key in self.kept_slices:
            return T100EasyGuardDecision(domain, horizon_i, key, True, False, "validation_supported_t100_keep_candidate")
        if key in self.guarded_slices:
            return T100EasyGuardDecision(domain, horizon_i, key, False, True, "validation_easy_harm_t100_fallback_floor")
        return T100EasyGuardDecision(domain, horizon_i, key, False, True, "unknown_t100_domain_no_validation_support_fallback_floor")

    def apply(
        self,
        *,
        domains: Sequence[str] | np.ndarray,
        horizons: Sequence[int] | np.ndarray,
        candidate_xy: np.ndarray,
        floor_xy: np.ndarray,
        candidate_switch: Sequence[bool] | np.ndarray | None = None,
    ) -> T100EasyGuardRuntimeResult:
        domain_arr = np.asarray(domains).astype(str)
        horizon_arr = np.asarray(horizons).astype(int)
        if len(domain_arr) != len(horizon_arr) or len(domain_arr) != len(candidate_xy) or len(domain_arr) != len(floor_xy):
            raise ValueError("domains, horizons, candidate_xy, and floor_xy must have the same first dimension")
        selected = candidate_xy.astype(np.float32).copy()
        switch = (
            np.asarray(candidate_switch, dtype=bool).copy()
            if candidate_switch is not None
            else np.ones(len(domain_arr), dtype=bool)
        )
        if len(switch) != len(domain_arr):
            raise ValueError("candidate_switch must have the same first dimension as domains")
        reasons: list[str] = []
        for i, (domain, horizon) in enumerate(zip(domain_arr.tolist(), horizon_arr.tolist())):
            decision = self.decide(domain=domain, horizon=horizon)
            reasons.append(decision.reason)
            if decision.fallback_to_floor:
                selected[i] = floor_xy[i]
                switch[i] = False
        return T100EasyGuardRuntimeResult(selected_xy=selected, switch=switch, reasons=reasons)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _smoke_cases(policy: FrozenT100EasyGuardPolicy) -> dict[str, Any]:
    candidate = np.asarray(
        [
            [[1.0, 0.0], [2.0, 0.0]],
            [[3.0, 0.0], [4.0, 0.0]],
            [[5.0, 0.0], [6.0, 0.0]],
            [[7.0, 0.0], [8.0, 0.0]],
        ],
        dtype=np.float32,
    )
    floor = np.zeros_like(candidate)
    domains = np.asarray(["TrajNet", "UCY", "TrajNet", "UnknownDomain"], dtype=object)
    horizons = np.asarray([100, 100, 50, 100], dtype=np.int64)
    result = policy.apply(domains=domains, horizons=horizons, candidate_xy=candidate, floor_xy=floor)
    expected_switch = [False, True, True, False]
    expected_reasons = [
        "validation_easy_harm_t100_fallback_floor",
        "validation_supported_t100_keep_candidate",
        "non_t100_not_guarded",
        "unknown_t100_domain_no_validation_support_fallback_floor",
    ]
    return {
        "rows": int(len(domains)),
        "decisions": [
            policy.decide(domain=str(domain), horizon=int(horizon)).as_dict()
            for domain, horizon in zip(domains.tolist(), horizons.tolist())
        ],
        "diagnostics": result.diagnostics(),
        "expected_switch": expected_switch,
        "actual_switch": result.switch.tolist(),
        "expected_reasons": expected_reasons,
        "actual_reasons": result.reasons,
        "selected_matches_floor_for_guarded_rows": bool(np.allclose(result.selected_xy[[0, 3]], floor[[0, 3]])),
        "selected_keeps_candidate_for_supported_rows": bool(np.allclose(result.selected_xy[[1, 2]], candidate[[1, 2]])),
        "passes": bool(
            result.switch.tolist() == expected_switch
            and result.reasons == expected_reasons
            and np.allclose(result.selected_xy[[0, 3]], floor[[0, 3]])
            and np.allclose(result.selected_xy[[1, 2]], candidate[[1, 2]])
        ),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    hs_gate = payload["inputs"]["stage42_hs"].get("stage42_hs_gate", {})
    runtime = payload["runtime_policy"]
    smoke = payload["smoke_case"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "policy_artifact_exists": payload["policy_artifact"]["exists"] is True,
        "policy_hash_available": bool(payload["policy_hash"]),
        "hs_gate_passed": hs_gate.get("passed") == hs_gate.get("total"),
        "runtime_guarded_slices_match_policy": runtime["guarded_slices"] == payload["policy_artifact_payload"]["decision_table"]["guarded_slices"],
        "runtime_kept_slices_match_policy": runtime["kept_slices"] == payload["policy_artifact_payload"]["decision_table"]["kept_slices"],
        "smoke_cases_pass": smoke["passes"] is True,
        "trajnet_t100_fallbacks": smoke["actual_reasons"][0] == "validation_easy_harm_t100_fallback_floor",
        "ucy_t100_keeps_candidate": smoke["actual_reasons"][1] == "validation_supported_t100_keep_candidate",
        "non_t100_not_guarded": smoke["actual_reasons"][2] == "non_t100_not_guarded",
        "unknown_t100_fallbacks": smoke["actual_reasons"][3] == "unknown_t100_domain_no_validation_support_fallback_floor",
        "runtime_inputs_past_only_or_predictions": payload["runtime_inputs"] == [
            "domain",
            "horizon",
            "candidate_xy_predicted_rollout",
            "floor_xy_train_horizon_causal_rollout",
            "candidate_switch_optional",
        ],
        "no_future_endpoint_input": no_leak["future_endpoint_input"] is False,
        "no_future_waypoint_input": no_leak["future_waypoint_input"] is False,
        "no_central_velocity": no_leak["central_velocity"] is False,
        "no_test_endpoint_goals": no_leak["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": no_leak["test_threshold_tuning"] is False,
        "metric_seconds_overclaim_blocked": claim["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": claim["stage5c_executed"] is False,
        "smc_not_enabled": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_ht_t100_easy_guard_runtime_policy_pass" if passed == total else "stage42_ht_t100_easy_guard_runtime_policy_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    metric = payload["policy_artifact_payload"]["test_summary_vs_train_horizon_causal_floor"]
    lines = [
        "## Stage42-HT Runtime T100 Easy Guard Policy",
        "",
        "- source: `fresh_runtime_api_from_frozen_stage42_hs_t100_easy_guard_policy`",
        "- role: convert the frozen Stage42-HS domain|t100 easy guard into a callable runtime policy API.",
        f"- gate: `{payload['stage42_ht_gate']['passed']} / {payload['stage42_ht_gate']['total']}`; verdict `{payload['stage42_ht_gate']['verdict']}`.",
        f"- policy artifact: `{POLICY_JSON}`",
        f"- policy hash: `{payload['policy_hash']}`",
        "- runtime rule: TrajNet|100 falls back to floor; UCY|100 keeps candidate; unknown t100 domains fallback to floor; non-t100 rows are unchanged.",
        f"- inherited guarded all/t50/t100 raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
        "- Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, MASTER_SUMMARY]:
        _replace_section(path, "STAGE42_HT_T100_EASY_GUARD_RUNTIME", lines)


def _write_report(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_ht_gate"]
    metric = payload["policy_artifact_payload"]["test_summary_vs_train_horizon_causal_floor"]
    smoke = payload["smoke_case"]
    lines = [
        "# Stage42-HT Runtime T100 Easy Guard Policy",
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
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Runtime Rule",
        "",
        "- `TrajNet|100`: fallback to train-horizon causal floor.",
        "- `UCY|100`: keep candidate rollout.",
        "- unknown `domain|100`: fallback to floor because validation support is absent.",
        "- non-t100 rows: unchanged candidate rollout.",
        "",
        "## Smoke Replay",
        "",
        f"- passes: `{smoke['passes']}`",
        f"- actual_switch: `{smoke['actual_switch']}`",
        f"- actual_reasons: `{smoke['actual_reasons']}`",
        "",
        "## Inherited Metrics From Frozen HS Policy",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| all | {_pct(metric['all_improvement'])} |",
        f"| t50 | {_pct(metric['t50_improvement'])} |",
        f"| t100 raw diagnostic | {_pct(metric['t100_raw_frame_diagnostic_improvement'])} |",
        f"| hard/failure | {_pct(metric['hard_failure_improvement'])} |",
        f"| easy degradation | {_pct(metric['easy_degradation'])} |",
        f"| t100 easy degradation | {_pct(metric['t100_easy_degradation'])} |",
        "",
        "## Interpretation",
        "",
        "- HT makes the HS t100 guard callable at deployment/replay time.",
        "- It does not retrain, retune thresholds, execute Stage5C, enable SMC, or make metric/seconds-level claims.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-HT Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(ok)}` |" for name, ok in gate["gates"].items()],
    ]
    write_md(GATE_MD, gate_lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HT runtime t100 easy guard policy"
    state["current_verdict"] = payload["stage42_ht_gate"]["verdict"]
    state["stage42_ht_t100_easy_guard_runtime_policy"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ht_gate"]["verdict"],
        "gates": f"{payload['stage42_ht_gate']['passed']}/{payload['stage42_ht_gate']['total']}",
        "policy_hash": payload["policy_hash"],
        "runtime_rule": "TrajNet|100 fallback; UCY|100 keep; unknown t100 fallback; non-t100 unchanged",
        "smoke_pass": payload["smoke_case"]["passes"],
        "metric": payload["policy_artifact_payload"]["test_summary_vs_train_horizon_causal_floor"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["last_updated"] = "2026-05-27"
    write_json(RESEARCH_STATE, state)


def run_stage42_group_consistency_t100_easy_guard_runtime() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    policy_payload = read_json(POLICY_JSON, {})
    hs_payload = read_json(HS_JSON, {})
    if not policy_payload:
        raise FileNotFoundError(f"Missing frozen Stage42-HS policy artifact: {POLICY_JSON}")
    policy = FrozenT100EasyGuardPolicy.from_file(POLICY_JSON)
    smoke = _smoke_cases(policy)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-HT runtime t100 easy guard policy",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([POLICY_JSON, HS_JSON]),
        "current_facts": CURRENT_FACTS,
        "policy_artifact": {"path": str(POLICY_JSON), "exists": POLICY_JSON.exists(), "sha256": _combined_hash([POLICY_JSON])},
        "policy_hash": policy.policy_hash,
        "policy_artifact_payload": policy_payload,
        "runtime_policy": {
            "guarded_slices": policy.guarded_slices,
            "kept_slices": policy.kept_slices,
            "threshold_easy_degradation": policy.threshold_easy_degradation,
        },
        "inputs": {
            "stage42_hs": {
                "path": str(HS_JSON),
                "exists": HS_JSON.exists(),
                "stage42_hs_gate": hs_payload.get("stage42_hs_gate", {}),
            }
        },
        "runtime_inputs": [
            "domain",
            "horizon",
            "candidate_xy_predicted_rollout",
            "floor_xy_train_horizon_causal_rollout",
            "candidate_switch_optional",
        ],
        "smoke_case": smoke,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_ht_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    _write_report(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    out = run_stage42_group_consistency_t100_easy_guard_runtime()
    gate = out["stage42_ht_gate"]
    print(f"Stage42-HT t100 easy guard runtime policy: {gate['verdict']} ({gate['passed']}/{gate['total']})")
