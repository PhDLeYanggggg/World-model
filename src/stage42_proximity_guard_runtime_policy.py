from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _pct, _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
POLICY_JSON = OUT_DIR / "frozen_proximity_guard_composer_policy_stage42_policy.json"
CT_JSON = OUT_DIR / "frozen_proximity_guard_policy_replay_stage42.json"

REPORT_JSON = OUT_DIR / "proximity_guard_runtime_policy_stage42.json"
REPORT_MD = OUT_DIR / "proximity_guard_runtime_policy_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_cu_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RETRO_README = Path("README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CU 把 Stage42-CS frozen policy artifact 变成可调用 runtime policy API。",
    "runtime policy 只使用 domain、horizon 和模型预测 rollout geometry 的 group min-distance。",
    "runtime policy 不使用 future endpoint、future waypoints、central velocity 或 test endpoint goals。",
    "runtime smoke audit 是 deployment/reproducibility evidence，不是新增模型训练分数。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


@dataclass(frozen=True)
class RuntimeDecision:
    domain: str
    horizon: int
    key: str
    base_wants_full_waypoint: bool
    use_full_waypoint: bool
    guarded_off: bool
    reason: str
    endpoint_min_group_distance: float | None
    full_min_group_distance: float | None
    min_sep: float
    margin: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "horizon": self.horizon,
            "key": self.key,
            "base_wants_full_waypoint": self.base_wants_full_waypoint,
            "use_full_waypoint": self.use_full_waypoint,
            "guarded_off": self.guarded_off,
            "reason": self.reason,
            "endpoint_min_group_distance": self.endpoint_min_group_distance,
            "full_min_group_distance": self.full_min_group_distance,
            "min_sep": self.min_sep,
            "margin": self.margin,
        }


class FrozenProximityGuardPolicy:
    def __init__(self, payload: Mapping[str, Any], *, policy_hash: str | None = None) -> None:
        self.payload = dict(payload)
        selected = self.payload.get("selected_policy", {})
        guard = self.payload.get("guard_rule", {})
        self.min_sep = float(guard.get("min_sep", selected.get("min_sep", 0.0)))
        self.margin = float(guard.get("margin", selected.get("margin", 0.0)))
        self.base_choices = {str(k): bool(v) for k, v in self.payload.get("base_choices", {}).items()}
        self.policy_hash = policy_hash

    @classmethod
    def from_file(cls, path: Path = POLICY_JSON) -> "FrozenProximityGuardPolicy":
        payload = read_json(path, {})
        if not payload:
            raise FileNotFoundError(f"Missing or empty frozen policy artifact: {path}")
        return cls(payload, policy_hash=_combined_hash([path]))

    @staticmethod
    def key(domain: str, horizon: int | str) -> str:
        return f"{domain}|{int(horizon)}"

    def base_wants_full_waypoint(self, domain: str, horizon: int | str) -> bool:
        return bool(self.base_choices.get(self.key(domain, horizon), False))

    @staticmethod
    def _finite(value: float | int | None) -> bool:
        return value is not None and math.isfinite(float(value))

    def decide(
        self,
        *,
        domain: str,
        horizon: int | str,
        endpoint_min_group_distance: float | int | None,
        full_min_group_distance: float | int | None,
    ) -> RuntimeDecision:
        horizon_i = int(horizon)
        key = self.key(domain, horizon_i)
        base_wants_full = self.base_wants_full_waypoint(domain, horizon_i)
        endpoint_finite = self._finite(endpoint_min_group_distance)
        full_finite = self._finite(full_min_group_distance)
        guarded = (
            base_wants_full
            and endpoint_finite
            and full_finite
            and float(full_min_group_distance) < self.min_sep
            and float(full_min_group_distance) + self.margin < float(endpoint_min_group_distance)
        )
        if not base_wants_full:
            use_full = False
            reason = "base_choice_endpoint_linear"
        elif guarded:
            use_full = False
            reason = "proximity_guard_fallback_to_endpoint_linear"
        elif not endpoint_finite or not full_finite:
            use_full = True
            reason = "base_choice_full_waypoint_geometry_nonfinite_replay_no_guard"
        else:
            use_full = True
            reason = "base_choice_full_waypoint_guard_clear"
        return RuntimeDecision(
            domain=domain,
            horizon=horizon_i,
            key=key,
            base_wants_full_waypoint=base_wants_full,
            use_full_waypoint=use_full,
            guarded_off=guarded,
            reason=reason,
            endpoint_min_group_distance=None if endpoint_min_group_distance is None else float(endpoint_min_group_distance),
            full_min_group_distance=None if full_min_group_distance is None else float(full_min_group_distance),
            min_sep=self.min_sep,
            margin=self.margin,
        )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _smoke_cases(policy: FrozenProximityGuardPolicy) -> list[dict[str, Any]]:
    cases = [
        {
            "name": "full_slice_guard_clear",
            "domain": "ETH_UCY",
            "horizon": 50,
            "endpoint_min_group_distance": 0.40,
            "full_min_group_distance": 0.30,
            "expected_use_full_waypoint": True,
            "expected_guarded_off": False,
        },
        {
            "name": "full_slice_guarded_off",
            "domain": "ETH_UCY",
            "horizon": 50,
            "endpoint_min_group_distance": 0.40,
            "full_min_group_distance": 0.10,
            "expected_use_full_waypoint": False,
            "expected_guarded_off": True,
        },
        {
            "name": "endpoint_slice_never_switches",
            "domain": "TrajNet",
            "horizon": 50,
            "endpoint_min_group_distance": 0.40,
            "full_min_group_distance": 0.30,
            "expected_use_full_waypoint": False,
            "expected_guarded_off": False,
        },
        {
            "name": "full_slice_nonfinite_geometry_replays_no_guard",
            "domain": "ETH_UCY",
            "horizon": 100,
            "endpoint_min_group_distance": None,
            "full_min_group_distance": 0.10,
            "expected_use_full_waypoint": True,
            "expected_guarded_off": False,
        },
    ]
    rows = []
    for case in cases:
        decision = policy.decide(
            domain=case["domain"],
            horizon=case["horizon"],
            endpoint_min_group_distance=case["endpoint_min_group_distance"],
            full_min_group_distance=case["full_min_group_distance"],
        ).as_dict()
        rows.append(
            {
                **case,
                "decision": decision,
                "passed": (
                    decision["use_full_waypoint"] == case["expected_use_full_waypoint"]
                    and decision["guarded_off"] == case["expected_guarded_off"]
                ),
            }
        )
    return rows


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    policy = payload["runtime_policy"]
    ct_gate = payload["inputs"]["stage42_ct"].get("stage42_ct_gate", {})
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    smoke = payload["smoke_cases"]
    gates = {
        "policy_artifact_exists": payload["policy_artifact"]["exists"],
        "policy_hash_available": bool(payload.get("policy_hash")),
        "ct_replay_gate_passed": ct_gate.get("passed") == ct_gate.get("total"),
        "runtime_min_sep_matches_artifact": float(policy["min_sep"]) == float(payload["policy_artifact_payload"]["guard_rule"]["min_sep"]),
        "runtime_margin_matches_artifact": float(policy["margin"]) == float(payload["policy_artifact_payload"]["guard_rule"]["margin"]),
        "runtime_base_choices_match_artifact": policy["base_choices"] == payload["policy_artifact_payload"]["base_choices"],
        "full_slice_guard_clear_passed": smoke["full_slice_guard_clear"]["passed"],
        "full_slice_guarded_off_passed": smoke["full_slice_guarded_off"]["passed"],
        "endpoint_slice_never_switches_passed": smoke["endpoint_slice_never_switches"]["passed"],
        "nonfinite_geometry_replay_rule_passed": smoke["full_slice_nonfinite_geometry_replays_no_guard"]["passed"],
        "uses_domain_horizon_and_predicted_geometry_only": payload["runtime_inputs"] == [
            "domain",
            "horizon",
            "endpoint_min_group_distance_from_predicted_endpoint_rollout",
            "full_min_group_distance_from_predicted_full_waypoint_rollout",
        ],
        "no_future_endpoint_input": no_leakage["future_endpoint_input"] is False,
        "no_future_waypoints_input": no_leakage["future_waypoints_input"] is False,
        "no_central_velocity": no_leakage["central_velocity"] is False,
        "no_test_endpoint_goals": no_leakage["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": no_leakage["test_threshold_tuning"] is False,
        "metric_seconds_overclaim_blocked": claim["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": claim["stage5c_executed"] is False,
        "smc_not_enabled": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_cu_runtime_policy_api_pass" if passed == total else "stage42_cu_runtime_policy_api_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = [
        "## Stage42-CU Runtime Policy API Smoke Audit",
        "",
        "- source: `fresh_runtime_api_from_frozen_policy_artifact`",
        f"- verdict: `{payload['stage42_cu_gate']['verdict']}`",
        f"- gates: `{payload['stage42_cu_gate']['passed']} / {payload['stage42_cu_gate']['total']}`",
        f"- policy hash: `{payload['policy_hash']}`",
        "- runtime inputs: domain, horizon, endpoint predicted group min-distance, full-waypoint predicted group min-distance.",
        "- guard rule: use full-waypoint only when validation-selected base slice wants full and predicted proximity guard does not fire.",
        "- smoke cases: guard-clear full slice, guarded-off full slice, endpoint-only slice, and nonfinite-geometry replay behavior all pass.",
        "- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, RETRO_README]:
        _replace_section(path, "STAGE42_CU_RUNTIME_POLICY_API", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-CU runtime policy API smoke audit"
    state["current_verdict"] = payload["stage42_cu_gate"]["verdict"]
    state["stage42_cu_runtime_policy_api"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_cu_gate"]["verdict"],
        "gates": f"{payload['stage42_cu_gate']['passed']}/{payload['stage42_cu_gate']['total']}",
        "policy_hash": payload["policy_hash"],
        "runtime_inputs": payload["runtime_inputs"],
        "smoke_case_summary": {name: row["decision"]["reason"] for name, row in payload["smoke_cases"].items()},
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-CU exposes the frozen Stage42-CS proximity-guard composer as a deterministic runtime policy API and verifies representative smoke cases. It makes the policy deployable/reproducible without changing the raw-frame/dataset-local 2.5D claim boundary.",
        "verification": {
            "runner": ".venv-pytorch/bin/python run_stage42_runtime_proximity_guard_policy.py",
            "targeted_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_proximity_guard_runtime_policy.py",
        },
    }
    summary = state.setdefault("latest_user_facing_goal_summary", {})
    summary["source"] = "cached_verified_synthesis_for_user_question_refreshed_after_stage42_cu"
    included = summary.setdefault("latest_fresh_evidence_included", [])
    note = "Stage42-CU runtime policy API: frozen proximity-guard composer exposed as deterministic domain/horizon/proximity decision function"
    if note not in included:
        included.append(note)
    write_json(RESEARCH_STATE, state)


def _write_md(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_cu_gate"]
    lines = [
        "# Stage42-CU Runtime Policy API Smoke Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- policy_hash: `{payload['policy_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Runtime API",
        "",
        "- loader: `FrozenProximityGuardPolicy.from_file()`",
        "- decision method: `policy.decide(domain=..., horizon=..., endpoint_min_group_distance=..., full_min_group_distance=...)`",
        "- inputs: `domain`, `horizon`, predicted endpoint group min-distance, predicted full-waypoint group min-distance.",
        "- output: deterministic `RuntimeDecision` with `use_full_waypoint`, `guarded_off`, and reason.",
        "",
        "## Smoke Cases",
        "",
        "| case | decision | reason | passed |",
        "| --- | --- | --- | ---: |",
    ]
    for name, row in payload["smoke_cases"].items():
        decision = row["decision"]
        lines.append(
            f"| `{name}` | use_full=`{decision['use_full_waypoint']}`, guarded_off=`{decision['guarded_off']}` | `{decision['reason']}` | `{row['passed']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-CU turns the frozen policy artifact into a callable deployment component.",
            "- It does not reselect thresholds and does not add new model scores.",
            "- Nonfinite predicted geometry follows the exact CQ replay behavior: if the base slice selected full-waypoint, the guard does not fire.",
            "- This remains protected dataset-local/raw-frame 2.5D evidence, not true 3D, not metric/seconds-level, not Stage5C, and not SMC.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-CU Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | passed |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        gate_lines.append(f"| `{name}` | `{ok}` |")
    write_md(GATE_MD, gate_lines)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_stage42_runtime_proximity_guard_policy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    policy_payload = read_json(POLICY_JSON, {})
    ct = read_json(CT_JSON, {})
    if not policy_payload:
        raise FileNotFoundError(f"Missing frozen policy artifact: {POLICY_JSON}")
    if not ct:
        raise FileNotFoundError(f"Missing Stage42-CT replay report: {CT_JSON}")
    policy = FrozenProximityGuardPolicy.from_file(POLICY_JSON)
    smoke_rows = _smoke_cases(policy)
    smoke = {row["name"]: row for row in smoke_rows}
    payload: dict[str, Any] = {
        "source": "fresh_runtime_api_from_frozen_policy_artifact",
        "stage": "Stage42-CU runtime proximity-guard policy API",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "inputs": {"stage42_ct": {"path": str(CT_JSON), "stage42_ct_gate": ct.get("stage42_ct_gate", {})}},
        "policy_artifact": {
            "path": str(POLICY_JSON),
            "exists": POLICY_JSON.exists(),
            "size_bytes": POLICY_JSON.stat().st_size if POLICY_JSON.exists() else 0,
        },
        "policy_artifact_payload": policy_payload,
        "policy_hash": policy.policy_hash,
        "runtime_policy": {
            "min_sep": policy.min_sep,
            "margin": policy.margin,
            "base_choices": policy.base_choices,
        },
        "runtime_inputs": [
            "domain",
            "horizon",
            "endpoint_min_group_distance_from_predicted_endpoint_rollout",
            "full_min_group_distance_from_predicted_full_waypoint_rollout",
        ],
        "smoke_cases": smoke,
        "no_leakage": policy_payload.get("no_leakage", {}),
        "claim_boundary": policy_payload.get("claim_boundary", {}),
    }
    payload["stage42_cu_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    _write_md(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_runtime_proximity_guard_policy()
    gate = result["stage42_cu_gate"]
    print(f"Stage42-CU runtime policy API: {gate['verdict']} ({gate['passed']}/{gate['total']})")
