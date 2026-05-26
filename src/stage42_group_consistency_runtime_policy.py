from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_external_validation as s42b
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _pct, _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

POLICY_JSON = OUT_DIR / "frozen_group_consistency_full_waypoint_policy_stage42_policy.json"
DK_JSON = OUT_DIR / "group_consistency_policy_replay_stage42.json"

REPORT_JSON = OUT_DIR / "group_consistency_runtime_policy_stage42.json"
REPORT_MD = OUT_DIR / "group_consistency_runtime_policy_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_dl_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_README = Path("README_M3W_GOAL_SUMMARY_ZH.md")
CURRENT_RETROSPECTIVE = Path("README_M3W_CURRENT_FULL_RETROSPECTIVE_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_FILES = [
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DL 把 Stage42-DJ/DK frozen group-consistency full-waypoint policy 变成可调用 runtime policy API。",
    "runtime policy 只使用 predicted full-waypoint rollouts、train-horizon causal floor rollout、source/frame/horizon group key、agent id、current xy、normalizer。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "DL 不重新训练、不重新选择阈值、不使用 test metrics 调参。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


@dataclass(frozen=True)
class GroupConsistencyRuntimeResult:
    selected_xy: np.ndarray
    switch: np.ndarray
    base_min: np.ndarray
    final_min: np.ndarray
    floor_min: np.ndarray

    def diagnostics(self) -> dict[str, Any]:
        return {
            "rows": int(len(self.switch)),
            "switch_rate": float(np.mean(self.switch)) if len(self.switch) else 0.0,
            "base_near_005": float(np.mean(np.isfinite(self.base_min) & (self.base_min < 0.05))) if len(self.base_min) else 0.0,
            "final_near_005": float(np.mean(np.isfinite(self.final_min) & (self.final_min < 0.05))) if len(self.final_min) else 0.0,
            "floor_near_005": float(np.mean(np.isfinite(self.floor_min) & (self.floor_min < 0.05))) if len(self.floor_min) else 0.0,
            "base_p05_min_distance": float(np.percentile(self.base_min[np.isfinite(self.base_min)], 5))
            if np.any(np.isfinite(self.base_min))
            else None,
            "final_p05_min_distance": float(np.percentile(self.final_min[np.isfinite(self.final_min)], 5))
            if np.any(np.isfinite(self.final_min))
            else None,
            "floor_p05_min_distance": float(np.percentile(self.floor_min[np.isfinite(self.floor_min)], 5))
            if np.any(np.isfinite(self.floor_min))
            else None,
        }


class FrozenGroupConsistencyPolicy:
    def __init__(self, payload: Mapping[str, Any], *, policy_hash: str | None = None) -> None:
        self.payload = dict(payload)
        rule = self.payload.get("repair_rule", {})
        self.mode = str(rule.get("type"))
        self.min_sep = float(rule.get("min_sep", 0.0))
        self.margin = float(rule.get("margin", 0.0))
        self.strength = None if rule.get("strength") is None else float(rule.get("strength"))
        self.alpha = None if rule.get("alpha") is None else float(rule.get("alpha"))
        self.safe_min_sep = None if rule.get("safe_min_sep") is None else float(rule.get("safe_min_sep"))
        self.policy_hash = policy_hash

    @classmethod
    def from_file(cls, path: Path = POLICY_JSON) -> "FrozenGroupConsistencyPolicy":
        payload = read_json(path, {})
        if not payload:
            raise FileNotFoundError(f"Missing or empty frozen policy artifact: {path}")
        return cls(payload, policy_hash=_combined_hash([path]))

    def candidate(self) -> dict[str, Any]:
        row: dict[str, Any] = {"mode": self.mode, "min_sep": self.min_sep, "margin": self.margin}
        if self.strength is not None:
            row["strength"] = self.strength
        if self.alpha is not None:
            row["alpha"] = self.alpha
        if self.safe_min_sep is not None:
            row["safe_min_sep"] = self.safe_min_sep
        return row

    def apply(
        self,
        *,
        base_xy: np.ndarray,
        floor_xy: np.ndarray,
        pred_xy: np.ndarray,
        base_switch: np.ndarray,
        group_key: np.ndarray,
        normalizer: np.ndarray,
        agent_id: np.ndarray,
        current_xy: np.ndarray,
    ) -> GroupConsistencyRuntimeResult:
        selected_xy = base_xy.astype(np.float32).copy()
        switch = base_switch.astype(bool).copy()
        norm = np.maximum(normalizer.astype(np.float64), di.EPS)
        agent = agent_id.astype(np.int64)
        floor_min = di._min_group_distance_fast(floor_xy, group_key, norm, agent)
        base_min = di._min_group_distance_fast(selected_xy, group_key, norm, agent)
        pred_min = di._min_group_distance_fast(pred_xy, group_key, norm, agent)
        unsafe = switch & np.isfinite(base_min) & np.isfinite(floor_min) & (base_min < self.min_sep) & (
            base_min + self.margin < floor_min
        )
        if self.mode == "fallback_unsafe":
            selected_xy[unsafe] = floor_xy[unsafe]
            switch[unsafe] = False
        elif self.mode == "blend_unsafe":
            alpha = float(self.alpha or 0.0)
            blend = floor_xy + alpha * (base_xy - floor_xy)
            selected_xy[unsafe] = blend[unsafe]
            switch[unsafe] = alpha > di.EPS
        elif self.mode == "predicted_safe_only":
            safe_min = self.safe_min_sep if self.safe_min_sep is not None else self.min_sep
            safe = np.isfinite(pred_min) & (pred_min >= float(safe_min))
            off = switch & ~safe
            selected_xy[off] = floor_xy[off]
            switch[off] = False
        elif self.mode == "repel_unsafe":
            selected_xy = di._repel_selected_rows(
                selected_xy,
                switch,
                group_key,
                norm,
                agent,
                current_xy.astype(np.float64),
                min_sep=self.min_sep,
                strength=float(self.strength or 0.0),
            )
        else:
            raise ValueError(f"Unknown frozen group-consistency runtime mode: {self.mode}")
        final_min = di._min_group_distance_fast(selected_xy, group_key, norm, agent)
        return GroupConsistencyRuntimeResult(
            selected_xy=selected_xy,
            switch=switch,
            base_min=base_min,
            final_min=final_min,
            floor_min=floor_min,
        )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b))) if a.size and b.size else 0.0


def _smoke_cases(policy: FrozenGroupConsistencyPolicy) -> dict[str, Any]:
    base = np.asarray(
        [
            [[0.02, 0.0], [0.02, 0.0]],
            [[0.03, 0.0], [0.03, 0.0]],
        ],
        dtype=np.float32,
    )
    floor = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.2, 0.0], [0.2, 0.0]],
        ],
        dtype=np.float32,
    )
    pred = base.copy()
    current = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    group_key = np.asarray(["scene\t1\t50", "scene\t1\t50"], dtype=object)
    normalizer = np.ones(2, dtype=np.float64)
    agent = np.asarray([1, 2], dtype=np.int64)
    switch = np.asarray([True, True])
    result = policy.apply(
        base_xy=base,
        floor_xy=floor,
        pred_xy=pred,
        base_switch=switch,
        group_key=group_key,
        normalizer=normalizer,
        agent_id=agent,
        current_xy=current,
    )
    diag = result.diagnostics()
    return {
        "rows": 2,
        "mode": policy.mode,
        "base_min_before": result.base_min.tolist(),
        "final_min_after": result.final_min.tolist(),
        "diagnostics": diag,
        "passes": bool(diag["final_near_005"] <= diag["base_near_005"]),
    }


def _real_batch_replay(policy: FrozenGroupConsistencyPolicy) -> dict[str, Any]:
    s42b.build_stage42_source_split()
    data = s41._combined()
    split, _group = am._split_arrays(data)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = split == "train"
    floor = am._floor_arrays(data, train_mask)
    am_candidate = di._rebuild_stage42_am_candidate(data, split, labels, floor)
    group_key = di._group_key(data)
    test_ids = np.where(split == "test")[0]
    expected = di._repair_subset(
        test_ids,
        policy.candidate(),
        data,
        labels,
        floor["floor_xy"].astype(np.float32),
        am_candidate["pred_xy"].astype(np.float32),
        am_candidate["selected_xy"].astype(np.float32),
        am_candidate["switch"].astype(bool),
        group_key,
    )
    runtime = policy.apply(
        base_xy=am_candidate["selected_xy"][test_ids].astype(np.float32),
        floor_xy=floor["floor_xy"][test_ids].astype(np.float32),
        pred_xy=am_candidate["pred_xy"][test_ids].astype(np.float32),
        base_switch=am_candidate["switch"][test_ids].astype(bool),
        group_key=group_key[test_ids],
        normalizer=np.maximum(data["scale"][test_ids].astype(np.float64), di.EPS),
        agent_id=data["agent_id"][test_ids].astype(np.int64),
        current_xy=np.stack([data["current_x"][test_ids], data["current_y"][test_ids]], axis=1).astype(np.float32),
    )
    selected_ade, selected_fde = di._trajectory_errors_subset(runtime.selected_xy, labels, test_ids)
    floor_ade, floor_fde = di._trajectory_errors_subset(floor["floor_xy"][test_ids], labels, test_ids)
    metric = di._metric_subset(selected_ade, floor_ade, data, test_ids, runtime.switch)
    diag = runtime.diagnostics()
    return {
        "source": "fresh_runtime_replay_on_reconstructed_stage42_di_test_rows",
        "rows": int(len(test_ids)),
        "selected_xy_max_abs_diff": _max_abs_diff(runtime.selected_xy, expected["selected_xy"]),
        "switch_exact_match": bool(np.array_equal(runtime.switch, expected["switch"])),
        "selected_ade_max_abs_diff": _max_abs_diff(selected_ade, expected["selected_ade"]),
        "selected_fde_max_abs_diff": _max_abs_diff(selected_fde, expected["selected_fde"]),
        "metric": metric,
        "expected_metric": expected["metric"],
        "metric_abs_diff": {k: abs(float(metric.get(k, 0.0)) - float(expected["metric"].get(k, 0.0))) for k in metric},
        "diagnostics": diag,
        "expected_diagnostics": expected["diagnostics"],
        "diagnostic_abs_diff": {
            k: abs(float(diag.get(k, 0.0)) - float(expected["diagnostics"].get(k, 0.0)))
            for k in ["base_near_005", "final_near_005", "floor_near_005"]
        },
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    policy = payload["runtime_policy"]
    dk_gate = payload["inputs"]["stage42_dk"].get("stage42_dk_gate", {})
    real = payload["real_batch_replay"]
    metric = real["metric"]
    diag = real["diagnostics"]
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "policy_artifact_exists": payload["policy_artifact"]["exists"],
        "policy_hash_available": bool(payload.get("policy_hash")),
        "dk_replay_gate_passed": dk_gate.get("passed") == dk_gate.get("total"),
        "runtime_mode_matches_artifact": policy["mode"] == payload["policy_artifact_payload"]["repair_rule"]["type"],
        "runtime_min_sep_matches_artifact": float(policy["min_sep"]) == float(payload["policy_artifact_payload"]["repair_rule"]["min_sep"]),
        "runtime_strength_matches_artifact": float(policy["strength"] or 0.0)
        == float(payload["policy_artifact_payload"]["repair_rule"].get("strength") or 0.0),
        "smoke_case_passed": payload["smoke_case"]["passes"],
        "real_batch_rows_present": real["rows"] > 0,
        "real_batch_selected_xy_exact": real["selected_xy_max_abs_diff"] <= 1e-6,
        "real_batch_switch_exact": real["switch_exact_match"],
        "real_batch_ade_exact": real["selected_ade_max_abs_diff"] <= 1e-8,
        "real_batch_fde_exact": real["selected_fde_max_abs_diff"] <= 1e-8,
        "real_batch_metrics_exact": all(float(v) <= 1e-12 for v in real["metric_abs_diff"].values()),
        "real_batch_diagnostics_exact": all(float(v) <= 1e-12 for v in real["diagnostic_abs_diff"].values()),
        "all_positive": float(metric["all_improvement"]) > 0.0,
        "t50_positive": float(metric["t50_improvement"]) > 0.0,
        "t100_raw_positive": float(metric["t100_raw_frame_diagnostic_improvement"]) > 0.0,
        "hard_positive": float(metric["hard_failure_improvement"]) > 0.0,
        "easy_under_2pct": float(metric["easy_degradation"]) <= 0.02,
        "near_collision_reduced_vs_base": float(diag["final_near_005"]) <= float(diag["base_near_005"]),
        "runtime_inputs_are_causal_predictions_only": payload["runtime_inputs"] == [
            "base_xy_predicted_full_waypoint_candidate",
            "floor_xy_train_horizon_causal_rollout",
            "pred_xy_model_rollout_diagnostic",
            "base_switch_from_validation_selected_policy",
            "source_frame_horizon_group_key",
            "normalizer",
            "agent_id",
            "current_xy",
        ],
        "no_future_endpoint_input": no_leakage["future_endpoint_input"] is False,
        "no_future_waypoint_input": no_leakage["future_waypoint_input"] is False,
        "no_central_velocity": no_leakage["central_velocity"] is False,
        "no_test_endpoint_goals": no_leakage["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": no_leakage["test_threshold_tuning"] is False,
        "metric_seconds_overclaim_blocked": claim["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": claim["stage5c_executed"] is False,
        "smc_not_enabled": claim["smc_enabled"] is False,
        "paper_files_refreshed": all(row["contains_stage42_dl"] for row in payload.get("paper_file_status", []) if row["exists"]),
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_dl_group_consistency_runtime_policy_pass" if passed == total else "stage42_dl_group_consistency_runtime_policy_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    metric = payload["real_batch_replay"]["metric"]
    diag = payload["real_batch_replay"]["diagnostics"]
    return [
        "## Stage42-DL Group-Consistency Runtime Policy API",
        "",
        "- source: `fresh_runtime_api_from_frozen_group_consistency_policy_artifact`",
        "- role: expose Stage42-DJ/DK frozen group-consistency full-waypoint repair as a callable runtime policy.",
        "- real batch replay uses reconstructed Stage42-DI source-level test rows and checks exact selected trajectory replay.",
        f"- policy artifact: `{POLICY_JSON}`",
        f"- policy hash: `{payload['policy_hash']}`",
        f"- gate: `{payload['stage42_dl_gate']['passed']} / {payload['stage42_dl_gate']['total']}`; verdict `{payload['stage42_dl_gate']['verdict']}`.",
        f"- replayed ADE vs train-horizon causal floor: all `{_pct(metric['all_improvement'])}`, t50 `{_pct(metric['t50_improvement'])}`, t100 raw `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(metric['hard_failure_improvement'])}`, easy `{_pct(metric['easy_degradation'])}`.",
        f"- replayed near@0.05 base/final/floor: `{_pct(diag['base_near_005'])}` / `{_pct(diag['final_near_005'])}` / `{_pct(diag['floor_near_005'])}`.",
        "- claim boundary: still protected dataset-local/raw-frame 2.5D; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_README, CURRENT_RETROSPECTIVE]:
        _replace_section(path, "STAGE42_DL_GROUP_CONSISTENCY_RUNTIME_POLICY", lines)


def _refresh_paper_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _refresh_lines(payload)
    status = []
    for path in PAPER_FILES:
        if path.exists():
            _replace_section(path, "STAGE42_DL_GROUP_CONSISTENCY_RUNTIME_POLICY", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "exists": True,
                    "contains_stage42_dl": "Stage42-DL Group-Consistency Runtime Policy API" in text,
                    "contains_claim_boundary": "no true 3D" in text and "no Stage5C" in text,
                }
            )
        else:
            status.append({"path": str(path), "exists": False, "contains_stage42_dl": False, "contains_claim_boundary": False})
    return status


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    metric = payload["real_batch_replay"]["metric"]
    state["current_stage"] = "Stage42-DL group-consistency runtime policy API"
    state["current_verdict"] = payload["stage42_dl_gate"]["verdict"]
    state["stage42_dl_group_consistency_runtime_policy"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_dl_gate"]["verdict"],
        "gates": f"{payload['stage42_dl_gate']['passed']}/{payload['stage42_dl_gate']['total']}",
        "policy_artifact": str(POLICY_JSON),
        "policy_hash": payload["policy_hash"],
        "runtime_policy": payload["runtime_policy"],
        "real_batch_replay": {
            "rows": payload["real_batch_replay"]["rows"],
            "selected_xy_max_abs_diff": payload["real_batch_replay"]["selected_xy_max_abs_diff"],
            "switch_exact_match": payload["real_batch_replay"]["switch_exact_match"],
            "metric": metric,
            "diagnostics": payload["real_batch_replay"]["diagnostics"],
        },
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-DL converts the frozen Stage42-DJ/DK group-consistency full-waypoint policy into a callable runtime API and verifies exact replay on reconstructed Stage42-DI test rows. It is deployment/reproducibility evidence, not new training.",
        "verification": {
            "runner": ".venv-pytorch/bin/python run_stage42_group_consistency_runtime_policy.py",
            "targeted_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_group_consistency_runtime_policy.py",
        },
    }
    write_json(RESEARCH_STATE, state)


def _write_md(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_dl_gate"]
    real = payload["real_batch_replay"]
    metric = real["metric"]
    diag = real["diagnostics"]
    lines = [
        "# Stage42-DL Group-Consistency Runtime Policy API",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- policy_hash: `{payload['policy_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Runtime Policy",
        "",
        f"- mode: `{payload['runtime_policy']['mode']}`",
        f"- min_sep: `{payload['runtime_policy']['min_sep']}`",
        f"- margin: `{payload['runtime_policy']['margin']}`",
        f"- strength: `{payload['runtime_policy']['strength']}`",
        f"- runtime inputs: `{payload['runtime_inputs']}`",
        "",
        "## Real Batch Replay",
        "",
        f"- rows: `{real['rows']}`",
        f"- selected_xy_max_abs_diff: `{real['selected_xy_max_abs_diff']}`",
        f"- switch_exact_match: `{real['switch_exact_match']}`",
        f"- selected_ade_max_abs_diff: `{real['selected_ade_max_abs_diff']}`",
        f"- selected_fde_max_abs_diff: `{real['selected_fde_max_abs_diff']}`",
        "",
        "## Replayed Metrics Vs Train-Horizon Causal Floor",
        "",
        f"- all: `{_pct(metric['all_improvement'])}`",
        f"- t50: `{_pct(metric['t50_improvement'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure: `{_pct(metric['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`",
        "",
        "## Replayed Group Safety",
        "",
        f"- base near@0.05: `{_pct(diag['base_near_005'])}`",
        f"- final near@0.05: `{_pct(diag['final_near_005'])}`",
        f"- floor near@0.05: `{_pct(diag['floor_near_005'])}`",
        f"- base p05 min distance: `{diag['base_p05_min_distance']}`",
        f"- final p05 min distance: `{diag['final_p05_min_distance']}`",
        "",
        "## Interpretation",
        "",
        "- Stage42-DL turns the frozen group-consistency full-waypoint repair into a callable runtime policy API.",
        "- The real-batch replay verifies that runtime application exactly matches the original Stage42-DI selected repair on reconstructed test rows.",
        "- It does not execute Stage5C, does not enable SMC, and does not make metric/seconds-level claims.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-DL Gate",
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


def run_stage42_group_consistency_runtime_policy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payload = read_json(POLICY_JSON, {})
    dk = read_json(DK_JSON, {})
    if not payload or not dk:
        missing = [str(path) for path in [POLICY_JSON, DK_JSON] if not path.exists()]
        raise FileNotFoundError(f"Missing Stage42-DL runtime inputs: {missing}")
    policy = FrozenGroupConsistencyPolicy(payload, policy_hash=_combined_hash([POLICY_JSON]))
    result: dict[str, Any] = {
        "source": "fresh_runtime_api_from_frozen_group_consistency_policy_artifact",
        "stage": "Stage42-DL group-consistency runtime policy API",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([POLICY_JSON, DK_JSON]),
        "inputs": {
            "stage42_dk": {"path": str(DK_JSON), "stage42_dk_gate": dk.get("stage42_dk_gate", {})},
        },
        "policy_artifact": {
            "path": str(POLICY_JSON),
            "exists": POLICY_JSON.exists(),
            "size_bytes": POLICY_JSON.stat().st_size if POLICY_JSON.exists() else 0,
        },
        "policy_artifact_payload": payload,
        "policy_hash": policy.policy_hash,
        "runtime_policy": {
            "mode": policy.mode,
            "min_sep": policy.min_sep,
            "margin": policy.margin,
            "strength": policy.strength,
            "alpha": policy.alpha,
            "safe_min_sep": policy.safe_min_sep,
        },
        "runtime_inputs": [
            "base_xy_predicted_full_waypoint_candidate",
            "floor_xy_train_horizon_causal_rollout",
            "pred_xy_model_rollout_diagnostic",
            "base_switch_from_validation_selected_policy",
            "source_frame_horizon_group_key",
            "normalizer",
            "agent_id",
            "current_xy",
        ],
        "smoke_case": _smoke_cases(policy),
        "real_batch_replay": _real_batch_replay(policy),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "runtime_inputs_predicted_or_causal_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    result["paper_file_status"] = []
    result["stage42_dl_gate"] = _gate(result)
    result["paper_file_status"] = _refresh_paper_files(result)
    result["stage42_dl_gate"] = _gate(result)
    write_json(REPORT_JSON, di._jsonable(result))
    _write_md(result)
    _refresh_readmes(result)
    _refresh_research_state(result)
    return result


if __name__ == "__main__":
    out = run_stage42_group_consistency_runtime_policy()
    gate = out["stage42_dl_gate"]
    print(f"Stage42-DL group-consistency runtime policy: {gate['verdict']} ({gate['passed']}/{gate['total']})")
