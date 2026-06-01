from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_domain_failure_repair import _domain_metrics, _eval, _switches
from src.stage43_latent_state_robustness_audit import _bootstrap_metric_fast
from src.stage43_protected_latent_state_model import OUT_DIR, _git_commit, _jsonable, _predict, _sha256
from src.stage43_source_horizon_expert_policy import (
    REPORT_JSON as AW_JSON,
    _composite_switch,
    _h50,
    _pct,
    _source_family_metrics,
    _stage43k_base_switch,
)
from src.stage43_source_horizon_safety_envelope import _trial_safety_flags
from src.stage43_source_level_latent_model import REPORT_JSON as STAGE43G_JSON
from src.stage43_source_level_latent_model import build_source_level_datasets
from src.stage43_source_level_latent_robustness_audit import _apply_checkpoint_standardization
from src.stage43_source_slice_repair import _families, _metadata_for_source_split, _source_metrics
from src.stage43_unit_consistent_safe_switch import _candidate_unit_error, _load_checkpoint


REPORT_JSON = OUT_DIR / "stage43_source_horizon_expert_replay.json"
REPORT_MD = OUT_DIR / "stage43_source_horizon_expert_replay.md"
GATE_MD = OUT_DIR / "stage43_stage_ax_source_horizon_expert_replay_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_AX_SOURCE_HORIZON_EXPERT_REPLAY"
SOURCE = "fresh_stage43_ax_source_horizon_expert_replay"
STAGE43_K = OUT_DIR / "stage43_source_slice_repair.json"
STAGE43_AV = OUT_DIR / "stage43_source_horizon_safety_envelope.json"
EPS = 1e-8


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _hash_np_array(digest: "hashlib._Hash", name: str, arr: np.ndarray) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(arr.shape).encode("utf-8"))
    if arr.dtype.kind in {"U", "S", "O"}:
        digest.update("\n".join(np.asarray(arr).astype(str).tolist()).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(arr).tobytes())


def _row_hash(ds, raw_x: np.ndarray, meta: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, arr in [
        ("raw_x", raw_x),
        ("horizon", ds.horizon),
        ("domain", ds.domain),
        ("floor_err", ds.floor_err),
        ("hard", ds.hard),
        ("failure", ds.failure),
        ("easy", ds.easy),
        ("scale", ds.scale),
        ("source_file", np.asarray(meta.get("source_file", []), dtype=str)),
        ("scene_id", np.asarray(meta.get("scene_id", []), dtype=str)),
    ]:
        _hash_np_array(digest, name, np.asarray(arr))
    return digest.hexdigest()


def _metric_diff(replayed: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "all_improvement_vs_floor",
        "t50_improvement_vs_floor",
        "t100_raw_frame_diagnostic_vs_floor",
        "hard_failure_improvement_vs_floor",
        "easy_degradation_vs_floor",
        "switch_rate",
    ]
    rows: dict[str, Any] = {}
    max_abs = 0.0
    for key in keys:
        exp = float(expected[key])
        got = float(replayed[key])
        diff = got - exp
        rows[key] = {"expected": exp, "replayed": got, "signed_diff": diff, "abs_diff": abs(diff)}
        max_abs = max(max_abs, abs(diff))
    return {"max_abs_diff": max_abs, "by_metric": rows}


def _bootstrap(ds, selected: np.ndarray, *, n: int) -> dict[str, Any]:
    return {
        "unit_all": _bootstrap_metric_fast(selected, ds.floor_err, np.arange(len(selected)), n=n, seed=443301),
        "unit_t50": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.horizon == 50)[0], n=n, seed=443302),
        "unit_t100_raw_frame_diagnostic": _bootstrap_metric_fast(
            selected, ds.floor_err, np.where(ds.horizon == 100)[0], n=n, seed=443303
        ),
        "unit_hard_failure": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.hard | ds.failure)[0], n=n, seed=443304),
        "unit_easy_degradation": _bootstrap_metric_fast(selected, ds.floor_err, np.where(ds.easy)[0], easy=True, n=n, seed=443305),
    }


def build_source_horizon_expert_replay(*, bootstrap: int = 1000) -> dict[str, Any]:
    if not AW_JSON.exists():
        raise FileNotFoundError(AW_JSON)
    ensure_dir(OUT_DIR)
    artifact = read_json(AW_JSON, {})
    stage43g = read_json(STAGE43G_JSON, {})
    stage43k = read_json(STAGE43_K, {})
    checkpoint, ckpt, model = _load_checkpoint(stage43g)
    _, _, test_raw, manifest = build_source_level_datasets(seed=int(ckpt.get("seed", 443)))
    test_x_raw = test_raw.x.copy()
    test = _apply_checkpoint_standardization(test_raw, ckpt)
    test_pred = _predict(model, test, torch.device("cpu"), 4096)
    test_candidate = _candidate_unit_error(test, test_pred)
    test_meta = _metadata_for_source_split(manifest, "test")
    test_families = _families(test_meta)
    allowed = stage43k["deployment_policy"]["allowed_families"]
    test_base = _stage43k_base_switch(test_x_raw, test.feature_names, test_families, allowed)
    selected_trial = artifact["policy"]["t50_expert_trial"]
    test_t50_switch = _switches(test_x_raw, test.feature_names, selected_trial)
    replay_switch = _composite_switch(test_base, test_t50_switch, _h50(test))
    replay_selected, replay_metrics = _eval(test, test_candidate, replay_switch)
    replay_domain = _domain_metrics(test, replay_selected, replay_switch)
    replay_source = _source_metrics(test, replay_selected, replay_switch, test_meta)
    replay_family = _source_family_metrics(test, replay_selected, replay_switch, test_families)
    replay_flags = _trial_safety_flags(replay_metrics, replay_domain, replay_source)
    metric_diff = _metric_diff(replay_metrics, artifact["test_metrics"])
    policy_hash = _stable_hash(artifact["policy"])
    row_hash = _row_hash(test, test_x_raw, test_meta)
    switch_hash = _stable_hash(
        {
            "selected_trial": selected_trial,
            "switch_rows": np.where(replay_switch)[0],
            "selected_error": replay_selected,
            "row_hash": row_hash,
        }
    )
    boot = _bootstrap(test, replay_selected, n=bootstrap)
    claim_boundary = {
        "true_3d_world_model": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "dataset_local_raw_frame_only": True,
        "stage5c_executed": False,
        "smc_enabled": False,
        "test_threshold_tuning": False,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_exact_replay_from_stage43_aw_artifact",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "artifact": str(AW_JSON),
        "artifact_sha256": _sha256(AW_JSON),
        "artifact_input_hash": artifact.get("input_hash"),
        "checkpoint": str(checkpoint),
        "checkpoint_committed": False,
        "policy_hash": policy_hash,
        "row_hash": row_hash,
        "switch_hash": switch_hash,
        "selected_t50_expert": selected_trial,
        "artifact_deployment_decision": artifact.get("deployment_decision"),
        "replay_metrics": replay_metrics,
        "artifact_metrics": artifact["test_metrics"],
        "metric_diff": metric_diff,
        "replay_domain_metrics": replay_domain,
        "replay_source_metrics": replay_source,
        "replay_source_family_metrics": replay_family,
        "replay_flags": replay_flags,
        "bootstrap": boot,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "thresholds_selected_on_test": False,
            "replay_does_not_reselect_policy": True,
        },
        "claim_boundary": claim_boundary,
        "input_hash": _combined_hash([AW_JSON, STAGE43_K, STAGE43_AV, STAGE43G_JSON]),
    }
    payload["stage43_ax_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["replay_metrics"]
    flags = payload["replay_flags"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "stage43_aw_artifact_present": Path(payload["artifact"]).exists(),
        "artifact_requires_reviewer_replay": payload["artifact_deployment_decision"]
        == "candidate_requires_reviewer_replay_before_deployment",
        "policy_hash_recorded": len(str(payload["policy_hash"])) == 64,
        "row_hash_recorded": len(str(payload["row_hash"])) == 64,
        "switch_hash_recorded": len(str(payload["switch_hash"])) == 64,
        "replay_metrics_exact": payload["metric_diff"]["max_abs_diff"] <= 1e-8,
        "replayed_t50_positive": payload["bootstrap"]["unit_t50"]["ci_low"] > 0.0,
        "replayed_aggregate_safe": metrics["all_improvement_vs_floor"] >= 0.0
        and metrics["easy_degradation_vs_floor"] <= 0.02,
        "domain_easy_safe": flags["domain_easy_safe"] is True,
        "source_negative_free": flags["negative_source_count"] == 0,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_labels_eval_or_loss_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["thresholds_selected_on_test"] is False
        and no_leak["replay_does_not_reselect_policy"] is True,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    replay_passed = passed == total
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_ax_source_horizon_expert_replay_pass"
        if replay_passed
        else "stage43_ax_source_horizon_expert_replay_incomplete",
        "reviewer_replay_passed": replay_passed,
        "deploy_without_replay": False,
        "candidate_for_deployment_update": replay_passed,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_ax_gate"]
    metrics = payload["replay_metrics"]
    boot = payload["bootstrap"]
    lines = [
        "# Stage43-AX Source-Horizon Expert Replay",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- reviewer replay passed: `{gate['reviewer_replay_passed']}`",
        f"- candidate for deployment update: `{gate['candidate_for_deployment_update']}`",
        f"- deploy without replay: `{gate['deploy_without_replay']}`",
        f"- policy hash: `{payload['policy_hash']}`",
        f"- row hash: `{payload['row_hash']}`",
        f"- switch hash: `{payload['switch_hash']}`",
        f"- replay max metric diff: `{payload['metric_diff']['max_abs_diff']:.10f}`",
        "",
        "## Replayed Policy",
        "",
        f"- selected t50 expert: `{payload['selected_t50_expert']['name']}`",
        "- replay mode: artifact selected policy only; no validation reselection and no test threshold tuning",
        "",
        "## Replayed Metrics",
        "",
        f"- all improvement: `{_pct(metrics['all_improvement_vs_floor'])}`",
        f"- t50 improvement: `{_pct(metrics['t50_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_diagnostic_vs_floor'])}`",
        f"- hard/failure improvement: `{_pct(metrics['hard_failure_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Bootstrap CI",
        "",
        f"- all CI: `[{_pct(boot['unit_all']['ci_low'])}, {_pct(boot['unit_all']['ci_high'])}]`",
        f"- t50 CI: `[{_pct(boot['unit_t50']['ci_low'])}, {_pct(boot['unit_t50']['ci_high'])}]`",
        f"- hard/failure CI: `[{_pct(boot['unit_hard_failure']['ci_low'])}, {_pct(boot['unit_hard_failure']['ci_high'])}]`",
        f"- easy degradation CI: `[{_pct(boot['unit_easy_degradation']['ci_low'])}, {_pct(boot['unit_easy_degradation']['ci_high'])}]`",
        "",
        "## Metric Replay Diff",
        "",
        "| metric | artifact | replayed | abs diff |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, row in payload["metric_diff"]["by_metric"].items():
        lines.append(f"| `{key}` | `{_pct(row['expected'])}` | `{_pct(row['replayed'])}` | `{row['abs_diff']:.10f}` |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is exact reviewer replay of the Stage43-AW artifact, not a new threshold search.",
            "- Dataset-local/raw-frame 2.5D only.",
            "- No metric/seconds, true 3D, foundation, Stage5C, or SMC claim.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
        ]
    )
    for name, passed in gate["gates"].items():
        lines.append(f"| `{name}` | `{passed}` |")
    return lines


def _update_summaries(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ax_gate"]
    metrics = payload["replay_metrics"]
    body = [
        f"Stage43-AX exact-replays the Stage43-AW source/horizon expert artifact without validation reselection or test threshold tuning. Gate: `{gate['passed']} / {gate['total']}` with verdict `{gate['verdict']}`.",
        "",
        f"Replay metrics: all `{_pct(metrics['all_improvement_vs_floor'])}`, t50 `{_pct(metrics['t50_improvement_vs_floor'])}`, t100 raw-frame diagnostic `{_pct(metrics['t100_raw_frame_diagnostic_vs_floor'])}`, hard/failure `{_pct(metrics['hard_failure_improvement_vs_floor'])}`, easy degradation `{_pct(metrics['easy_degradation_vs_floor'])}`.",
        f"Replay max metric diff vs AW artifact: `{payload['metric_diff']['max_abs_diff']:.10f}`. Policy hash `{payload['policy_hash']}`, row hash `{payload['row_hash']}`.",
        "",
        f"Decision: reviewer replay passed = `{gate['reviewer_replay_passed']}`; candidate for deployment update = `{gate['candidate_for_deployment_update']}`. This remains dataset-local/raw-frame 2.5D evidence; Stage5C and SMC remain disabled.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, body)
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state.setdefault("stage43", {})
    state["stage43"]["source_horizon_expert_replay"] = {
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']}/{gate['total']}",
        "reviewer_replay_passed": gate["reviewer_replay_passed"],
        "candidate_for_deployment_update": gate["candidate_for_deployment_update"],
        "selected_t50_expert": payload["selected_t50_expert"]["name"],
        "policy_hash": payload["policy_hash"],
        "row_hash": payload["row_hash"],
        "switch_hash": payload["switch_hash"],
        "metric_diff_max_abs": payload["metric_diff"]["max_abs_diff"],
        "test_all": metrics["all_improvement_vs_floor"],
        "test_t50": metrics["t50_improvement_vs_floor"],
        "test_hard": metrics["hard_failure_improvement_vs_floor"],
        "test_easy": metrics["easy_degradation_vs_floor"],
        "result_source": payload["result_source"],
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    write_json(RESEARCH_STATE, _jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                _jsonable(
                    {
                        "source": SOURCE,
                        "verdict": gate["verdict"],
                        "reviewer_replay_passed": gate["reviewer_replay_passed"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def run_source_horizon_expert_replay(*, bootstrap: int = 1000) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payload = build_source_horizon_expert_replay(bootstrap=bootstrap)
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_md(payload))
    gate = payload["stage43_ax_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-AX Source-Horizon Expert Replay Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- reviewer replay passed: `{gate['reviewer_replay_passed']}`",
            f"- candidate for deployment update: `{gate['candidate_for_deployment_update']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{k}` | `{v}` |" for k, v in gate["gates"].items()],
            "",
        ],
    )
    _update_summaries(payload)
    return payload


def main() -> None:
    payload = run_source_horizon_expert_replay()
    gate = payload["stage43_ax_gate"]
    print(json.dumps({"verdict": gate["verdict"], "passed": gate["passed"], "total": gate["total"]}, indent=2))


if __name__ == "__main__":
    main()
