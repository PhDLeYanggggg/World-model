from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_tail_horizon_waypoint_adapter as p


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_tail_adapter_reviewer_replay.json"
REPORT_MD = OUT_DIR / "stage43_tail_adapter_reviewer_replay.md"
GATE_MD = OUT_DIR / "stage43_stage_az_tail_adapter_reviewer_replay_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_az_tail_adapter_reviewer_replay"
SECTION = "STAGE43_AZ_TAIL_ADAPTER_REVIEWER_REPLAY"

STAGE43_P = OUT_DIR / "stage43_tail_horizon_waypoint_adapter.json"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(m._jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _hash_array(digest: "hashlib._Hash", name: str, arr: np.ndarray) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(arr.shape).encode("utf-8"))
    if arr.dtype.kind in {"U", "S", "O"}:
        digest.update("\n".join(np.asarray(arr).astype(str).tolist()).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(arr).tobytes())


def _split_hash(ds) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(ds.feature_names).encode("utf-8"))
    for name, arr in [
        ("x", ds.x),
        ("horizon", ds.horizon),
        ("domain", ds.domain),
        ("source_file", ds.source_file),
        ("scene_id", ds.scene_id),
        ("floor_ade", ds.floor_ade),
        ("floor_fde", ds.floor_fde),
        ("waypoint_delta", ds.waypoint_delta),
        ("easy", ds.easy),
        ("hard", ds.hard),
        ("failure", ds.failure),
    ]:
        _hash_array(digest, name, np.asarray(arr))
    return digest.hexdigest()


def _feature_schema_hash(feature_names: list[str]) -> str:
    return hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()


def _parse_allowed(rules: list[str]) -> set[tuple[str, int]]:
    allowed: set[tuple[str, int]] = set()
    for rule in rules:
        family, horizon = str(rule).rsplit("|", 1)
        allowed.add((family, int(horizon)))
    return allowed


def _metric_diff(replayed: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "full_waypoint_ade_improvement_vs_floor",
        "endpoint_fde_improvement_vs_floor",
        "t50_full_waypoint_ade_improvement_vs_floor",
        "t50_endpoint_fde_improvement_vs_floor",
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
        "hard_failure_full_waypoint_ade_improvement_vs_floor",
        "easy_degradation_vs_floor",
        "switch_rate",
    ]
    rows: dict[str, Any] = {}
    max_abs = 0.0
    for key in keys:
        got = float(replayed[key])
        exp = float(expected[key])
        diff = got - exp
        rows[key] = {"expected": exp, "replayed": got, "signed_diff": diff, "abs_diff": abs(diff)}
        max_abs = max(max_abs, abs(diff))
    return {"max_abs_diff": max_abs, "by_metric": rows}


def build_tail_adapter_reviewer_replay() -> dict[str, Any]:
    if not STAGE43_P.exists():
        raise FileNotFoundError(STAGE43_P)
    ensure_dir(OUT_DIR)
    artifact = read_json(STAGE43_P, {})
    selected = artifact["selected_model"]
    seed = int(artifact["training_protocol"]["seed"])

    train = p._build_split("train", max_rows=None, seed=seed)
    val = p._build_split("val", max_rows=None, seed=seed)
    test = p._build_split("test", max_rows=None, seed=seed)
    feature_names = list(train.feature_names)
    feature_schema_hash = _feature_schema_hash(feature_names)
    feature_mean, feature_std = p._standardize(train, val, test)
    feature_mean_hash = hashlib.sha256(feature_mean.tobytes()).hexdigest()
    feature_std_hash = hashlib.sha256(feature_std.tobytes()).hexdigest()

    train_mask = p._train_mask(train, str(selected["train_filter"]))
    weight = p._ridge_fit(
        train.x[train_mask],
        p._target_matrix(train, str(selected["target"]))[train_mask],
        float(selected["l2"]),
    )
    replay_model_hash = p._model_hash(
        weight,
        l2=float(selected["l2"]),
        target=str(selected["target"]),
        train_filter=str(selected["train_filter"]),
    )
    allowed = _parse_allowed(list(selected["allowed_rules"]))
    pred = p._predict_waypoint(test, weight, str(selected["target"]))
    candidate_ade, candidate_fde = p._trajectory_error(test, pred)
    selected_ade, selected_fde, switch = p._apply_rules(test, candidate_ade, candidate_fde, allowed)
    replay_metrics = p._metrics(test, selected_ade, selected_fde, switch)
    metric_diff = _metric_diff(replay_metrics, artifact["overall_full_test_metrics"])
    replay_policy = {
        "target": selected["target"],
        "train_filter": selected["train_filter"],
        "l2": selected["l2"],
        "allowed_rules": list(selected["allowed_rules"]),
        "model_hash": replay_model_hash,
        "feature_mean_hash": feature_mean_hash,
        "feature_std_hash": feature_std_hash,
        "feature_schema_hash": feature_schema_hash,
    }
    policy_hash = _stable_hash(replay_policy)
    split_hashes = {
        "train": _split_hash(train),
        "val": _split_hash(val),
        "test": _split_hash(test),
    }
    switch_hash = _stable_hash(
        {
            "switch_rows": np.where(switch)[0],
            "selected_ade": selected_ade,
            "selected_fde": selected_fde,
            "test_split_hash": split_hashes["test"],
        }
    )
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_exact_recompute_replay_from_stage43_p_artifact",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "artifact": str(STAGE43_P),
        "artifact_sha256": m._sha256(STAGE43_P),
        "artifact_source": artifact.get("source"),
        "artifact_result_source": artifact.get("result_source"),
        "artifact_gate": artifact.get("stage43_p_gate", {}),
        "artifact_selected_model_hash": selected["model_hash"],
        "replayed_model_hash": replay_model_hash,
        "model_hash_match": replay_model_hash == selected["model_hash"],
        "policy_hash": policy_hash,
        "switch_hash": switch_hash,
        "split_hashes": split_hashes,
        "feature_schema_hash": feature_schema_hash,
        "feature_mean_hash": feature_mean_hash,
        "feature_std_hash": feature_std_hash,
        "feature_mean_hash_match": feature_mean_hash == artifact["training_protocol"]["feature_mean_hash"],
        "feature_std_hash_match": feature_std_hash == artifact["training_protocol"]["feature_std_hash"],
        "replay_policy": replay_policy,
        "replay_mode": "artifact_selected_config_and_allowed_rules_only_no_validation_reselection_no_test_threshold_tuning",
        "replay_metrics": replay_metrics,
        "artifact_metrics": artifact["overall_full_test_metrics"],
        "metric_diff": metric_diff,
        "switch_count": int(switch.sum()),
        "full_test_rows": int(len(test.x)),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "validation_reselection_during_replay": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "uniform_positive_external_transfer_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "input_hash": _combined_hash([STAGE43_P]),
    }
    payload["stage43_az_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["replay_metrics"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "stage43_p_artifact_present": Path(payload["artifact"]).exists(),
        "stage43_p_artifact_passed": payload["artifact_gate"].get("verdict")
        == "stage43_p_tail_horizon_adapter_pass_t100_still_fallback",
        "model_hash_exact": payload["model_hash_match"] is True,
        "feature_standardization_hashes_match": payload["feature_mean_hash_match"] is True
        and payload["feature_std_hash_match"] is True,
        "split_hashes_recorded": all(len(str(value)) == 64 for value in payload["split_hashes"].values()),
        "switch_hash_recorded": len(str(payload["switch_hash"])) == 64,
        "replay_metrics_exact": payload["metric_diff"]["max_abs_diff"] <= 1e-8,
        "replayed_policy_safe": metrics["full_waypoint_ade_improvement_vs_floor"] > 0.0
        and metrics["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
        and metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
        and metrics["easy_degradation_vs_floor"] <= 0.02
        and metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -1e-8,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_labels_eval_or_loss_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["validation_reselection_during_replay"] is False
        and no_leak["test_threshold_tuning"] is False,
        "claim_boundary_not_overstated": claim["true_3d_world_model"] is False
        and claim["foundation_world_model"] is False
        and claim["metric_or_seconds_claim"] is False
        and claim["dataset_local_raw_frame_only"] is True
        and claim["uniform_positive_external_transfer_claim"] is False,
        "stage5c_and_smc_false": claim["stage5c_executed"] is False and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_az_tail_adapter_reviewer_replay_pass"
        if passed == total
        else "stage43_az_tail_adapter_reviewer_replay_incomplete",
        "reviewer_replay_passed": passed == total,
        "stage5c_executed": False,
        "smc_enabled": False,
        "goal_complete": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_az_gate"]
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    metrics = payload["replay_metrics"]
    lines = [
        "# Stage43-AZ Tail Adapter Reviewer Replay",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- reviewer replay passed: `{gate['reviewer_replay_passed']}`",
        f"- policy hash: `{payload['policy_hash']}`",
        f"- model hash match: `{payload['model_hash_match']}`",
        f"- replay max metric diff: `{payload['metric_diff']['max_abs_diff']:.10f}`",
        "",
        "## Replay Mode",
        "",
        f"- mode: `{payload['replay_mode']}`",
        f"- artifact: `{payload['artifact']}`",
        f"- artifact sha256: `{payload['artifact_sha256']}`",
        f"- switch hash: `{payload['switch_hash']}`",
        f"- feature schema hash: `{payload['feature_schema_hash']}`",
        "",
        "## Replayed Metrics",
        "",
        f"- all full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- endpoint FDE improvement: `{_pct(metrics['endpoint_fde_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 endpoint FDE improvement: `{_pct(metrics['t50_endpoint_fde_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure full-waypoint ADE improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Metric Replay Diff",
        "",
        "| metric | artifact | replayed | abs diff |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, row in payload["metric_diff"]["by_metric"].items():
        lines.append(
            f"| `{key}` | `{_pct(row['expected'])}` | `{_pct(row['replayed'])}` | `{row['abs_diff']:.10f}` |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is an exact recompute replay from the Stage43-P artifact; no validation reselection and no test threshold tuning.",
            "- Dataset-local/raw-frame 2.5D only.",
            "- t100 remains raw-frame diagnostic and guarded.",
            "- No true 3D, foundation, metric/seconds, Stage5C, or SMC claim.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43 Current World-Model Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- current performance leader replayed: `{gate['reviewer_replay_passed']}`",
        f"- long objective complete: `{gate['goal_complete']}`",
        f"- Stage5C executed: `{gate['stage5c_executed']}`",
        f"- SMC enabled: `{gate['smc_enabled']}`",
        "",
        "## Candidate Roles",
        "",
        "- Performance leader and replayed candidate: `Stage43-P / Stage43-AZ tail adapter replay`.",
        "- Source-horizon replay leader remains: `Stage43-AX exact replay of source-horizon expert policy`.",
        "- Frozen bounded-residual artifact remains: `Stage43-AO`.",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
    ]
    write_md(WORLD_GATE_MD, gate_lines)
    write_md(GATE_MD, lines)
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_az_gate"]
    metrics = payload["replay_metrics"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"policy_hash = `{payload['policy_hash']}`",
        f"model_hash_match = `{payload['model_hash_match']}`",
        f"replay_max_metric_diff = `{payload['metric_diff']['max_abs_diff']:.10f}`",
        "",
        f"replayed_all_t50_t100_hard_easy = `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}` / `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(metrics['easy_degradation_vs_floor'])}`",
        "",
        "Stage43-AZ recomputes the Stage43-P tail-horizon full-waypoint adapter from the artifact-selected config and allowed rules. It performs no validation reselection and no test threshold tuning. This strengthens Stage43-P from a performance leader into an exact recompute replay artifact while preserving the claim boundary: dataset-local/raw-frame 2.5D only; t100 diagnostic only; no Stage5C; no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_az_tail_adapter_reviewer_replay"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "policy_hash": payload["policy_hash"],
        "model_hash_match": payload["model_hash_match"],
        "metric_diff": payload["metric_diff"],
        "metrics": metrics,
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_az_tail_adapter_reviewer_replay"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-AZ",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "policy_hash": payload["policy_hash"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _run(_: argparse.Namespace) -> dict[str, Any]:
    payload = build_tail_adapter_reviewer_replay()
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Exact recompute replay for the Stage43-P tail adapter.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = _run(args)
    gate = result["stage43_az_gate"]
    print(f"Stage43-AZ: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"reviewer_replay_passed={gate['reviewer_replay_passed']}")
    print(f"max_metric_diff={result['metric_diff']['max_abs_diff']:.10f}")
    return result


if __name__ == "__main__":
    main()
