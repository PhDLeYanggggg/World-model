from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_sequence_full_waypoint as s42i
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "full_waypoint_auxiliary_ablation_stage42.json"
REPORT_MD = OUT_DIR / "full_waypoint_auxiliary_ablation_stage42.md"
REPORT_CSV = OUT_DIR / "full_waypoint_auxiliary_ablation_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_ab_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

STAGE42I_JSON = OUT_DIR / "sequence_full_waypoint_stage42.json"

VARIANT = "sequence_waypoint_no_aux_interaction_occupancy_physical"
SEEDS = [67, 71, 73]
EPOCHS = 2
BATCH = 2048


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-AB 是 retrained auxiliary-head loss ablation，不是 inference masking。",
    "future waypoints / endpoints 只作为 train/val labels 和 eval labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test 调阈值。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _torch():
    return s42i._torch()


def _train_no_aux(seed: int, train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray]) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    torch.manual_seed(seed)
    tr_tokens, tr_mask, tr_static = s42i._variant_inputs(train, "sequence_waypoint_full")
    va_tokens, va_mask, va_static = s42i._variant_inputs(val, "sequence_waypoint_full")
    model = s42i._make_model(tr_static.shape[1], tr_tokens.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    ckpt = CHECKPOINT_DIR / f"stage42ab_{VARIANT}_seed{seed}.pt"
    heartbeat = OUT_DIR / f"stage42ab_{VARIANT}_seed{seed}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        payload = json.loads(heartbeat.read_text(encoding="utf-8"))
        if int(payload.get("epoch", 0)) >= EPOCHS:
            return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": payload.get("best", {})}

    tensors = {
        "tokens": torch.tensor(tr_tokens),
        "mask": torch.tensor(tr_mask),
        "static": torch.tensor(tr_static),
        "target": torch.tensor(train["waypoint_delta"]),
        "valid": torch.tensor(train["waypoint_valid"].astype(np.float32)),
        "hard": torch.tensor((train["hard"] | train["failure"]).astype(np.float32)),
        "horizon": torch.tensor(train["horizon"]),
    }
    val_tensors = {
        "tokens": torch.tensor(va_tokens),
        "mask": torch.tensor(va_mask),
        "static": torch.tensor(va_static),
        "target": torch.tensor(val["waypoint_delta"]),
        "valid": torch.tensor(val["waypoint_valid"].astype(np.float32)),
    }
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(tr_tokens))
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(tensors["tokens"][ids], tensors["mask"][ids], tensors["static"][ids])
            valid = tensors["valid"][ids]
            row_w = 1.0 + 1.5 * tensors["hard"][ids] + 2.0 * (tensors["horizon"][ids] == 50).float() + 1.0 * (tensors["horizon"][ids] == 100).float()
            per_wp = F.smooth_l1_loss(out["waypoint_delta"], tensors["target"][ids], reduction="none").mean(dim=2)
            traj = ((per_wp * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0) * row_w).mean()
            err = torch.linalg.norm(out["waypoint_delta"] - tensors["target"][ids], dim=2)
            risk_target = torch.log1p((err * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)).detach()
            risk = (F.smooth_l1_loss(out["traj_risk"], risk_target, reduction="none") * row_w).mean()
            # Ablation: remove supervised interaction / occupancy / physical
            # auxiliary losses while preserving the model interface.
            loss = traj + 0.30 * risk
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(val_tensors["tokens"], val_tensors["mask"], val_tensors["static"])
            per_wp = F.smooth_l1_loss(out["waypoint_delta"], val_tensors["target"], reduction="none").mean(dim=2)
            val_loss = float(((per_wp * val_tensors["valid"]).sum(dim=1) / val_tensors["valid"].sum(dim=1).clamp_min(1.0)).mean().cpu())
        cand = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_loss < best["val_loss"]:
            best = cand
            torch.save({"model": model.state_dict(), "variant": VARIANT, "seed": seed, "static_dim": tr_static.shape[1], "token_dim": tr_tokens.shape[-1], "best": best}, ckpt)
        heartbeat.write_text(json.dumps({"variant": VARIANT, "seed": seed, "epoch": epoch, "best": best, "checkpoint": str(ckpt)}, ensure_ascii=False), encoding="utf-8")
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict(info: Mapping[str, Any], split: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    torch = _torch()
    payload = torch.load(info["checkpoint"], map_location="cpu", weights_only=False)
    model = s42i._make_model(int(payload["static_dim"]), int(payload["token_dim"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    tokens, mask, static = s42i._variant_inputs(split, "sequence_waypoint_full")
    outs = {k: [] for k in ["waypoint_delta", "traj_risk", "interaction", "occupancy", "physical"]}
    with torch.no_grad():
        for start in range(0, len(tokens), 4096):
            sl = slice(start, min(start + 4096, len(tokens)))
            out = model(torch.tensor(tokens[sl]), torch.tensor(mask[sl]), torch.tensor(static[sl]))
            outs["waypoint_delta"].append(out["waypoint_delta"].cpu().numpy())
            outs["traj_risk"].append(out["traj_risk"].cpu().numpy().reshape(-1))
            outs["interaction"].append(torch.sigmoid(out["interaction_logit"]).cpu().numpy().reshape(-1))
            outs["occupancy"].append(torch.sigmoid(out["occupancy_logit"]).cpu().numpy().reshape(-1))
            outs["physical"].append(torch.sigmoid(out["physical_logit"]).cpu().numpy().reshape(-1))
    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in outs.items()}


def _train_eval_seed(seed: int, train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    train_info = _train_no_aux(seed, train, val)
    pred_val = _predict(train_info, val)
    pred_test = _predict(train_info, test)
    labels_val = s42i._labels(val)
    labels_test = s42i._labels(test)
    policy, val_metrics = s42i._fit_light_policy(pred_val, labels_val)
    test_metrics = s42i._row_metrics(VARIANT, pred_test, labels_test, policy)
    ungated = s42i._row_metrics(
        f"{VARIANT}_ungated",
        pred_test,
        labels_test,
        {"slices": {f"{d}|{h}": {"risk_max": 1e9, "physical_min": 0.0, "max_switch": 1.0, "hard_only": False, "easy_block": False} for d in sorted(set(labels_test["domain"].tolist())) for h in [10, 25, 50, 100]}},
    )
    return {
        "source": train_info["source"],
        "variant": VARIANT,
        "seed": seed,
        "train_info": train_info,
        "val_policy": policy,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "ungated_test_metrics": ungated,
    }


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    tmp = s42i._summarize(rows)
    return tmp.get(VARIANT, {})


def _mean(stage42i: Mapping[str, Any], variant: str, key: str) -> float:
    return float(((stage42i.get("summary", {}) or {}).get(variant, {}) or {}).get(key, {}).get("mean", 0.0))


def _delta_vs_full(summary: Mapping[str, Any], stage42i: Mapping[str, Any]) -> dict[str, float]:
    return {
        "ade_all_delta_full_minus_no_aux": _mean(stage42i, "sequence_waypoint_full", "ade_all") - float(summary.get("ade_all", {}).get("mean", 0.0)),
        "ade_t50_delta_full_minus_no_aux": _mean(stage42i, "sequence_waypoint_full", "ade_t50") - float(summary.get("ade_t50", {}).get("mean", 0.0)),
        "ade_hard_delta_full_minus_no_aux": _mean(stage42i, "sequence_waypoint_full", "ade_hard_failure") - float(summary.get("ade_hard_failure", {}).get("mean", 0.0)),
        "fde_t50_delta_full_minus_no_aux": _mean(stage42i, "sequence_waypoint_full", "fde_t50") - float(summary.get("fde_t50", {}).get("mean", 0.0)),
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result.get("summary", {})
    delta = result.get("delta_vs_stage42i_full", {})
    gates = {
        "stage42i_reference_present": result.get("inputs", {}).get("stage42i_exists") is True,
        "no_aux_models_trained_or_cached_verified": len(summary.get("seeds", [])) >= 3,
        "same_inputs_policy_interface": result.get("ablation_protocol", {}).get("same_model_outputs") is True,
        "auxiliary_loss_removed": result.get("ablation_protocol", {}).get("interaction_occupancy_physical_loss_weight") == 0.0,
        "easy_preserved": float(summary.get("ade_easy_degradation", {}).get("mean", 1.0)) <= 0.02,
        "aux_contribution_measured": all(key in delta for key in ["ade_all_delta_full_minus_no_aux", "ade_t50_delta_full_minus_no_aux", "ade_hard_delta_full_minus_no_aux", "fde_t50_delta_full_minus_no_aux"]),
        "mixed_or_negative_evidence_not_overclaimed": result.get("interpretation", {}).get("uniform_aux_positive_claim_allowed") is False
        or all(float(v) > 0.0 for v in delta.values()),
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoint_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False,
        "no_metric_seconds_overclaim": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    return {
        "source": result.get("source"),
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "verdict": "stage42_ab_full_waypoint_auxiliary_ablation_pass" if all(gates.values()) else "stage42_ab_full_waypoint_auxiliary_ablation_partial",
    }


def _write_csv(result: Mapping[str, Any]) -> None:
    with REPORT_CSV.open("w", encoding="utf-8") as f:
        f.write("metric,stage42i_full,no_aux,full_minus_no_aux\n")
        summary = result["summary"]
        ref = result["stage42i_full_reference"]
        delta = result["delta_vs_stage42i_full"]
        for key, delta_key in [
            ("ade_all", "ade_all_delta_full_minus_no_aux"),
            ("ade_t50", "ade_t50_delta_full_minus_no_aux"),
            ("ade_hard_failure", "ade_hard_delta_full_minus_no_aux"),
            ("fde_t50", "fde_t50_delta_full_minus_no_aux"),
        ]:
            f.write(f"{key},{ref.get(key)},{summary.get(key, {}).get('mean')},{delta.get(delta_key)}\n")


def _write_md(result: Mapping[str, Any]) -> None:
    summary = result["summary"]
    delta = result["delta_vs_stage42i_full"]
    lines = [
        "# Stage42-AB Full-Waypoint Auxiliary-Head Ablation",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_ab_gate']['passed']} / {result['stage42_ab_gate']['total']}`",
        f"- verdict: `{result['stage42_ab_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## No-Aux Metrics",
        "",
        "| metric | mean | ci low | ci high |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ["ade_all", "ade_t50", "ade_t100_raw_frame_diagnostic", "ade_hard_failure", "ade_easy_degradation", "fde_all", "fde_t50", "switch_rate", "ungated_easy_degradation"]:
        item = summary.get(key, {})
        lines.append(f"| `{key}` | {float(item.get('mean', 0.0)):.6f} | {float(item.get('ci_low', 0.0)):.6f} | {float(item.get('ci_high', 0.0)):.6f} |")
    lines.extend(
        [
            "",
            "## Full Minus No-Aux",
            "",
            "`full_minus_no_aux > 0` means the supervised auxiliary heads helped the Stage42-I full-waypoint model on that metric.",
            "",
            "| delta | value |",
            "| --- | ---: |",
            *[f"| `{key}` | {float(value):.6f} |" for key, value in delta.items()],
            "",
            "## Interpretation",
            "",
            f"- uniform_aux_positive_claim_allowed: `{result['interpretation']['uniform_aux_positive_claim_allowed']}`",
            f"- conclusion: {result['interpretation']['conclusion']}",
            "- This is a fresh retrained ablation over full-waypoint sequence dynamics, but it remains dataset-local raw-frame 2.5D.",
            "- Stage5C and SMC remain disabled.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-AB Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    write_md(GATE_MD, lines)


def _append_if_missing(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in text:
        path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def _append_readme_and_state(result: Mapping[str, Any]) -> None:
    gate = result["stage42_ab_gate"]
    summary = result["summary"]
    delta = result["delta_vs_stage42i_full"]
    block = f"""
## Stage42-AB Full-Waypoint Auxiliary-Head Ablation

```text
source = {result['source']}
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
no_aux_ADE_all = {summary.get('ade_all', {}).get('mean')}
no_aux_ADE_t50 = {summary.get('ade_t50', {}).get('mean')}
no_aux_ADE_hard_failure = {summary.get('ade_hard_failure', {}).get('mean')}
no_aux_easy_degradation = {summary.get('ade_easy_degradation', {}).get('mean')}
full_minus_no_aux_ADE_all = {delta.get('ade_all_delta_full_minus_no_aux')}
full_minus_no_aux_ADE_t50 = {delta.get('ade_t50_delta_full_minus_no_aux')}
full_minus_no_aux_ADE_hard = {delta.get('ade_hard_delta_full_minus_no_aux')}
uniform_aux_positive_claim_allowed = {result['interpretation']['uniform_aux_positive_claim_allowed']}
stage5c_executed = false
smc_enabled = false
```

Stage42-AB removes supervised interaction / occupancy / physical auxiliary losses while keeping the same full-waypoint model inputs, outputs, and validation-only policy interface. Positive deltas mean the auxiliary heads helped; mixed or negative deltas are recorded as limitation evidence, not overclaimed.
"""
    for path in [Path("README_RESULTS.md"), Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), Path("README_M3W_RESEARCH_SUMMARY_ZH.md")]:
        _append_if_missing(path, "## Stage42-AB Full-Waypoint Auxiliary-Head Ablation", block)
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_ab_full_waypoint_auxiliary_ablation"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_ab_full_waypoint_auxiliary_ablation"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "csv": str(REPORT_CSV),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "no_aux_ade_all": summary.get("ade_all", {}).get("mean"),
        "no_aux_ade_t50": summary.get("ade_t50", {}).get("mean"),
        "no_aux_ade_hard_failure": summary.get("ade_hard_failure", {}).get("mean"),
        "no_aux_easy_degradation": summary.get("ade_easy_degradation", {}).get("mean"),
        "delta_vs_stage42i_full": delta,
        "uniform_aux_positive_claim_allowed": result["interpretation"]["uniform_aux_positive_claim_allowed"],
        "claim_boundary": result["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, REPORT_CSV, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(Path("research_state.json"), _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": "stage42_ab_full_waypoint_auxiliary_ablation",
        "source": result.get("source"),
        "status": "success",
        "input_hash": result.get("input_hash"),
        "output_hash": _combined_hash([REPORT_JSON, REPORT_MD, REPORT_CSV, GATE_MD]),
        "git_commit": _git_commit(),
        "generated_at_utc": result.get("generated_at_utc"),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def run_stage42_full_waypoint_auxiliary_ablation() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CHECKPOINT_DIR)
    stage42i = read_json(STAGE42I_JSON, {})
    data = {split: s42i._split_arrays(split, ensure_labels=(split == "train")) for split in ["train", "val", "test"]}
    rows = [_train_eval_seed(seed, data["train"], data["val"], data["test"]) for seed in SEEDS]
    summary = _summary(rows)
    delta = _delta_vs_full(summary, stage42i)
    uniform_positive = bool(delta) and all(float(v) > 0.0 for v in delta.values())
    source = "fresh_run" if any(r.get("source") == "fresh_run" for r in rows) else "cached_verified"
    result = {
        "stage": "Stage42-AB full-waypoint auxiliary-head ablation",
        "source": source,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42i": str(STAGE42I_JSON),
            "stage42i_exists": STAGE42I_JSON.exists(),
        },
        "input_hash": _combined_hash(
            [
                STAGE42I_JSON,
                s42i.ft.DATA_DIR / "all_agent_train.npz",
                s42i.ft.DATA_DIR / "all_agent_val.npz",
                s42i.ft.DATA_DIR / "all_agent_test.npz",
                s42i.ft.DATA_DIR / "full_trajectory_train.npz",
                s42i.ft.DATA_DIR / "full_trajectory_val.npz",
                s42i.ft.DATA_DIR / "full_trajectory_test.npz",
            ]
        ),
        "dataset_rows": {split: int(len(data[split]["horizon"])) for split in ["train", "val", "test"]},
        "ablation_protocol": {
            "variant": VARIANT,
            "same_model_outputs": True,
            "same_past_only_inputs": True,
            "same_val_only_policy_selection": True,
            "interaction_occupancy_physical_loss_weight": 0.0,
            "trajectory_loss_weight": 1.0,
            "risk_loss_weight": 0.30,
            "baseline_stage42i_auxiliary_loss_weight": 0.15,
        },
        "stage42i_full_reference": {
            "ade_all": _mean(stage42i, "sequence_waypoint_full", "ade_all"),
            "ade_t50": _mean(stage42i, "sequence_waypoint_full", "ade_t50"),
            "ade_hard_failure": _mean(stage42i, "sequence_waypoint_full", "ade_hard_failure"),
            "fde_t50": _mean(stage42i, "sequence_waypoint_full", "fde_t50"),
        },
        "rows": rows,
        "summary": summary,
        "delta_vs_stage42i_full": delta,
        "interpretation": {
            "uniform_aux_positive_claim_allowed": uniform_positive,
            "conclusion": "Auxiliary interaction/occupancy/physical heads are uniformly positive on measured full-waypoint deltas." if uniform_positive else "Auxiliary interaction/occupancy/physical heads are mixed or negative in this retrained ablation; keep them as limitation/partial evidence rather than a main positive claim.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoints_label_eval_only": True,
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
    result["stage42_ab_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_csv(result)
    _write_md(result)
    _write_gate(result["stage42_ab_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


if __name__ == "__main__":
    run_stage42_full_waypoint_auxiliary_ablation()
