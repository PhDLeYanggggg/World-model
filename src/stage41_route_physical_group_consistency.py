from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_group_consistency_distiller as gcd
from src import stage41_goal_route_physical_repair as gr
from src import stage41_route_physical_policy_integration as rpi
from src.stage41_group_consistency_multiseed_repair import _fit_safety_buffer_policy, _deployable


OUT_DIR = gcd.OUT_DIR
CHECKPOINT_DIR = gcd.CHECKPOINT_DIR
REPORT_JSON = OUT_DIR / "stage41_route_physical_group_consistency.json"
REPORT_MD = OUT_DIR / "stage41_route_physical_group_consistency.md"
SEED = 4159
BATCH = gcd.BATCH
EPOCHS = gcd.EPOCHS


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


def _append_ledger(step: str, status: str, started: float, inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    ensure_dir(OUT_DIR)
    entry = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": step,
        "source": "fresh_run",
        "status": status,
        "wall_time_s": time.perf_counter() - started,
        "input_hash": _combined_hash(inputs),
        "output_hash": _combined_hash(outputs),
        "git_commit": _git_commit(),
    }
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _route_checkpoint_paths() -> list[str]:
    report = read_json(OUT_DIR / "stage41_goal_route_physical_repair.json", {})
    if not report:
        from src.stage41_goal_route_physical_repair import train_goal_route_physical_repair

        train_goal_route_physical_repair()
    return rpi._checkpoint_paths(OUT_DIR / "stage41_goal_route_physical_repair.json", "goal_route")


def _route_aux_features(split: str, n_rows: int | None = None) -> np.ndarray:
    pred, labels = gr._predict_ensemble(_route_checkpoint_paths(), split)
    if n_rows is not None and len(labels["route"]) != n_rows:
        raise ValueError(f"route/physical rows are not aligned for {split}: {len(labels['route'])} != {n_rows}")
    prob = gr._softmax(pred["route_logits"].astype(np.float64)).astype(np.float32)
    feat = rpi._route_features(pred)
    entropy = (-np.sum(prob * np.log(np.maximum(prob, 1e-6)), axis=1, keepdims=True)).astype(np.float32)
    aux = np.concatenate(
        [
            prob,
            feat["route_conf"].astype(np.float32)[:, None],
            feat["non_straight"].astype(np.float32)[:, None],
            feat["hard_route"].astype(np.float32)[:, None],
            feat["physical_challenge"].astype(np.float32)[:, None],
            entropy,
        ],
        axis=1,
    ).astype(np.float32)
    return aux


def _bundle(split: str, checkpoint: str | Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    base = gcd._bundle(split, checkpoint, policy)
    aux = _route_aux_features(split, len(base["x"]))
    out = dict(base)
    out["x"] = np.concatenate([base["x"].astype(np.float32), aux], axis=1).astype(np.float32)
    out["route_physical_aux"] = aux
    out["feature_dim"] = np.asarray([out["x"].shape[1]], dtype=np.int64)
    return out


def _train_model(train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray]) -> dict[str, Any]:
    torch = gcd._torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    ckpt = CHECKPOINT_DIR / "stage41_route_physical_group_consistency.pt"
    heartbeat = OUT_DIR / "route_physical_group_consistency_heartbeat.json"
    x = torch.tensor(train["x"])
    y_gain = torch.tensor(train["gain"])
    y_safe = torch.tensor(train["safe_switch"])
    y_unsafe = torch.tensor(train["unsafe"])
    hard = torch.tensor((train["hard"].astype(bool) | train["failure"].astype(bool)).astype(np.float32))
    vx = torch.tensor(val["x"])
    vg = torch.tensor(val["gain"])
    vs = torch.tensor(val["safe_switch"])
    vu = torch.tensor(val["unsafe"])
    torch.manual_seed(SEED)
    model = gcd._make_model(x.shape[1], width=104, dropout=0.07)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(x.shape[0])
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(x[ids])
            row_w = 1.0 + 1.8 * hard[ids] + 2.0 * torch.clamp(torch.abs(y_gain[ids]), max=2.0)
            gain_loss = (F.smooth_l1_loss(out["gain"], y_gain[ids], reduction="none") * row_w).mean()
            safe_loss = (F.binary_cross_entropy_with_logits(out["safe_logit"], y_safe[ids], reduction="none") * row_w).mean()
            unsafe_loss = (F.binary_cross_entropy_with_logits(out["unsafe_logit"], y_unsafe[ids], reduction="none") * row_w).mean()
            loss = gain_loss + 1.25 * safe_loss + 1.1 * unsafe_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(vx)
            val_loss = float(
                (
                    F.smooth_l1_loss(out["gain"], vg)
                    + 1.25 * F.binary_cross_entropy_with_logits(out["safe_logit"], vs)
                    + 1.1 * F.binary_cross_entropy_with_logits(out["unsafe_logit"], vu)
                ).cpu()
            )
        candidate = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
        heartbeat.write_text(
            json.dumps({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "checkpoint": str(ckpt), "best": candidate}, ensure_ascii=False),
            encoding="utf-8",
        )
        if val_loss < best["val_loss"]:
            best = candidate
            torch.save({"model": model.state_dict(), "dim": int(x.shape[1]), "best": best, "width": 104, "dropout": 0.07, "seed": SEED}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    return {
        "all_delta": float(a.get("all_improvement", 0.0) - b.get("all_improvement", 0.0)),
        "t50_delta": float(a.get("t50_improvement", 0.0) - b.get("t50_improvement", 0.0)),
        "t100_delta": float(a.get("t100_improvement", 0.0) - b.get("t100_improvement", 0.0)),
        "hard_delta": float(a.get("hard_failure_improvement", 0.0) - b.get("hard_failure_improvement", 0.0)),
        "easy_delta": float(a.get("easy_degradation", 0.0) - b.get("easy_degradation", 0.0)),
    }


def run_route_physical_group_consistency() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    checkpoint, repaired_policy, policy_source = gcd._load_policy_and_checkpoint()
    train = _bundle("train", checkpoint, repaired_policy)
    val = _bundle("val", checkpoint, repaired_policy)
    test = _bundle("test", checkpoint, repaired_policy)
    train_result = _train_model(train, val)
    scores_val = gcd._predict(train_result["checkpoint"], val)
    policy, candidates = _fit_safety_buffer_policy(scores_val, val)
    policy["type"] = "route_physical_group_consistency_safety_buffer"
    scores_test = gcd._predict(train_result["checkpoint"], test)
    test_metrics, test_switch = gcd._policy_metrics(scores_test, test, policy)
    deployable = _deployable(test_metrics, float(np.mean(test_switch)))
    group_report = read_json(OUT_DIR / "stage41_group_consistency_distiller.json", {})
    group_metrics = group_report.get("test_metrics") or {}
    multiseed_repair = read_json(OUT_DIR / "stage41_group_consistency_multiseed_repair.json", {})
    multiseed_summary = multiseed_repair.get("metric_summary") or {}
    multiseed_mean = {
        "all_improvement": (multiseed_summary.get("all_improvement") or {}).get("mean", 0.0),
        "t50_improvement": (multiseed_summary.get("t50_improvement") or {}).get("mean", 0.0),
        "t100_improvement": (multiseed_summary.get("t100_improvement") or {}).get("mean", 0.0),
        "hard_failure_improvement": (multiseed_summary.get("hard_failure_improvement") or {}).get("mean", 0.0),
        "easy_degradation": (multiseed_summary.get("easy_degradation") or {}).get("max", 0.0),
    }
    lift_over_group_distiller = _delta(test_metrics, group_metrics)
    lift_over_multiseed_mean = _delta(test_metrics, multiseed_mean)
    route_physical_contributes = bool(
        deployable
        and (
            lift_over_group_distiller["all_delta"] > 0
            or lift_over_group_distiller["t50_delta"] > 0
            or lift_over_group_distiller["hard_delta"] > 0
        )
    )
    result = {
        "source": "fresh_run",
        "protocol": "route_physical_augmented_group_consistency",
        "policy_source": policy_source,
        "train": train_result,
        "selected_policy": policy,
        "val_candidates_count": len(candidates),
        "test_metrics": test_metrics,
        "route_physical_group_consistency_deployable": deployable,
        "route_physical_contributes_to_group_policy": route_physical_contributes,
        "lift_over_group_consistency_distiller": lift_over_group_distiller,
        "lift_over_multiseed_safety_buffer_mean": lift_over_multiseed_mean,
        "route_physical_feature_dim": int(train["route_physical_aux"].shape[1]),
        "no_leakage": {
            "route_physical_predictions_from_past_only_model": True,
            "future_route_label_input": False,
            "future_physical_label_input": False,
            "future_waypoints_input": False,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "Route/physical labels are supervised targets only. This tests whether their past-only predictions improve the joint-safe group consistency head; it is not metric, true 3D, Stage5C, or SMC.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Route/Physical Group Consistency",
            "",
            "- source: `fresh_run`",
            "- protocol: `route_physical_augmented_group_consistency`",
            f"- deployable: `{deployable}`",
            f"- route/physical contributes to group policy: `{route_physical_contributes}`",
            f"- selected policy: `{policy}`",
            f"- test metrics: `{test_metrics}`",
            f"- lift over group consistency distiller: `{lift_over_group_distiller}`",
            f"- lift over multiseed safety-buffer mean: `{lift_over_multiseed_mean}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "This is a deployment-policy contribution test. If negative, route/physical heads remain diagnostic-only.",
        ],
    )
    return result


def main_route_physical_group_consistency() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_route_physical_group_consistency()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_route_physical_group_consistency",
            status,
            started,
            [
                OUT_DIR / "stage41_group_consistency_distiller.json",
                OUT_DIR / "stage41_group_consistency_multiseed_repair.json",
                OUT_DIR / "stage41_goal_route_physical_repair.json",
            ],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_route_physical_group_consistency()
