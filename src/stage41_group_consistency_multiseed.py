from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_group_consistency_distiller as gcd


OUT_DIR = gcd.OUT_DIR
CHECKPOINT_DIR = gcd.CHECKPOINT_DIR
REPORT_JSON = OUT_DIR / "stage41_group_consistency_multiseed.json"
REPORT_MD = OUT_DIR / "stage41_group_consistency_multiseed.md"
SEEDS = [11, 17, 23]


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


def _summarize_metric(rows: list[Mapping[str, Any]], key: str) -> dict[str, float]:
    vals = np.asarray([float(row["test_metrics"].get(key, 0.0)) for row in rows], dtype=np.float64)
    return {
        "mean": float(vals.mean()) if len(vals) else 0.0,
        "std": float(vals.std(ddof=0)) if len(vals) else 0.0,
        "min": float(vals.min()) if len(vals) else 0.0,
        "max": float(vals.max()) if len(vals) else 0.0,
    }


def _positive_domains(metrics: Mapping[str, Any]) -> int:
    return sum(
        1
        for row in (metrics.get("by_domain") or {}).values()
        if row.get("all_improvement", 0.0) > 0
        or row.get("t50_improvement", 0.0) > 0
        or row.get("hard_failure_improvement", 0.0) > 0
    )


def _train_seed(seed: int, train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray]) -> dict[str, Any]:
    torch = gcd._torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    ckpt = CHECKPOINT_DIR / f"stage41_group_consistency_distiller_seed{seed}.pt"
    heartbeat = OUT_DIR / f"group_consistency_distiller_seed{seed}_heartbeat.json"
    x = torch.tensor(train["x"])
    y_gain = torch.tensor(train["gain"])
    y_safe = torch.tensor(train["safe_switch"])
    y_unsafe = torch.tensor(train["unsafe"])
    hard = torch.tensor((train["hard"].astype(bool) | train["failure"].astype(bool)).astype(np.float32))
    vx = torch.tensor(val["x"])
    vg = torch.tensor(val["gain"])
    vs = torch.tensor(val["safe_switch"])
    vu = torch.tensor(val["unsafe"])
    torch.manual_seed(int(seed))
    model = gcd._make_model(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    rng = np.random.default_rng(gcd.SEED + int(seed))
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, gcd.EPOCHS + 1):
        order = rng.permutation(x.shape[0])
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), gcd.BATCH):
            ids = torch.tensor(order[start : start + gcd.BATCH], dtype=torch.long)
            out = model(x[ids])
            row_w = 1.0 + 1.5 * hard[ids] + 2.0 * torch.clamp(torch.abs(y_gain[ids]), max=2.0)
            gain_loss = (F.smooth_l1_loss(out["gain"], y_gain[ids], reduction="none") * row_w).mean()
            safe_loss = (F.binary_cross_entropy_with_logits(out["safe_logit"], y_safe[ids], reduction="none") * row_w).mean()
            unsafe_loss = (F.binary_cross_entropy_with_logits(out["unsafe_logit"], y_unsafe[ids], reduction="none") * row_w).mean()
            loss = gain_loss + 1.2 * safe_loss + unsafe_loss
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
                    + 1.2 * F.binary_cross_entropy_with_logits(out["safe_logit"], vs)
                    + F.binary_cross_entropy_with_logits(out["unsafe_logit"], vu)
                ).cpu()
            )
        heartbeat.write_text(
            json.dumps(
                _jsonable(
                    {
                        "seed": int(seed),
                        "epoch": epoch,
                        "train_loss": float(np.mean(losses)),
                        "val_loss": val_loss,
                        "checkpoint": str(ckpt),
                        "best": best,
                    }
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
            torch.save({"model": model.state_dict(), "dim": int(x.shape[1]), "best": best, "width": 96, "dropout": 0.06, "seed": int(seed)}, ckpt)
    return {"source": "fresh_run", "seed": int(seed), "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _replicate_seed(seed: int, train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    trained = _train_seed(seed, train, val)
    scores_val = gcd._predict(trained["checkpoint"], val)
    policy, val_candidates = gcd._fit_policy(scores_val, val)
    scores_test = gcd._predict(trained["checkpoint"], test)
    test_metrics, test_switch = gcd._policy_metrics(scores_test, test, policy)
    deployable = bool(
        test_metrics.get("all_improvement", 0.0) > 0
        and test_metrics.get("t50_improvement", 0.0) > 0
        and test_metrics.get("t100_improvement", 0.0) > 0
        and test_metrics.get("hard_failure_improvement", 0.0) > 0
        and test_metrics.get("easy_degradation", 1.0) <= 0.02
        and test_metrics.get("collision_delta_vs_floor_005", 1.0) <= 0.01
        and float(np.mean(test_switch)) > 0.0
    )
    return {
        "seed": int(seed),
        "train": trained,
        "selected_policy": policy,
        "val_candidates_count": len(val_candidates),
        "test_metrics": test_metrics,
        "positive_external_domains": _positive_domains(test_metrics),
        "deployable": deployable,
        "no_leakage": {
            "future_waypoints_input": False,
            "future_labels_eval_only": True,
            "train_gain_safe_unsafe_labels_only": True,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def run_group_consistency_multiseed() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    checkpoint, repaired_policy, policy_source = gcd._load_policy_and_checkpoint()
    train = gcd._bundle("train", checkpoint, repaired_policy)
    val = gcd._bundle("val", checkpoint, repaired_policy)
    test = gcd._bundle("test", checkpoint, repaired_policy)
    seed_results = [_replicate_seed(seed, train, val, test) for seed in SEEDS]
    metric_summary = {
        key: _summarize_metric(seed_results, key)
        for key in [
            "all_improvement",
            "t50_improvement",
            "t100_improvement",
            "hard_failure_improvement",
            "easy_degradation",
            "switch_rate",
            "collision_delta_vs_floor_005",
        ]
    }
    positive_domain_counts = [int(row["positive_external_domains"]) for row in seed_results]
    replication_pass = bool(
        all(row.get("deployable") for row in seed_results)
        and metric_summary["all_improvement"]["min"] > 0
        and metric_summary["t50_improvement"]["min"] > 0
        and metric_summary["t100_improvement"]["min"] > 0
        and metric_summary["hard_failure_improvement"]["min"] > 0
        and metric_summary["easy_degradation"]["max"] <= 0.02
        and metric_summary["collision_delta_vs_floor_005"]["max"] <= 0.01
        and min(positive_domain_counts or [0]) >= 2
    )
    result = {
        "source": "fresh_run",
        "protocol": "group_consistency_distiller_multiseed",
        "policy_source": policy_source,
        "seeds": SEEDS,
        "seed_results": seed_results,
        "metric_summary": metric_summary,
        "positive_domain_counts": positive_domain_counts,
        "replication_pass": replication_pass,
        "no_leakage": {
            "future_waypoints_input": False,
            "future_labels_eval_only": True,
            "train_gain_safe_unsafe_labels_only": True,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "This is multi-seed replication of the neural group-consistency head. It remains dataset-local raw-frame 2.5D and does not execute Stage5C or SMC.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Group Consistency Distiller Multi-Seed Replication",
            "",
            "- source: `fresh_run`",
            f"- seeds: `{SEEDS}`",
            f"- replication pass: `{replication_pass}`",
            f"- metric summary: `{metric_summary}`",
            f"- positive domain counts: `{positive_domain_counts}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "All policies are selected on validation and evaluated on test once. Coordinates remain dataset-local/raw-frame; this is not true 3D, foundation-scale, Stage5C, or SMC.",
        ],
    )
    return result


def main_group_consistency_multiseed() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_group_consistency_multiseed()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_group_consistency_multiseed",
            status,
            started,
            [
                OUT_DIR / "stage41_group_consistency_distiller.json",
                OUT_DIR / "stage41_group_consistency_evidence.json",
            ],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_group_consistency_multiseed()
