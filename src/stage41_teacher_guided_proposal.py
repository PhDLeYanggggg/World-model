from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_group_consistency_distiller as gcd
from src import stage41_joint_residual_rollout as jrr
from src import stage41_joint_rollout_consistency as jrc


OUT_DIR = gcd.OUT_DIR
CHECKPOINT_DIR = gcd.CHECKPOINT_DIR
REPORT_JSON = OUT_DIR / "stage41_teacher_guided_proposal.json"
REPORT_MD = OUT_DIR / "stage41_teacher_guided_proposal.md"
THREADS = 4
BATCH = 2048
EPOCHS = 4
SEED = 4189
EPS = 1e-6
USE_RESIDUAL_SIGNALS = False

TRIALS = [
    {"name": "teacher_proposal_balanced", "width": 96, "dropout": 0.08, "lr": 8e-4, "teacher_w": 1.2, "gain_w": 0.6, "harm_w": 1.0, "hard_w": 1.5, "seed": 1},
    {"name": "teacher_proposal_conservative", "width": 80, "dropout": 0.12, "lr": 7e-4, "teacher_w": 1.5, "gain_w": 0.4, "harm_w": 1.5, "hard_w": 1.2, "seed": 2},
    {"name": "teacher_proposal_hard_expand", "width": 112, "dropout": 0.08, "lr": 7e-4, "teacher_w": 1.0, "gain_w": 1.0, "harm_w": 1.2, "hard_w": 2.5, "seed": 3},
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
    return value


def _torch():
    torch = gcd._torch()
    torch.set_num_threads(THREADS)
    return torch


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


def _teacher_seed_rows() -> list[Mapping[str, Any]]:
    report = read_json(OUT_DIR / "stage41_group_consistency_multiseed_repair.json", {})
    return list(report.get("seed_results") or [])


def _selected_residual_trial() -> tuple[str | None, float | None]:
    report = read_json(OUT_DIR / "stage41_joint_residual_rollout.json", {})
    name = report.get("selected_trial")
    trial = ((report.get("trained_trials") or {}).get(str(name)) or {}).get("train", {}).get("trial", {})
    ckpt = ((report.get("trained_trials") or {}).get(str(name)) or {}).get("train", {}).get("checkpoint")
    return ckpt, float(trial.get("clip", 1.0)) if trial else None


def _base_data(split: str) -> dict[str, Any]:
    checkpoint, repaired_policy, _policy_source = gcd._load_policy_and_checkpoint()
    return gcd._bundle(split, checkpoint, repaired_policy)


def _teacher_switches(data: Mapping[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    switches = []
    diagnostics = []
    for row in _teacher_seed_rows():
        scores = gcd._predict(row["checkpoint"], data)
        metrics, switch = gcd._policy_metrics(scores, data, row["selected_policy"])
        switches.append(switch.astype(np.float32))
        diagnostics.append({"seed": row.get("seed"), "switch_rate": float(np.mean(switch)), "metrics": metrics})
    if not switches:
        raise FileNotFoundError("Missing Stage41 group-consistency multiseed repair teacher switches.")
    return np.mean(switches, axis=0).astype(np.float32), diagnostics


def _residual_signals(split: str, rows: int) -> np.ndarray:
    if not USE_RESIDUAL_SIGNALS:
        return np.zeros((rows, 8), dtype=np.float32)
    ckpt, clip = _selected_residual_trial()
    if not ckpt or clip is None:
        return np.zeros((rows, 8), dtype=np.float32)
    data = jrr._residual_bundle(split, clip)
    pred = jrr._predict(ckpt, data)
    return np.stack(
        [
            pred["gain"],
            pred["harm"],
            pred["uncertainty"],
            pred["traj_risk"],
            pred["interaction"],
            pred["occupancy"],
            pred["physical"],
            pred["future_close"],
        ],
        axis=1,
    ).astype(np.float32)


def _bundle(split: str) -> dict[str, Any]:
    data = _base_data(split)
    teacher_prob, teacher_diag = _teacher_switches(data)
    residual = _residual_signals(split, len(data["x"]))
    features = np.concatenate([data["x"].astype(np.float32), residual], axis=1).astype(np.float32)
    teacher_label = (teacher_prob >= 0.5).astype(np.float32)
    gain = (data["floor_ade"].astype(np.float64) - data["neural_ade"].astype(np.float64)).astype(np.float32)
    harm = (data["neural_ade"].astype(np.float64) > data["floor_ade"].astype(np.float64)).astype(np.float32)
    return {
        **data,
        "x_teacher": features,
        "teacher_prob": teacher_prob,
        "teacher_label": teacher_label,
        "teacher_diagnostics": teacher_diag,
        "gain": gain,
        "harm": harm,
        "feature_dim": int(features.shape[1]),
    }


def _make_model(dim: int, width: int, dropout: float):
    torch = _torch()
    import torch.nn as nn

    class TeacherGuidedProposal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, width),
                nn.GELU(),
                nn.LayerNorm(width),
                nn.Dropout(dropout),
                nn.Linear(width, width),
                nn.GELU(),
                nn.LayerNorm(width),
            )
            self.teacher = nn.Linear(width, 1)
            self.gain = nn.Linear(width, 1)
            self.harm = nn.Linear(width, 1)
            self.uncertainty = nn.Linear(width, 1)

        def forward(self, x):
            h = self.net(x)
            return {
                "teacher_logit": self.teacher(h).squeeze(-1),
                "gain": self.gain(h).squeeze(-1),
                "harm_logit": self.harm(h).squeeze(-1),
                "uncertainty_logit": self.uncertainty(h).squeeze(-1),
            }

    return TeacherGuidedProposal()


def _train_trial(trial: Mapping[str, Any], train: Mapping[str, Any], val: Mapping[str, Any]) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    ckpt = CHECKPOINT_DIR / f"stage41_{trial['name']}.pt"
    heartbeat = OUT_DIR / f"{trial['name']}_heartbeat.json"
    x = torch.tensor(train["x_teacher"])
    y_teacher = torch.tensor(train["teacher_label"])
    y_gain = torch.tensor(train["gain"])
    y_harm = torch.tensor(train["harm"])
    hard = torch.tensor((train["hard"].astype(bool) | train["failure"].astype(bool)).astype(np.float32))
    vx = torch.tensor(val["x_teacher"])
    vt = torch.tensor(val["teacher_label"])
    vg = torch.tensor(val["gain"])
    vh = torch.tensor(val["harm"])
    model = _make_model(x.shape[1], int(trial["width"]), float(trial["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=1e-4)
    rng = np.random.default_rng(SEED + int(trial["seed"]))
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(x.shape[0])
        losses = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(x[ids])
            row_w = 1.0 + float(trial["hard_w"]) * hard[ids] + 1.5 * torch.clamp(torch.abs(y_gain[ids]), max=2.0)
            teacher_loss = (F.binary_cross_entropy_with_logits(out["teacher_logit"], y_teacher[ids], reduction="none") * row_w).mean()
            gain_loss = (F.smooth_l1_loss(out["gain"], y_gain[ids], reduction="none") * row_w).mean()
            harm_loss = (F.binary_cross_entropy_with_logits(out["harm_logit"], y_harm[ids], reduction="none") * row_w).mean()
            uncertainty_loss = F.binary_cross_entropy_with_logits(out["uncertainty_logit"], y_harm[ids])
            loss = float(trial["teacher_w"]) * teacher_loss + float(trial["gain_w"]) * gain_loss + float(trial["harm_w"]) * harm_loss + 0.25 * uncertainty_loss
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
                    F.binary_cross_entropy_with_logits(out["teacher_logit"], vt)
                    + 0.5 * F.smooth_l1_loss(out["gain"], vg)
                    + F.binary_cross_entropy_with_logits(out["harm_logit"], vh)
                ).cpu()
            )
        heartbeat.write_text(
            json.dumps({"trial": dict(trial), "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "checkpoint": str(ckpt), "best": best}, ensure_ascii=False),
            encoding="utf-8",
        )
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
            torch.save({"model": model.state_dict(), "dim": int(x.shape[1]), "trial": dict(trial), "best": best}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best, "trial": dict(trial)}


def _predict(path: str | Path, data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    trial = payload["trial"]
    model = _make_model(int(payload["dim"]), int(trial["width"]), float(trial["dropout"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    x = torch.tensor(data["x_teacher"])
    outs = {"teacher_prob": [], "gain": [], "harm": [], "uncertainty": []}
    with torch.no_grad():
        for start in range(0, x.shape[0], 4096):
            out = model(x[start : start + 4096])
            outs["teacher_prob"].append(torch.sigmoid(out["teacher_logit"]).cpu().numpy())
            outs["gain"].append(out["gain"].cpu().numpy())
            outs["harm"].append(torch.sigmoid(out["harm_logit"]).cpu().numpy())
            outs["uncertainty"].append(torch.sigmoid(out["uncertainty_logit"]).cpu().numpy())
    return {k: np.concatenate(v).astype(np.float32) for k, v in outs.items()}


def _evaluate_switch(data: Mapping[str, Any], switch: np.ndarray, name: str) -> dict[str, Any]:
    bundle = {
        "labels": data["labels"],
        "keys": data["keys"],
        "floor_xy": data["floor_xy"],
        "neural_xy": data["neural_xy"],
        "floor_ade": data["floor_ade"],
    }
    return jrc._evaluate_split_rollout(bundle, switch.astype(bool), name)


def _policy_switch(pred: Mapping[str, np.ndarray], policy: Mapping[str, float]) -> np.ndarray:
    mode = str(policy.get("mode", "teacher_and_gain"))
    teacher = pred["teacher_prob"] >= float(policy.get("teacher_min", 0.5))
    gain = pred["gain"] >= float(policy.get("gain_min", 0.0))
    safe = (pred["harm"] <= float(policy.get("harm_max", 1.0))) & (pred["uncertainty"] <= float(policy.get("uncertainty_max", 1.0)))
    if mode == "teacher_only":
        return teacher & safe
    if mode == "teacher_or_gain":
        return (teacher | gain) & safe
    return teacher & gain & safe


def _policy_grid(pred: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    teacher_grid = [0.45, 0.50, 0.60, 0.70, 0.80]
    gain_values = pred["gain"]
    gain_grid = [0.0] + [float(v) for v in np.quantile(gain_values, [0.50, 0.70, 0.85])]
    harm_grid = [float(v) for v in np.quantile(pred["harm"], [0.20, 0.40, 0.60])]
    out: list[dict[str, Any]] = []
    for mode in ["teacher_only", "teacher_and_gain", "teacher_or_gain"]:
        for teacher_min in teacher_grid:
            for gain_min in gain_grid:
                for harm_max in harm_grid:
                    out.append({"mode": mode, "teacher_min": teacher_min, "gain_min": gain_min, "harm_max": harm_max, "uncertainty_max": harm_max})
    return out


def _score(metrics: Mapping[str, Any], collision_delta: float) -> float:
    return (
        float(metrics.get("all_improvement", 0.0))
        + 1.25 * float(metrics.get("t50_improvement", 0.0))
        + float(metrics.get("t100_improvement", 0.0))
        + 1.15 * float(metrics.get("hard_failure_improvement", 0.0))
        - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 8.0 * max(0.0, collision_delta - 0.01)
    )


def _fit_policy(pred: Mapping[str, np.ndarray], val: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    floor_ade = val["floor_ade"].astype(np.float64)
    neural_ade = val["neural_ade"].astype(np.float64)
    metric_ds = {
        "horizon": val["horizon"],
        "hard": val["hard"],
        "failure": val["failure"],
        "easy": val["easy"],
        "domain": val["domain"],
        "candidate_fde": val["candidate_fde"],
    }
    for policy in _policy_grid(pred):
        switch = _policy_switch(pred, policy)
        selected = floor_ade.copy()
        selected[switch] = neural_ade[switch]
        metrics = s41._metrics(selected, floor_ade, metric_ds, switch)
        eligible = bool(
            metrics.get("all_improvement", 0.0) > 0
            and metrics.get("t50_improvement", 0.0) > 0
            and metrics.get("hard_failure_improvement", 0.0) > 0
            and metrics.get("easy_degradation", 1.0) <= 0.02
            and float(np.mean(switch)) > 0.0
        )
        candidates.append({"policy": dict(policy), "metrics": metrics, "switch_rate": float(np.mean(switch)), "eligible": eligible, "score": _score(metrics, 0.0)})
    pool = [row for row in candidates if row["eligible"]] or candidates
    best = max(pool, key=lambda row: row["score"])
    return {"type": "teacher_guided_proposal", **best["policy"], "val_eligible": bool([row for row in candidates if row["eligible"]])}, candidates


def _teacher_reference_eval(data: Mapping[str, Any]) -> dict[str, Any]:
    switch = data["teacher_prob"] >= 0.5
    ev = _evaluate_switch(data, switch, "teacher_consensus_reference")
    return {"metrics": ev["selected_metrics"], "collision_delta_005": ev["collision_delta_005"], "switch_rate": float(np.mean(switch))}


def run_teacher_guided_proposal() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    train = _bundle("train")
    val = _bundle("val")
    test = _bundle("test")
    trial_reports: dict[str, Any] = {}
    rank = []
    for trial in TRIALS:
        train_result = _train_trial(trial, train, val)
        pred_val = _predict(train_result["checkpoint"], val)
        policy, candidates = _fit_policy(pred_val, val)
        best = max(candidates, key=lambda row: row["score"])
        trial_reports[str(trial["name"])] = {"train": train_result, "policy": policy, "val_best": best, "val_candidate_count": len(candidates)}
        rank.append({"trial": trial["name"], "score": best["score"]})
    selected = max(rank, key=lambda row: row["score"])
    selected_train = trial_reports[str(selected["trial"])]["train"]
    selected_policy = trial_reports[str(selected["trial"])]["policy"]
    pred_test = _predict(selected_train["checkpoint"], test)
    switch = _policy_switch(pred_test, selected_policy)
    test_eval = _evaluate_switch(test, switch, "test_teacher_guided_proposal")
    teacher_ref = _teacher_reference_eval(test)
    group_repair = read_json(OUT_DIR / "stage41_group_consistency_multiseed_repair.json", {})
    group_summary = group_repair.get("metric_summary") or {}
    group_basis = {
        "all_improvement": (group_summary.get("all_improvement") or {}).get("mean", 0.0),
        "t50_improvement": (group_summary.get("t50_improvement") or {}).get("mean", 0.0),
        "t100_improvement": (group_summary.get("t100_improvement") or {}).get("mean", 0.0),
        "hard_failure_improvement": (group_summary.get("hard_failure_improvement") or {}).get("mean", 0.0),
        "easy_degradation": (group_summary.get("easy_degradation") or {}).get("max", 1.0),
        "collision_delta_vs_floor_005": (group_summary.get("collision_delta_vs_floor_005") or {}).get("max", 1.0),
    }
    metrics = test_eval["selected_metrics"]
    lift = {
        "all_delta": float(metrics.get("all_improvement", 0.0) - float(group_basis.get("all_improvement") or 0.0)),
        "t50_delta": float(metrics.get("t50_improvement", 0.0) - float(group_basis.get("t50_improvement") or 0.0)),
        "t100_delta": float(metrics.get("t100_improvement", 0.0) - float(group_basis.get("t100_improvement") or 0.0)),
        "hard_delta": float(metrics.get("hard_failure_improvement", 0.0) - float(group_basis.get("hard_failure_improvement") or 0.0)),
        "easy_delta": float(metrics.get("easy_degradation", 0.0) - float(group_basis.get("easy_degradation") or 0.0)),
    }
    deployable = bool(
        metrics.get("all_improvement", 0.0) > 0
        and metrics.get("t50_improvement", 0.0) > 0
        and metrics.get("hard_failure_improvement", 0.0) > 0
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and test_eval["collision_delta_005"] <= 0.01
        and float(np.mean(switch)) > 0.0
    )
    improves_current = bool(deployable and (lift["all_delta"] > 0 or lift["t50_delta"] > 0 or lift["hard_delta"] > 0))
    result = {
        "source": "fresh_run",
        "protocol_status": "teacher_guided_neural_proposal_policy",
        "hypothesis": "Distilling the group-consistency safety-buffer teacher into a neural proposal policy can produce positive neural switch-rate without residual-regression harm.",
        "use_residual_signals": USE_RESIDUAL_SIGNALS,
        "selected_trial": selected["trial"],
        "selected_policy": selected_policy,
        "trial_reports": trial_reports,
        "teacher_reference": teacher_ref,
        "test_metrics": metrics,
        "multi_agent_metrics": test_eval["multi_agent_metrics"],
        "collision_delta_vs_floor_005": test_eval["collision_delta_005"],
        "switch_rate": float(np.mean(switch)),
        "teacher_agreement_rate": float(np.mean((switch.astype(bool)) == (test["teacher_prob"] >= 0.5))),
        "current_best_group_consistency_basis": group_basis,
        "lift_over_current_group_consistency_basis": lift,
        "teacher_guided_proposal_deployable": deployable,
        "teacher_guided_proposal_improves_current_deployable": improves_current,
        "no_leakage": {
            "teacher_switch_inference_input": False,
            "teacher_switch_train_label_only": True,
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "This is teacher-guided neural proposal selection under the Stage37 safety floor. It remains dataset-local raw-frame 2.5D, not metric/seconds/true 3D/foundation.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Teacher-Guided Neural Proposal",
            "",
            "- source: `fresh_run`",
            f"- selected trial: `{selected['trial']}`",
            f"- selected policy: `{selected_policy}`",
            f"- residual signals enabled: `{USE_RESIDUAL_SIGNALS}`",
            f"- deployable: `{deployable}`",
            f"- improves current deployable: `{improves_current}`",
            f"- test metrics: `{metrics}`",
            f"- teacher reference: `{teacher_ref}`",
            f"- lift over current group-consistency basis: `{lift}`",
            f"- collision delta vs floor @0.05 normalized: `{test_eval['collision_delta_005']}`",
            f"- switch rate: `{float(np.mean(switch))}`",
            f"- teacher agreement rate: `{result['teacher_agreement_rate']}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "Teacher switches are train labels only, not inference inputs. The model sees past/static/prediction/residual signals and is selected on validation.",
        ],
    )
    return result


def main_teacher_guided_proposal() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_teacher_guided_proposal()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_teacher_guided_proposal",
            status,
            started,
            [OUT_DIR / "stage41_group_consistency_multiseed_repair.json", OUT_DIR / "stage41_joint_residual_rollout.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_teacher_guided_proposal()
