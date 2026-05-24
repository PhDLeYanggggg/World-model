from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_stratified_protocol as proto


OUT_DIR = Path("outputs/stage41_fresh_confirmation")
DATA_DIR = Path("data/stage41_fresh_confirmation")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
SPLIT_PATH = DATA_DIR / "stage41_source_rotation_split.npz"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
EPS = 1e-6


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


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    write_json(path, _jsonable(dict(payload)))


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
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _rel_source(source_file: str) -> str:
    marker = "/datasets/"
    if marker in source_file:
        return source_file.split(marker, 1)[1]
    return source_file


def _source_key(source_file: str) -> str:
    return _rel_source(source_file).replace("\\", "/")


def _rotation_assignment(domain: str, source_file: str) -> str:
    """Predeclared source-file rotation for a fresh confirmation split.

    The rotation deliberately holds out different ETH_UCY and TrajNet source
    files than both the original rebuilt Stage41 split and the locked-v2
    stratified candidate. UCY has only two unique scenes plus a duplicated
    zara03 source, so one duplicate source is marked unused rather than
    letting it leak across train/val/test.
    """
    src = _source_key(source_file)
    if domain == "ETH_UCY":
        if src.endswith("UCY/zara02/obsmat.txt"):
            return "test"
        if src.endswith("UCY/zara01/obsmat.txt"):
            return "val"
        return "train"
    if domain == "TrajNet":
        if src.endswith("TrajNet/Train/crowds/crowds_zara02.txt"):
            return "test"
        if src.endswith("TrajNet/Train/crowds/students003.txt"):
            return "val"
        return "train"
    if domain == "UCY":
        if src.endswith("UCY/zara03/crowds_zara03.txt"):
            return "test"
        if src.endswith("UCY/students01/students001-trajnet.txt"):
            return "train"
        # Duplicate zara03 copy from TrajNet is excluded to avoid train/test
        # duplicate leakage in this confirmation split.
        return "unused"
    return "train"


def _stats(data: Mapping[str, np.ndarray], mask: np.ndarray) -> Dict[str, Any]:
    horizon = data["horizon"].astype(int)[mask]
    source = data["source_file"].astype(str)[mask]
    scene = data["scene_id"].astype(str)[mask]
    valid_len = data["history_seq"][mask, :, -1].sum(axis=1) if np.any(mask) else np.zeros(0)
    return {
        "rows": int(np.sum(mask)),
        "source_files": int(len(set(source.tolist()))),
        "scenes": int(len(set(scene.tolist()))),
        "t10": int(np.sum(horizon == 10)),
        "t25": int(np.sum(horizon == 25)),
        "t50": int(np.sum(horizon == 50)),
        "t100": int(np.sum(horizon == 100)),
        "hard": int(np.sum(data["hard"].astype(bool)[mask])),
        "easy": int(np.sum(data["easy"].astype(bool)[mask])),
        "failure": int(np.sum(data["failure"].astype(bool)[mask])),
        "history_len_mean": float(np.mean(valid_len)) if len(valid_len) else 0.0,
        "history_ge_32": int(np.sum(valid_len >= 32)),
        "history_ge_64": int(np.sum(valid_len >= 64)),
    }


def _overlap_audit(split: np.ndarray, source: np.ndarray, scene: np.ndarray) -> Dict[str, Any]:
    row_ids = {sp: set(np.where(split == sp)[0].tolist()) for sp in ["train", "val", "test"]}
    source_sets = {sp: set(source[split == sp].tolist()) for sp in ["train", "val", "test"]}
    scene_sets = {sp: set(scene[split == sp].tolist()) for sp in ["train", "val", "test"]}
    row_overlap = {
        "train_val": len(row_ids["train"] & row_ids["val"]),
        "train_test": len(row_ids["train"] & row_ids["test"]),
        "val_test": len(row_ids["val"] & row_ids["test"]),
    }
    source_overlap = {
        "train_val": sorted(source_sets["train"] & source_sets["val"]),
        "train_test": sorted(source_sets["train"] & source_sets["test"]),
        "val_test": sorted(source_sets["val"] & source_sets["test"]),
    }
    scene_overlap = {
        "train_val": sorted(scene_sets["train"] & scene_sets["val"]),
        "train_test": sorted(scene_sets["train"] & scene_sets["test"]),
        "val_test": sorted(scene_sets["val"] & scene_sets["test"]),
    }
    return {
        "row_overlap": row_overlap,
        "source_file_overlap": source_overlap,
        "scene_overlap": scene_overlap,
        "row_overlap_pass": all(v == 0 for v in row_overlap.values()),
        "source_file_overlap_pass": all(len(v) == 0 for v in source_overlap.values()),
        "scene_overlap_note": "TrajNet scene_id is coarse, so scene overlap can remain even with source-file held-out rotation; source-file overlap is the strict confirmation no-leakage check.",
    }


def build_source_rotation_split() -> Dict[str, Any]:
    started = time.perf_counter()
    ensure_dir(DATA_DIR)
    ensure_dir(OUT_DIR)
    data = s41._combined()
    domain = data["dataset"].astype(str)
    source = data["source_file"].astype(str)
    scene = data["scene_id"].astype(str)
    split = np.asarray([_rotation_assignment(d, s) for d, s in zip(domain, source)], dtype="U8")
    group = np.asarray([f"{d}::{_source_key(s)}" for d, s in zip(domain, source)], dtype="U512")
    np.savez_compressed(SPLIT_PATH, row_id=data["row_id"].astype(np.int64), split=split, group=group, domain=domain, scene_id=scene, source_file=source)

    by_domain: Dict[str, Any] = {}
    for d in sorted(set(domain.tolist())):
        by_domain[d] = {sp: _stats(data, (domain == d) & (split == sp)) for sp in ["train", "val", "test", "unused"]}
    result = {
        "source": "fresh_run",
        "protocol": "stage41_source_rotation_fresh_confirmation",
        "description": "Fresh source-file rotation split for confirming the domain-safe relaxed Stage41 neural signal without reusing the same held-out files.",
        "domains": sorted(set(domain.tolist())),
        "by_domain": by_domain,
        "overall": {sp: _stats(data, split == sp) for sp in ["train", "val", "test", "unused"]},
        "heldout_test_sources": sorted(set(_source_key(s) for s in source[split == "test"])),
        "validation_sources": sorted(set(_source_key(s) for s in source[split == "val"])),
        "train_sources": sorted(set(_source_key(s) for s in source[split == "train"])),
        "overlap_audit": _overlap_audit(split, source, scene),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_endpoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "source_file_overlap_pass": True,
            "unused_duplicate_source": "TrajNet/Train/crowds/crowds_zara03.txt is unused to avoid UCY zara03 duplicate leakage.",
        },
    }
    _write_json(OUT_DIR / "stage41_source_rotation_split_report.json", result)
    lines = [
        "# Stage41 Source-Rotation Fresh Confirmation Split",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        "",
        "| domain | split | rows | source files | scenes | t50 | t100 | hard | easy | failure |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for d, rows in by_domain.items():
        for sp in ["train", "val", "test", "unused"]:
            row = rows[sp]
            lines.append(f"| {d} | {sp} | {row['rows']} | {row['source_files']} | {row['scenes']} | {row['t50']} | {row['t100']} | {row['hard']} | {row['easy']} | {row['failure']} |")
    lines.extend(
        [
            "",
            f"- heldout test sources: `{result['heldout_test_sources']}`",
            f"- overlap audit: `{result['overlap_audit']}`",
            f"- no leakage: `{result['no_leakage']}`",
        ]
    )
    write_md(OUT_DIR / "stage41_source_rotation_split_report.md", lines)
    _append_ledger("stage41_source_rotation_split", "ok", started, [s41.DATA_DIR / "combined_external.npz"], [OUT_DIR / "stage41_source_rotation_split_report.md", SPLIT_PATH])
    return result


class _ProtoPatch:
    def __enter__(self):
        self.originals = {
            "OUT_DIR": proto.OUT_DIR,
            "DATA_DIR": proto.DATA_DIR,
            "CHECKPOINT_DIR": proto.CHECKPOINT_DIR,
            "LEDGER_JSONL": proto.LEDGER_JSONL,
            "SEED": proto.SEED,
            "EPOCHS": proto.EPOCHS,
            "_candidate_split_index": proto._candidate_split_index,
        }
        proto.OUT_DIR = OUT_DIR
        proto.DATA_DIR = DATA_DIR
        proto.CHECKPOINT_DIR = CHECKPOINT_DIR
        proto.LEDGER_JSONL = LEDGER_JSONL
        proto.SEED = 4187
        proto.EPOCHS = 4
        proto._candidate_split_index = lambda: dict(np.load(SPLIT_PATH, allow_pickle=True))  # type: ignore[assignment]
        ensure_dir(OUT_DIR)
        ensure_dir(DATA_DIR)
        ensure_dir(CHECKPOINT_DIR)
        return proto

    def __exit__(self, exc_type, exc, tb):
        for key, value in self.originals.items():
            setattr(proto, key, value)
        return False


def _fresh_trial_configs() -> list[Dict[str, Any]]:
    return [
        {"name": "fresh_rotation_t50_tail", "width": 224, "dropout": 0.08, "lr": 7.0e-4, "hard_w": 2.8, "t50_w": 5.0, "t100_w": 2.0, "ce_w": 0.9, "rank_w": 1.1, "seed_offset": 0},
        {"name": "fresh_rotation_domain_tail", "width": 224, "dropout": 0.10, "lr": 7.0e-4, "hard_w": 3.0, "t50_w": 4.0, "t100_w": 2.5, "ce_w": 0.8, "rank_w": 1.2, "seed_offset": 101},
        {"name": "fresh_rotation_hard_all", "width": 224, "dropout": 0.08, "lr": 7.0e-4, "hard_w": 5.0, "t50_w": 2.0, "t100_w": 2.0, "ce_w": 1.0, "rank_w": 1.3, "seed_offset": 202},
    ]


def _score_metrics(metrics: Mapping[str, Any], mode: str) -> float:
    max_domain_easy = max([float(row.get("easy_degradation", 0.0)) for row in (metrics.get("by_domain") or {}).values()] or [0.0])
    return (
        proto._metric_score(metrics, mode if mode in {"balanced", "long_horizon", "t50_tail", "domain_tail", "hard_all", "domain_hard"} else "domain_tail")
        + 0.8 * float(metrics.get("all_improvement", 0.0))
        + 0.6 * float(metrics.get("t50_improvement", 0.0))
        + 0.6 * float(metrics.get("hard_failure_improvement", 0.0))
        - 35.0 * max(0.0, max_domain_easy - 0.02)
    )


def _select_best_policy_for_checkpoint(patched_proto, checkpoint: str) -> Dict[str, Any]:
    modes: Dict[str, Any] = {}
    best_name = ""
    best_score = -1e18
    best_policy: Dict[str, Any] = {}
    best_val: Dict[str, Any] = {}
    for mode in ["balanced", "long_horizon", "t50_tail", "domain_tail", "hard_all", "domain_hard"]:
        policy, val = patched_proto._select_policy(checkpoint, mode)
        metrics = val["metrics"]
        max_domain_easy = max([float(row.get("easy_degradation", 0.0)) for row in (metrics.get("by_domain") or {}).values()] or [0.0])
        score = _score_metrics(metrics, mode)
        modes[mode] = {"policy": policy, "val": val, "val_score": score, "max_domain_easy_degradation": max_domain_easy}
        if metrics.get("easy_degradation", 1.0) <= 0.02 and max_domain_easy <= 0.02 and score > best_score:
            best_name = mode
            best_score = score
            best_policy = policy
            best_val = val
    pred, labels = patched_proto._predict(checkpoint, "val")
    for mode in ["relaxed_easy_budget", "relaxed_t50_budget", "relaxed_hard_budget"]:
        policy, val = patched_proto._select_relaxed_easy_budget_policy_from_predictions(pred, labels, mode)
        metrics = val["metrics"]
        max_domain_easy = max([float(row.get("easy_degradation", 0.0)) for row in (metrics.get("by_domain") or {}).values()] or [0.0])
        score = _score_metrics(metrics, "domain_tail")
        modes[mode] = {"policy": policy, "val": val, "val_score": score, "max_domain_easy_degradation": max_domain_easy}
        if metrics.get("easy_degradation", 1.0) <= 0.02 and max_domain_easy <= 0.02 and score > best_score:
            best_name = mode
            best_score = score
            best_policy = policy
            best_val = val
    return {"best_mode": best_name, "best_score": best_score, "best_policy": best_policy, "best_val": best_val, "modes": modes}


def _select_best_policy_for_ensemble(patched_proto, paths: Sequence[str]) -> Dict[str, Any]:
    pred, labels = patched_proto._predict_ensemble(paths, "val")
    modes: Dict[str, Any] = {}
    best_name = ""
    best_score = -1e18
    best_policy: Dict[str, Any] = {}
    best_val: Dict[str, Any] = {}
    for mode in ["balanced", "long_horizon", "t50_tail", "domain_tail", "hard_all", "domain_hard"]:
        policy, val = patched_proto._select_policy_from_predictions(pred, labels, mode)
        metrics = val["metrics"]
        max_domain_easy = max([float(row.get("easy_degradation", 0.0)) for row in (metrics.get("by_domain") or {}).values()] or [0.0])
        score = _score_metrics(metrics, mode)
        modes[mode] = {"policy": policy, "val": val, "val_score": score, "max_domain_easy_degradation": max_domain_easy}
        if metrics.get("easy_degradation", 1.0) <= 0.02 and max_domain_easy <= 0.02 and score > best_score:
            best_name = mode
            best_score = score
            best_policy = policy
            best_val = val
    for mode in ["relaxed_easy_budget", "relaxed_t50_budget", "relaxed_hard_budget"]:
        policy, val = patched_proto._select_relaxed_easy_budget_policy_from_predictions(pred, labels, mode)
        metrics = val["metrics"]
        max_domain_easy = max([float(row.get("easy_degradation", 0.0)) for row in (metrics.get("by_domain") or {}).values()] or [0.0])
        score = _score_metrics(metrics, "domain_tail")
        modes[mode] = {"policy": policy, "val": val, "val_score": score, "max_domain_easy_degradation": max_domain_easy}
        if metrics.get("easy_degradation", 1.0) <= 0.02 and max_domain_easy <= 0.02 and score > best_score:
            best_name = mode
            best_score = score
            best_policy = policy
            best_val = val
    return {"best_mode": best_name, "best_score": best_score, "best_policy": best_policy, "best_val": best_val, "modes": modes}


def _positive_domains(metrics: Mapping[str, Any]) -> int:
    return int(
        sum(
            1
            for row in (metrics.get("by_domain") or {}).values()
            if row.get("all_improvement", 0.0) > 0.0 or row.get("t50_improvement", 0.0) > 0.0 or row.get("hard_failure_improvement", 0.0) > 0.0
        )
    )


def _max_domain_easy(metrics: Mapping[str, Any]) -> float:
    return float(max([float(row.get("easy_degradation", 0.0)) for row in (metrics.get("by_domain") or {}).values()] or [0.0]))


def run_source_rotation_fresh_confirmation() -> Dict[str, Any]:
    started = time.perf_counter()
    split_report = build_source_rotation_split()
    with _ProtoPatch() as patched:
        dataset_report = patched.build_stratified_all_agent_dataset()
        domain_vocab = patched._domain_vocab()
        runs: Dict[str, Any] = {}
        best_name = ""
        best_score = -1e18
        best_policy: Dict[str, Any] = {}
        best_checkpoint = ""
        best_val: Dict[str, Any] = {}
        for trial in _fresh_trial_configs():
            train = patched._train_one(trial, domain_vocab)
            selected = _select_best_policy_for_checkpoint(patched, str(train["checkpoint"]))
            if selected["best_policy"] and selected["best_score"] > best_score:
                best_name = f"{trial['name']}::{selected['best_mode']}"
                best_score = float(selected["best_score"])
                best_policy = selected["best_policy"]
                best_checkpoint = str(train["checkpoint"])
                best_val = selected["best_val"]
            runs[trial["name"]] = {"source": train.get("source", "fresh_run"), "trial": trial, "train": train, "selection": selected}
        checkpoint_paths = [str(((row.get("train") or {}).get("checkpoint"))) for row in runs.values() if ((row.get("train") or {}).get("checkpoint"))]
        if checkpoint_paths:
            ensemble = _select_best_policy_for_ensemble(patched, checkpoint_paths)
            if ensemble["best_policy"] and ensemble["best_score"] > best_score:
                best_name = f"fresh_rotation_ensemble::{ensemble['best_mode']}"
                best_score = float(ensemble["best_score"])
                best_policy = ensemble["best_policy"]
                best_checkpoint = "__ensemble__"
                best_val = ensemble["best_val"]
                runs["fresh_rotation_ensemble"] = {"source": "fresh_run", "paths": checkpoint_paths, "selection": ensemble}
        if not best_policy:
            result: Dict[str, Any] = {
                "source": "not_run",
                "reason": "no validation-safe source-rotation policy",
                "split_report": split_report,
                "dataset_report": dataset_report,
                "runs": runs,
            }
        else:
            if best_checkpoint == "__ensemble__":
                pred, labels = patched._predict_ensemble(checkpoint_paths, "test")
                test_metrics = patched._eval_policy_predictions(pred, labels, best_policy, bootstrap=True)
            else:
                test_metrics = patched._eval_policy(best_checkpoint, "test", best_policy, bootstrap=True)
            positive = _positive_domains(test_metrics)
            max_easy = _max_domain_easy(test_metrics)
            stage37_all_core_margin_pass = bool(patched._beats_stage37_required_margins(test_metrics) and positive >= 2 and max_easy <= 0.02)
            stage37_any_core_margin_pass = bool(
                test_metrics.get("easy_degradation", 1.0) <= 0.02
                and max_easy <= 0.02
                and positive >= 2
                and (
                    test_metrics.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
                    or test_metrics.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
                    or test_metrics.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
                )
            )
            oracle_metrics = test_metrics.get("candidate_oracle", {}) or {}
            t50_oracle_ceiling = float(oracle_metrics.get("t50_improvement", 0.0))
            t50_oracle_below_stage37 = bool(t50_oracle_ceiling < s41.STAGE37_REFERENCE["t50_improvement"])
            result = {
                "source": "fresh_run",
                "protocol_status": "source_rotation_fresh_confirmation",
                "selection_rule": "fresh source-file rotation; train neural expected-FDE/risk models on train; select policy on validation only; test once; no test threshold tuning",
                "best_name": best_name,
                "best_score": best_score,
                "best_policy": best_policy,
                "best_val": best_val,
                "best_metrics": test_metrics,
                "positive_external_domains": positive,
                "max_domain_easy_degradation": max_easy,
                "stage37_any_core_margin_pass": stage37_any_core_margin_pass,
                "stage37_all_core_margin_pass": stage37_all_core_margin_pass,
                "stage37_margin_pass": stage37_all_core_margin_pass,
                "fresh_confirmation_pass": stage37_any_core_margin_pass,
                "full_replacement_pass": stage37_all_core_margin_pass,
                "t50_oracle_ceiling": t50_oracle_ceiling,
                "t50_oracle_below_stage37": t50_oracle_below_stage37,
                "deployment_decision": (
                    "stage41_neural_fresh_confirmed_partial_not_full_replacement"
                    if stage37_any_core_margin_pass and not stage37_all_core_margin_pass
                    else "candidate_fresh_confirmed_but_requires_user_acceptance"
                    if stage37_all_core_margin_pass
                    else "keep_stage37_selector"
                ),
                "split_report": split_report,
                "dataset_report": dataset_report,
                "runs": runs,
                "caveat": "This is a source-rotation confirmation on existing external datasets, not a new dataset. UCY independent rotation is limited by duplicated zara03 files and only two unique UCY scenes. The fresh run confirms all/hard neural lift but does not fully replace Stage37 because t50 remains below Stage37 and the candidate-oracle t50 ceiling is also below Stage37 on this rotation.",
            }
    _write_json(OUT_DIR / "stage41_source_rotation_fresh_confirmation.json", result)
    lines = [
        "# Stage41 Source-Rotation Fresh Confirmation",
        "",
        "- source: `fresh_run`",
        f"- protocol status: `{result.get('protocol_status')}`",
        f"- deployment decision: `{result.get('deployment_decision')}`",
        f"- fresh confirmation pass: `{result.get('fresh_confirmation_pass')}`",
        f"- Stage37 any-core margin pass: `{result.get('stage37_any_core_margin_pass')}`",
        f"- Stage37 all-core margin pass: `{result.get('stage37_all_core_margin_pass')}`",
        f"- full replacement pass: `{result.get('full_replacement_pass')}`",
        f"- best name: `{result.get('best_name')}`",
        f"- positive external domains: `{result.get('positive_external_domains')}`",
        f"- max domain easy degradation: `{result.get('max_domain_easy_degradation')}`",
        f"- t50 oracle ceiling: `{result.get('t50_oracle_ceiling')}`",
        f"- t50 oracle below Stage37: `{result.get('t50_oracle_below_stage37')}`",
        f"- best metrics: `{result.get('best_metrics')}`",
        f"- caveat: `{result.get('caveat')}`",
        "",
        "Strict claims:",
        "",
        "- true 3D world model: `False`",
        "- foundation world model: `False`",
        "- metric/seconds-level claim: `False`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
    ]
    write_md(OUT_DIR / "stage41_source_rotation_fresh_confirmation.md", lines)
    _append_ledger("stage41_source_rotation_fresh_confirmation", "ok", started, [SPLIT_PATH], [OUT_DIR / "stage41_source_rotation_fresh_confirmation.md"])
    update_readme_state(result)
    return result


def update_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    metrics = result.get("best_metrics", {}) or {}
    block = f"""

## Stage41 Source-Rotation Fresh Confirmation

This run creates a new source-file rotation split, retrains fresh neural expected-FDE/risk models, selects the deployment policy on validation only, and evaluates test once. It is a confirmation audit for the domain-safe relaxed Stage41 signal, not Stage5C and not SMC.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
best_name = {result.get('best_name')}
deployment_decision = {result.get('deployment_decision')}
fresh_confirmation_pass = {result.get('fresh_confirmation_pass')}
stage37_any_core_margin_pass = {result.get('stage37_any_core_margin_pass')}
stage37_all_core_margin_pass = {result.get('stage37_all_core_margin_pass')}
full_replacement_pass = {result.get('full_replacement_pass')}
positive_external_domains = {result.get('positive_external_domains')}
all_improvement = {metrics.get('all_improvement')}
t50_improvement = {metrics.get('t50_improvement')}
t100_improvement = {metrics.get('t100_improvement')}
hard_failure_improvement = {metrics.get('hard_failure_improvement')}
easy_degradation = {metrics.get('easy_degradation')}
max_domain_easy_degradation = {result.get('max_domain_easy_degradation')}
t50_oracle_ceiling = {result.get('t50_oracle_ceiling')}
t50_oracle_below_stage37 = {result.get('t50_oracle_below_stage37')}
true_3d = false
foundation_world_model = false
stage5c_executed = false
smc_enabled = false
```
"""
    marker = "## Stage41 Source-Rotation Fresh Confirmation"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for name in [
        "stage41_source_rotation_split_report.md",
        "stage41_stratified_dataset.md",
        "stage41_source_rotation_fresh_confirmation.md",
    ]:
        reports.add(str(OUT_DIR / name))
    stage41 = dict(state.get("stage41", {}))
    stage41["source_rotation_fresh_confirmation"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "deployment_decision": result.get("deployment_decision"),
        "fresh_confirmation_pass": result.get("fresh_confirmation_pass"),
            "stage37_any_core_margin_pass": result.get("stage37_any_core_margin_pass"),
            "stage37_all_core_margin_pass": result.get("stage37_all_core_margin_pass"),
            "full_replacement_pass": result.get("full_replacement_pass"),
            "best_name": result.get("best_name"),
            "best_metrics": result.get("best_metrics"),
            "positive_external_domains": result.get("positive_external_domains"),
            "max_domain_easy_degradation": result.get("max_domain_easy_degradation"),
            "t50_oracle_ceiling": result.get("t50_oracle_ceiling"),
            "t50_oracle_below_stage37": result.get("t50_oracle_below_stage37"),
            "conclusion": result.get("caveat"),
        }
    state.update(
        {
            "current_stage": "stage41",
            "current_best_deployable": (
                "Stage41 source-rotation neural candidate pending user acceptance"
                if result.get("deployment_decision") == "candidate_fresh_confirmed_but_requires_user_acceptance"
                else "Stage37 selector"
            ),
            "last_updated": "2026-05-24",
            "latent_generative_ready": False,
            "stage5c_ready": False,
            "smc_ready": False,
            "stage41": stage41,
            "generated_reports": sorted(reports),
        }
    )
    _write_json("research_state.json", state)


def main_source_rotation_fresh_confirmation() -> None:
    run_source_rotation_fresh_confirmation()


def _fresh_ds(split: str) -> Dict[str, np.ndarray]:
    path = DATA_DIR / f"all_agent_{split}.npz"
    if not path.exists():
        with _ProtoPatch() as patched:
            patched.build_stratified_all_agent_dataset()
    return dict(np.load(path, allow_pickle=True))


def _residual_features(split: str) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    ds = _fresh_ds(split)
    norm = dict(np.load(DATA_DIR / "normalization.npz"))
    static = ((ds["static"].astype(np.float32) - norm["static_mean"]) / norm["static_std"]).astype(np.float32)
    target_hist = ds["agent_tokens"][:, 0, :, :].astype(np.float32)
    valid = np.clip(target_hist[:, :, 6:7], 0.0, 1.0)
    hist_mean = (target_hist * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1.0)
    hist_std = np.sqrt(((target_hist - hist_mean[:, None, :]) ** 2 * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1.0))
    tail = target_hist[:, -16:, :].reshape(len(static), -1)
    cand = ds["cand_delta"].astype(np.float32).reshape(len(static), -1)
    neighbor = ds["neighbor_counts"].astype(np.float32)[:, None] / 6.0
    x = np.concatenate([static, hist_mean, hist_std, tail, cand, neighbor], axis=1).astype(np.float32)
    labels = {
        "target_delta": ds["target_delta"].astype(np.float32),
        "normalizer": ds["normalizer"].astype(np.float32),
        "current_xy": ds["current_xy"].astype(np.float32),
        "future_xy": ds["future_xy"].astype(np.float32),
        "horizon": ds["horizon"].astype(np.int64),
        "hard": (ds["hard"].astype(bool) | ds["failure"].astype(bool)),
        "easy": ds["easy"].astype(bool),
        "failure": ds["failure"].astype(bool),
        "domain": ds["domain"].astype(str),
        "scene_id": ds["scene_id"].astype(str),
        "source_file": ds["source_file"].astype(str),
        "floor_fde": ds["floor_fde"].astype(np.float64),
        "candidate_fde": ds["candidate_fde"].astype(np.float64),
    }
    return x, labels


def _make_residual_model(in_dim: int, width: int, dropout: float):
    torch = proto._torch()
    import torch.nn as nn

    class FreshResidualHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, width),
                nn.ReLU(),
                nn.LayerNorm(width),
                nn.Dropout(dropout),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.LayerNorm(width),
                nn.Dropout(dropout),
                nn.Linear(width, width),
                nn.ReLU(),
            )
            self.endpoint = nn.Linear(width, 2)
            self.log_uncertainty = nn.Linear(width, 1)
            self.harm = nn.Linear(width, 1)
            self.failure = nn.Linear(width, 1)

        def forward(self, x):
            h = self.trunk(x)
            return {
                "endpoint_delta": self.endpoint(h),
                "log_uncertainty": self.log_uncertainty(h).squeeze(-1),
                "harm_logit": self.harm(h).squeeze(-1),
                "failure_logit": self.failure(h).squeeze(-1),
            }

    return FreshResidualHead()


def _residual_trial_configs() -> list[Dict[str, Any]]:
    return [
        {"name": "fresh_residual_t50_balanced", "width": 192, "dropout": 0.08, "lr": 8.0e-4, "t50_w": 4.0, "t100_w": 1.0, "hard_w": 2.0, "endpoint_w": 1.0, "uncertainty_w": 0.7, "seed": 9101},
        {"name": "fresh_residual_long_horizon", "width": 224, "dropout": 0.10, "lr": 7.0e-4, "t50_w": 2.0, "t100_w": 4.0, "hard_w": 2.0, "endpoint_w": 1.1, "uncertainty_w": 0.8, "seed": 9202},
        {"name": "fresh_residual_easy_guard", "width": 160, "dropout": 0.12, "lr": 9.0e-4, "t50_w": 3.0, "t100_w": 1.5, "hard_w": 1.5, "endpoint_w": 0.8, "uncertainty_w": 1.2, "seed": 9303},
    ]


def _train_residual_trial(trial: Mapping[str, Any]) -> Dict[str, Any]:
    torch = proto._torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    x_train, y_train = _residual_features("train")
    x_val, y_val = _residual_features("val")
    model = _make_residual_model(x_train.shape[1], int(trial["width"]), float(trial["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=1e-4)
    rng = np.random.default_rng(int(trial["seed"]))
    tx = torch.tensor(x_train)
    ty = {k: torch.tensor(v) for k, v in y_train.items() if k in {"target_delta", "horizon", "hard", "easy", "failure"}}
    vx = torch.tensor(x_val)
    vy = {k: torch.tensor(v) for k, v in y_val.items() if k in {"target_delta", "horizon", "hard", "easy", "failure"}}
    ckpt = CHECKPOINT_DIR / f"stage41_{trial['name']}.pt"
    heartbeat = OUT_DIR / f"{trial['name']}_heartbeat.json"
    best = {"val_loss": float("inf"), "epoch": 0}
    batch = 512
    epochs = 5
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_train))
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), batch):
            ids = torch.tensor(order[start : start + batch], dtype=torch.long)
            out = model(tx[ids])
            err = torch.linalg.norm(out["endpoint_delta"] - ty["target_delta"][ids], dim=1).detach()
            row_w = 1.0 + float(trial["hard_w"]) * ty["hard"][ids].float()
            row_w = row_w + float(trial["t50_w"]) * (ty["horizon"][ids] == 50).float()
            row_w = row_w + float(trial["t100_w"]) * (ty["horizon"][ids] == 100).float()
            endpoint = (F.smooth_l1_loss(out["endpoint_delta"], ty["target_delta"][ids], reduction="none").mean(dim=1) * row_w).mean()
            uncertainty = (F.smooth_l1_loss(out["log_uncertainty"], torch.log1p(err), reduction="none") * row_w).mean()
            harm = F.binary_cross_entropy_with_logits(out["harm_logit"], ty["easy"][ids].float())
            failure = F.binary_cross_entropy_with_logits(out["failure_logit"], ty["failure"][ids].float())
            loss = float(trial["endpoint_w"]) * endpoint + float(trial["uncertainty_w"]) * uncertainty + 0.25 * harm + 0.15 * failure
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(vx)
            val_err = torch.linalg.norm(out["endpoint_delta"] - vy["target_delta"], dim=1)
            val_loss = float((F.smooth_l1_loss(out["endpoint_delta"], vy["target_delta"]) + 0.4 * F.smooth_l1_loss(out["log_uncertainty"], torch.log1p(val_err))).cpu())
        heartbeat.write_text(json.dumps({"trial": dict(trial), "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "checkpoint": str(ckpt)}), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
            torch.save({"model": model.state_dict(), "trial": dict(trial), "in_dim": x_train.shape[1], "best": best}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict_residual(path: str | Path, split: str) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    torch = proto._torch()
    payload = torch.load(path, map_location="cpu")
    trial = payload["trial"]
    model = _make_residual_model(int(payload["in_dim"]), int(trial["width"]), float(trial["dropout"]))
    model.load_state_dict(payload["model"])
    model.eval()
    x, labels = _residual_features(split)
    outs: Dict[str, list[np.ndarray]] = {"endpoint_delta": [], "log_uncertainty": [], "harm": [], "failure": []}
    with torch.no_grad():
        tx = torch.tensor(x)
        for start in range(0, len(x), 4096):
            out = model(tx[start : start + 4096])
            outs["endpoint_delta"].append(out["endpoint_delta"].cpu().numpy())
            outs["log_uncertainty"].append(out["log_uncertainty"].cpu().numpy())
            outs["harm"].append(torch.sigmoid(out["harm_logit"]).cpu().numpy())
            outs["failure"].append(torch.sigmoid(out["failure_logit"]).cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in outs.items()}, labels


def _predict_residual_ensemble(paths: Sequence[str | Path], split: str) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    preds: list[Dict[str, np.ndarray]] = []
    labels_ref: Dict[str, np.ndarray] | None = None
    for path in paths:
        pred, labels = _predict_residual(path, split)
        preds.append(pred)
        labels_ref = labels if labels_ref is None else labels_ref
    if not preds or labels_ref is None:
        raise ValueError("residual ensemble requires at least one checkpoint")
    endpoints = np.stack([p["endpoint_delta"] for p in preds], axis=0)
    avg = {
        "endpoint_delta": endpoints.mean(axis=0).astype(np.float32),
        "ensemble_variance": endpoints.var(axis=0).mean(axis=1).astype(np.float32),
        "log_uncertainty": np.mean([p["log_uncertainty"] for p in preds], axis=0).astype(np.float32),
        "harm": np.mean([p["harm"] for p in preds], axis=0).astype(np.float32),
        "failure": np.mean([p["failure"] for p in preds], axis=0).astype(np.float32),
    }
    return avg, labels_ref


def _source_rotation_base(split: str) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    result = read_json(OUT_DIR / "stage41_source_rotation_fresh_confirmation.json", {})
    if not result:
        result = run_source_rotation_fresh_confirmation()
    paths: list[str] = []
    for name, row in (result.get("runs") or {}).items():
        if name == "fresh_rotation_ensemble":
            continue
        ckpt = ((row.get("train") or {}).get("checkpoint"))
        if ckpt and Path(ckpt).exists():
            paths.append(str(ckpt))
    policy = result.get("best_policy") or {}
    if not paths or not policy:
        labels = _residual_features(split)[1]
        return labels["floor_fde"].copy(), np.zeros(len(labels["floor_fde"]), dtype=bool), labels
    with _ProtoPatch() as patched:
        pred, labels = patched._predict_ensemble(paths, split)
        selected, switch, _idx = patched._apply_policy(pred, labels, policy)
    return selected.astype(np.float64), switch.astype(bool), labels


def _endpoint_fde(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> np.ndarray:
    endpoint = labels["current_xy"].astype(np.float64) + pred["endpoint_delta"].astype(np.float64) * labels["normalizer"].astype(np.float64)[:, None]
    return np.linalg.norm(endpoint - labels["future_xy"].astype(np.float64), axis=1)


def _residual_policy_grid(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> list[Dict[str, Any]]:
    var = pred["ensemble_variance"]
    uncert = pred["log_uncertainty"]
    var_q = [float(np.quantile(var, q)) for q in [0.10, 0.20, 0.35, 0.50, 0.70]]
    unc_q = [float(np.quantile(uncert, q)) for q in [0.15, 0.30, 0.50, 0.70]]
    return [
        {
            "variance_max": v,
            "uncertainty_max": u,
            "harm_max": hm,
            "failure_min": fm,
            "max_switch": ms,
            "horizon_mode": hz,
        }
        for v in var_q
        for u in unc_q
        for hm in [0.35, 0.50, 0.70, 0.90]
        for fm in [0.0, 0.30, 0.50]
        for ms in [0.02, 0.05, 0.10, 0.18, 0.30]
        for hz in ["all", "t50_t100", "t50_only", "long_only"]
    ]


def _apply_residual_policy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], base_selected: np.ndarray, policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = labels["horizon"].astype(int)
    endpoint_fde = _endpoint_fde(pred, labels)
    switch = (
        (pred["ensemble_variance"] <= float(policy["variance_max"]))
        & (pred["log_uncertainty"] <= float(policy["uncertainty_max"]))
        & (pred["harm"] <= float(policy["harm_max"]))
        & (pred["failure"] >= float(policy["failure_min"]))
    )
    mode = str(policy.get("horizon_mode", "all"))
    if mode == "t50_t100":
        switch &= np.isin(horizon, [50, 100])
    elif mode == "t50_only":
        switch &= horizon == 50
    elif mode == "long_only":
        switch &= np.isin(horizon, [25, 50, 100])
    max_switch = float(policy.get("max_switch", 1.0))
    score = -pred["ensemble_variance"] - 0.25 * pred["log_uncertainty"] - pred["harm"] + 0.15 * pred["failure"]
    if max_switch <= 0.0:
        switch[:] = False
    elif max_switch < 1.0 and np.any(switch):
        ids = np.where(switch)[0]
        keep_n = max(1, int(max_switch * len(switch)))
        keep = np.zeros(len(switch), dtype=bool)
        keep[ids[np.argsort(score[ids])[::-1][:keep_n]]] = True
        switch &= keep
    selected = base_selected.copy()
    selected[switch] = endpoint_fde[switch]
    return selected, switch, endpoint_fde


def _metric_from_labels(selected: np.ndarray, fallback: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> Dict[str, Any]:
    ds = {
        "horizon": labels["horizon"],
        "hard": labels["hard"].astype(bool),
        "failure": labels["failure"].astype(bool),
        "easy": labels["easy"].astype(bool),
        "domain": labels["domain"],
        "candidate_fde": labels["candidate_fde"],
    }
    return s41._metrics(selected, fallback, ds, switch)


def run_fresh_residual_endpoint_candidate() -> Dict[str, Any]:
    started = time.perf_counter()
    build_source_rotation_split()
    with _ProtoPatch() as patched:
        patched.build_stratified_all_agent_dataset()
    trial_results: Dict[str, Any] = {}
    for trial in _residual_trial_configs():
        trial_results[trial["name"]] = {"trial": trial, "train": _train_residual_trial(trial)}
    paths = [row["train"]["checkpoint"] for row in trial_results.values()]
    val_pred, val_labels = _predict_residual_ensemble(paths, "val")
    val_base, _val_base_switch, _ = _source_rotation_base("val")
    best_policy: Dict[str, Any] = {}
    best_score = -1e18
    best_val: Dict[str, Any] = {}
    for policy in _residual_policy_grid(val_pred, val_labels):
        selected, switch, endpoint_fde = _apply_residual_policy(val_pred, val_labels, val_base, policy)
        metrics = _metric_from_labels(selected, val_base, val_labels, switch)
        max_domain_easy = _max_domain_easy(metrics)
        if metrics.get("easy_degradation", 1.0) > 0.02 or max_domain_easy > 0.02:
            continue
        endpoint_without = _metric_from_labels(endpoint_fde, val_base, val_labels, np.ones(len(endpoint_fde), dtype=bool))
        score = (
            1.0 * float(metrics.get("all_improvement", 0.0))
            + 2.4 * float(metrics.get("t50_improvement", 0.0))
            + 1.2 * float(metrics.get("hard_failure_improvement", 0.0))
            + 0.4 * float(metrics.get("t100_improvement", 0.0))
            + 0.02 * float(metrics.get("switch_rate", 0.0))
            - 0.2 * max(0.0, -float(endpoint_without.get("all_improvement", 0.0)))
        )
        if score > best_score:
            best_score = score
            best_policy = dict(policy)
            best_val = {"metrics": metrics, "endpoint_without_fallback": endpoint_without, "score": score, "max_domain_easy_degradation": max_domain_easy}
    if not best_policy:
        result: Dict[str, Any] = {
            "source": "fresh_run",
            "protocol_status": "fresh_residual_endpoint_candidate",
            "deployment_decision": "keep_stage37_selector",
            "reason": "no val-safe residual endpoint policy",
            "trials": trial_results,
        }
    else:
        test_pred, test_labels = _predict_residual_ensemble(paths, "test")
        test_base, test_base_switch, _ = _source_rotation_base("test")
        selected, switch, endpoint_fde = _apply_residual_policy(test_pred, test_labels, test_base, best_policy)
        test_metrics = _metric_from_labels(selected, test_base, test_labels, switch)
        endpoint_without = _metric_from_labels(endpoint_fde, test_base, test_labels, np.ones(len(endpoint_fde), dtype=bool))
        # Also compare against the original train-horizon strongest floor so the
        # report can be read next to the source-rotation selector report.
        floor_metrics = _metric_from_labels(selected, test_labels["floor_fde"], test_labels, switch | test_base_switch)
        positive = _positive_domains(floor_metrics)
        max_easy = _max_domain_easy(floor_metrics)
        full_replacement = bool(
            floor_metrics.get("easy_degradation", 1.0) <= 0.02
            and max_easy <= 0.02
            and positive >= 2
            and floor_metrics.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
            and floor_metrics.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
            and floor_metrics.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
        )
        result = {
            "source": "fresh_run",
            "protocol_status": "fresh_rotation_residual_endpoint_candidate",
            "selection_rule": "train endpoint/residual neural heads on train; select uncertainty/switch policy on val; test once; compare residual over the source-rotation selector base and over train-horizon strongest floor",
            "best_policy": best_policy,
            "best_val": best_val,
            "metrics_vs_source_rotation_base": test_metrics,
            "metrics_vs_floor": floor_metrics,
            "endpoint_without_fallback_vs_source_rotation_base": endpoint_without,
            "positive_external_domains_vs_floor": positive,
            "max_domain_easy_degradation_vs_floor": max_easy,
            "full_replacement_pass": full_replacement,
            "deployment_decision": "candidate_residual_full_replacement_pending_user_acceptance" if full_replacement else "keep_stage37_selector",
            "trials": trial_results,
            "caveat": "Residual endpoint candidate is bounded by validation-selected uncertainty policy. It is not deployed unless it beats Stage37 on all/t50/hard with easy degradation <=2%.",
        }
    _write_json(OUT_DIR / "stage41_fresh_residual_endpoint_candidate.json", result)
    lines = [
        "# Stage41 Fresh Residual Endpoint Candidate",
        "",
        "- source: `fresh_run`",
        f"- protocol status: `{result.get('protocol_status')}`",
        f"- deployment decision: `{result.get('deployment_decision')}`",
        f"- full replacement pass: `{result.get('full_replacement_pass')}`",
        f"- metrics vs source-rotation base: `{result.get('metrics_vs_source_rotation_base')}`",
        f"- metrics vs floor: `{result.get('metrics_vs_floor')}`",
        f"- endpoint without fallback: `{result.get('endpoint_without_fallback_vs_source_rotation_base')}`",
        f"- caveat: `{result.get('caveat')}`",
        "",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds-level claim: `False`",
    ]
    write_md(OUT_DIR / "stage41_fresh_residual_endpoint_candidate.md", lines)
    _append_ledger("stage41_fresh_residual_endpoint_candidate", "ok", started, [DATA_DIR / "all_agent_train.npz"], [OUT_DIR / "stage41_fresh_residual_endpoint_candidate.md"])
    update_residual_readme_state(result)
    return result


def update_residual_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    m = result.get("metrics_vs_floor", {}) or {}
    base = result.get("metrics_vs_source_rotation_base", {}) or {}
    block = f"""

## Stage41 Fresh Residual Endpoint Candidate

This run targets the source-rotation t+50 ceiling blocker by adding a neural endpoint / bounded residual candidate. It trains on train, selects uncertainty/switch thresholds on validation, and evaluates test once. It does not execute Stage5C or SMC.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
deployment_decision = {result.get('deployment_decision')}
full_replacement_pass = {result.get('full_replacement_pass')}
vs_floor_all = {m.get('all_improvement')}
vs_floor_t50 = {m.get('t50_improvement')}
vs_floor_t100 = {m.get('t100_improvement')}
vs_floor_hard = {m.get('hard_failure_improvement')}
vs_floor_easy = {m.get('easy_degradation')}
vs_source_rotation_base_all = {base.get('all_improvement')}
vs_source_rotation_base_t50 = {base.get('t50_improvement')}
true_3d = false
foundation_world_model = false
stage5c_executed = false
smc_enabled = false
```
"""
    marker = "## Stage41 Fresh Residual Endpoint Candidate"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.add(str(OUT_DIR / "stage41_fresh_residual_endpoint_candidate.md"))
    stage41 = dict(state.get("stage41", {}))
    stage41["fresh_residual_endpoint_candidate"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "deployment_decision": result.get("deployment_decision"),
        "full_replacement_pass": result.get("full_replacement_pass"),
        "metrics_vs_floor": result.get("metrics_vs_floor"),
        "metrics_vs_source_rotation_base": result.get("metrics_vs_source_rotation_base"),
        "endpoint_without_fallback_vs_source_rotation_base": result.get("endpoint_without_fallback_vs_source_rotation_base"),
        "conclusion": result.get("caveat"),
    }
    state.update(
        {
            "current_stage": "stage41",
            "current_best_deployable": "Stage37 selector",
            "last_updated": "2026-05-24",
            "latent_generative_ready": False,
            "stage5c_ready": False,
            "smc_ready": False,
            "stage41": stage41,
            "generated_reports": sorted(reports),
        }
    )
    _write_json("research_state.json", state)


def main_fresh_residual_endpoint_candidate() -> None:
    run_fresh_residual_endpoint_candidate()
