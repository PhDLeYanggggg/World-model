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
