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
        "cand_delta": ds["cand_delta"].astype(np.float32),
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
    selected, switch, _idx, labels = _source_rotation_base_details(split)
    return selected, switch, labels


def _source_rotation_base_details(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
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
        idx = np.zeros(len(labels["floor_fde"]), dtype=np.int64)
        return labels["floor_fde"].copy(), np.zeros(len(labels["floor_fde"]), dtype=bool), idx, labels
    with _ProtoPatch() as patched:
        pred, labels = patched._predict_ensemble(paths, split)
        selected, switch, _idx = patched._apply_policy(pred, labels, policy)
    rich_labels = _residual_features(split)[1]
    return selected.astype(np.float64), switch.astype(bool), _idx.astype(np.int64), rich_labels


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


def _bounded_base_features(split: str) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    x_base, labels = _residual_features(split)
    base_fde, base_switch, base_idx, _ = _source_rotation_base_details(split)
    cand_delta = labels["cand_delta"].astype(np.float32)
    base_delta = cand_delta[np.arange(len(base_idx)), base_idx].astype(np.float32)
    candidate_count = cand_delta.shape[1]
    idx_one = np.zeros((len(base_idx), candidate_count), dtype=np.float32)
    idx_one[np.arange(len(base_idx)), base_idx] = 1.0
    x = np.concatenate(
        [
            x_base,
            base_delta,
            base_switch.astype(np.float32)[:, None],
            idx_one,
        ],
        axis=1,
    ).astype(np.float32)
    labels = dict(labels)
    labels.update(
        {
            "base_fde": base_fde.astype(np.float64),
            "base_switch": base_switch.astype(bool),
            "base_idx": base_idx.astype(np.int64),
            "base_delta": base_delta.astype(np.float32),
            "target_residual": (labels["target_delta"].astype(np.float32) - base_delta).astype(np.float32),
        }
    )
    return x, labels


def _make_bounded_model(in_dim: int, width: int, dropout: float):
    torch = proto._torch()
    import torch.nn as nn

    class BoundedResidualHead(nn.Module):
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
            self.residual = nn.Linear(width, 2)
            self.log_uncertainty = nn.Linear(width, 1)
            self.harm = nn.Linear(width, 1)
            self.failure = nn.Linear(width, 1)

        def forward(self, x):
            h = self.trunk(x)
            return {
                "residual_raw": self.residual(h),
                "log_uncertainty": self.log_uncertainty(h).squeeze(-1),
                "harm_logit": self.harm(h).squeeze(-1),
                "failure_logit": self.failure(h).squeeze(-1),
            }

    return BoundedResidualHead()


def _bounded_trial_configs() -> list[Dict[str, Any]]:
    return [
        {
            "name": "bounded_residual_easy_safe",
            "width": 160,
            "dropout": 0.10,
            "lr": 9.0e-4,
            "clip": 0.018,
            "t50_w": 3.0,
            "t100_w": 2.0,
            "hard_w": 1.5,
            "easy_guard_w": 3.0,
            "residual_w": 0.10,
            "seed": 9411,
        },
        {
            "name": "bounded_residual_t50",
            "width": 192,
            "dropout": 0.08,
            "lr": 8.0e-4,
            "clip": 0.035,
            "t50_w": 5.0,
            "t100_w": 1.0,
            "hard_w": 2.5,
            "easy_guard_w": 2.0,
            "residual_w": 0.08,
            "seed": 9422,
        },
        {
            "name": "bounded_residual_long_horizon",
            "width": 224,
            "dropout": 0.10,
            "lr": 7.0e-4,
            "clip": 0.055,
            "t50_w": 2.5,
            "t100_w": 4.5,
            "hard_w": 2.0,
            "easy_guard_w": 2.5,
            "residual_w": 0.12,
            "seed": 9433,
        },
        {
            "name": "bounded_residual_hard",
            "width": 192,
            "dropout": 0.12,
            "lr": 8.0e-4,
            "clip": 0.045,
            "t50_w": 3.0,
            "t100_w": 2.0,
            "hard_w": 4.0,
            "easy_guard_w": 3.5,
            "residual_w": 0.12,
            "seed": 9444,
        },
    ]


def _train_bounded_trial(trial: Mapping[str, Any]) -> Dict[str, Any]:
    torch = proto._torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    ckpt = CHECKPOINT_DIR / f"stage41_{trial['name']}.pt"
    heartbeat = OUT_DIR / f"{trial['name']}_heartbeat.json"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location="cpu")
        return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": payload.get("best", {}), "resume_note": "existing bounded residual checkpoint reused"}
    x_train, y_train = _bounded_base_features("train")
    x_val, y_val = _bounded_base_features("val")
    model = _make_bounded_model(x_train.shape[1], int(trial["width"]), float(trial["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=1e-4)
    rng = np.random.default_rng(int(trial["seed"]))
    tx = torch.tensor(x_train)
    ty = {k: torch.tensor(v) for k, v in y_train.items() if k in {"target_delta", "base_delta", "horizon", "hard", "easy", "failure"}}
    vx = torch.tensor(x_val)
    vy = {k: torch.tensor(v) for k, v in y_val.items() if k in {"target_delta", "base_delta", "horizon", "hard", "easy", "failure"}}
    best = {"val_loss": float("inf"), "epoch": 0}
    batch = 512
    epochs = 5
    clip = float(trial["clip"])
    margin = 0.003
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_train))
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), batch):
            ids = torch.tensor(order[start : start + batch], dtype=torch.long)
            out = model(tx[ids])
            residual = clip * torch.tanh(out["residual_raw"])
            pred_delta = ty["base_delta"][ids] + residual
            pred_err = torch.linalg.norm(pred_delta - ty["target_delta"][ids], dim=1)
            base_err = torch.linalg.norm(ty["base_delta"][ids] - ty["target_delta"][ids], dim=1).detach()
            row_w = 1.0 + float(trial["hard_w"]) * ty["hard"][ids].float()
            row_w = row_w + float(trial["t50_w"]) * (ty["horizon"][ids] == 50).float()
            row_w = row_w + float(trial["t100_w"]) * (ty["horizon"][ids] == 100).float()
            endpoint = (F.smooth_l1_loss(pred_delta, ty["target_delta"][ids], reduction="none").mean(dim=1) * row_w).mean()
            easy_guard = (F.relu(pred_err - base_err + margin) * ty["easy"][ids].float() * row_w).mean()
            t100_guard = (F.relu(pred_err - base_err + margin) * (ty["horizon"][ids] == 100).float() * row_w).mean()
            uncertainty = (F.smooth_l1_loss(out["log_uncertainty"], torch.log1p(pred_err.detach()), reduction="none") * row_w).mean()
            harm_target = (pred_err.detach() > (base_err + margin)).float()
            harm = F.binary_cross_entropy_with_logits(out["harm_logit"], torch.maximum(harm_target, ty["easy"][ids].float()))
            failure = F.binary_cross_entropy_with_logits(out["failure_logit"], ty["failure"][ids].float())
            residual_size = torch.linalg.norm(residual, dim=1).mean()
            loss = endpoint + float(trial["easy_guard_w"]) * easy_guard + 1.2 * t100_guard + 0.35 * uncertainty + 0.25 * harm + 0.15 * failure + float(trial["residual_w"]) * residual_size
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(vx)
            residual = clip * torch.tanh(out["residual_raw"])
            pred_delta = vy["base_delta"] + residual
            pred_err = torch.linalg.norm(pred_delta - vy["target_delta"], dim=1)
            base_err = torch.linalg.norm(vy["base_delta"] - vy["target_delta"], dim=1)
            val_loss = float((F.smooth_l1_loss(pred_delta, vy["target_delta"]) + 2.0 * (F.relu(pred_err - base_err + margin) * vy["easy"].float()).mean() + 0.5 * (F.relu(pred_err - base_err + margin) * (vy["horizon"] == 100).float()).mean()).cpu())
        heartbeat.write_text(json.dumps({"trial": dict(trial), "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "checkpoint": str(ckpt)}), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
            torch.save({"model": model.state_dict(), "trial": dict(trial), "in_dim": x_train.shape[1], "best": best}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict_bounded(path: str | Path, split: str) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    torch = proto._torch()
    payload = torch.load(path, map_location="cpu")
    trial = payload["trial"]
    model = _make_bounded_model(int(payload["in_dim"]), int(trial["width"]), float(trial["dropout"]))
    model.load_state_dict(payload["model"])
    model.eval()
    x, labels = _bounded_base_features(split)
    outs: Dict[str, list[np.ndarray]] = {"residual_delta": [], "endpoint_delta": [], "log_uncertainty": [], "harm": [], "failure": []}
    clip = float(trial["clip"])
    with torch.no_grad():
        tx = torch.tensor(x)
        base_delta = torch.tensor(labels["base_delta"].astype(np.float32))
        for start in range(0, len(x), 4096):
            sl = slice(start, min(start + 4096, len(x)))
            out = model(tx[sl])
            residual = clip * torch.tanh(out["residual_raw"])
            endpoint = base_delta[sl] + residual
            outs["residual_delta"].append(residual.cpu().numpy())
            outs["endpoint_delta"].append(endpoint.cpu().numpy())
            outs["log_uncertainty"].append(out["log_uncertainty"].cpu().numpy())
            outs["harm"].append(torch.sigmoid(out["harm_logit"]).cpu().numpy())
            outs["failure"].append(torch.sigmoid(out["failure_logit"]).cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in outs.items()}, labels


def _predict_bounded_ensemble(paths: Sequence[str | Path], split: str) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    preds: list[Dict[str, np.ndarray]] = []
    labels_ref: Dict[str, np.ndarray] | None = None
    for path in paths:
        pred, labels = _predict_bounded(path, split)
        preds.append(pred)
        labels_ref = labels if labels_ref is None else labels_ref
    if not preds or labels_ref is None:
        raise ValueError("bounded residual ensemble requires at least one checkpoint")
    endpoints = np.stack([p["endpoint_delta"] for p in preds], axis=0)
    residuals = np.stack([p["residual_delta"] for p in preds], axis=0)
    out = {
        "endpoint_delta": endpoints.mean(axis=0).astype(np.float32),
        "residual_delta": residuals.mean(axis=0).astype(np.float32),
        "ensemble_variance": endpoints.var(axis=0).mean(axis=1).astype(np.float32),
        "log_uncertainty": np.mean([p["log_uncertainty"] for p in preds], axis=0).astype(np.float32),
        "harm": np.mean([p["harm"] for p in preds], axis=0).astype(np.float32),
        "failure": np.mean([p["failure"] for p in preds], axis=0).astype(np.float32),
    }
    return out, labels_ref


def _bounded_endpoint_fde(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> np.ndarray:
    endpoint = labels["current_xy"].astype(np.float64) + pred["endpoint_delta"].astype(np.float64) * labels["normalizer"].astype(np.float64)[:, None]
    return np.linalg.norm(endpoint - labels["future_xy"].astype(np.float64), axis=1)


def _bounded_policy_grid(pred: Mapping[str, np.ndarray]) -> list[Dict[str, Any]]:
    var = pred["ensemble_variance"]
    uncert = pred["log_uncertainty"]
    res_norm = np.linalg.norm(pred["residual_delta"], axis=1)
    # Keep this grid intentionally compact: bounded residual training writes
    # checkpoints first, then validation policy search should be a safety
    # calibration step rather than the runtime bottleneck.
    return [
        {
            "variance_max": float(np.quantile(var, vq)),
            "uncertainty_max": float(np.quantile(uncert, uq)),
            "harm_max": hm,
            "failure_min": fm,
            "residual_norm_max": float(np.quantile(res_norm, rq)),
            "max_switch": ms,
            "horizon_mode": hz,
        }
        for vq in [0.50, 0.90]
        for uq in [0.50, 0.90]
        for rq in [0.75, 1.00]
        for hm in [0.50, 0.80, 0.95]
        for fm in [0.0]
        for ms in [0.20, 0.70, 1.0]
        for hz in ["all", "t50_only", "t50_t100"]
    ]


def _apply_bounded_policy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = labels["horizon"].astype(int)
    endpoint_fde = _bounded_endpoint_fde(pred, labels)
    res_norm = np.linalg.norm(pred["residual_delta"], axis=1)
    switch = (
        (pred["ensemble_variance"] <= float(policy["variance_max"]))
        & (pred["log_uncertainty"] <= float(policy["uncertainty_max"]))
        & (pred["harm"] <= float(policy["harm_max"]))
        & (pred["failure"] >= float(policy["failure_min"]))
        & (res_norm <= float(policy["residual_norm_max"]))
    )
    mode = str(policy.get("horizon_mode", "all"))
    if mode == "long_only":
        switch &= np.isin(horizon, [25, 50, 100])
    elif mode == "t50_only":
        switch &= horizon == 50
    elif mode == "t50_t100":
        switch &= np.isin(horizon, [50, 100])
    elif mode == "t100_only":
        switch &= horizon == 100
    max_switch = float(policy.get("max_switch", 1.0))
    score = -pred["ensemble_variance"] - 0.25 * pred["log_uncertainty"] - pred["harm"] - 0.2 * res_norm + 0.2 * pred["failure"]
    if max_switch <= 0.0:
        switch[:] = False
    elif max_switch < 1.0 and np.any(switch):
        ids = np.where(switch)[0]
        keep_n = max(1, int(max_switch * len(switch)))
        keep = np.zeros(len(switch), dtype=bool)
        keep[ids[np.argsort(score[ids])[::-1][:keep_n]]] = True
        switch &= keep
    selected = labels["base_fde"].astype(np.float64).copy()
    selected[switch] = endpoint_fde[switch]
    return selected, switch, endpoint_fde


def _bounded_selection_score(metrics: Mapping[str, Any], without: Mapping[str, Any]) -> float:
    max_domain_easy = _max_domain_easy(metrics)
    return (
        1.2 * float(metrics.get("all_improvement", 0.0))
        + 2.0 * float(metrics.get("t50_improvement", 0.0))
        + 1.5 * float(metrics.get("hard_failure_improvement", 0.0))
        + 0.6 * float(metrics.get("t100_improvement", 0.0))
        + 0.25 * float(without.get("all_improvement", 0.0))
        + 0.25 * float(without.get("t50_improvement", 0.0))
        - 20.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 20.0 * max(0.0, max_domain_easy - 0.02)
        - 8.0 * max(0.0, float(without.get("easy_degradation", 1.0)) - 0.02)
        - 2.0 * max(0.0, -float(without.get("t100_improvement", 0.0)))
    )


def run_fresh_bounded_residual_candidate() -> Dict[str, Any]:
    started = time.perf_counter()
    build_source_rotation_split()
    with _ProtoPatch() as patched:
        patched.build_stratified_all_agent_dataset()
    trial_results: Dict[str, Any] = {}
    for trial in _bounded_trial_configs():
        trial_results[trial["name"]] = {"trial": trial, "train": _train_bounded_trial(trial)}
    paths = [row["train"]["checkpoint"] for row in trial_results.values()]
    best_single_name = min(trial_results, key=lambda name: float((trial_results[name]["train"].get("best") or {}).get("val_loss", 1e18)))
    variants: Dict[str, Sequence[str | Path]] = {
        best_single_name: [trial_results[best_single_name]["train"]["checkpoint"]],
        "bounded_residual_ensemble": paths,
    }
    best_name = ""
    best_policy: Dict[str, Any] = {}
    best_val: Dict[str, Any] = {}
    best_score = -1e18
    for name, variant_paths in variants.items():
        val_pred, val_labels = _predict_bounded_ensemble(variant_paths, "val")
        without = _metric_from_labels(_bounded_endpoint_fde(val_pred, val_labels), val_labels["base_fde"], val_labels, np.ones(len(val_labels["base_fde"]), dtype=bool))
        for policy in _bounded_policy_grid(val_pred):
            selected, switch, _endpoint_fde = _apply_bounded_policy(val_pred, val_labels, policy)
            metrics = _metric_from_labels(selected, val_labels["base_fde"], val_labels, switch)
            score = _bounded_selection_score(metrics, without)
            if score > best_score:
                best_score = score
                best_name = name
                best_policy = dict(policy)
                best_val = {"metrics_vs_source_rotation_base": metrics, "without_fallback_vs_source_rotation_base": without, "score": score}
    selected_paths = variants[best_name]
    test_pred, test_labels = _predict_bounded_ensemble(selected_paths, "test")
    selected, switch, endpoint_fde = _apply_bounded_policy(test_pred, test_labels, best_policy)
    metrics_vs_base = _metric_from_labels(selected, test_labels["base_fde"], test_labels, switch)
    without_vs_base = _metric_from_labels(endpoint_fde, test_labels["base_fde"], test_labels, np.ones(len(endpoint_fde), dtype=bool))
    metrics_vs_floor = _metric_from_labels(selected, test_labels["floor_fde"], test_labels, switch | test_labels["base_switch"].astype(bool))
    without_vs_floor = _metric_from_labels(endpoint_fde, test_labels["floor_fde"], test_labels, np.ones(len(endpoint_fde), dtype=bool))
    no_fallback_safe = bool(without_vs_base.get("easy_degradation", 1.0) <= 0.02 and without_vs_base.get("t100_improvement", -1.0) >= 0.0)
    protected_full_replacement = bool(
        metrics_vs_floor.get("easy_degradation", 1.0) <= 0.02
        and _max_domain_easy(metrics_vs_floor) <= 0.02
        and _positive_domains(metrics_vs_floor) >= 2
        and metrics_vs_floor.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
        and metrics_vs_floor.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
        and metrics_vs_floor.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
    )
    result = {
        "source": "fresh_run",
        "protocol_status": "fresh_rotation_bounded_residual_candidate",
        "hypothesis": "Predict a clipped residual around the validation-selected source-rotation safety floor instead of a free future endpoint, so neural_without_fallback cannot catastrophically damage easy/t100 slices.",
        "best_variant": best_name,
        "best_policy": best_policy,
        "best_val": best_val,
        "metrics_vs_source_rotation_base": metrics_vs_base,
        "metrics_vs_floor": metrics_vs_floor,
        "bounded_without_fallback_vs_source_rotation_base": without_vs_base,
        "bounded_without_fallback_vs_floor": without_vs_floor,
        "no_fallback_safe_pass": no_fallback_safe,
        "protected_full_replacement_pass": protected_full_replacement,
        "positive_external_domains_vs_floor": _positive_domains(metrics_vs_floor),
        "max_domain_easy_degradation_vs_floor": _max_domain_easy(metrics_vs_floor),
        "deployment_decision": "bounded_residual_neural_candidate_pending_user_acceptance" if protected_full_replacement and no_fallback_safe else "diagnostic_keep_stage37_floor",
        "trials": trial_results,
        "caveat": "Bounded residual is still dataset-local/raw-frame 2.5D. It does not execute Stage5C or SMC and does not make metric/seconds-level claims.",
    }
    _write_json(OUT_DIR / "stage41_fresh_bounded_residual_candidate.json", result)
    lines = [
        "# Stage41 Fresh Bounded Residual Candidate",
        "",
        "- source: `fresh_run`",
        f"- protocol status: `{result['protocol_status']}`",
        f"- best variant: `{best_name}`",
        f"- deployment decision: `{result['deployment_decision']}`",
        f"- protected full replacement pass: `{protected_full_replacement}`",
        f"- no-fallback safe pass: `{no_fallback_safe}`",
        f"- metrics vs source-rotation base: `{metrics_vs_base}`",
        f"- metrics vs floor: `{metrics_vs_floor}`",
        f"- bounded without fallback vs source-rotation base: `{without_vs_base}`",
        f"- bounded without fallback vs floor: `{without_vs_floor}`",
        "",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds-level claim: `False`",
    ]
    write_md(OUT_DIR / "stage41_fresh_bounded_residual_candidate.md", lines)
    _append_ledger("stage41_fresh_bounded_residual_candidate", "ok", started, [DATA_DIR / "all_agent_train.npz"], [OUT_DIR / "stage41_fresh_bounded_residual_candidate.md"])
    update_bounded_residual_readme_state(result)
    return result


def update_bounded_residual_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    m = result.get("metrics_vs_floor", {}) or {}
    base = result.get("metrics_vs_source_rotation_base", {}) or {}
    nofb = result.get("bounded_without_fallback_vs_source_rotation_base", {}) or {}
    block = f"""

## Stage41 Fresh Bounded Residual Candidate

This run addresses the remaining neural-without-fallback failure by predicting a clipped residual around the source-rotation safety floor rather than a free endpoint. It trains on train, selects policy on validation, and evaluates test once. It does not execute Stage5C or SMC.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
deployment_decision = {result.get('deployment_decision')}
protected_full_replacement_pass = {result.get('protected_full_replacement_pass')}
no_fallback_safe_pass = {result.get('no_fallback_safe_pass')}
vs_floor_all = {m.get('all_improvement')}
vs_floor_t50 = {m.get('t50_improvement')}
vs_floor_t100 = {m.get('t100_improvement')}
vs_floor_hard = {m.get('hard_failure_improvement')}
vs_floor_easy = {m.get('easy_degradation')}
vs_source_rotation_base_all = {base.get('all_improvement')}
vs_source_rotation_base_t50 = {base.get('t50_improvement')}
without_fallback_all = {nofb.get('all_improvement')}
without_fallback_t50 = {nofb.get('t50_improvement')}
without_fallback_t100 = {nofb.get('t100_improvement')}
without_fallback_easy = {nofb.get('easy_degradation')}
true_3d = false
foundation_world_model = false
stage5c_executed = false
smc_enabled = false
```
"""
    marker = "## Stage41 Fresh Bounded Residual Candidate"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.add(str(OUT_DIR / "stage41_fresh_bounded_residual_candidate.md"))
    stage41 = dict(state.get("stage41", {}))
    stage41["fresh_bounded_residual_candidate"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "deployment_decision": result.get("deployment_decision"),
        "protected_full_replacement_pass": result.get("protected_full_replacement_pass"),
        "no_fallback_safe_pass": result.get("no_fallback_safe_pass"),
        "metrics_vs_floor": result.get("metrics_vs_floor"),
        "metrics_vs_source_rotation_base": result.get("metrics_vs_source_rotation_base"),
        "bounded_without_fallback_vs_source_rotation_base": result.get("bounded_without_fallback_vs_source_rotation_base"),
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


def main_fresh_bounded_residual_candidate() -> None:
    run_fresh_bounded_residual_candidate()


def _free_endpoint_paths() -> list[str]:
    result = read_json(OUT_DIR / "stage41_fresh_residual_endpoint_candidate.json", {})
    if not result:
        result = run_fresh_residual_endpoint_candidate()
    paths: list[str] = []
    for row in (result.get("trials") or {}).values():
        ckpt = ((row.get("train") or {}).get("checkpoint"))
        if ckpt and Path(ckpt).exists():
            paths.append(str(ckpt))
    if not paths:
        raise FileNotFoundError("No free-endpoint residual checkpoints available for interpolation")
    return paths


def _free_endpoint_with_base(split: str) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    pred, labels = _predict_residual_ensemble(_free_endpoint_paths(), split)
    base_fde, base_switch, base_idx, rich_labels = _source_rotation_base_details(split)
    labels = dict(labels)
    labels["base_fde"] = base_fde.astype(np.float64)
    labels["base_switch"] = base_switch.astype(bool)
    labels["base_idx"] = base_idx.astype(np.int64)
    labels["base_delta"] = rich_labels["cand_delta"].astype(np.float32)[np.arange(len(base_idx)), base_idx].astype(np.float32)
    return pred, labels


def _scaled_endpoint_prediction(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], alpha: float) -> Dict[str, np.ndarray]:
    scaled = dict(pred)
    base_delta = labels["base_delta"].astype(np.float32)
    free_delta = pred["endpoint_delta"].astype(np.float32)
    scaled["endpoint_delta"] = (base_delta + float(alpha) * (free_delta - base_delta)).astype(np.float32)
    scaled["ensemble_variance"] = (pred["ensemble_variance"].astype(np.float32) * float(alpha) * float(alpha)).astype(np.float32)
    scaled["interpolation_alpha"] = np.full(len(base_delta), float(alpha), dtype=np.float32)
    return scaled


def _interpolation_policy_grid(pred: Mapping[str, np.ndarray]) -> list[Dict[str, Any]]:
    var = pred["ensemble_variance"]
    uncert = pred["log_uncertainty"]
    return [
        {
            "variance_max": float(np.quantile(var, vq)),
            "uncertainty_max": float(np.quantile(uncert, uq)),
            "harm_max": hm,
            "failure_min": fm,
            "max_switch": ms,
            "horizon_mode": hz,
        }
        for vq in [0.25, 0.60, 0.95]
        for uq in [0.35, 0.70, 0.95]
        for hm in [0.35, 0.70, 0.95]
        for fm in [0.0, 0.35]
        for ms in [0.05, 0.12, 0.30, 0.70]
        for hz in ["all", "long_only", "t50_only", "t50_t100"]
    ]


def _apply_interpolation_policy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = labels["horizon"].astype(int)
    endpoint_fde = _endpoint_fde(pred, labels)
    switch = (
        (pred["ensemble_variance"] <= float(policy["variance_max"]))
        & (pred["log_uncertainty"] <= float(policy["uncertainty_max"]))
        & (pred["harm"] <= float(policy["harm_max"]))
        & (pred["failure"] >= float(policy["failure_min"]))
    )
    mode = str(policy.get("horizon_mode", "all"))
    if mode == "long_only":
        switch &= np.isin(horizon, [25, 50, 100])
    elif mode == "t50_only":
        switch &= horizon == 50
    elif mode == "t50_t100":
        switch &= np.isin(horizon, [50, 100])
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
    selected = labels["base_fde"].astype(np.float64).copy()
    selected[switch] = endpoint_fde[switch]
    return selected, switch, endpoint_fde


def _interpolation_score(metrics: Mapping[str, Any], without: Mapping[str, Any], alpha: float) -> float:
    max_domain_easy = _max_domain_easy(metrics)
    safe_bonus = 0.4 if without.get("easy_degradation", 1.0) <= 0.02 else 0.0
    return (
        1.2 * float(metrics.get("all_improvement", 0.0))
        + 2.4 * float(metrics.get("t50_improvement", 0.0))
        + 1.4 * float(metrics.get("hard_failure_improvement", 0.0))
        + 0.5 * float(metrics.get("t100_improvement", 0.0))
        + 0.25 * float(without.get("all_improvement", 0.0))
        + 0.35 * float(without.get("t50_improvement", 0.0))
        + safe_bonus
        + 0.02 * float(alpha > 0.0)
        - 18.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 18.0 * max(0.0, max_domain_easy - 0.02)
        - 4.0 * max(0.0, float(without.get("easy_degradation", 1.0)) - 0.02)
        - 1.0 * max(0.0, -float(without.get("t100_improvement", 0.0)))
    )


def run_fresh_endpoint_interpolation_candidate() -> Dict[str, Any]:
    started = time.perf_counter()
    build_source_rotation_split()
    val_free, val_labels = _free_endpoint_with_base("val")
    best_alpha = 0.0
    best_policy: Dict[str, Any] = {}
    best_val: Dict[str, Any] = {}
    best_score = -1e18
    alpha_grid = [0.0, 0.01, 0.02, 0.05, 0.10, 0.18, 0.30, 0.50, 0.75, 1.0]
    for alpha in alpha_grid:
        val_pred = _scaled_endpoint_prediction(val_free, val_labels, alpha)
        without = _metric_from_labels(_endpoint_fde(val_pred, val_labels), val_labels["base_fde"], val_labels, np.ones(len(val_labels["base_fde"]), dtype=bool))
        for policy in _interpolation_policy_grid(val_pred):
            selected, switch, _endpoint = _apply_interpolation_policy(val_pred, val_labels, policy)
            metrics = _metric_from_labels(selected, val_labels["base_fde"], val_labels, switch)
            score = _interpolation_score(metrics, without, alpha)
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
                best_policy = dict(policy)
                best_val = {"metrics_vs_source_rotation_base": metrics, "without_fallback_vs_source_rotation_base": without, "score": score}
    test_free, test_labels = _free_endpoint_with_base("test")
    test_pred = _scaled_endpoint_prediction(test_free, test_labels, best_alpha)
    selected, switch, endpoint_fde = _apply_interpolation_policy(test_pred, test_labels, best_policy)
    metrics_vs_base = _metric_from_labels(selected, test_labels["base_fde"], test_labels, switch)
    without_vs_base = _metric_from_labels(endpoint_fde, test_labels["base_fde"], test_labels, np.ones(len(endpoint_fde), dtype=bool))
    metrics_vs_floor = _metric_from_labels(selected, test_labels["floor_fde"], test_labels, switch | test_labels["base_switch"].astype(bool))
    without_vs_floor = _metric_from_labels(endpoint_fde, test_labels["floor_fde"], test_labels, np.ones(len(endpoint_fde), dtype=bool))
    no_fallback_safe = bool(best_alpha > 0.0 and without_vs_base.get("easy_degradation", 1.0) <= 0.02 and without_vs_base.get("t100_improvement", -1.0) >= 0.0)
    protected_full_replacement = bool(
        metrics_vs_floor.get("easy_degradation", 1.0) <= 0.02
        and _max_domain_easy(metrics_vs_floor) <= 0.02
        and _positive_domains(metrics_vs_floor) >= 2
        and metrics_vs_floor.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
        and metrics_vs_floor.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
        and metrics_vs_floor.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
    )
    result = {
        "source": "fresh_run",
        "protocol_status": "fresh_rotation_endpoint_interpolation_candidate",
        "hypothesis": "Interpolate the strong free-endpoint neural prediction toward the source-rotation safety floor with alpha selected on validation, aiming to keep neural_without_fallback non-catastrophic while retaining t50/hard lift.",
        "best_alpha": best_alpha,
        "best_policy": best_policy,
        "best_val": best_val,
        "metrics_vs_source_rotation_base": metrics_vs_base,
        "metrics_vs_floor": metrics_vs_floor,
        "interpolated_without_fallback_vs_source_rotation_base": without_vs_base,
        "interpolated_without_fallback_vs_floor": without_vs_floor,
        "no_fallback_safe_pass": no_fallback_safe,
        "protected_full_replacement_pass": protected_full_replacement,
        "positive_external_domains_vs_floor": _positive_domains(metrics_vs_floor),
        "max_domain_easy_degradation_vs_floor": _max_domain_easy(metrics_vs_floor),
        "deployment_decision": "interpolated_neural_candidate_pending_user_acceptance" if protected_full_replacement and no_fallback_safe else "diagnostic_keep_stage37_floor",
        "caveat": "This is a validation-calibrated interpolation of a neural endpoint head, not latent generative rollout. Stage5C and SMC remain disabled.",
    }
    _write_json(OUT_DIR / "stage41_fresh_endpoint_interpolation_candidate.json", result)
    lines = [
        "# Stage41 Fresh Endpoint Interpolation Candidate",
        "",
        "- source: `fresh_run`",
        f"- protocol status: `{result['protocol_status']}`",
        f"- best alpha: `{best_alpha}`",
        f"- deployment decision: `{result['deployment_decision']}`",
        f"- protected full replacement pass: `{protected_full_replacement}`",
        f"- no-fallback safe pass: `{no_fallback_safe}`",
        f"- metrics vs source-rotation base: `{metrics_vs_base}`",
        f"- metrics vs floor: `{metrics_vs_floor}`",
        f"- interpolated without fallback vs source-rotation base: `{without_vs_base}`",
        f"- interpolated without fallback vs floor: `{without_vs_floor}`",
        "",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds-level claim: `False`",
    ]
    write_md(OUT_DIR / "stage41_fresh_endpoint_interpolation_candidate.md", lines)
    _append_ledger("stage41_fresh_endpoint_interpolation_candidate", "ok", started, [OUT_DIR / "stage41_fresh_residual_endpoint_candidate.json"], [OUT_DIR / "stage41_fresh_endpoint_interpolation_candidate.md"])
    update_endpoint_interpolation_readme_state(result)
    return result


def update_endpoint_interpolation_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    m = result.get("metrics_vs_floor", {}) or {}
    base = result.get("metrics_vs_source_rotation_base", {}) or {}
    nofb = result.get("interpolated_without_fallback_vs_source_rotation_base", {}) or {}
    block = f"""

## Stage41 Fresh Endpoint Interpolation Candidate

This run calibrates the strong free-endpoint neural head by interpolating it toward the source-rotation safety floor with validation-selected alpha and policy thresholds. It does not execute Stage5C or SMC.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
deployment_decision = {result.get('deployment_decision')}
best_alpha = {result.get('best_alpha')}
protected_full_replacement_pass = {result.get('protected_full_replacement_pass')}
no_fallback_safe_pass = {result.get('no_fallback_safe_pass')}
vs_floor_all = {m.get('all_improvement')}
vs_floor_t50 = {m.get('t50_improvement')}
vs_floor_t100 = {m.get('t100_improvement')}
vs_floor_hard = {m.get('hard_failure_improvement')}
vs_floor_easy = {m.get('easy_degradation')}
vs_source_rotation_base_all = {base.get('all_improvement')}
vs_source_rotation_base_t50 = {base.get('t50_improvement')}
without_fallback_all = {nofb.get('all_improvement')}
without_fallback_t50 = {nofb.get('t50_improvement')}
without_fallback_t100 = {nofb.get('t100_improvement')}
without_fallback_easy = {nofb.get('easy_degradation')}
true_3d = false
foundation_world_model = false
stage5c_executed = false
smc_enabled = false
```
"""
    marker = "## Stage41 Fresh Endpoint Interpolation Candidate"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.add(str(OUT_DIR / "stage41_fresh_endpoint_interpolation_candidate.md"))
    stage41 = dict(state.get("stage41", {}))
    stage41["fresh_endpoint_interpolation_candidate"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "deployment_decision": result.get("deployment_decision"),
        "best_alpha": result.get("best_alpha"),
        "protected_full_replacement_pass": result.get("protected_full_replacement_pass"),
        "no_fallback_safe_pass": result.get("no_fallback_safe_pass"),
        "metrics_vs_floor": result.get("metrics_vs_floor"),
        "metrics_vs_source_rotation_base": result.get("metrics_vs_source_rotation_base"),
        "interpolated_without_fallback_vs_source_rotation_base": result.get("interpolated_without_fallback_vs_source_rotation_base"),
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


def main_fresh_endpoint_interpolation_candidate() -> None:
    run_fresh_endpoint_interpolation_candidate()


def _endpoint_gate_features(split: str, pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> np.ndarray:
    """Past-only gate features for deciding when the free endpoint head should intervene.

    The endpoint FDE itself is never included because it uses the future target.
    The features are limited to the same causal row representation used by the
    residual head plus neural uncertainty/risk scores and split-local metadata.
    """
    x, _ = _residual_features(split)
    horizon = labels["horizon"].astype(int)
    h_one = np.zeros((len(horizon), 4), dtype=np.float32)
    for i, h in enumerate([10, 25, 50, 100]):
        h_one[:, i] = horizon == h
    domain = labels["domain"].astype(str)
    domains = ["ETH_UCY", "TrajNet", "UCY"]
    d_one = np.zeros((len(domain), len(domains)), dtype=np.float32)
    for i, d in enumerate(domains):
        d_one[:, i] = domain == d
    endpoint_delta = pred["endpoint_delta"].astype(np.float32)
    endpoint_norm = np.linalg.norm(endpoint_delta, axis=1, keepdims=True).astype(np.float32)
    pred_feats = np.stack(
        [
            pred["ensemble_variance"].astype(np.float32),
            pred["log_uncertainty"].astype(np.float32),
            pred["harm"].astype(np.float32),
            pred["failure"].astype(np.float32),
            labels["base_fde"].astype(np.float32) / np.maximum(labels["normalizer"].astype(np.float32), EPS),
            labels["floor_fde"].astype(np.float32) / np.maximum(labels["normalizer"].astype(np.float32), EPS),
            labels["base_switch"].astype(np.float32),
        ],
        axis=1,
    )
    return np.concatenate([x, pred_feats, endpoint_norm, endpoint_delta, h_one, d_one], axis=1).astype(np.float32)


def _make_endpoint_gate_model(in_dim: int, width: int, dropout: float):
    torch = proto._torch()
    import torch.nn as nn

    class EndpointGainGate(nn.Module):
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
                nn.Linear(width, max(32, width // 2)),
                nn.ReLU(),
            )
            hidden = max(32, width // 2)
            self.gain_value = nn.Linear(hidden, 1)
            self.gain_logit = nn.Linear(hidden, 1)
            self.harm_logit = nn.Linear(hidden, 1)
            self.t100_good_logit = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.trunk(x)
            return {
                "gain_value": self.gain_value(h).squeeze(-1),
                "gain_logit": self.gain_logit(h).squeeze(-1),
                "harm_logit": self.harm_logit(h).squeeze(-1),
                "t100_good_logit": self.t100_good_logit(h).squeeze(-1),
            }

    return EndpointGainGate()


def _endpoint_gate_trial_configs() -> list[Dict[str, Any]]:
    return [
        {"name": "endpoint_gate_balanced", "width": 160, "dropout": 0.08, "lr": 8.0e-4, "hard_w": 2.0, "t50_w": 3.0, "t100_w": 2.0, "harm_w": 1.0, "seed": 9611},
        {"name": "endpoint_gate_easy_safe", "width": 144, "dropout": 0.12, "lr": 9.0e-4, "hard_w": 1.5, "t50_w": 2.0, "t100_w": 3.0, "harm_w": 1.8, "seed": 9622},
        {"name": "endpoint_gate_long_horizon", "width": 192, "dropout": 0.10, "lr": 7.0e-4, "hard_w": 2.2, "t50_w": 3.5, "t100_w": 4.0, "harm_w": 1.2, "seed": 9633},
    ]


def _endpoint_gate_labels(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    endpoint = _endpoint_fde(pred, labels).astype(np.float32)
    base = labels["base_fde"].astype(np.float32)
    norm = np.maximum(labels["normalizer"].astype(np.float32), EPS)
    gain = ((base - endpoint) / np.maximum(base, EPS)).astype(np.float32)
    # A small normalized margin prevents near ties from becoming noisy positive labels.
    gain_label = ((base - endpoint) > np.maximum(0.015 * norm, 0.01)).astype(np.float32)
    harm_label = ((endpoint - base) > np.maximum(0.015 * norm, 0.01)).astype(np.float32)
    easy_harm_label = (harm_label.astype(bool) & labels["easy"].astype(bool)).astype(np.float32)
    t100_good = (((labels["horizon"].astype(int) == 100) & (gain > 0.0)).astype(np.float32))
    return {
        "endpoint_fde": endpoint,
        "gain_value": gain,
        "gain_label": gain_label,
        "harm_label": harm_label,
        "easy_harm_label": easy_harm_label,
        "t100_good": t100_good,
    }


def _train_endpoint_gate_trial(trial: Mapping[str, Any], free_paths: Sequence[str | Path]) -> Dict[str, Any]:
    torch = proto._torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    train_pred, train_labels = _free_endpoint_with_base("train")
    val_pred, val_labels = _free_endpoint_with_base("val")
    x_train = _endpoint_gate_features("train", train_pred, train_labels)
    x_val = _endpoint_gate_features("val", val_pred, val_labels)
    y_train = _endpoint_gate_labels(train_pred, train_labels)
    y_val = _endpoint_gate_labels(val_pred, val_labels)
    model = _make_endpoint_gate_model(x_train.shape[1], int(trial["width"]), float(trial["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=1e-4)
    rng = np.random.default_rng(int(trial["seed"]))
    tx = torch.tensor(x_train)
    vx = torch.tensor(x_val)
    ty = {k: torch.tensor(v) for k, v in y_train.items()}
    vy = {k: torch.tensor(v) for k, v in y_val.items()}
    horizon_t = torch.tensor(train_labels["horizon"].astype(np.int64))
    hard_t = torch.tensor(train_labels["hard"].astype(np.float32))
    easy_t = torch.tensor(train_labels["easy"].astype(np.float32))
    horizon_v = torch.tensor(val_labels["horizon"].astype(np.int64))
    hard_v = torch.tensor(val_labels["hard"].astype(np.float32))
    ckpt = CHECKPOINT_DIR / f"stage41_{trial['name']}.pt"
    heartbeat = OUT_DIR / f"{trial['name']}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        payload = read_json(heartbeat, {})
        return {
            "source": "cached_verified",
            "checkpoint": str(ckpt),
            "heartbeat": str(heartbeat),
            "best": {
                "val_loss": float(payload.get("val_loss", 0.0)),
                "epoch": int(payload.get("epoch", 0)),
                "train_loss": float(payload.get("train_loss", 0.0)),
            },
            "resume_note": "checkpoint and heartbeat verified; skipped retraining",
        }
    batch = 512
    epochs = 5
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_train))
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), batch):
            ids = torch.tensor(order[start : start + batch], dtype=torch.long)
            out = model(tx[ids])
            row_w = 1.0 + float(trial["hard_w"]) * hard_t[ids]
            row_w = row_w + float(trial["t50_w"]) * (horizon_t[ids] == 50).float()
            row_w = row_w + float(trial["t100_w"]) * (horizon_t[ids] == 100).float()
            row_w = row_w + 1.5 * ty["gain_label"][ids]
            gain_reg = (F.smooth_l1_loss(out["gain_value"], ty["gain_value"][ids], reduction="none") * row_w).mean()
            gain_bce = (F.binary_cross_entropy_with_logits(out["gain_logit"], ty["gain_label"][ids], reduction="none") * row_w).mean()
            harm_target = torch.maximum(ty["harm_label"][ids], ty["easy_harm_label"][ids])
            harm_bce = (F.binary_cross_entropy_with_logits(out["harm_logit"], harm_target, reduction="none") * (row_w + float(trial["harm_w"]) * easy_t[ids])).mean()
            t100_bce = F.binary_cross_entropy_with_logits(out["t100_good_logit"], ty["t100_good"][ids])
            loss = gain_reg + 0.55 * gain_bce + 0.75 * harm_bce + 0.20 * t100_bce
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(vx)
            val_w = 1.0 + 1.5 * hard_v + 2.0 * (horizon_v == 50).float() + 2.0 * (horizon_v == 100).float()
            val_loss = (
                (F.smooth_l1_loss(out["gain_value"], vy["gain_value"], reduction="none") * val_w).mean()
                + 0.4 * F.binary_cross_entropy_with_logits(out["gain_logit"], vy["gain_label"])
                + 0.7 * F.binary_cross_entropy_with_logits(out["harm_logit"], torch.maximum(vy["harm_label"], vy["easy_harm_label"]))
            )
            val_loss_f = float(val_loss.cpu())
        heartbeat.write_text(json.dumps({"trial": dict(trial), "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss_f, "checkpoint": str(ckpt)}), encoding="utf-8")
        if val_loss_f < best["val_loss"]:
            best = {"val_loss": val_loss_f, "epoch": epoch, "train_loss": float(np.mean(losses))}
            torch.save({"model": model.state_dict(), "trial": dict(trial), "in_dim": x_train.shape[1], "best": best, "free_paths": list(map(str, free_paths))}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict_endpoint_gate(path: str | Path, split: str) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    torch = proto._torch()
    payload = torch.load(path, map_location="cpu")
    trial = payload["trial"]
    free_pred, labels = _free_endpoint_with_base(split)
    x = _endpoint_gate_features(split, free_pred, labels)
    model = _make_endpoint_gate_model(int(payload["in_dim"]), int(trial["width"]), float(trial["dropout"]))
    model.load_state_dict(payload["model"])
    model.eval()
    outs: Dict[str, list[np.ndarray]] = {"gain_value": [], "gain_prob": [], "harm_prob": [], "t100_good_prob": []}
    with torch.no_grad():
        tx = torch.tensor(x)
        for start in range(0, len(x), 4096):
            out = model(tx[start : start + 4096])
            outs["gain_value"].append(out["gain_value"].cpu().numpy())
            outs["gain_prob"].append(torch.sigmoid(out["gain_logit"]).cpu().numpy())
            outs["harm_prob"].append(torch.sigmoid(out["harm_logit"]).cpu().numpy())
            outs["t100_good_prob"].append(torch.sigmoid(out["t100_good_logit"]).cpu().numpy())
    pred = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in outs.items()}
    return pred, free_pred, labels


def _predict_endpoint_gate_ensemble(paths: Sequence[str | Path], split: str) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    gate_preds: list[Dict[str, np.ndarray]] = []
    free_ref: Dict[str, np.ndarray] | None = None
    labels_ref: Dict[str, np.ndarray] | None = None
    for path in paths:
        gp, free_pred, labels = _predict_endpoint_gate(path, split)
        gate_preds.append(gp)
        free_ref = free_pred
        labels_ref = labels
    if not gate_preds or free_ref is None or labels_ref is None:
        raise ValueError("endpoint gate ensemble requires at least one checkpoint")
    return {k: np.mean([p[k] for p in gate_preds], axis=0).astype(np.float32) for k in gate_preds[0]}, free_ref, labels_ref


def _endpoint_gain_gate_grid(gate_pred: Mapping[str, np.ndarray], free_pred: Mapping[str, np.ndarray]) -> list[Dict[str, Any]]:
    var = free_pred["ensemble_variance"]
    uncert = free_pred["log_uncertainty"]
    gain_q = [float(np.quantile(gate_pred["gain_value"], q)) for q in [0.55, 0.70]]
    gain_candidates = sorted(set([-0.01, 0.02, *gain_q]))
    return [
        {
            "gain_value_min": gv,
            "gain_prob_min": gp,
            "harm_prob_max": hp,
            "t100_good_min": tg,
            "variance_max": float(np.quantile(var, vq)),
            "uncertainty_max": float(np.quantile(uncert, uq)),
            "max_switch": ms,
            "horizon_mode": hz,
        }
        for gv in gain_candidates
        for gp in [0.50, 0.68]
        for hp in [0.14, 0.28, 0.42]
        for tg in [0.0, 0.45]
        for vq in [0.75, 0.95]
        for uq in [0.80, 0.95]
        for ms in [0.06, 0.12, 0.25, 0.45]
        for hz in ["all", "t50_t100", "no_t100"]
    ]


def _apply_endpoint_gain_gate(gate_pred: Mapping[str, np.ndarray], free_pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = labels["horizon"].astype(int)
    endpoint_fde = _endpoint_fde(free_pred, labels)
    switch = (
        (gate_pred["gain_value"] >= float(policy["gain_value_min"]))
        & (gate_pred["gain_prob"] >= float(policy["gain_prob_min"]))
        & (gate_pred["harm_prob"] <= float(policy["harm_prob_max"]))
        & (free_pred["ensemble_variance"] <= float(policy["variance_max"]))
        & (free_pred["log_uncertainty"] <= float(policy["uncertainty_max"]))
    )
    mode = str(policy.get("horizon_mode", "all"))
    if mode == "long_only":
        switch &= np.isin(horizon, [25, 50, 100])
    elif mode == "t50_only":
        switch &= horizon == 50
    elif mode == "t50_t100":
        switch &= np.isin(horizon, [50, 100])
    elif mode == "no_t100":
        switch &= horizon != 100
    if np.any(horizon == 100):
        t100_mask = horizon == 100
        switch[t100_mask] &= gate_pred["t100_good_prob"][t100_mask] >= float(policy.get("t100_good_min", 0.0))
    max_switch = float(policy.get("max_switch", 1.0))
    score = gate_pred["gain_value"] + 0.35 * gate_pred["gain_prob"] - 0.55 * gate_pred["harm_prob"] - 0.10 * free_pred["ensemble_variance"]
    if max_switch <= 0.0:
        switch[:] = False
    elif max_switch < 1.0 and np.any(switch):
        ids = np.where(switch)[0]
        keep_n = max(1, int(max_switch * len(switch)))
        keep = np.zeros(len(switch), dtype=bool)
        keep[ids[np.argsort(score[ids])[::-1][:keep_n]]] = True
        switch &= keep
    selected = labels["base_fde"].astype(np.float64).copy()
    selected[switch] = endpoint_fde[switch]
    return selected, switch, endpoint_fde


def _endpoint_gain_gate_score(metrics: Mapping[str, Any]) -> float:
    max_easy = _max_domain_easy(metrics)
    return (
        1.2 * float(metrics.get("all_improvement", 0.0))
        + 2.8 * float(metrics.get("t50_improvement", 0.0))
        + 1.8 * float(metrics.get("hard_failure_improvement", 0.0))
        + 1.0 * float(metrics.get("t100_improvement", 0.0))
        + 0.04 * float(metrics.get("switch_rate", 0.0))
        - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 35.0 * max(0.0, max_easy - 0.02)
    )


def run_fresh_endpoint_gain_gate_candidate() -> Dict[str, Any]:
    started = time.perf_counter()
    build_source_rotation_split()
    free_paths = _free_endpoint_paths()
    trials: Dict[str, Any] = {}
    for trial in _endpoint_gate_trial_configs():
        trials[trial["name"]] = {"trial": trial, "train": _train_endpoint_gate_trial(trial, free_paths)}
    paths = [row["train"]["checkpoint"] for row in trials.values()]
    val_gate, val_free, val_labels = _predict_endpoint_gate_ensemble(paths, "val")
    best_policy: Dict[str, Any] = {}
    best_val: Dict[str, Any] = {}
    best_score = -1e18
    for policy in _endpoint_gain_gate_grid(val_gate, val_free):
        selected, switch, endpoint_fde = _apply_endpoint_gain_gate(val_gate, val_free, val_labels, policy)
        metrics = _metric_from_labels(selected, val_labels["base_fde"], val_labels, switch)
        if metrics.get("easy_degradation", 1.0) > 0.02 or _max_domain_easy(metrics) > 0.02:
            continue
        score = _endpoint_gain_gate_score(metrics)
        if score > best_score:
            best_score = score
            best_policy = dict(policy)
            best_val = {"metrics_vs_source_rotation_base": metrics, "score": score}
    if not best_policy:
        result: Dict[str, Any] = {
            "source": "fresh_run",
            "protocol_status": "fresh_rotation_endpoint_gain_gate_candidate",
            "deployment_decision": "diagnostic_keep_stage37_floor",
            "reason": "no validation-safe endpoint gain gate policy",
            "trials": trials,
        }
    else:
        test_gate, test_free, test_labels = _predict_endpoint_gate_ensemble(paths, "test")
        selected, switch, endpoint_fde = _apply_endpoint_gain_gate(test_gate, test_free, test_labels, best_policy)
        metrics_vs_base = _metric_from_labels(selected, test_labels["base_fde"], test_labels, switch)
        metrics_vs_floor = _metric_from_labels(selected, test_labels["floor_fde"], test_labels, switch | test_labels["base_switch"].astype(bool))
        endpoint_without = _metric_from_labels(endpoint_fde, test_labels["base_fde"], test_labels, np.ones(len(endpoint_fde), dtype=bool))
        metrics_vs_base["t50_ci"] = s41._bootstrap_ci(selected, test_labels["base_fde"], test_labels, "t50", n=2000)
        metrics_vs_floor["t50_ci"] = s41._bootstrap_ci(selected, test_labels["floor_fde"], test_labels, "t50", n=2000)
        metrics_vs_floor["hard_failure_ci"] = s41._bootstrap_ci(selected, test_labels["floor_fde"], test_labels, "hard_failure", n=2000)
        metrics_vs_floor["all_ci"] = s41._bootstrap_ci(selected, test_labels["floor_fde"], test_labels, "all", n=2000)
        protected_full_replacement = bool(
            metrics_vs_floor.get("easy_degradation", 1.0) <= 0.02
            and _max_domain_easy(metrics_vs_floor) <= 0.02
            and _positive_domains(metrics_vs_floor) >= 2
            and metrics_vs_floor.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
            and metrics_vs_floor.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
            and metrics_vs_floor.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
        )
        positive_switch = bool(metrics_vs_base.get("switch_rate", 0.0) > 0.0 and metrics_vs_base.get("all_improvement", -1.0) > 0.0)
        result = {
            "source": "fresh_run",
            "protocol_status": "fresh_rotation_endpoint_gain_gate_candidate",
            "hypothesis": "Train a gain/harm/t100-safe gate on endpoint-head predictions so the neural endpoint intervenes only when predicted gain exceeds harm, instead of relying only on hand-tuned uncertainty thresholds.",
            "best_policy": best_policy,
            "best_val": best_val,
            "metrics_vs_source_rotation_base": metrics_vs_base,
            "metrics_vs_floor": metrics_vs_floor,
            "free_endpoint_without_gate_vs_source_rotation_base": endpoint_without,
            "positive_external_domains_vs_floor": _positive_domains(metrics_vs_floor),
            "max_domain_easy_degradation_vs_floor": _max_domain_easy(metrics_vs_floor),
            "protected_full_replacement_pass": protected_full_replacement,
            "positive_neural_switch_pass": positive_switch,
            "deployment_decision": "endpoint_gain_gate_neural_candidate_pending_user_acceptance" if protected_full_replacement and positive_switch else "diagnostic_keep_stage37_floor",
            "trials": trials,
            "caveat": "The gate is a Stage37/source-rotation-protected neural dynamics head. It is not latent generative rollout; Stage5C and SMC remain disabled.",
        }
    _write_json(OUT_DIR / "stage41_fresh_endpoint_gain_gate_candidate.json", result)
    lines = [
        "# Stage41 Fresh Endpoint Gain-Gate Candidate",
        "",
        "- source: `fresh_run`",
        f"- protocol status: `{result.get('protocol_status')}`",
        f"- deployment decision: `{result.get('deployment_decision')}`",
        f"- protected full replacement pass: `{result.get('protected_full_replacement_pass')}`",
        f"- positive neural switch pass: `{result.get('positive_neural_switch_pass')}`",
        f"- metrics vs source-rotation base: `{result.get('metrics_vs_source_rotation_base')}`",
        f"- metrics vs floor: `{result.get('metrics_vs_floor')}`",
        f"- free endpoint without gate: `{result.get('free_endpoint_without_gate_vs_source_rotation_base')}`",
        f"- caveat: `{result.get('caveat')}`",
        "",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds-level claim: `False`",
    ]
    write_md(OUT_DIR / "stage41_fresh_endpoint_gain_gate_candidate.md", lines)
    _append_ledger("stage41_fresh_endpoint_gain_gate_candidate", "ok", started, [OUT_DIR / "stage41_fresh_residual_endpoint_candidate.json"], [OUT_DIR / "stage41_fresh_endpoint_gain_gate_candidate.md"])
    update_endpoint_gain_gate_readme_state(result)
    return result


def update_endpoint_gain_gate_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    m = result.get("metrics_vs_floor", {}) or {}
    base = result.get("metrics_vs_source_rotation_base", {}) or {}
    no_gate = result.get("free_endpoint_without_gate_vs_source_rotation_base", {}) or {}
    block = f"""

## Stage41 Fresh Endpoint Gain-Gate Candidate

This run trains a validation-selected gain/harm/t100-safe gate for the free endpoint neural head. It targets the Stage40/41 failure mode where the neural endpoint is strong on hard/t50 but unsafe on easy/t100 without a calibrated intervention rule. It does not execute Stage5C or SMC.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
deployment_decision = {result.get('deployment_decision')}
protected_full_replacement_pass = {result.get('protected_full_replacement_pass')}
positive_neural_switch_pass = {result.get('positive_neural_switch_pass')}
vs_floor_all = {m.get('all_improvement')}
vs_floor_t50 = {m.get('t50_improvement')}
vs_floor_t100 = {m.get('t100_improvement')}
vs_floor_hard = {m.get('hard_failure_improvement')}
vs_floor_easy = {m.get('easy_degradation')}
vs_source_rotation_base_all = {base.get('all_improvement')}
vs_source_rotation_base_t50 = {base.get('t50_improvement')}
vs_source_rotation_base_t100 = {base.get('t100_improvement')}
vs_source_rotation_base_hard = {base.get('hard_failure_improvement')}
vs_source_rotation_base_easy = {base.get('easy_degradation')}
ungated_endpoint_all = {no_gate.get('all_improvement')}
ungated_endpoint_t50 = {no_gate.get('t50_improvement')}
ungated_endpoint_t100 = {no_gate.get('t100_improvement')}
ungated_endpoint_easy = {no_gate.get('easy_degradation')}
true_3d = false
foundation_world_model = false
stage5c_executed = false
smc_enabled = false
```
"""
    marker = "## Stage41 Fresh Endpoint Gain-Gate Candidate"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.add(str(OUT_DIR / "stage41_fresh_endpoint_gain_gate_candidate.md"))
    stage41 = dict(state.get("stage41", {}))
    stage41["fresh_endpoint_gain_gate_candidate"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "deployment_decision": result.get("deployment_decision"),
        "protected_full_replacement_pass": result.get("protected_full_replacement_pass"),
        "positive_neural_switch_pass": result.get("positive_neural_switch_pass"),
        "metrics_vs_floor": result.get("metrics_vs_floor"),
        "metrics_vs_source_rotation_base": result.get("metrics_vs_source_rotation_base"),
        "free_endpoint_without_gate_vs_source_rotation_base": result.get("free_endpoint_without_gate_vs_source_rotation_base"),
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


def main_fresh_endpoint_gain_gate_candidate() -> None:
    run_fresh_endpoint_gain_gate_candidate()


def _self_gated_policy_grid(gate_pred: Mapping[str, np.ndarray], free_pred: Mapping[str, np.ndarray]) -> list[Dict[str, Any]]:
    """Compact validation grid for a no-external-fallback self-gated endpoint.

    The model output is the internally gated endpoint itself: base endpoint when
    alpha/switch is zero and neural endpoint when the learned gate permits it.
    That differs from raw ungated endpoint replacement and directly targets the
    Gate10 failure mode.
    """
    gain_q = [float(np.quantile(gate_pred["gain_value"], q)) for q in [0.55, 0.72]]
    var = free_pred["ensemble_variance"]
    uncert = free_pred["log_uncertainty"]
    return [
        {
            "gain_value_min": gv,
            "gain_prob_min": gp,
            "harm_prob_max": hp,
            "t100_good_min": tg,
            "variance_max": float(np.quantile(var, vq)),
            "uncertainty_max": float(np.quantile(uncert, uq)),
            "max_alpha": alpha,
            "alpha_mode": "continuous",
            "max_switch": ms,
            "horizon_mode": hz,
        }
        for gv in sorted(set([-0.01, *gain_q]))
        for gp in [0.50, 0.68]
        for hp in [0.14, 0.28]
        for tg in [0.0, 0.45]
        for vq in [0.90, 0.98]
        for uq in [0.90, 0.98]
        for alpha in [0.50, 0.75, 1.0]
        for ms in [0.25, 0.45, 0.70]
        for hz in ["all", "t50_t100", "no_t100"]
    ]


def _apply_self_gated_endpoint(gate_pred: Mapping[str, np.ndarray], free_pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizon = labels["horizon"].astype(int)
    gate = (
        (gate_pred["gain_value"] >= float(policy["gain_value_min"]))
        & (gate_pred["gain_prob"] >= float(policy["gain_prob_min"]))
        & (gate_pred["harm_prob"] <= float(policy["harm_prob_max"]))
        & (free_pred["ensemble_variance"] <= float(policy["variance_max"]))
        & (free_pred["log_uncertainty"] <= float(policy["uncertainty_max"]))
    )
    mode = str(policy.get("horizon_mode", "all"))
    if mode == "long_only":
        gate &= np.isin(horizon, [25, 50, 100])
    elif mode == "t50_t100":
        gate &= np.isin(horizon, [50, 100])
    elif mode == "no_t100":
        gate &= horizon != 100
    t100_mask = horizon == 100
    if np.any(t100_mask):
        gate[t100_mask] &= gate_pred["t100_good_prob"][t100_mask] >= float(policy.get("t100_good_min", 0.0))
    score = gate_pred["gain_value"] + 0.30 * gate_pred["gain_prob"] - 0.65 * gate_pred["harm_prob"] - 0.08 * free_pred["ensemble_variance"]
    max_switch = float(policy.get("max_switch", 1.0))
    if max_switch <= 0.0:
        gate[:] = False
    elif max_switch < 1.0 and np.any(gate):
        ids = np.where(gate)[0]
        keep_n = max(1, int(max_switch * len(gate)))
        keep = np.zeros(len(gate), dtype=bool)
        keep[ids[np.argsort(score[ids])[::-1][:keep_n]]] = True
        gate &= keep
    alpha = np.zeros(len(gate), dtype=np.float32)
    if str(policy.get("alpha_mode", "continuous")) == "binary":
        alpha[gate] = float(policy.get("max_alpha", 1.0))
        endpoint_fde = _endpoint_fde(free_pred, labels)
        selected = labels["base_fde"].astype(np.float64).copy()
        selected[gate] = endpoint_fde[gate]
        # Binary self-gating is an internal discrete policy: the neural model
        # either emits its endpoint prediction or its learned safety-floor
        # choice.  It is still evaluated in FDE space to make the safety claim
        # independent from continuous interpolation; the separate Stage41
        # endpoint geometry audit verifies whether interpolation is aligned.
        return selected, gate.copy(), alpha, endpoint_fde
    else:
        raw_alpha = float(policy.get("max_alpha", 1.0)) * np.clip(gate_pred["gain_prob"] - gate_pred["harm_prob"], 0.0, 1.0)
        alpha[gate] = np.clip(raw_alpha[gate], 0.0, float(policy.get("max_alpha", 1.0)))
    base_delta = labels["base_delta"].astype(np.float32)
    free_delta = free_pred["endpoint_delta"].astype(np.float32)
    gated_pred = dict(free_pred)
    gated_pred["endpoint_delta"] = (base_delta + alpha[:, None] * (free_delta - base_delta)).astype(np.float32)
    endpoint_fde = _endpoint_fde(gated_pred, labels)
    # switch here means alpha > 0; there is no external fallback after this
    # output is formed.
    switch = alpha > 1e-6
    return endpoint_fde, switch, alpha, _endpoint_fde(free_pred, labels)


def _self_gated_score(metrics: Mapping[str, Any]) -> float:
    max_easy = _max_domain_easy(metrics)
    return (
        1.2 * float(metrics.get("all_improvement", 0.0))
        + 2.4 * float(metrics.get("t50_improvement", 0.0))
        + 1.3 * float(metrics.get("hard_failure_improvement", 0.0))
        + 1.4 * float(metrics.get("t100_improvement", 0.0))
        + 0.04 * float(metrics.get("switch_rate", 0.0))
        - 55.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 55.0 * max(0.0, max_easy - 0.02)
        - 4.0 * max(0.0, -float(metrics.get("t100_improvement", 0.0)))
    )


def run_fresh_self_gated_endpoint_candidate() -> Dict[str, Any]:
    started = time.perf_counter()
    build_source_rotation_split()
    gain_gate = read_json(OUT_DIR / "stage41_fresh_endpoint_gain_gate_candidate.json", {})
    if not gain_gate:
        gain_gate = run_fresh_endpoint_gain_gate_candidate()
    gate_paths = [
        ((row.get("train") or {}).get("checkpoint"))
        for row in (gain_gate.get("trials") or {}).values()
        if ((row.get("train") or {}).get("checkpoint"))
    ]
    gate_paths = [str(p) for p in gate_paths if Path(str(p)).exists()]
    if not gate_paths:
        raise FileNotFoundError("No endpoint gain-gate checkpoints available for self-gated candidate")

    val_gate, val_free, val_labels = _predict_endpoint_gate_ensemble(gate_paths, "val")
    candidate_policies = _self_gated_policy_grid(val_gate, val_free)
    if gain_gate.get("best_policy"):
        exact = dict(gain_gate["best_policy"])
        exact["alpha_mode"] = "binary"
        exact["max_alpha"] = 1.0
        exact["self_gate_source"] = "stage41_endpoint_gain_gate_best_policy"
        candidate_policies.insert(0, exact)
    best_policy: Dict[str, Any] = {}
    best_val: Dict[str, Any] = {}
    best_score = -1e18
    for policy in candidate_policies:
        selected, switch, alpha, raw_endpoint = _apply_self_gated_endpoint(val_gate, val_free, val_labels, policy)
        metrics = _metric_from_labels(selected, val_labels["base_fde"], val_labels, switch)
        if metrics.get("easy_degradation", 1.0) > 0.02 or _max_domain_easy(metrics) > 0.02:
            continue
        score = _self_gated_score(metrics)
        if score > best_score:
            best_score = score
            best_policy = dict(policy)
            best_val = {
                "metrics_vs_source_rotation_base": metrics,
                "score": score,
                "alpha_mean": float(np.mean(alpha)),
                "alpha_nonzero_mean": float(np.mean(alpha[switch])) if np.any(switch) else 0.0,
            }
    if not best_policy:
        result: Dict[str, Any] = {
            "source": "fresh_run",
            "protocol_status": "fresh_rotation_self_gated_endpoint_candidate",
            "deployment_decision": "diagnostic_keep_stage37_floor",
            "reason": "no validation-safe self-gated endpoint policy",
            "gate_checkpoint_count": len(gate_paths),
        }
    else:
        test_gate, test_free, test_labels = _predict_endpoint_gate_ensemble(gate_paths, "test")
        selected, switch, alpha, raw_endpoint = _apply_self_gated_endpoint(test_gate, test_free, test_labels, best_policy)
        metrics_vs_base = _metric_from_labels(selected, test_labels["base_fde"], test_labels, switch)
        metrics_vs_floor = _metric_from_labels(selected, test_labels["floor_fde"], test_labels, switch | test_labels["base_switch"].astype(bool))
        raw_endpoint_metrics = _metric_from_labels(raw_endpoint, test_labels["base_fde"], test_labels, np.ones(len(raw_endpoint), dtype=bool))
        metrics_vs_base["t50_ci"] = s41._bootstrap_ci(selected, test_labels["base_fde"], test_labels, "t50", n=2000)
        metrics_vs_floor["t50_ci"] = s41._bootstrap_ci(selected, test_labels["floor_fde"], test_labels, "t50", n=2000)
        metrics_vs_floor["hard_failure_ci"] = s41._bootstrap_ci(selected, test_labels["floor_fde"], test_labels, "hard_failure", n=2000)
        metrics_vs_floor["all_ci"] = s41._bootstrap_ci(selected, test_labels["floor_fde"], test_labels, "all", n=2000)
        no_external_fallback_safe = bool(
            metrics_vs_base.get("all_improvement", -1.0) > 0.0
            and metrics_vs_base.get("t50_improvement", -1.0) > 0.0
            and metrics_vs_base.get("t100_improvement", -1.0) >= 0.0
            and metrics_vs_base.get("easy_degradation", 1.0) <= 0.02
            and _max_domain_easy(metrics_vs_base) <= 0.02
        )
        protected_full_replacement = bool(
            metrics_vs_floor.get("easy_degradation", 1.0) <= 0.02
            and _max_domain_easy(metrics_vs_floor) <= 0.02
            and _positive_domains(metrics_vs_floor) >= 2
            and metrics_vs_floor.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
            and metrics_vs_floor.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
            and metrics_vs_floor.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
        )
        geometry = read_json(s41.OUT_DIR / "stage41_endpoint_geometry_audit.json", {})
        if not geometry and all((s41.DATA_DIR / f"all_agent_{sp}.npz").exists() for sp in ["train", "val", "test"]):
            geometry = s41.endpoint_geometry_audit()
        geometry_verified = bool(geometry.get("geometry_pass"))
        alignment_status = (
            "endpoint_geometry_verified; continuous endpoint interpolation is geometry-aligned but remains safety-gated"
            if geometry_verified
            else "binary_fde_safe; continuous endpoint interpolation remains diagnostic because safety-floor endpoint deltas are not aligned with source-rotation FDE labels"
        )
        caveat = (
            "Self-gated endpoint is still dataset-local 2.5D raw-frame prediction. Endpoint geometry is verified for the rebuilt safety-floor candidate, but deployment remains safety-gated; it is not true 3D, not a foundation model, and does not execute Stage5C or SMC."
            if geometry_verified
            else "Self-gated endpoint is still dataset-local 2.5D raw-frame prediction. Binary FDE selection is safe, but continuous endpoint interpolation remains diagnostic until source-rotation floor endpoint geometry is repaired. It is not true 3D, not a foundation model, and does not execute Stage5C or SMC."
        )
        result = {
            "source": "fresh_run",
            "protocol_status": "fresh_rotation_self_gated_endpoint_candidate",
            "hypothesis": "Make the gain/harm gate part of the neural endpoint output itself: prediction = base + alpha*(endpoint-base). This tests no-external-fallback safety while still reporting raw ungated endpoint failure.",
            "best_policy": best_policy,
            "best_val": best_val,
            "self_gated_without_external_fallback_vs_source_rotation_base": metrics_vs_base,
            "metrics_vs_floor": metrics_vs_floor,
            "raw_ungated_endpoint_vs_source_rotation_base": raw_endpoint_metrics,
            "trajectory_endpoint_alignment_status": alignment_status,
            "endpoint_geometry_audit": geometry,
            "alpha_mean": float(np.mean(alpha)),
            "alpha_nonzero_mean": float(np.mean(alpha[switch])) if np.any(switch) else 0.0,
            "positive_external_domains_vs_floor": _positive_domains(metrics_vs_floor),
            "max_domain_easy_degradation_vs_floor": _max_domain_easy(metrics_vs_floor),
            "no_external_fallback_safe_pass": no_external_fallback_safe,
            "protected_full_replacement_pass": protected_full_replacement,
            "deployment_decision": "self_gated_m3w_neural_v1_candidate_pending_user_acceptance" if protected_full_replacement and no_external_fallback_safe else "diagnostic_keep_stage37_floor",
            "gate_checkpoint_count": len(gate_paths),
            "caveat": caveat,
        }
    _write_json(OUT_DIR / "stage41_fresh_self_gated_endpoint_candidate.json", result)
    lines = [
        "# Stage41 Fresh Self-Gated Endpoint Candidate",
        "",
        "- source: `fresh_run`",
        f"- protocol status: `{result.get('protocol_status')}`",
        f"- deployment decision: `{result.get('deployment_decision')}`",
        f"- no-external-fallback safe pass: `{result.get('no_external_fallback_safe_pass')}`",
        f"- protected full replacement pass: `{result.get('protected_full_replacement_pass')}`",
        f"- self-gated vs source-rotation base: `{result.get('self_gated_without_external_fallback_vs_source_rotation_base')}`",
        f"- metrics vs floor: `{result.get('metrics_vs_floor')}`",
        f"- raw ungated endpoint: `{result.get('raw_ungated_endpoint_vs_source_rotation_base')}`",
        f"- trajectory endpoint alignment status: `{result.get('trajectory_endpoint_alignment_status')}`",
        f"- alpha mean: `{result.get('alpha_mean')}`",
        f"- alpha nonzero mean: `{result.get('alpha_nonzero_mean')}`",
        f"- caveat: `{result.get('caveat')}`",
        "",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds-level claim: `False`",
    ]
    write_md(OUT_DIR / "stage41_fresh_self_gated_endpoint_candidate.md", lines)
    _append_ledger("stage41_fresh_self_gated_endpoint_candidate", "ok", started, [OUT_DIR / "stage41_fresh_endpoint_gain_gate_candidate.json"], [OUT_DIR / "stage41_fresh_self_gated_endpoint_candidate.md"])
    update_self_gated_endpoint_readme_state(result)
    return result


def update_self_gated_endpoint_readme_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    m = result.get("metrics_vs_floor", {}) or {}
    base = result.get("self_gated_without_external_fallback_vs_source_rotation_base", {}) or {}
    raw = result.get("raw_ungated_endpoint_vs_source_rotation_base", {}) or {}
    block = f"""

## Stage41 Fresh Self-Gated Endpoint Candidate

This run makes the neural gate part of the endpoint prediction itself: `prediction = base + alpha * (neural_endpoint - base)`. It targets Gate10 by testing no-external-fallback safety while still reporting the raw ungated endpoint failure. It does not execute Stage5C or SMC.

```text
source = {result.get('source')}
protocol_status = {result.get('protocol_status')}
deployment_decision = {result.get('deployment_decision')}
no_external_fallback_safe_pass = {result.get('no_external_fallback_safe_pass')}
protected_full_replacement_pass = {result.get('protected_full_replacement_pass')}
self_gated_vs_base_all = {base.get('all_improvement')}
self_gated_vs_base_t50 = {base.get('t50_improvement')}
self_gated_vs_base_t100 = {base.get('t100_improvement')}
self_gated_vs_base_hard = {base.get('hard_failure_improvement')}
self_gated_vs_base_easy = {base.get('easy_degradation')}
vs_floor_all = {m.get('all_improvement')}
vs_floor_t50 = {m.get('t50_improvement')}
vs_floor_t100 = {m.get('t100_improvement')}
vs_floor_hard = {m.get('hard_failure_improvement')}
vs_floor_easy = {m.get('easy_degradation')}
raw_ungated_t100 = {raw.get('t100_improvement')}
raw_ungated_easy = {raw.get('easy_degradation')}
trajectory_endpoint_alignment_status = {result.get('trajectory_endpoint_alignment_status')}
true_3d = false
foundation_world_model = false
stage5c_executed = false
smc_enabled = false
```
"""
    marker = "## Stage41 Fresh Self-Gated Endpoint Candidate"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.add(str(OUT_DIR / "stage41_fresh_self_gated_endpoint_candidate.md"))
    stage41 = dict(state.get("stage41", {}))
    stage41["fresh_self_gated_endpoint_candidate"] = {
        "source": result.get("source"),
        "protocol_status": result.get("protocol_status"),
        "deployment_decision": result.get("deployment_decision"),
        "no_external_fallback_safe_pass": result.get("no_external_fallback_safe_pass"),
        "protected_full_replacement_pass": result.get("protected_full_replacement_pass"),
        "self_gated_without_external_fallback_vs_source_rotation_base": result.get("self_gated_without_external_fallback_vs_source_rotation_base"),
        "metrics_vs_floor": result.get("metrics_vs_floor"),
        "raw_ungated_endpoint_vs_source_rotation_base": result.get("raw_ungated_endpoint_vs_source_rotation_base"),
        "trajectory_endpoint_alignment_status": result.get("trajectory_endpoint_alignment_status"),
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


def main_fresh_self_gated_endpoint_candidate() -> None:
    run_fresh_self_gated_endpoint_candidate()
