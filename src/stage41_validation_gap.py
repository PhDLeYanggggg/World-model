from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41


OUT_DIR = Path("outputs/stage41_external_split")
DATA_DIR = s41.DATA_DIR
LEDGER_JSONL = s41.OUT_DIR / "run_ledger.jsonl"
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


def _append_ledger(step: str, status: str, started: float, inputs: list[str], outputs: list[str]) -> None:
    ensure_dir(s41.OUT_DIR)
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


def _load_split_rows() -> Dict[str, np.ndarray]:
    if not (DATA_DIR / "all_agent_train.npz").exists():
        s41.build_seq2seq_dataset()
    arrays: Dict[str, list[np.ndarray]] = defaultdict(list)
    for split in ["train", "val", "test"]:
        ds = dict(np.load(DATA_DIR / f"all_agent_{split}.npz", allow_pickle=True))
        n = len(ds["horizon"])
        arrays["split"].append(np.full(n, split, dtype="U8"))
        for key in ["ids", "domain", "scene_id", "source_file", "horizon", "hard", "easy", "failure", "candidate_fde", "floor_fde"]:
            arrays[key].append(ds[key])
    return {k: np.concatenate(v, axis=0) for k, v in arrays.items()}


def _stats(rows: Mapping[str, np.ndarray], mask: np.ndarray) -> Dict[str, Any]:
    if not np.any(mask):
        return {
            "rows": 0,
            "t50_rows": 0,
            "t100_rows": 0,
            "hard": 0,
            "easy": 0,
            "failure": 0,
            "oracle_headroom_all": 0.0,
            "oracle_headroom_t50": 0.0,
            "candidate6_mean_improvement_t50": 0.0,
            "best_candidate_distribution_t50": {},
        }
    candidate = rows["candidate_fde"][mask].astype(np.float64)
    floor = rows["floor_fde"][mask].astype(np.float64)
    horizon = rows["horizon"][mask].astype(int)
    t50 = horizon == 50
    best = np.argmin(candidate, axis=1)
    oracle_all = 1.0 - float(candidate.min(axis=1).mean()) / max(float(floor.mean()), EPS)
    oracle_t50 = 0.0
    cand6_t50 = 0.0
    best_dist: Dict[str, int] = {}
    if np.any(t50):
        oracle_t50 = 1.0 - float(candidate[t50].min(axis=1).mean()) / max(float(floor[t50].mean()), EPS)
        if candidate.shape[1] > 6:
            cand6_t50 = 1.0 - float(candidate[t50, 6].mean()) / max(float(floor[t50].mean()), EPS)
        keys, vals = np.unique(best[t50], return_counts=True)
        best_dist = {str(int(k)): int(v) for k, v in zip(keys, vals)}
    return {
        "rows": int(mask.sum()),
        "source_files": int(len(set(rows["source_file"][mask].astype(str).tolist()))),
        "scenes": int(len(set(rows["scene_id"][mask].astype(str).tolist()))),
        "t50_rows": int(np.sum(horizon == 50)),
        "t100_rows": int(np.sum(horizon == 100)),
        "hard": int(np.sum(rows["hard"][mask].astype(bool))),
        "easy": int(np.sum(rows["easy"][mask].astype(bool))),
        "failure": int(np.sum(rows["failure"][mask].astype(bool))),
        "oracle_headroom_all": float(oracle_all),
        "oracle_headroom_t50": float(oracle_t50),
        "candidate6_mean_improvement_t50": float(cand6_t50),
        "best_candidate_distribution_t50": best_dist,
    }


def audit_validation_gap() -> Dict[str, Any]:
    started = time.perf_counter()
    ensure_dir(OUT_DIR)
    rows = _load_split_rows()
    split = rows["split"].astype(str)
    domain = rows["domain"].astype(str)
    by_domain: Dict[str, Dict[str, Any]] = {}
    blockers: list[str] = []
    for d in sorted(set(domain.tolist())):
        by_domain[d] = {sp: _stats(rows, (domain == d) & (split == sp)) for sp in ["train", "val", "test"]}
        val = by_domain[d]["val"]
        test = by_domain[d]["test"]
        gap = test["oracle_headroom_t50"] - val["oracle_headroom_t50"]
        ratio = test["oracle_headroom_t50"] / max(val["oracle_headroom_t50"], EPS)
        by_domain[d]["validation_test_gap"] = {"t50_oracle_gap": float(gap), "t50_oracle_ratio": float(ratio)}
        if test["t50_rows"] >= 500 and gap > 0.10 and ratio > 3.0:
            blockers.append(f"{d} t50 validation headroom is not representative: val={val['oracle_headroom_t50']:.4f}, test={test['oracle_headroom_t50']:.4f}")
    result = {
        "source": "fresh_run",
        "purpose": "audit whether validation can select safe neural switching policies for the same t50 failure slices seen at test",
        "by_domain": by_domain,
        "blockers": blockers,
        "largest_blocker": blockers[0] if blockers else "",
        "no_leakage": {
            "future_endpoint_input": False,
            "test_endpoint_goals": False,
            "central_velocity": False,
            "note": "future labels are used only for split diagnostics/oracle headroom audit, not as model features or goal construction",
        },
    }
    _write_json(OUT_DIR / "stage41_validation_gap_audit.json", result)
    lines = [
        "# Stage41 Validation Gap Audit",
        "",
        "- source: `fresh_run`",
        "- purpose: check whether validation represents held-out t+50 switchability.",
        "- no future endpoint is used as an inference feature; oracle headroom here is a diagnostic label/eval statistic.",
        "",
        "| domain | split | rows | t50 rows | t100 rows | oracle all | oracle t50 | candidate6 t50 | hard | easy | failure |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for d, splits in by_domain.items():
        for sp in ["train", "val", "test"]:
            row = splits[sp]
            lines.append(
                f"| {d} | {sp} | {row['rows']} | {row['t50_rows']} | {row['t100_rows']} | "
                f"{row['oracle_headroom_all']:.4f} | {row['oracle_headroom_t50']:.4f} | {row['candidate6_mean_improvement_t50']:.4f} | "
                f"{row['hard']} | {row['easy']} | {row['failure']} |"
            )
        lines.append(f"| {d} | val-test gap |  |  |  |  | {splits['validation_test_gap']['t50_oracle_gap']:.4f} | ratio {splits['validation_test_gap']['t50_oracle_ratio']:.2f} |  |  |  |")
    lines.extend(["", "## Blockers", "", *[f"- {b}" for b in blockers or ["No severe validation/test t50 headroom gap detected."]]])
    write_md(OUT_DIR / "stage41_validation_gap_audit.md", lines)
    _append_ledger("stage41_validation_gap_audit", "ok", started, [str(DATA_DIR / "all_agent_train.npz"), str(DATA_DIR / "all_agent_val.npz"), str(DATA_DIR / "all_agent_test.npz")], [str(OUT_DIR / "stage41_validation_gap_audit.md")])
    return result


def build_stratified_split_candidate() -> Dict[str, Any]:
    started = time.perf_counter()
    ensure_dir(OUT_DIR)
    rows = _load_split_rows()
    # Candidate only: do not overwrite the locked Stage41 split. It exists so
    # the next retraining loop can rebuild datasets with a validation slice
    # that contains t50 switchability instead of silently tuning on test.
    ids = rows["ids"].astype(np.int64)
    domain = rows["domain"].astype(str)
    group = np.asarray([f"{d}::{s}" for d, s in zip(domain, rows["source_file"].astype(str))], dtype="U512")
    split = np.full(len(ids), "train", dtype="U8")
    assignment: Dict[str, str] = {}
    group_stats: Dict[str, Any] = {}
    for d in sorted(set(domain.tolist())):
        groups = sorted(set(group[domain == d].tolist()))
        scored = []
        for g in groups:
            mask = group == g
            st = _stats(rows, mask)
            score = st["t50_rows"] * max(st["oracle_headroom_t50"], 0.0) + 0.1 * st["hard"]
            scored.append((score, g, st))
            group_stats[g] = st
        scored.sort(reverse=True)
        if len(scored) < 3:
            for i, (_score, g, _st) in enumerate(scored):
                assignment[g] = "test" if i == 0 else "train"
            continue
        loads = {"train": 0.0, "val": 0.0, "test": 0.0}
        counts = {"train": 0, "val": 0, "test": 0}
        # Seed val/test with the largest two t50-headroom groups, then greedily
        # fill train/val/test while preserving held-out source-file isolation.
        for target, item in zip(["test", "val"], scored[:2]):
            score, g, _st = item
            assignment[g] = target
            loads[target] += score
            counts[target] += 1
        for score, g, _st in scored[2:]:
            target = min(["train", "val", "test"], key=lambda sp: (loads[sp] / max(counts[sp], 1), counts[sp] if sp != "train" else counts[sp] - 1))
            assignment[g] = target
            loads[target] += score
            counts[target] += 1
    for g, sp in assignment.items():
        split[group == g] = sp
    np.savez_compressed(DATA_DIR / "stage41_stratified_split_candidate.npz", row_id=ids, split=split, group=group, domain=domain, source_file=rows["source_file"].astype(str), scene_id=rows["scene_id"].astype(str))
    by_domain = {
        d: {sp: _stats(rows, (domain == d) & (split == sp)) for sp in ["train", "val", "test"]}
        for d in sorted(set(domain.tolist()))
    }
    result = {
        "source": "fresh_run",
        "status": "candidate_protocol_not_used_for_stage41_claims",
        "split_strategy": "domain-wise source-file held-out split with t50 oracle-headroom stratification for validation/test balance",
        "assignment": assignment,
        "group_stats": group_stats,
        "by_domain": by_domain,
        "no_leakage": {
            "overwrites_current_split": False,
            "future_endpoint_input": False,
            "test_endpoint_goals": False,
            "central_velocity": False,
            "diagnostic_label_use": "oracle headroom is used only to design a candidate validation protocol; models must still train on train, select on val, test once",
        },
    }
    _write_json(OUT_DIR / "stage41_stratified_split_candidate.json", result)
    lines = [
        "# Stage41 Stratified Split Candidate",
        "",
        "- source: `fresh_run`",
        "- status: `candidate_protocol_not_used_for_stage41_claims`",
        "- this does not overwrite `stage41_split_index.npz`; it is an input for the next retraining loop.",
        "",
        "| domain | split | rows | files | t50 rows | oracle t50 | candidate6 t50 | hard | easy | failure |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for d, splits in by_domain.items():
        for sp in ["train", "val", "test"]:
            row = splits[sp]
            lines.append(
                f"| {d} | {sp} | {row['rows']} | {row['source_files']} | {row['t50_rows']} | "
                f"{row['oracle_headroom_t50']:.4f} | {row['candidate6_mean_improvement_t50']:.4f} | {row['hard']} | {row['easy']} | {row['failure']} |"
            )
    write_md(OUT_DIR / "stage41_stratified_split_candidate.md", lines)
    _append_ledger("stage41_stratified_split_candidate", "ok", started, [str(DATA_DIR / "all_agent_train.npz"), str(DATA_DIR / "all_agent_val.npz"), str(DATA_DIR / "all_agent_test.npz")], [str(OUT_DIR / "stage41_stratified_split_candidate.md")])
    return result


def run_validation_gap_and_split_candidate() -> Dict[str, Any]:
    audit = audit_validation_gap()
    split = build_stratified_split_candidate()
    return {"source": "fresh_run", "audit": audit, "stratified_split_candidate": split}


def main_validation_gap() -> None:
    run_validation_gap_and_split_candidate()
