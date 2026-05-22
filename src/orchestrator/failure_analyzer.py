from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from .research_state import read_json, read_text


def _contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def analyze_failures(report_text: str, gate_text: str) -> Dict[str, List[str]]:
    text = report_text + "\n" + gate_text
    failures: List[str] = []
    fixes: List[str] = []
    blockers: List[str] = []

    if _contains(text, "deterministic learned model 是否超过 strongest causal baseline：否") or _contains(text, "deterministic_5pct_gate = false"):
        failures.append("Deterministic residual still does not beat strongest causal baseline by the required margin.")
        fixes.append("Repair deterministic residual around baseline-failure cases before latent/stochastic work.")
    if _contains(text, "Stage 5C latent generative：否") or _contains(text, "latent_stage5c_ready = false"):
        failures.append("Latent generative readiness is false; deterministic gates remain the blocker.")
        fixes.append("Keep latent generative disabled and generate a plan only after deterministic gates pass.")
    if _contains(text, "SMC：否") or _contains(text, "smc_ready = false"):
        failures.append("SMC readiness is false; no strong stochastic proposal exists yet.")
        fixes.append("Keep SMC disabled until stochastic coverage improves.")
    if _contains(text, "AerialMPT remains pixel-space"):
        failures.append("AerialMPT remains pixel-space without homography or meter scale.")
        blockers.append("Provide AerialMPT homography/control points or keep AerialMPT qualitative/pixel-space only.")
    if _contains(text, "only ETH/UCY EWAP currently provides"):
        failures.append("Verified pedestrian/drone long-horizon coverage is narrow: currently ETH/UCY EWAP only.")
        blockers.append("Provide Stanford Drone Dataset/OpenTraj local paths if available.")
    if _contains(text, "silver_rule_confirmed"):
        failures.append("Scene labels still rely heavily on rule-confirmed silver annotations, not human gold.")
        fixes.append("Upgrade high-value scenes from silver_rule_confirmed to silver_human_confirmed/gold_human.")

    if not failures:
        failures.append("No explicit failure pattern was parsed; inspect latest gate report manually.")
    return {
        "top_failures": failures[:6],
        "recommended_fixes": fixes[:6],
        "user_blockers": blockers[:6],
    }


def strongest_baseline_summary() -> Dict[str, Dict]:
    summary_path = Path("outputs/reports/stage12_rebenchmark/report_stage12_deterministic_rebenchmark.json")
    payload = read_json(summary_path, default={})
    best = payload.get("best_by_dataset", {}) if isinstance(payload, dict) else {}
    result = {}
    for dataset, row in best.items():
        result[dataset] = {
            "target_horizon": row.get("horizon"),
            "strongest_baseline_FDE": row.get("baseline_FDE"),
            "best_learned_variant": row.get("variant"),
            "best_learned_FDE": row.get("FDE"),
            "improvement": row.get("improvement"),
        }
    return result


def datasets_from_stage12() -> Dict[str, List[str]]:
    report = read_text("outputs/reports/report_stage12_final.md")
    loaded = []
    match = re.search(r"是否接入真实 pedestrian/drone 数据：是 \(\[([^\]]+)\]\)", report)
    if match:
        loaded = [item.strip(" '\"") for item in match.group(1).split(",")]
    verified = []
    match = re.search(r"是否补上 verified t\+50/t\+100：是 \(\[([^\]]+)\]\)", report)
    if match:
        verified = [item.strip(" '\"") for item in match.group(1).split(",")]
    return {"loaded": loaded, "verified_long_horizon": verified}

