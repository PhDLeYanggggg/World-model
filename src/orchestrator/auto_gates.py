from __future__ import annotations

from typing import Any, Dict, List

from .research_state import write_json, write_md


def evaluate_auto_gates(state: Dict[str, Any]) -> Dict[str, Any]:
    gates = [
        ("Data Gate", bool(state.get("datasets_converted")), "At least one real pedestrian/drone dataset loaded and converted."),
        ("Long-Horizon Gate", bool(state.get("pedestrian_long_horizon_ready")), "At least one pedestrian/drone source supports verified t+50/t+100."),
        ("Annotation Gate", bool(state.get("scene_annotation_ready")), "At least 3 human-confirmed or high-quality silver scenes exist."),
        ("Scene/Goal Gate", bool(state.get("goalbench_beats_majority")), "GoalBench beats majority or improves calibrated goal metrics."),
        ("Multi-Agent Gate", bool(state.get("multi_agent_ready")), "At least 300 multi-agent episodes with >=2 agents."),
        ("Strong Baseline Gate", bool(state.get("strongest_causal_baselines")), "Strongest causal baseline computed."),
        ("Deterministic Improvement Gate", bool(state.get("learned_model_beats_strongest_baseline") == "是"), "Learned deterministic model beats strongest causal baseline."),
        ("Easy Preservation Gate", False, "Easy subset degradation <=2% is not proven by the latest report."),
        ("Scene/Goal Ablation Gate", False, "Scene/goal ablation is not proven to improve trajectory metrics."),
        ("Interaction Gate", False, "Interaction module does not yet improve hard/failure trajectory metrics."),
        ("Physical Validity Gate", True, "No major physical-validity degradation reported in Stage 12 summary."),
    ]
    rows: List[Dict[str, Any]] = []
    for name, passed, evidence in gates:
        rows.append({"gate": name, "pass": bool(passed), "evidence": evidence})
    latent_ready = all(row["pass"] for row in rows if row["gate"] in {
        "Scene/Goal Gate",
        "Deterministic Improvement Gate",
        "Easy Preservation Gate",
        "Scene/Goal Ablation Gate",
        "Interaction Gate",
        "Physical Validity Gate",
    })
    rows.append({"gate": "Latent Generative Readiness Gate", "pass": latent_ready, "evidence": "Must pass deterministic scene/goal/interaction gates first."})
    rows.append({"gate": "SMC Readiness Gate", "pass": False, "evidence": "No stochastic proposal with coverage lift exists."})
    return {
        "passed": [row["gate"] for row in rows if row["pass"]],
        "failed": [row["gate"] for row in rows if not row["pass"]],
        "rows": rows,
        "latent_generative_ready": latent_ready,
        "smc_ready": False,
    }


def write_auto_gate_report(gates: Dict[str, Any]) -> None:
    write_json("outputs/reports/auto_gate_report.json", gates)
    lines = [
        "# Auto Gate Report",
        "",
        "| gate | pass | evidence |",
        "| --- | --- | --- |",
    ]
    for row in gates["rows"]:
        lines.append(f"| {row['gate']} | {row['pass']} | {row['evidence']} |")
    lines += [
        "",
        f"latent_generative_ready: `{gates['latent_generative_ready']}`",
        f"smc_ready: `{gates['smc_ready']}`",
    ]
    write_md("outputs/reports/auto_gate_report.md", lines)

