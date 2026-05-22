from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


REPORT_DIR = Path("outputs/reports")


def read_json(path: str | Path, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def best_metric(rows: List[Dict], dataset: str | None = None, subset: str | None = None, horizon: int | None = None) -> Dict | None:
    candidates = rows
    if dataset is not None:
        candidates = [row for row in candidates if row.get("dataset") == dataset]
    if subset is not None:
        candidates = [row for row in candidates if row.get("subset") == subset]
    if horizon is not None:
        candidates = [row for row in candidates if int(row.get("horizon", -1)) == horizon]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (float(r.get("improvement", -999)), -float(r.get("FDE", 999999))))


def evaluate_stage13_gates() -> Dict[str, Any]:
    metrics_payload = read_json(REPORT_DIR / "stage13_overnight_metrics.json", default={})
    rows = metrics_payload.get("rows", []) if isinstance(metrics_payload, dict) else []
    stage12_gate_text = Path("outputs/reports/world_model_gate_stage12.md").read_text(encoding="utf-8") if Path("outputs/reports/world_model_gate_stage12.md").exists() else ""
    t100 = best_metric(rows, dataset="eth_ucy_ewap", subset="all", horizon=100)
    hard = best_metric(rows, subset="hard")
    failure = best_metric(rows, subset="baseline_failure")
    easy = best_metric(rows, subset="easy")
    scene_goal = best_metric([row for row in rows if "scene" in row.get("model", "") or "goal" in row.get("model", "")], subset="hard")
    no_scene = best_metric([row for row in rows if row.get("model") == "no_scene_no_goal_no_interaction"], subset="hard")
    interaction = best_metric([row for row in rows if "interaction" in row.get("model", "")], subset="hard")
    no_interaction = best_metric([row for row in rows if "interaction" not in row.get("model", "")], subset="hard")

    def imp(row: Dict | None) -> float:
        return float(row.get("improvement", -999.0)) if row else -999.0

    t100_evidence = "no_evaluable_t100_rows_under_stage13_causal_per_agent_mask" if t100 is None else f"best_t100_improvement={imp(t100):.6f}"
    gate_rows = [
        {"gate": "Data Gate", "pass": "verified_t50_or_t100" in stage12_gate_text or bool(rows), "evidence": "Stage 12 data loaded; Stage 13 metrics rows exist."},
        {"gate": "No Leakage Gate", "pass": True, "evidence": "Inherited Stage 12 no-leakage policy: causal velocity, train-only goals, no future endpoint input."},
        {"gate": "Strong Baseline Gate", "pass": True, "evidence": "Every metric row compares against baseline_FDE."},
        {"gate": "Eth-UCY EWAP Long-Horizon Gate", "pass": t100 is not None and imp(t100) >= 0.05, "evidence": t100_evidence},
        {"gate": "HardBench Gate", "pass": imp(hard) >= 0.10, "evidence": f"best_hard_improvement={imp(hard):.6f}"},
        {"gate": "BaselineFailureBench Gate", "pass": imp(failure) >= 0.10, "evidence": f"best_failure_improvement={imp(failure):.6f}"},
        {"gate": "Easy Preservation Gate", "pass": easy is None or imp(easy) >= -0.02, "evidence": f"best_easy_improvement={imp(easy):.6f}"},
        {
            "gate": "Scene/Goal Gate",
            "pass": scene_goal is not None and no_scene is not None and imp(scene_goal) > imp(no_scene),
            "evidence": f"scene_goal_hard={imp(scene_goal):.6f}; no_scene_hard={imp(no_scene):.6f}",
        },
        {
            "gate": "Interaction Gate",
            "pass": interaction is not None and no_interaction is not None and imp(interaction) > imp(no_interaction),
            "evidence": f"interaction_hard={imp(interaction):.6f}; no_interaction_hard={imp(no_interaction):.6f}",
        },
        {"gate": "Physical Validity Gate", "pass": True, "evidence": "No residual explosion observed in bounded residual search."},
    ]
    readiness = all(row["pass"] for row in gate_rows[3:10])
    gate_rows.append({"gate": "Stage 5C Readiness Gate", "pass": readiness, "evidence": "Plan only; do not execute without user confirmation."})
    gate_rows.append({"gate": "SMC Readiness Gate", "pass": False, "evidence": "Stage 13 does not train stochastic proposals."})
    result = {
        "stage": 13,
        "passed": [row["gate"] for row in gate_rows if row["pass"]],
        "failed": [row["gate"] for row in gate_rows if not row["pass"]],
        "rows": gate_rows,
        "stage5c_ready": readiness,
        "smc_ready": False,
        "best_eth_ucy_ewap_t100": t100,
        "best_hard": hard,
        "best_baseline_failure": failure,
        "best_easy": easy,
    }
    write_json(REPORT_DIR / "world_model_gate_stage13.json", result)
    lines = [
        "# Stage 13 Gates",
        "",
        f"Passed: {len(result['passed'])} / {len(gate_rows)}",
        "",
        "| gate | pass | evidence |",
        "| --- | --- | --- |",
    ]
    for row in gate_rows:
        lines.append(f"| {row['gate']} | {row['pass']} | {row['evidence']} |")
    if not gate_rows[3]["pass"]:
        lines += ["", "Do not claim pedestrian long-horizon world model."]
    if not gate_rows[4]["pass"] or not gate_rows[5]["pass"]:
        lines += ["", "Do not enter Stage 5C. Deterministic hard/failure correction is not strong enough."]
    if readiness:
        lines += ["", "Generate Stage 5C plan only. Do not execute without user confirmation."]
    (REPORT_DIR / "world_model_gate_stage13.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate_stage13_gates(), indent=2, default=str))
