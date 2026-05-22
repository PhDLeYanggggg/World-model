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


def mine_stage13_failures() -> Dict[str, Any]:
    payload = read_json(REPORT_DIR / "stage13_overnight_metrics.json", default={})
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    worse = [row for row in rows if float(row.get("improvement", 0.0)) < -0.10]
    easy_degraded = [row for row in rows if row.get("subset") == "easy" and float(row.get("improvement", 0.0)) < -0.02]
    alpha_high_easy = [row for row in rows if row.get("subset") == "easy" and float(row.get("alpha", 0.0)) > 0.5]
    t100_drift = [row for row in rows if row.get("dataset") == "eth_ucy_ewap" and int(row.get("horizon", 0)) == 100 and float(row.get("improvement", 0.0)) < 0.05]
    interaction_fail = [row for row in rows if "interaction" in row.get("model", "") and row.get("subset") == "hard" and float(row.get("improvement", 0.0)) <= 0]
    cases = []
    for label, items, reason in [
        ("baseline_better_than_model", worse, "Model correction made FDE worse than baseline by >10%."),
        ("easy_degraded", easy_degraded, "Easy subset degraded beyond preservation tolerance."),
        ("alpha_high_on_easy", alpha_high_easy, "Alpha intervention is too high on easy samples."),
        ("t100_drift", t100_drift, "EWAP t+100 did not improve enough over strongest baseline."),
        ("interaction_ineffective", interaction_fail, "Interaction family failed hard subset."),
    ]:
        for row in items[:20]:
            cases.append(
                {
                    "failure_type": label,
                    "dataset": row.get("dataset"),
                    "scene_id": "aggregate",
                    "episode_id": "aggregate",
                    "agent_ids": "aggregate",
                    "subset": row.get("subset"),
                    "horizon": row.get("horizon"),
                    "baseline_FDE": row.get("baseline_FDE"),
                    "model_FDE": row.get("FDE"),
                    "improvement": row.get("improvement"),
                    "alpha": row.get("alpha"),
                    "residual_norm": row.get("residual_magnitude"),
                    "goal_candidates": "not inspected in aggregate miner",
                    "chosen_goal": "not applicable",
                    "scene_annotation_quality": "mixed silver/human-silver",
                    "likely_failure_reason": reason,
                    "recommended_fix": "Increase fallback-to-baseline discipline, reduce residual clipping/alpha, or gather stronger scene/goal labels.",
                }
            )
    result = {
        "failure_case_count": len(cases),
        "worse_than_baseline_count": len(worse),
        "easy_degraded_count": len(easy_degraded),
        "alpha_high_easy_count": len(alpha_high_easy),
        "t100_drift_count": len(t100_drift),
        "interaction_failure_count": len(interaction_fail),
        "cases": cases,
    }
    (REPORT_DIR / "stage13_failure_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Stage 13 Failure Analysis",
        "",
        f"- failure_case_count: `{len(cases)}`",
        f"- worse_than_baseline_count: `{len(worse)}`",
        f"- easy_degraded_count: `{len(easy_degraded)}`",
        f"- alpha_high_easy_count: `{len(alpha_high_easy)}`",
        f"- t100_drift_count: `{len(t100_drift)}`",
        f"- interaction_failure_count: `{len(interaction_fail)}`",
        "",
        "| failure_type | dataset | subset | horizon | improvement | likely reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases[:50]:
        lines.append(
            f"| {case['failure_type']} | {case['dataset']} | {case['subset']} | {case['horizon']} | "
            f"{case['improvement']} | {case['likely_failure_reason']} |"
        )
    (REPORT_DIR / "stage13_failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(mine_stage13_failures(), indent=2))

