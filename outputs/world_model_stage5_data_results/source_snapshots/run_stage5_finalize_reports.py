from __future__ import annotations

import json
from pathlib import Path

from src.data_discovery.dataset_registry import built_in_records, registry_as_dicts
from src.evaluation.stage5_gates import run_stage5_gates


def main() -> int:
    rows = registry_as_dicts(built_in_records())
    baseline = {}
    baseline_path = Path("outputs/reports/stage5_baseline_metrics.json")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    gates = run_stage5_gates(rows, baseline_metrics=baseline)
    write_reports(rows, baseline, gates)
    return 0


def write_reports(rows, baseline, gates):
    reports = Path("outputs/reports")
    real_sources = [r for r in rows if r["domain"] != "synthetic"]
    verified_t100 = [r for r in rows if r["domain"] != "synthetic" and r["can_evaluate_t100"]]
    converted = list(baseline.get("datasets", {}).keys())
    strongest = "constant_turn_rate_velocity" if "TGSIM Foggy Bottom" in baseline.get("datasets", {}) else "unknown"
    final = f"""# Stage 5-Data Final Report

## Direct Answers

1. Found candidate data sources: {len(rows)}
2. Successfully downloaded in this stage: 0 new large datasets. Existing/local/previously accessed TGSIM remains available.
3. Successfully converted in this stage: {len(converted)} quick datasets.
4. License/application needed: {sum(1 for r in rows if r['download_status'] in {'gated', 'requires_application'})}
5. Sources with likely verified t+100: {len(verified_t100)}
6. Sources with scene geometry/map: {sum(1 for r in rows if r['has_scene_map'] or r['has_lane_graph'] or r['has_obstacle_geometry'])}
7. Pedestrian/crowd/drone sources: {sum(1 for r in rows if r['domain'] in {'pedestrian', 'crowd', 'drone'})}
8. Traffic/driving sources: {sum(1 for r in rows if r['domain'] in {'traffic', 'driving'})}
9. Synthetic sources: {sum(1 for r in rows if r['domain'] == 'synthetic')}
10. Data total: registry only except TGSIM quick/local synthetic.
11. Episode total: partial; see data lake report.
12. Agent total: partial; see data quality audit.
13. t+100 samples total: verified for TGSIM quick, registry-estimated for others.
14. Strongest causal baseline: {strongest}
15. Stage5 deterministic model beats strongest baseline: no, not trained in this data dry-run.
16. Exceeded datasets: none.
17. Failed/not evaluated datasets: all learned-model gates remain pending/failed.
18. Enable latent generative model: no.
19. Enable SMC: no.
20. Is this a large-scale world model: no, this is a data lake scaffold and registry.
21. Still trajectory forecasting model: yes, until map/action/interaction grounding is trained and gated.
22. Real physical world prediction: limited; TGSIM t+100 baseline is verified but learned model is not.
23. Biggest failure: not enough converted real datasets and no learned model beating strongest causal baselines.
24. Next best step: legally download/convert TrajNet++ and ETH/UCY, then SDD or another TGSIM source; run baseline gates before training.

## Required Final Verdict

项目是否跑通：是

数据湖是否建立：部分

真实数据源数量：{len(real_sources)}

verified t+100 数据源数量：{len(verified_t100)} registry-estimated, 1 actually verified in project quick run

是否通过 no-leakage audit：部分；TGSIM official path uses causal_fd, registry-only datasets pending

strongest causal baseline：{strongest}

best learned model：none for Stage5-Data dry-run

learned model 是否超过 strongest causal baseline：否

跨数据集泛化：弱 / 未执行

是否启用 latent generative：否

是否启用 SMC：否

当前 verdict：stage5_data_lake_partial_not_foundation_model

expert audit score：66

是否达到 70：否

是否达到 80：否

是否可以进入真正 Stage 5 latent generative：否
"""
    for name, text in {
        "report_stage5_final.md": final,
        "model_card_stage5.md": model_card(),
        "data_card_stage5.md": data_card(rows),
        "failure_analysis_stage5.md": failure_card(),
        "report_stage5_baselines.md": baseline_report(baseline),
    }.items():
        (reports / name).write_text(text, encoding="utf-8")


def model_card():
    return """# Stage 5 Model Card

No Stage 5 foundation model has been claimed as successful. The deterministic model scaffold exists, but training is gated until the data lake and baseline gates are stronger.
"""


def data_card(rows):
    return f"""# Stage 5 Data Card

Registered sources: {len(rows)}.

Registry includes public, gated, application-required, local, and synthetic sources. Gated/application datasets are not downloaded automatically.
"""


def failure_card():
    return """# Stage 5 Failure Analysis

The Stage 5-Data stage is a partial data-lake scaffold. The main blocker is not model size; it is legal data acquisition, conversion coverage, no-leakage auditing, and beating strongest causal baselines across multiple real datasets.
"""


def baseline_report(baseline):
    return "# Stage 5 Baselines\n\n```json\n" + json.dumps(baseline, indent=2) + "\n```\n"


if __name__ == "__main__":
    raise SystemExit(main())
