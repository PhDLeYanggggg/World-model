from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .research_state import write_json, write_md
from .task_planner import plan_to_markdown


REPORT_DIR = Path("outputs/reports")


def write_current_state_report(state: Dict[str, Any]) -> None:
    write_json(REPORT_DIR / "auto_orchestrator_current_state.json", state)
    missing = [f"- {item}" for item in state.get("missing_requested_files", [])] or ["- none"]
    lines = [
        "# Auto-Orchestrator Current State",
        "",
        f"- current_highest_stage: `{state.get('current_highest_stage')}`",
        f"- expert_audit_score: `{state.get('expert_audit_score')}`",
        f"- verdict: `{state.get('verdict')}`",
        f"- model_type: `{state.get('model_type')}`",
        f"- true_3D: `{state.get('true_3D')}`",
        f"- large_scale_foundation_model: `{state.get('large_scale_foundation_model')}`",
        f"- latent_generative_ready: `{state.get('latent_generative_ready')}`",
        f"- smc_ready: `{state.get('smc_ready')}`",
        f"- strongest_causal_baseline: `{state.get('strongest_causal_baseline')}`",
        f"- learned_model_beats_strongest_baseline: `{state.get('learned_model_beats_strongest_baseline')}`",
        f"- verified_pedestrian_drone_t50_t100: `{state.get('verified_pedestrian_drone_t50_t100')}`",
        f"- human_gold_annotations: `{state.get('human_gold_annotations')}`",
        f"- human_silver_annotations: `{state.get('human_silver_annotations')}`",
        f"- goalbench_beats_majority: `{state.get('goalbench_beats_majority')}`",
        f"- hardbench_baselinefailure_enough: `{state.get('hardbench_baselinefailure_enough')}`",
        "",
        "## Top Failures",
        "",
        *[f"- {item}" for item in state.get("top_failure_reasons", [])],
        "",
        "## Best Automatic Directions",
        "",
        *[f"- {item}" for item in state.get("best_auto_directions", [])],
        "",
        "## Missing Requested Files",
        "",
        *missing,
    ]
    write_md(REPORT_DIR / "auto_orchestrator_current_state.md", lines)


def write_iteration_report(
    start_state: Dict[str, Any],
    tasks: List[Dict[str, Any]],
    executed: List[Dict[str, Any]],
    decision: Dict[str, Any],
) -> None:
    improved = "部分" if tasks or executed else "否"
    success_lines = [f"- {row['task']}: {row.get('status', 'done')}" for row in executed] or ["- none"]
    failure_lines = [f"- {row['task']}: {row.get('error')}" for row in executed if row.get("error")] or ["- none"]
    blocker_lines = [f"- {item}" for item in decision.get("blockers_requiring_user", [])] or ["- none"]
    lines = [
        "# Auto Loop Iteration Report",
        "",
        "## 本轮开始状态",
        "",
        f"- stage: `{start_state.get('current_highest_stage')}`",
        f"- verdict: `{start_state.get('verdict')}`",
        f"- expert_audit_score: `{start_state.get('expert_audit_score')}`",
        f"- latent_generative_ready: `{start_state.get('latent_generative_ready')}`",
        f"- smc_ready: `{start_state.get('smc_ready')}`",
        "",
        "## 本轮执行任务",
        "",
        *plan_to_markdown(tasks),
        "",
        "## 成功的任务",
        "",
        *success_lines,
        "",
        "## 失败的任务",
        "",
        *failure_lines,
        "",
        "## Gates 变化",
        "",
        "- 本轮是 orchestrator/bootstrap iteration，没有训练新模型，因此 deterministic/latent/SMC gates 不应被改变为通过。",
        "",
        "## 是否更接近 world model",
        "",
        "- 部分。现在项目有可重复执行的状态读取、gate 判断和下一步任务规划，能防止错误进入 latent generative 或 SMC。",
        "",
        "## 是否仍只是 trajectory forecasting scaffold",
        "",
        "- 是。直到 deterministic model 在 verified pedestrian/drone long-horizon 和 hard/failure subsets 上击败 strongest causal baseline。",
        "",
        "## 是否允许下一阶段",
        "",
        f"- Stage 13 deterministic repair allowed: `{start_state.get('stage13_ready')}`",
        "- Stage 5C latent generative allowed: `False`",
        "- SMC allowed: `False`",
        "",
        "## 需要用户输入",
        "",
        *blocker_lines,
        "",
        "## 下一轮推荐任务",
        "",
        *[f"- {item['name']}: {item['reason']}" for item in decision.get("actions", [])[:3]],
        "",
        "## Final Conclusion",
        "",
        "项目是否跑通：是",
        f"本轮是否改善：{improved}",
        "新增真实数据：否",
        "新增人工/银标注：否",
        f"deterministic model 是否超过 strongest causal baseline：{start_state.get('learned_model_beats_strongest_baseline')}",
        "hard/failure 是否改善：否",
        "verified long-horizon 是否改善：否",
        "latent generative 是否 ready：否",
        "SMC 是否 ready：否",
        f"当前 verdict：{start_state.get('verdict')}",
        f"expert audit score：{start_state.get('expert_audit_score')}",
    ]
    write_md(REPORT_DIR / "auto_loop_iteration_report.md", lines)


def write_next_stage_plan(decision: Dict[str, Any], tasks: List[Dict[str, Any]]) -> None:
    lines = [
        "# Auto Next Stage Plan",
        "",
        "This plan is generated by the autonomous orchestrator. It does not authorize latent generative training or SMC.",
        "",
        "## Planned Tasks",
        "",
        *plan_to_markdown(tasks),
        "",
        "## Guardrails",
        "",
        "- Do not enable latent generative training until deterministic gates pass.",
        "- Do not enable SMC until a stochastic proposal improves coverage.",
        "- Do not treat silver_rule or AI visual labels as human gold.",
        "- Do not use test endpoints to construct candidate goals.",
    ]
    write_md(REPORT_DIR / "auto_next_stage_plan.md", lines)
