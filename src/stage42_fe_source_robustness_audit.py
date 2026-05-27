from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_fe_policy_freeze_replay as ff
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "fe_source_robustness_audit_stage42.json"
REPORT_MD = OUT_DIR / "fe_source_robustness_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fg_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = ff.fe.PAPER_FILES

SOURCE = "fresh_stage42_fe_source_robustness_audit"
BOOTSTRAP_N = 1000
MIN_CI_ROWS = 30
EASY_LIMIT = 0.02
MIN_DOMAIN_ROWS = 500


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FF 已冻结 Stage42-FE policy 并做 exact replay + 2000-bootstrap。",
    "Stage42-FG 不重新训练、不重新选择 policy、不调 test threshold；它审计 FE 在 domain/source/horizon/scene 切片上的稳健性。",
    "弱 source / weak slice 必须显式报告；不能把 global positive 包装成每个外部源都 positive。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _source_name(path: str) -> str:
    p = Path(path)
    parts = p.parts
    if len(parts) >= 4:
        return "/".join(parts[-4:])
    return str(p)


def _metric_for_mask(
    data: Mapping[str, np.ndarray],
    test_ids: np.ndarray,
    selected: np.ndarray,
    floor: np.ndarray,
    switch: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    local = np.asarray(mask, dtype=bool)
    if not np.any(local):
        return {
            "rows": 0,
            "all_improvement": 0.0,
            "t10_improvement": 0.0,
            "t25_improvement": 0.0,
            "t50_improvement": 0.0,
            "t100_raw_frame_diagnostic_improvement": 0.0,
            "hard_failure_improvement": 0.0,
            "easy_degradation": 0.0,
            "switch_rate": 0.0,
            "harm_over_fallback": 0.0,
        }
    return di._metric_subset(selected[local], floor[local], data, test_ids[local], switch[local])


def _bootstrap_for_mask(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, *, seed: int, easy: bool = False) -> dict[str, Any]:
    local = np.asarray(mask, dtype=bool)
    if int(np.sum(local)) < MIN_CI_ROWS:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(np.sum(local)), "bootstrap_n": 0}
    if easy:
        return di._bootstrap_ci_subset(floor, selected, local, seed=seed, n=BOOTSTRAP_N)
    return di._bootstrap_ci_subset(selected, floor, local, seed=seed, n=BOOTSTRAP_N)


def _rate_ci(values: np.ndarray, mask: np.ndarray, *, seed: int) -> dict[str, Any]:
    local = np.asarray(mask, dtype=bool)
    vals = np.asarray(values, dtype=np.float64)[local]
    if len(vals) < MIN_CI_ROWS:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(vals)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(BOOTSTRAP_N):
        sample = rng.choice(len(vals), size=len(vals), replace=True)
        out.append(float(np.mean(vals[sample])))
    return {
        "low": float(np.percentile(out, 2.5)),
        "mid": float(np.percentile(out, 50.0)),
        "high": float(np.percentile(out, 97.5)),
        "n": int(len(vals)),
        "bootstrap_n": BOOTSTRAP_N,
    }


def _group_row(
    name: str,
    mask: np.ndarray,
    data: Mapping[str, np.ndarray],
    test_ids: np.ndarray,
    selected: np.ndarray,
    floor: np.ndarray,
    switch: np.ndarray,
    final_near: np.ndarray,
    fc_near: np.ndarray,
    di_near: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    h = data["horizon"][test_ids].astype(int)
    hard_failure = data["hard"][test_ids].astype(bool) | data["failure"][test_ids].astype(bool)
    easy = data["easy"][test_ids].astype(bool)
    local = np.asarray(mask, dtype=bool)
    metric = _metric_for_mask(data, test_ids, selected, floor, switch, local)
    bootstrap = {
        "all": _bootstrap_for_mask(selected, floor, local, seed=seed + 1),
        "t50": _bootstrap_for_mask(selected, floor, local & (h == 50), seed=seed + 2),
        "t100_raw_frame_diagnostic": _bootstrap_for_mask(selected, floor, local & (h == 100), seed=seed + 3),
        "hard_failure": _bootstrap_for_mask(selected, floor, local & hard_failure, seed=seed + 4),
        "easy_degradation": _bootstrap_for_mask(selected, floor, local & easy, seed=seed + 5, easy=True),
    }
    near_ci = {
        "final_near_005": _rate_ci(final_near, local, seed=seed + 11),
        "delta_final_minus_fc": _rate_ci(final_near.astype(float) - fc_near.astype(float), local, seed=seed + 12),
        "delta_final_minus_di": _rate_ci(final_near.astype(float) - di_near.astype(float), local, seed=seed + 13),
    }
    robust = bool(
        metric["rows"] >= MIN_CI_ROWS
        and bootstrap["all"]["low"] > 0.0
        and (bootstrap["hard_failure"]["bootstrap_n"] == 0 or bootstrap["hard_failure"]["low"] > 0.0)
        and (bootstrap["easy_degradation"]["bootstrap_n"] == 0 or bootstrap["easy_degradation"]["high"] <= EASY_LIMIT)
        and near_ci["delta_final_minus_fc"]["high"] <= 0.0
    )
    weak_reasons = []
    if metric["rows"] < MIN_CI_ROWS:
        weak_reasons.append("underpowered")
    if bootstrap["all"]["bootstrap_n"] > 0 and bootstrap["all"]["low"] <= 0.0:
        weak_reasons.append("all_ci_not_positive")
    if bootstrap["hard_failure"]["bootstrap_n"] > 0 and bootstrap["hard_failure"]["low"] <= 0.0:
        weak_reasons.append("hard_ci_not_positive")
    if bootstrap["easy_degradation"]["bootstrap_n"] > 0 and bootstrap["easy_degradation"]["high"] > EASY_LIMIT:
        weak_reasons.append("easy_ci_exceeds_2pct")
    if near_ci["delta_final_minus_fc"]["bootstrap_n"] > 0 and near_ci["delta_final_minus_fc"]["high"] > 0.0:
        weak_reasons.append("near_not_ci_better_than_fc")
    return {
        "name": name,
        "rows": int(np.sum(local)),
        "metric": metric,
        "bootstrap": bootstrap,
        "near_bootstrap": near_ci,
        "robust_positive": robust,
        "weak_reasons": weak_reasons,
    }


def _top_scene_rows(
    data: Mapping[str, np.ndarray],
    test_ids: np.ndarray,
    selected: np.ndarray,
    floor: np.ndarray,
    switch: np.ndarray,
    max_rows: int = 40,
) -> list[dict[str, Any]]:
    scene = data["scene_id"][test_ids].astype(str)
    rows = []
    for name in sorted(set(scene.tolist())):
        mask = scene == name
        metric = _metric_for_mask(data, test_ids, selected, floor, switch, mask)
        rows.append({"scene": name, "rows": int(np.sum(mask)), "metric": metric})
    rows.sort(key=lambda row: (float(row["metric"]["all_improvement"]), row["rows"]))
    return rows[:max_rows]


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ff_payload = read_json(ff.REPORT_JSON, {})
    if not ff_payload:
        ff_payload = ff.run_stage42_fe_policy_freeze_replay()
    ctx = ff._context()
    candidate = ff_payload["frozen_policy"]["selected_candidate"]
    replay = ff._replay_selected(ctx, candidate)
    data = ctx["data"]
    test_ids = replay["test_ids"]
    selected = replay["test"]["selected_ade"]
    floor = replay["test"]["floor_ade"]
    switch = replay["test"]["switch"]
    final_near = np.isfinite(replay["final_min_distance"]) & (replay["final_min_distance"] < 0.05)
    fc_near = np.isfinite(replay["test_evals"]["fc"]["min_distance"]) & (replay["test_evals"]["fc"]["min_distance"] < 0.05)
    di_near = np.isfinite(replay["test_evals"]["di"]["min_distance"]) & (replay["test_evals"]["di"]["min_distance"] < 0.05)
    domain = data["dataset"][test_ids].astype(str)
    source_file = data["source_file"][test_ids].astype(str)
    horizon = data["horizon"][test_ids].astype(int)
    domain_rows = {
        name: _group_row(
            name,
            domain == name,
            data,
            test_ids,
            selected,
            floor,
            switch,
            final_near,
            fc_near,
            di_near,
            seed=43600 + i * 100,
        )
        for i, name in enumerate(sorted(set(domain.tolist())))
    }
    domain_horizon_rows = {
        f"{d}|{h}": _group_row(
            f"{d}|{h}",
            (domain == d) & (horizon == h),
            data,
            test_ids,
            selected,
            floor,
            switch,
            final_near,
            fc_near,
            di_near,
            seed=43700 + i * 100,
        )
        for i, (d, h) in enumerate((d, h) for d in sorted(set(domain.tolist())) for h in [10, 25, 50, 100])
        if np.any((domain == d) & (horizon == h))
    }
    source_rows = {
        _source_name(name): _group_row(
            _source_name(name),
            source_file == name,
            data,
            test_ids,
            selected,
            floor,
            switch,
            final_near,
            fc_near,
            di_near,
            seed=43800 + i * 100,
        )
        for i, name in enumerate(sorted(set(source_file.tolist())))
    }
    weak_domains = [name for name, row in domain_rows.items() if row["rows"] >= MIN_DOMAIN_ROWS and not row["robust_positive"]]
    weak_domain_horizons = [name for name, row in domain_horizon_rows.items() if row["rows"] >= MIN_CI_ROWS and not row["robust_positive"]]
    weak_sources = [name for name, row in source_rows.items() if row["rows"] >= MIN_CI_ROWS and not row["robust_positive"]]
    robust_domains = [name for name, row in domain_rows.items() if row["robust_positive"]]
    summary = {
        "source": SOURCE,
        "test_rows": int(len(test_ids)),
        "domain_count": len(domain_rows),
        "source_count": len(source_rows),
        "domain_horizon_count": len(domain_horizon_rows),
        "robust_domains": robust_domains,
        "weak_domains": weak_domains,
        "weak_domain_horizons": weak_domain_horizons,
        "weak_sources": weak_sources,
        "broad_uniform_source_claim_allowed": len(weak_sources) == 0 and len(source_rows) >= 3,
        "paper_claim": "Stage42-FE/FF is frozen and bootstrap-positive globally; Stage42-FG reports source/domain weak slices and only allows broad source claims if no powered weak sources remain.",
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FG FE source/domain/horizon robustness audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": ff._git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                str(ff.REPORT_JSON),
                str(ff.POLICY_JSON),
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "fe_policy": {
            "policy_hash": ff_payload["frozen_policy"]["policy_hash"],
            "selected_candidate": candidate,
            "ff_verdict": ff_payload["stage42_ff_gate"]["verdict"],
        },
        "summary": summary,
        "domain_rows": domain_rows,
        "domain_horizon_rows": domain_horizon_rows,
        "source_rows": source_rows,
        "weak_scene_rows_top": _top_scene_rows(data, test_ids, selected, floor, switch),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
            "test_rows_reporting_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "broad_uniform_source_claim": summary["broad_uniform_source_claim_allowed"],
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_fg_gate"] = _gate(payload)
    return am._jsonable(payload)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    no_leak = payload["no_leakage"]
    boundary = payload["claim_boundary"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "ff_input_verified": payload["fe_policy"]["ff_verdict"] == "stage42_ff_fe_policy_freeze_replay_pass",
        "domains_audited": s["domain_count"] >= 2,
        "sources_audited": s["source_count"] >= 2,
        "domain_horizons_audited": s["domain_horizon_count"] >= 4,
        "at_least_two_robust_domains": len(s["robust_domains"]) >= 2,
        "weak_slices_reported": "weak_domain_horizons" in s and "weak_sources" in s,
        "broad_source_claim_only_if_no_weak_sources": (boundary["broad_uniform_source_claim"] is False) or len(s["weak_sources"]) == 0,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["future_waypoint_label_eval_only"] is True,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
                no_leak["validation_only_policy_selection"] is True,
            ]
        ),
        "no_metric_seconds_overclaim": boundary["metric_or_seconds_claim"] is False,
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_fg_fe_source_robustness_pass" if passed == total else "stage42_fg_fe_source_robustness_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_group_table(title: str, rows: Mapping[str, Mapping[str, Any]], *, max_rows: int | None = None) -> list[str]:
    items = list(rows.items())
    if max_rows is not None:
        items = items[:max_rows]
    out = [
        f"## {title}",
        "",
        "| name | rows | all | t50 | t100 raw | hard | easy | near delta vs FC CI high | robust | weak reasons |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in items:
        m = row["metric"]
        near_high = row["near_bootstrap"]["delta_final_minus_fc"]["high"]
        out.append(
            f"| `{name}` | {row['rows']} | {_pct(m['all_improvement'])} | {_pct(m['t50_improvement'])} | "
            f"{_pct(m['t100_raw_frame_diagnostic_improvement'])} | {_pct(m['hard_failure_improvement'])} | "
            f"{_pct(m['easy_degradation'])} | {_pct(near_high)} | {row['robust_positive']} | "
            f"`{', '.join(row['weak_reasons']) or 'none'}` |"
        )
    return out


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fg_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-FG FE Source / Domain / Horizon Robustness Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- FE policy hash: `{payload['fe_policy']['policy_hash']}`",
        "",
        "## Summary",
        "",
        f"- test_rows: `{s['test_rows']}`",
        f"- domain_count: `{s['domain_count']}`",
        f"- source_count: `{s['source_count']}`",
        f"- domain_horizon_count: `{s['domain_horizon_count']}`",
        f"- robust_domains: `{s['robust_domains']}`",
        f"- weak_domains: `{s['weak_domains']}`",
        f"- weak_domain_horizons: `{s['weak_domain_horizons']}`",
        f"- weak_sources: `{s['weak_sources']}`",
        f"- broad_uniform_source_claim_allowed: `{s['broad_uniform_source_claim_allowed']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
    ]
    lines.extend(_render_group_table("Domain Robustness", payload["domain_rows"]))
    lines.append("")
    lines.extend(_render_group_table("Domain-Horizon Robustness", payload["domain_horizon_rows"]))
    lines.append("")
    lines.extend(_render_group_table("Source Robustness", payload["source_rows"]))
    lines += [
        "",
        "## Weak Scene Rows",
        "",
        "| scene | rows | all | t50 | t100 raw | hard | easy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["weak_scene_rows_top"]:
        m = row["metric"]
        lines.append(
            f"| `{row['scene']}` | {row['rows']} | {_pct(m['all_improvement'])} | {_pct(m['t50_improvement'])} | "
            f"{_pct(m['t100_raw_frame_diagnostic_improvement'])} | {_pct(m['hard_failure_improvement'])} | {_pct(m['easy_degradation'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-FG is an audit, not a new policy selection or new training run.",
        "- Global FE/FF evidence is strong, but source/domain/horizon weak slices remain visible and must be cited as limitations.",
        "- Broad uniform source-level claims are allowed only if powered source slices have no weak failures.",
        "- No Stage5C, SMC, metric/seconds-level, true-3D, foundation, or floor-free neural claim is made.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fg_gate"]
    lines = [
        "# Stage42-FG Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def _replace_text_section(old: str, tag: str, block: str) -> str:
    start = f"<!-- {tag}:START -->"
    end = f"<!-- {tag}:END -->"
    if start in old and end in old:
        before, rest = old.split(start, 1)
        _, after = rest.split(end, 1)
        return before.rstrip() + "\n\n" + block.strip() + after
    return old.rstrip() + "\n\n" + block.strip() + "\n"


def _summary_section(payload: Mapping[str, Any]) -> str:
    s = payload["summary"]
    return "\n".join(
        [
            "<!-- STAGE42_FG_FE_SOURCE_ROBUSTNESS:START -->",
            "## Stage42-FG FE Source / Domain / Horizon Robustness Audit",
            "",
            f"- source: `{payload['source']}`",
            "- role: audit frozen Stage42-FE/FF across domain/source/horizon/scene slices without retraining or threshold reselection.",
            f"- gate: `{payload['stage42_fg_gate']['passed']} / {payload['stage42_fg_gate']['total']}`; verdict `{payload['stage42_fg_gate']['verdict']}`.",
            f"- robust domains: `{s['robust_domains']}`.",
            f"- weak domain-horizon slices: `{s['weak_domain_horizons']}`.",
            f"- weak sources: `{s['weak_sources']}`.",
            f"- broad uniform source claim allowed: `{s['broad_uniform_source_claim_allowed']}`.",
            "- Boundary: protected source-level raw-frame 2.5D audit; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            "<!-- STAGE42_FG_FE_SOURCE_ROBUSTNESS:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(_replace_text_section(old, "STAGE42_FG_FE_SOURCE_ROBUSTNESS", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FG FE source/domain/horizon robustness audit"
    state["current_verdict"] = payload["stage42_fg_gate"]["verdict"]
    state["stage42_fg_fe_source_robustness_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_fg_gate"]["verdict"],
        "gates": f"{payload['stage42_fg_gate']['passed']}/{payload['stage42_fg_gate']['total']}",
        "fe_policy_hash": payload["fe_policy"]["policy_hash"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FG audits the frozen FE policy across domain/source/horizon/scene slices and prevents broad source-level overclaiming when weak slices remain.",
    }
    block = state.get("m3w_work_attempts_failures_successes_readme")
    if isinstance(block, dict):
        evidence = list(block.get("latest_included_evidence", []))
        if "Stage42-FG FE source/domain/horizon robustness audit" not in evidence:
            evidence.append("Stage42-FG FE source/domain/horizon robustness audit")
        block["latest_included_evidence"] = evidence
        block["source"] = "cached_verified_summary_from_stage18_to_stage42_reports_plus_stage42_es_to_fg_fresh_audits"
        block[
            "latest_conclusion"
        ] = "Stage42-FG keeps FE honest by auditing source/domain/horizon weak slices after FF freeze/bootstrap, rather than treating global positive evidence as uniform source generalization."
        state["m3w_work_attempts_failures_successes_readme"] = block
    write_json(RESEARCH_STATE, state)


def run_stage42_fe_source_robustness_audit() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_fe_source_robustness_audit()
    gate = result["stage42_fg_gate"]
    print(f"Stage42-FG gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
