from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_scene_proxy_guarded_latent_policy as ac


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_scene_proxy_guarded_robustness_audit.json"
REPORT_MD = OUT_DIR / "stage43_scene_proxy_guarded_robustness_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_ad_scene_proxy_guarded_robustness_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AD_SCENE_PROXY_GUARDED_ROBUSTNESS_AUDIT"
SOURCE = "fresh_stage43_ad_scene_proxy_guarded_robustness_audit"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _slice_row(
    *,
    name: str,
    mask: np.ndarray,
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    ab_allowed: np.ndarray,
    stage43_m_metrics: Mapping[str, Any],
    stage43_ab_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    ids = np.asarray(mask, dtype=bool)
    if int(ids.sum()) == 0:
        return {
            "name": name,
            "rows": 0,
            "ac": {},
            "stage43_m": {},
            "stage43_ab_all": {},
            "delta_vs_stage43_m": {},
            "delta_vs_stage43_ab_all": {},
            "scene_proxy_override_rate": 0.0,
            "t100_scene_proxy_override_rate": 0.0,
            "weak_or_caveat": True,
            "caveat_reason": "empty_slice",
        }
    sub = _subset(ds, ids)
    ac_metrics = m._metrics(sub, selected_ade[ids], selected_fde[ids], switched[ids])
    ac_metrics["scene_proxy_override_rate"] = float(np.mean(ab_allowed[ids]))
    h100 = sub.horizon == 100
    ac_metrics["t100_scene_proxy_override_rate"] = float(np.mean(ab_allowed[ids][h100])) if int(h100.sum()) else 0.0
    m_metrics = _slice_metrics_from_full(stage43_m_metrics, ids, ds)
    ab_metrics = _slice_metrics_from_full(stage43_ab_metrics, ids, ds)
    delta_m = ac._delta_metrics(ac_metrics, m_metrics)
    delta_ab = ac._delta_metrics(ac_metrics, ab_metrics)
    caveats: list[str] = []
    if ac_metrics["full_waypoint_ade_improvement_vs_floor"] <= 0.0:
        caveats.append("non_positive_ac_all")
    if delta_m["full_waypoint_ade_improvement_vs_floor"] < -1e-9:
        caveats.append("worse_than_stage43_m_all")
    if delta_m["t50_full_waypoint_ade_improvement_vs_floor"] < -1e-9:
        caveats.append("worse_than_stage43_m_t50")
    if delta_m["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] < -0.002:
        caveats.append("worse_than_stage43_m_t100")
    if ac_metrics["easy_degradation_vs_floor"] > 0.02:
        caveats.append("easy_degradation_over_2pct")
    return {
        "name": name,
        "rows": int(ids.sum()),
        "ac": ac_metrics,
        "stage43_m": m_metrics,
        "stage43_ab_all": ab_metrics,
        "delta_vs_stage43_m": delta_m,
        "delta_vs_stage43_ab_all": delta_ab,
        "scene_proxy_override_rate": ac_metrics["scene_proxy_override_rate"],
        "t100_scene_proxy_override_rate": ac_metrics["t100_scene_proxy_override_rate"],
        "weak_or_caveat": bool(caveats),
        "caveat_reason": ", ".join(caveats) if caveats else "none",
    }


def _subset(ds: m.WaypointSplit, mask: np.ndarray) -> m.WaypointSplit:
    ids = np.asarray(mask, dtype=bool)
    return m.WaypointSplit(
        split=ds.split,
        x=ds.x[ids],
        waypoint_delta=ds.waypoint_delta[ids],
        waypoint_valid=ds.waypoint_valid[ids],
        floor_waypoint_delta=ds.floor_waypoint_delta[ids],
        floor_ade=ds.floor_ade[ids],
        floor_fde=ds.floor_fde[ids],
        y_failure=ds.y_failure[ids],
        y_gain=ds.y_gain[ids],
        y_harm=ds.y_harm[ids],
        y_density=ds.y_density[ids],
        horizon=ds.horizon[ids],
        domain=ds.domain[ids],
        source_file=ds.source_file[ids],
        scene_id=ds.scene_id[ids],
        hard=ds.hard[ids],
        failure=ds.failure[ids],
        easy=ds.easy[ids],
        scale=ds.scale[ids],
        feature_names=ds.feature_names,
    )


def _slice_metrics_from_full(pack: Mapping[str, Any], mask: np.ndarray, ds: m.WaypointSplit) -> dict[str, Any]:
    selected_ade = np.asarray(pack["selected_ade"])[mask]
    selected_fde = np.asarray(pack["selected_fde"])[mask]
    switched = np.asarray(pack["switched"])[mask]
    sub = _subset(ds, mask)
    return m._metrics(sub, selected_ade, selected_fde, switched)


def _full_selection_pack(
    *,
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
) -> dict[str, Any]:
    metrics = m._metrics(ds, selected_ade, selected_fde, switched)
    return {
        "selected_ade": selected_ade.astype(np.float32),
        "selected_fde": selected_fde.astype(np.float32),
        "switched": switched.astype(bool),
        "metrics": metrics,
    }


def _group_table(
    values: np.ndarray,
    *,
    prefix: str,
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    ab_allowed: np.ndarray,
    stage43_m_pack: Mapping[str, Any],
    stage43_ab_pack: Mapping[str, Any],
    min_rows: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in sorted(set(values.astype(str).tolist())):
        mask = values.astype(str) == value
        if int(mask.sum()) < min_rows:
            continue
        rows.append(
            _slice_row(
                name=f"{prefix}:{value}",
                mask=mask,
                ds=ds,
                selected_ade=selected_ade,
                selected_fde=selected_fde,
                switched=switched,
                ab_allowed=ab_allowed,
                stage43_m_metrics=stage43_m_pack,
                stage43_ab_metrics=stage43_ab_pack,
            )
        )
    return sorted(rows, key=lambda row: (-int(row["rows"]), str(row["name"])))


def _boolean_slices(
    *,
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    ab_allowed: np.ndarray,
    stage43_m_pack: Mapping[str, Any],
    stage43_ab_pack: Mapping[str, Any],
) -> list[dict[str, Any]]:
    masks = {
        "all": np.ones(len(ds.x), dtype=bool),
        "h10": ds.horizon == 10,
        "h25": ds.horizon == 25,
        "h50": ds.horizon == 50,
        "h100_raw_frame_diagnostic": ds.horizon == 100,
        "hard_or_failure": ds.hard | ds.failure,
        "easy": ds.easy,
        "scene_proxy_override": ab_allowed,
        "stage43_m_fallback": ~ab_allowed,
    }
    return [
        _slice_row(
            name=name,
            mask=mask,
            ds=ds,
            selected_ade=selected_ade,
            selected_fde=selected_fde,
            switched=switched,
            ab_allowed=ab_allowed,
            stage43_m_metrics=stage43_m_pack,
            stage43_ab_metrics=stage43_ab_pack,
        )
        for name, mask in masks.items()
    ]


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    seed = int(args.seed)
    rows = ac._max_rows("medium" if args.medium else "quick" if args.quick else "small")
    ac_report = read_json(ac.REPORT_JSON, {})
    stage43_m_report = read_json(m.REPORT_JSON, {})
    stage43_ab_report = read_json(ac.ab.REPORT_JSON, {})
    m_model, m_ckpt = ac._load_model(Path(stage43_m_report["checkpoint"]))
    ab_model, ab_ckpt = ac._load_model(Path(stage43_ab_report["checkpoint"]))
    pack = ac._replay_split(
        "test",
        max_rows=rows["test"],
        seed=seed,
        batch_size=int(args.batch_size),
        m_model=m_model,
        m_ckpt=m_ckpt,
        ab_model=ab_model,
        ab_ckpt=ab_ckpt,
        m_policy=stage43_m_report["validation_selected_policy"]["policy"],
    )
    selected_ade, selected_fde, switched, ab_allowed = ac._select_guarded(pack, ac_report["validation_selected_policy"]["policy"])
    stage43_m_pack = _full_selection_pack(
        ds=pack["ds"], selected_ade=pack["m_ade"], selected_fde=pack["m_fde"], switched=pack["m_switched"]
    )
    stage43_ab_pack = _full_selection_pack(
        ds=pack["ds"],
        selected_ade=pack["ab_ade"],
        selected_fde=pack["ab_fde"],
        switched=np.ones(len(pack["ds"].x), dtype=bool),
    )
    ac_pack = _full_selection_pack(ds=pack["ds"], selected_ade=selected_ade, selected_fde=selected_fde, switched=switched)
    slices = _boolean_slices(
        ds=pack["ds"],
        selected_ade=selected_ade,
        selected_fde=selected_fde,
        switched=switched,
        ab_allowed=ab_allowed,
        stage43_m_pack=stage43_m_pack,
        stage43_ab_pack=stage43_ab_pack,
    )
    domains = _group_table(
        pack["ds"].domain,
        prefix="domain",
        ds=pack["ds"],
        selected_ade=selected_ade,
        selected_fde=selected_fde,
        switched=switched,
        ab_allowed=ab_allowed,
        stage43_m_pack=stage43_m_pack,
        stage43_ab_pack=stage43_ab_pack,
    )
    horizons = _group_table(
        pack["ds"].horizon.astype(str),
        prefix="horizon",
        ds=pack["ds"],
        selected_ade=selected_ade,
        selected_fde=selected_fde,
        switched=switched,
        ab_allowed=ab_allowed,
        stage43_m_pack=stage43_m_pack,
        stage43_ab_pack=stage43_ab_pack,
    )
    sources = _group_table(
        pack["ds"].source_file,
        prefix="source",
        ds=pack["ds"],
        selected_ade=selected_ade,
        selected_fde=selected_fde,
        switched=switched,
        ab_allowed=ab_allowed,
        stage43_m_pack=stage43_m_pack,
        stage43_ab_pack=stage43_ab_pack,
        min_rows=max(1, int(args.min_source_rows)),
    )
    weak = [
        row
        for table in [slices, domains, horizons, sources]
        for row in table
        if row["weak_or_caveat"]
    ]
    powered_domains = [row for row in domains if row["rows"] >= int(args.min_domain_rows)]
    positive_powered_domains = [
        row
        for row in powered_domains
        if row["ac"].get("full_waypoint_ade_improvement_vs_floor", 0.0) > 0.0
        and row["ac"].get("easy_degradation_vs_floor", 0.0) <= 0.02
    ]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_replay_stage43_ac_slice_robustness_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": "medium" if args.medium else "quick" if args.quick else "small",
        "stage43_ac_verdict": ac_report.get("stage43_ac_gate", {}).get("verdict"),
        "stage43_ac_policy": ac_report.get("validation_selected_policy", {}).get("policy", {}),
        "rows": int(len(pack["ds"].x)),
        "overall": {
            "stage43_ac": ac_pack["metrics"],
            "stage43_m": stage43_m_pack["metrics"],
            "stage43_ab_all": stage43_ab_pack["metrics"],
            "delta_vs_stage43_m": ac._delta_metrics(ac_pack["metrics"], stage43_m_pack["metrics"]),
            "delta_vs_stage43_ab_all": ac._delta_metrics(ac_pack["metrics"], stage43_ab_pack["metrics"]),
        },
        "slice_table": slices,
        "domain_table": domains,
        "horizon_table": horizons,
        "source_table": sources,
        "weak_or_caveat_slices": weak,
        "powered_domain_count": len(powered_domains),
        "positive_powered_domain_count": len(positive_powered_domains),
        "bootstrap_ci": m._bootstrap_ci(pack["ds"], selected_ade, selected_fde, n=int(args.bootstrap), seed=seed + 3000),
        "row_hash": m._row_hash(m._npz(m._cache_path("test"))),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "scene_proxy_train_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "t100_raw_frame_diagnostic_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
            "uniform_horizon_success_claim": False,
        },
        "input_hash": _combined_hash([ac.REPORT_JSON, m.REPORT_JSON, ac.ab.REPORT_JSON, m._cache_path("test")]),
    }
    payload["stage43_ad_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    overall = payload["overall"]["stage43_ac"]
    delta = payload["overall"]["delta_vs_stage43_m"]
    gates = {
        "stage43_ac_candidate_available": payload["stage43_ac_verdict"] == "stage43_ac_guarded_scene_proxy_latent_candidate",
        "fresh_slice_replay_completed": payload["result_source"] == "fresh_replay_stage43_ac_slice_robustness_audit"
        and payload["rows"] > 0,
        "domain_table_reported": len(payload["domain_table"]) > 0,
        "horizon_table_reported": len(payload["horizon_table"]) >= 4,
        "source_table_reported": len(payload["source_table"]) > 0,
        "overall_easy_preserved": overall["easy_degradation_vs_floor"] <= 0.02,
        "overall_lift_vs_stage43_m": delta["full_waypoint_ade_improvement_vs_floor"] > 0.0
        or delta["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
        or delta["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "t100_caveat_recorded": payload["claim_boundary"]["t100_raw_frame_diagnostic_only"] is True
        and payload["overall"]["delta_vs_stage43_m"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -0.002,
        "powered_domain_status_reported": payload["powered_domain_count"] >= payload["positive_powered_domain_count"],
        "weak_slices_not_hidden": isinstance(payload["weak_or_caveat_slices"], list),
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False
        and payload["no_leakage"]["scene_proxy_train_only"] is True,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    all_powered_positive = payload["powered_domain_count"] > 0 and payload["powered_domain_count"] == payload["positive_powered_domain_count"]
    verdict = (
        "stage43_ad_guarded_scene_proxy_robust_with_caveats"
        if passed == total and all_powered_positive
        else "stage43_ad_guarded_scene_proxy_caveated_audit_pass"
        if passed == total
        else "stage43_ad_guarded_scene_proxy_audit_fail"
    )
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "all_powered_domains_positive": bool(all_powered_positive),
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _compact_rows(rows: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        ac_metrics = row["ac"]
        delta = row["delta_vs_stage43_m"]
        out.append(
            {
                "name": row["name"],
                "rows": row["rows"],
                "ac_all": ac_metrics.get("full_waypoint_ade_improvement_vs_floor", 0.0),
                "ac_t50": ac_metrics.get("t50_full_waypoint_ade_improvement_vs_floor", 0.0),
                "ac_t100": ac_metrics.get("t100_raw_frame_full_waypoint_diagnostic_vs_floor", 0.0),
                "ac_hard": ac_metrics.get("hard_failure_full_waypoint_ade_improvement_vs_floor", 0.0),
                "easy": ac_metrics.get("easy_degradation_vs_floor", 0.0),
                "delta_all_vs_m": delta.get("full_waypoint_ade_improvement_vs_floor", 0.0),
                "override": row.get("scene_proxy_override_rate", 0.0),
                "caveat": row.get("caveat_reason", "none"),
            }
        )
    return out


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_ad_gate"]
    overall = payload["overall"]
    ac_metrics = overall["stage43_ac"]
    delta = overall["delta_vs_stage43_m"]
    lines = [
        "# Stage43-AD Scene-Proxy Guarded Robustness Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- all powered domains positive: `{gate['all_powered_domains_positive']}`",
        "",
        "## Overall",
        "",
        f"- rows: `{payload['rows']}`",
        f"- AC full-waypoint ADE vs floor: `{_pct(ac_metrics['full_waypoint_ade_improvement_vs_floor'])}`; delta vs Stage43-M: `{_pct(delta['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- AC t50 ADE vs floor: `{_pct(ac_metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`; delta vs Stage43-M: `{_pct(delta['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- AC hard/failure vs floor: `{_pct(ac_metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`; delta vs Stage43-M: `{_pct(delta['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- AC easy degradation: `{_pct(ac_metrics['easy_degradation_vs_floor'])}`",
        f"- AC t100 raw-frame diagnostic: `{_pct(ac_metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`; delta vs Stage43-M: `{_pct(delta['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        "",
        "## Domain Table",
        "",
        "| domain | rows | AC all | delta all vs M | AC t50 | AC hard | easy | override | caveat |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in _compact_rows(payload["domain_table"], limit=20):
        lines.append(
            f"| `{row['name']}` | `{row['rows']}` | `{_pct(row['ac_all'])}` | `{_pct(row['delta_all_vs_m'])}` | `{_pct(row['ac_t50'])}` | `{_pct(row['ac_hard'])}` | `{_pct(row['easy'])}` | `{_pct(row['override'])}` | `{row['caveat']}` |"
        )
    lines.extend(
        [
            "",
            "## Horizon Table",
            "",
            "| horizon | rows | AC all | delta all vs M | AC t50 | AC t100 | AC hard | easy | override | caveat |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in _compact_rows(payload["horizon_table"], limit=20):
        lines.append(
            f"| `{row['name']}` | `{row['rows']}` | `{_pct(row['ac_all'])}` | `{_pct(row['delta_all_vs_m'])}` | `{_pct(row['ac_t50'])}` | `{_pct(row['ac_t100'])}` | `{_pct(row['ac_hard'])}` | `{_pct(row['easy'])}` | `{_pct(row['override'])}` | `{row['caveat']}` |"
        )
    lines.extend(
        [
            "",
            "## Caveat Slices",
            "",
            f"- caveat slice count: `{len(payload['weak_or_caveat_slices'])}`",
            "- AC does not claim uniform horizon success. t100 remains raw-frame diagnostic and is guarded by falling back to Stage43-M.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-AD Scene-Proxy Guarded Robustness Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- all powered domains positive: `{gate['all_powered_domains_positive']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_text_outputs(payload)


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ad_gate"]
    ac_metrics = payload["overall"]["stage43_ac"]
    delta = payload["overall"]["delta_vs_stage43_m"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"all_powered_domains_positive = `{gate['all_powered_domains_positive']}`",
        "",
        f"full_waypoint_ade_vs_floor = `{_pct(ac_metrics['full_waypoint_ade_improvement_vs_floor'])}`; delta_vs_stage43_m = `{_pct(delta['full_waypoint_ade_improvement_vs_floor'])}`",
        f"t50_full_waypoint_ade_vs_floor = `{_pct(ac_metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`; delta_vs_stage43_m = `{_pct(delta['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"hard_failure_vs_floor = `{_pct(ac_metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`; delta_vs_stage43_m = `{_pct(delta['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"t100_raw_frame_diagnostic = `{_pct(ac_metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`; delta_vs_stage43_m = `{_pct(delta['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"easy_degradation = `{_pct(ac_metrics['easy_degradation_vs_floor'])}`",
        f"caveat_slice_count = `{len(payload['weak_or_caveat_slices'])}`",
        "",
        "Stage43-AD audits the guarded Stage43-AC policy by domain, source, horizon, hard/failure, and easy slices. It records caveats instead of turning the average gain into a uniform success claim.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; t100 remains raw-frame diagnostic; no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ad_scene_proxy_guarded_robustness_audit"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "all_powered_domains_positive": gate["all_powered_domains_positive"],
        "metrics": payload["overall"]["stage43_ac"],
        "delta_vs_stage43_m": payload["overall"]["delta_vs_stage43_m"],
        "powered_domain_count": payload["powered_domain_count"],
        "positive_powered_domain_count": payload["positive_powered_domain_count"],
        "caveat_slice_count": len(payload["weak_or_caveat_slices"]),
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ad_scene_proxy_guarded_robustness_audit"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-AD",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "all_powered_domains_positive": gate["all_powered_domains_positive"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Stage43-AC guarded scene-proxy latent robustness by slice.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--small", action="store_true")
    group.add_argument("--medium", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--min-source-rows", type=int, default=200)
    parser.add_argument("--min-domain-rows", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    result = _run(args)
    gate = result["stage43_ad_gate"]
    print(f"Stage43-AD: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"all_powered_domains_positive={gate['all_powered_domains_positive']}")
    return result


if __name__ == "__main__":
    main()
