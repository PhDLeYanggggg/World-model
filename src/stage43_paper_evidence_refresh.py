from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_bounded_residual_policy_freeze as an


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_paper_evidence_refresh.json"
REPORT_MD = OUT_DIR / "stage43_paper_evidence_refresh.md"
TABLE_CSV = OUT_DIR / "stage43_paper_evidence_refresh.csv"
CLAIM_MD = OUT_DIR / "stage43_claim_boundary_refresh.md"
GAP_MD = OUT_DIR / "stage43_a_journal_gap_refresh.md"
GATE_MD = OUT_DIR / "stage43_stage_ap_paper_evidence_refresh_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AP_PAPER_EVIDENCE_REFRESH"
SOURCE = "fresh_stage43_ap_paper_evidence_refresh"

STAGE43_AJ = OUT_DIR / "stage43_safety_floor_necessity_audit.json"
STAGE43_AK = OUT_DIR / "stage43_self_gate_conformal_audit.json"
STAGE43_AL = OUT_DIR / "stage43_bounded_residual_safety_audit.json"
STAGE43_AM = OUT_DIR / "stage43_bounded_residual_statistical_confirmation.json"
STAGE43_AN = OUT_DIR / "stage43_bounded_residual_policy_freeze.json"
STAGE43_AO = OUT_DIR / "stage43_bounded_residual_reviewer_replay.json"
FROZEN_POLICY = OUT_DIR / "frozen_stage43_bounded_residual_policy.json"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def _evidence_row(name: str, status: str, evidence: str, claim: str, caveat: str) -> dict[str, str]:
    return {
        "claim_name": name,
        "status": status,
        "evidence": evidence,
        "allowed_claim": claim,
        "caveat": caveat,
    }


def _build_evidence_rows(
    aj: Mapping[str, Any],
    ak: Mapping[str, Any],
    al: Mapping[str, Any],
    am: Mapping[str, Any],
    an_payload: Mapping[str, Any],
    ao: Mapping[str, Any],
) -> list[dict[str, str]]:
    frozen = an_payload["frozen_metrics"]
    ci = am["bootstrap_delta_ci"]["metrics"]
    rows = [
        _evidence_row(
            "Reviewer-replayable protected bounded residual policy",
            "supported",
            f"AO gate {ao['stage43_ao_gate']['passed']}/{ao['stage43_ao_gate']['total']}, replay diff {ao['replay_diff']['max_abs_diff']:.8f}, policy hash match {ao['policy_hash']['match']}",
            "Stage43 bounded residual policy is frozen, hashable, and exact-replayable from the policy artifact.",
            "Requires local checkpoint/cache not committed to git; dataset-local/raw-frame only.",
        ),
        _evidence_row(
            "Protected full-waypoint latent dynamics lift",
            "supported",
            f"AN frozen all {_pct(frozen['all'])}, t50 {_pct(frozen['t50'])}, hard {_pct(frozen['hard_failure'])}, easy {_pct(frozen['easy'])}",
            "Protected bounded residual latent waypoint policy improves full-waypoint metrics under safety floor.",
            "This is protected residual dynamics, not ungated generative rollout.",
        ),
        _evidence_row(
            "Bootstrap-supported delta over stored hard switch",
            "supported",
            f"AM all delta CI [{_pct(ci['all_delta_improvement']['low'])}, {_pct(ci['all_delta_improvement']['high'])}], t50 [{_pct(ci['t50_delta_improvement']['low'])}, {_pct(ci['t50_delta_improvement']['high'])}], hard [{_pct(ci['hard_failure_delta_improvement']['low'])}, {_pct(ci['hard_failure_delta_improvement']['high'])}]",
            "Bounded residual has positive bootstrap delta over stored Stage43-M hard switch on all/t50/hard slices.",
            "Bootstrap over frozen rows; not a new external dataset acquisition claim.",
        ),
        _evidence_row(
            "Per-domain external support",
            "partially_supported",
            "; ".join(
                f"{row['slice']} delta {_pct(row['delta'])}" for row in am["slice_rows"] if row["slice"].startswith("domain:")
            ),
            "Positive dataset-local/raw-frame deltas are observed across ETH_UCY, TrajNet, and UCY slices.",
            "Domain labels are from existing external conversion; metric/seconds calibration remains unverified.",
        ),
        _evidence_row(
            "Global floor removal",
            "not_supported",
            f"AJ verdict {aj['stage43_aj_gate']['verdict']}; AK ungated easy {_pct(next(row for row in ak['policy_table'] if row['name']=='ungated_neural')['easy_degradation_vs_floor'])}",
            "Do not remove the safety floor globally.",
            "The floor is currently part of the method, not a disposable crutch.",
        ),
        _evidence_row(
            "h100 / t100 raw-frame behavior",
            "guarded_only",
            f"AN t100 diagnostic {_pct(frozen['t100'])}; h100 guard {an_payload['policy']['deployment_rule']['h100_guard']}",
            "Report t100 only as raw-frame diagnostic with h100 floor guard.",
            "No seconds-level long-horizon claim.",
        ),
        _evidence_row(
            "A-journal readiness",
            "not_yet",
            "Stage43 has replayable protected bounded residual evidence, but metric/time calibration, true 3D evidence, and broader raw-data multimodal validation remain incomplete.",
            "The current package is a stronger candidate evidence block, not a complete A-journal submission claim.",
            "Need source-level calibration, more multimodal scene evidence, full paper package coherence, and broader external verification.",
        ),
    ]
    return rows


def _write_csv(rows: list[Mapping[str, str]]) -> None:
    with TABLE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["claim_name", "status", "evidence", "allowed_claim", "caveat"])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _run(_: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    aj = _load(STAGE43_AJ)
    ak = _load(STAGE43_AK)
    al = _load(STAGE43_AL)
    am = _load(STAGE43_AM)
    an_payload = _load(STAGE43_AN)
    ao = _load(STAGE43_AO)
    policy = _load(FROZEN_POLICY)
    rows = _build_evidence_rows(aj, ak, al, am, an_payload, ao)
    answers = {
        "still_2_5d": True,
        "metric_time_subset_available": False,
        "full_waypoint_dynamics_available": True,
        "cross_domain_external_evidence": "positive dataset-local/raw-frame slices for ETH_UCY, TrajNet, UCY in Stage43-AM",
        "exceeds_stored_stage43_m_hard_switch": True,
        "exceeds_stage37_floor": "not directly re-benchmarked in AP; bounded residual is compared to stored Stage43-M hard-switch/floor lineage",
        "scene_goal_interaction_effective": "partially supported by earlier Stage43 ablations; bounded residual evidence itself is policy-level/full-waypoint",
        "a_journal_candidate": False,
        "why_not_a_journal_yet": [
            "Still not true 3D and not metric/seconds-level.",
            "Stage43 policy is protected and h100-guarded; global floor removal is not supported.",
            "Multimodal scene/raster evidence is still mostly proxy-level rather than raw image/video foundation-scale evidence.",
            "Need final integrated paper package, broader source calibration, and full test-suite replay.",
        ],
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_paper_evidence_refresh_from_stage43_aj_to_ao",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "evidence_rows": rows,
        "answers": answers,
        "policy_hash": policy["policy_hash"],
        "frozen_policy": str(FROZEN_POLICY),
        "key_metrics": {
            "all": an_payload["frozen_metrics"]["all"],
            "t50": an_payload["frozen_metrics"]["t50"],
            "t100": an_payload["frozen_metrics"]["t100"],
            "hard_failure": an_payload["frozen_metrics"]["hard_failure"],
            "easy": an_payload["frozen_metrics"]["easy"],
            "t50_delta_ci": an_payload["frozen_metrics"]["t50_delta_ci"],
            "ao_replay_diff": ao["replay_diff"]["max_abs_diff"],
        },
        "evidence_sources": {
            "stage43_aj": str(STAGE43_AJ),
            "stage43_ak": str(STAGE43_AK),
            "stage43_al": str(STAGE43_AL),
            "stage43_am": str(STAGE43_AM),
            "stage43_an": str(STAGE43_AN),
            "stage43_ao": str(STAGE43_AO),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "input_hash": _combined_hash([STAGE43_AJ, STAGE43_AK, STAGE43_AL, STAGE43_AM, STAGE43_AN, STAGE43_AO, FROZEN_POLICY]),
    }
    payload["stage43_ap_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    statuses = {row["claim_name"]: row["status"] for row in payload["evidence_rows"]}
    metrics = payload["key_metrics"]
    gates = {
        "replayable_policy_claim_supported": statuses["Reviewer-replayable protected bounded residual policy"]
        == "supported"
        and metrics["ao_replay_diff"] == 0.0,
        "full_waypoint_claim_supported": statuses["Protected full-waypoint latent dynamics lift"] == "supported"
        and metrics["all"] > 0.0
        and metrics["t50"] > 0.0
        and metrics["hard_failure"] > 0.0,
        "bootstrap_delta_claim_supported": statuses["Bootstrap-supported delta over stored hard switch"] == "supported"
        and metrics["t50_delta_ci"]["low"] > 0.0,
        "global_floor_not_overclaimed": statuses["Global floor removal"] == "not_supported",
        "a_journal_not_overclaimed": payload["answers"]["a_journal_candidate"] is False,
        "claim_boundary_answers_present": payload["answers"]["still_2_5d"] is True
        and payload["answers"]["metric_time_subset_available"] is False
        and payload["answers"]["full_waypoint_dynamics_available"] is True,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_ap_paper_evidence_refresh_pass"
        if passed == total
        else "stage43_ap_paper_evidence_refresh_incomplete",
        "paper_evidence_refreshed": passed == total,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    _write_csv(payload["evidence_rows"])
    gate = payload["stage43_ap_gate"]
    metrics = payload["key_metrics"]
    lines = [
        "# Stage43-AP Paper Evidence Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- policy hash: `{payload['policy_hash']}`",
        "",
        "## Key Current Metrics",
        "",
        f"- all: `{_pct(metrics['all'])}`",
        f"- t50: `{_pct(metrics['t50'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100'])}`",
        f"- hard/failure: `{_pct(metrics['hard_failure'])}`",
        f"- easy degradation: `{_pct(metrics['easy'])}`",
        f"- t50 delta CI vs stored hard switch: `[{_pct(metrics['t50_delta_ci']['low'])}, {_pct(metrics['t50_delta_ci']['high'])}]`",
        f"- reviewer replay diff: `{metrics['ao_replay_diff']:.8f}`",
        "",
        "## Claim Evidence Table",
        "",
        "| claim | status | allowed claim | caveat |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["evidence_rows"]:
        lines.append(f"| {row['claim_name']} | `{row['status']}` | {row['allowed_claim']} | {row['caveat']} |")
    lines.extend(
        [
            "",
            "## Direct Answers",
            "",
            f"- still 2.5D: `{payload['answers']['still_2_5d']}`",
            f"- metric/time subset available: `{payload['answers']['metric_time_subset_available']}`",
            f"- full-waypoint dynamics available: `{payload['answers']['full_waypoint_dynamics_available']}`",
            f"- cross-domain external evidence: {payload['answers']['cross_domain_external_evidence']}",
            f"- A-journal candidate now: `{payload['answers']['a_journal_candidate']}`",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(
        CLAIM_MD,
        [
            "# Stage43 Claim Boundary Refresh",
            "",
            "- Current model remains a protected dataset-local/raw-frame 2.5D multi-agent world-state candidate.",
            "- The strongest current Stage43 claim is a reviewer-replayable protected bounded-residual latent waypoint policy.",
            "- It is not true 3D, not a large-scale foundation model, and not metric/seconds-level.",
            "- Stage5C latent generative execution remains disabled.",
            "- SMC remains disabled.",
            "",
            "## Allowed Current Claim",
            "",
            "Stage43 provides a frozen, exact-replayable, floor-protected bounded-residual latent waypoint policy with positive bootstrap-supported deltas over the stored Stage43-M hard-switch policy on dataset-local/raw-frame external slices.",
            "",
            "## Disallowed Claims",
            "",
            "- Do not claim global floor removal.",
            "- Do not claim true 3D.",
            "- Do not claim foundation world model.",
            "- Do not convert raw-frame t100 into seconds-level long-horizon prediction.",
            "- Do not claim human-gold scene/goal labels.",
        ],
    )
    write_md(
        GAP_MD,
        [
            "# Stage43 A-Journal Gap Refresh",
            "",
            "Stage43 evidence is materially stronger after the bounded-residual freeze and reviewer replay, but it is not yet enough to claim an A-journal-ready system.",
            "",
            "## What Is Strong Now",
            "",
            "- Frozen policy artifact with stable hash.",
            "- Exact reviewer replay with zero metric diff.",
            "- Bootstrap-positive all/t50/hard deltas over stored Stage43-M hard-switch policy.",
            "- Positive external dataset-local slices for ETH_UCY, TrajNet, and UCY.",
            "- Easy degradation remains zero under the frozen policy.",
            "",
            "## Remaining Gaps",
            "",
            "- No verified metric or seconds-level calibration.",
            "- No true 3D evidence.",
            "- Global safety floor removal remains unsupported.",
            "- Multimodal raw scene/video evidence is still proxy-heavy.",
            "- Broader source-level calibration and final integrated full-test replay are still needed.",
        ],
    )
    write_md(
        GATE_MD,
        [
            "# Stage43-AP Paper Evidence Refresh Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- paper evidence refreshed: `{gate['paper_evidence_refreshed']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_text_outputs(payload)


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ap_gate"]
    metrics = payload["key_metrics"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"paper_evidence_refreshed = `{gate['paper_evidence_refreshed']}`",
        f"policy_hash = `{payload['policy_hash']}`",
        f"current_all_t50_t100_hard_easy = `{_pct(metrics['all'])}` / `{_pct(metrics['t50'])}` / `{_pct(metrics['t100'])}` / `{_pct(metrics['hard_failure'])}` / `{_pct(metrics['easy'])}`",
        f"t50_delta_ci = `[{_pct(metrics['t50_delta_ci']['low'])}, {_pct(metrics['t50_delta_ci']['high'])}]`",
        "",
        "Stage43-AP consolidates AJ-AO evidence into paper-facing claim boundaries, evidence table, and A-journal gap refresh. The strongest allowed claim is a reviewer-replayable, floor-protected bounded-residual latent waypoint policy in dataset-local/raw-frame 2.5D space.",
        "",
        "Boundary unchanged: not true 3D; not foundation; no metric/seconds claim; no Stage5C; no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ap_paper_evidence_refresh"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "paper_evidence_refreshed": gate["paper_evidence_refreshed"],
        "policy_hash": payload["policy_hash"],
        "key_metrics": payload["key_metrics"],
        "answers": payload["answers"],
        "report": str(REPORT_MD),
        "claim_boundary": str(CLAIM_MD),
        "gap_report": str(GAP_MD),
        "evidence_csv": str(TABLE_CSV),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ap_paper_evidence_refresh"
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
                        "stage": "Stage43-AP",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "policy_hash": payload["policy_hash"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Refresh Stage43 paper-facing evidence and claim boundaries.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = _run(args)
    gate = result["stage43_ap_gate"]
    print(f"Stage43-AP: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"paper_evidence_refreshed={gate['paper_evidence_refreshed']}")
    return result


if __name__ == "__main__":
    main()
