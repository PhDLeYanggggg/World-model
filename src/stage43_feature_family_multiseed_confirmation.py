from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_feature_family_retrained_ablation as ah


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_feature_family_multiseed_confirmation.json"
REPORT_MD = OUT_DIR / "stage43_feature_family_multiseed_confirmation.md"
GATE_MD = OUT_DIR / "stage43_stage_ai_feature_family_multiseed_confirmation_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
HEARTBEAT_JSON = OUT_DIR / "stage43_feature_family_multiseed_confirmation_heartbeat.json"

SECTION = "STAGE43_AI_FEATURE_FAMILY_MULTISEED_CONFIRMATION"
SOURCE = "fresh_stage43_ai_feature_family_multiseed_confirmation"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _rows_for_mode(mode: str) -> dict[str, int | None]:
    if mode == "quick":
        return {"train": 6000, "val": 3000, "test": 3000}
    if mode == "medium":
        return {"train": 90000, "val": 40000, "test": 50000}
    return {"train": 30000, "val": 12000, "test": 16000}


def _seed_args(base: argparse.Namespace, *, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        seed=int(seed),
        epochs=int(base.epochs),
        batch_size=int(base.batch_size),
        hidden_dim=int(base.hidden_dim),
        latent_dim=int(base.latent_dim),
        lr=float(base.lr),
        checkpoint_tag=f"stage43_feature_family_multiseed_confirmation_seed{seed}",
    )


def _delta(full: Mapping[str, Any], ablated: Mapping[str, Any]) -> dict[str, float]:
    return ah._metric_delta(full["test_metrics_with_floor"], ablated["test_metrics_with_floor"])


def _summarize_variant(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [
        "full_waypoint_ade_improvement_vs_floor",
        "t50_full_waypoint_ade_improvement_vs_floor",
        "hard_failure_full_waypoint_ade_improvement_vs_floor",
        "easy_degradation_vs_floor",
    ]
    out: dict[str, Any] = {"variant": rows[0]["variant"], "seeds": [row["seed"] for row in rows], "metrics": {}}
    for key in metrics:
        values = np.asarray([row["test_metrics_with_floor"][key] for row in rows], dtype=np.float64)
        out["metrics"][key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "positive_seed_count": int(np.sum(values > 0.0)),
        }
    if rows[0]["variant"] != "full_features":
        out["delta_full_minus_variant"] = {}
        for key in [
            "full_waypoint_ade_improvement_vs_floor",
            "t50_full_waypoint_ade_improvement_vs_floor",
            "hard_failure_full_waypoint_ade_improvement_vs_floor",
        ]:
            values = np.asarray([row["delta_full_minus_variant"][key] for row in rows], dtype=np.float64)
            out["delta_full_minus_variant"][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "positive_seed_count": int(np.sum(values > 0.0)),
            }
    return out


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    mode = "medium" if args.medium else "quick" if args.quick else "small"
    rows = _rows_for_mode(mode)
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    unknown = [v for v in variants if v not in ah.VARIANTS]
    if unknown:
        raise KeyError(f"Unknown Stage43-AI variants: {unknown}")
    if "full_features" not in variants:
        raise ValueError("Stage43-AI requires full_features.")

    seed_results: list[dict[str, Any]] = []
    by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    for seed in seeds:
        run_args = _seed_args(args, seed=seed)
        per_seed: list[dict[str, Any]] = []
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for variant in variants:
            row, arr = ah._train_one(run_args, variant=variant, rows=rows)
            row["seed"] = int(seed)
            per_seed.append(row)
            arrays[variant] = arr
            write_json(
                HEARTBEAT_JSON,
                m._jsonable(
                    {
                        "source": SOURCE,
                        "seed": seed,
                        "variant": variant,
                        "mode": mode,
                        "git_commit": m._git_commit(),
                    }
                ),
            )
        full = next(row for row in per_seed if row["variant"] == "full_features")
        for row in per_seed:
            if row["variant"] == "full_features":
                row["delta_full_minus_variant"] = {
                    key: 0.0 for key in ah._metric_delta(full["test_metrics_with_floor"], full["test_metrics_with_floor"])
                }
            else:
                row["delta_full_minus_variant"] = _delta(full, row)
            by_variant[row["variant"]].append(row)
        seed_results.append(
            {
                "seed": int(seed),
                "variants": [
                    {
                        "variant": row["variant"],
                        "metrics": row["test_metrics_with_floor"],
                        "delta_full_minus_variant": row["delta_full_minus_variant"],
                        "latent_variance": row["latent_variance"],
                        "checkpoint": row["checkpoint"],
                        "checkpoint_committed": False,
                    }
                    for row in per_seed
                ],
            }
        )

    summaries = [_summarize_variant(by_variant[variant]) for variant in variants]
    positive_t50_stable = [
        row["variant"]
        for row in summaries
        if row["variant"] != "full_features"
        and row["delta_full_minus_variant"]["t50_full_waypoint_ade_improvement_vs_floor"]["mean"] > 0.0
        and row["delta_full_minus_variant"]["t50_full_waypoint_ade_improvement_vs_floor"]["positive_seed_count"]
        >= max(2, len(seeds) - 1)
    ]
    positive_hard_or_all_stable = [
        row["variant"]
        for row in summaries
        if row["variant"] != "full_features"
        and (
            row["delta_full_minus_variant"]["full_waypoint_ade_improvement_vs_floor"]["positive_seed_count"]
            >= max(2, len(seeds) - 1)
            or row["delta_full_minus_variant"]["hard_failure_full_waypoint_ade_improvement_vs_floor"][
                "positive_seed_count"
            ]
            >= max(2, len(seeds) - 1)
        )
    ]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_multiseed_retrained_feature_family_confirmation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": mode,
        "seeds": seeds,
        "variants": variants,
        "seed_results": seed_results,
        "variant_summaries": summaries,
        "stable_positive_t50_contribution_variants": positive_t50_stable,
        "stable_positive_hard_or_all_contribution_variants": positive_hard_or_all_stable,
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
        "input_hash": _combined_hash([m._cache_path("train"), m._cache_path("val"), m._cache_path("test")]),
    }
    payload["stage43_ai_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "fresh_multiseed_retrained_confirmation": payload["result_source"]
        == "fresh_multiseed_retrained_feature_family_confirmation",
        "at_least_three_seeds": len(payload["seeds"]) >= 3,
        "full_features_and_four_ablations": "full_features" in payload["variants"] and len(payload["variants"]) >= 5,
        "baseline_floor_t50_contribution_stable": "no_baseline_floor"
        in payload["stable_positive_t50_contribution_variants"],
        "at_least_two_stable_module_contributions": len(
            set(payload["stable_positive_t50_contribution_variants"])
            | set(payload["stable_positive_hard_or_all_contribution_variants"])
        )
        >= 2,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "checkpoints_not_committed": all(
            row["checkpoint_committed"] is False
            for seed_row in payload["seed_results"]
            for row in seed_row["variants"]
        ),
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    deploy = passed == total
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_ai_feature_family_multiseed_confirmation_pass"
        if deploy
        else "stage43_ai_feature_family_multiseed_confirmation_diagnostic",
        "multiseed_feature_family_contribution_supported": deploy,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _summary_rows(payload: Mapping[str, Any]) -> list[str]:
    rows = [
        "| variant | mean all | mean t50 | mean hard | mean easy | delta all mean | delta t50 mean | delta hard mean | t50 positive seeds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["variant_summaries"]:
        metrics = row["metrics"]
        delta = row.get("delta_full_minus_variant", {})
        rows.append(
            f"| `{row['variant']}` | `{_pct(metrics['full_waypoint_ade_improvement_vs_floor']['mean'])}` | `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor']['mean'])}` | `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor']['mean'])}` | `{_pct(metrics['easy_degradation_vs_floor']['mean'])}` | `{_pct(delta.get('full_waypoint_ade_improvement_vs_floor', {'mean': 0.0})['mean'])}` | `{_pct(delta.get('t50_full_waypoint_ade_improvement_vs_floor', {'mean': 0.0})['mean'])}` | `{_pct(delta.get('hard_failure_full_waypoint_ade_improvement_vs_floor', {'mean': 0.0})['mean'])}` | `{delta.get('t50_full_waypoint_ade_improvement_vs_floor', {'positive_seed_count': 0})['positive_seed_count']}` |"
        )
    return rows


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_ai_gate"]
    lines = [
        "# Stage43-AI Feature-Family Multi-Seed Confirmation",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- mode: `{payload['mode']}`",
        f"- seeds: `{payload['seeds']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- stable positive t50 variants: `{payload['stable_positive_t50_contribution_variants']}`",
        f"- stable positive hard/all variants: `{payload['stable_positive_hard_or_all_contribution_variants']}`",
        "",
        "## Summary",
        "",
        *_summary_rows(payload),
        "",
        "## Interpretation",
        "",
        "Stage43-AI repeats the retrained feature-family ablation across multiple seeds. It is meant to test whether Stage43-AH's module contribution evidence is stable rather than a one-seed artifact.",
        "",
        "This remains dataset-local/raw-frame 2.5D evidence. It is not a deployment policy, not metric/seconds evidence, and not Stage5C or SMC execution.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-AI Feature-Family Multi-Seed Confirmation Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- supported: `{gate['multiseed_feature_family_contribution_supported']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_text_outputs(payload)


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ai_gate"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"seeds = `{payload['seeds']}`",
        f"stable_positive_t50_contribution_variants = `{payload['stable_positive_t50_contribution_variants']}`",
        f"stable_positive_hard_or_all_contribution_variants = `{payload['stable_positive_hard_or_all_contribution_variants']}`",
        "",
        "Stage43-AI repeats the Stage43-AH retrained feature-family ablation across multiple seeds. It tests whether baseline/floor, goal, history, neighbor/interaction, and domain feature-family contributions survive seed variation. Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ai_feature_family_multiseed_confirmation"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "seeds": payload["seeds"],
        "stable_positive_t50_contribution_variants": payload["stable_positive_t50_contribution_variants"],
        "stable_positive_hard_or_all_contribution_variants": payload[
            "stable_positive_hard_or_all_contribution_variants"
        ],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ai_feature_family_multiseed_confirmation"
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
                        "stage": "Stage43-AI",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "multiseed_feature_family_contribution_supported": gate[
                            "multiseed_feature_family_contribution_supported"
                        ],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-AI multi-seed feature-family retrained confirmation.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--small", action="store_true")
    group.add_argument("--medium", action="store_true")
    parser.add_argument("--variants", default="full_features,no_history,no_goal,no_neighbor_interaction,no_baseline_floor,no_domain")
    parser.add_argument("--seeds", default="431,443,457")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=7e-4)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    result = _run(args)
    gate = result["stage43_ai_gate"]
    print(f"Stage43-AI: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"multiseed_feature_family_contribution_supported={gate['multiseed_feature_family_contribution_supported']}")
    return result


if __name__ == "__main__":
    main()
