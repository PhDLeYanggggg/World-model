from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_full_suite_replay_audit.json"
REPORT_MD = OUT_DIR / "stage43_full_suite_replay_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_ar_full_suite_replay_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AR_FULL_SUITE_REPLAY_AUDIT"
SOURCE = "fresh_stage43_ar_full_suite_replay_audit"

SUMMARY_RE = re.compile(
    r"=+\s*(?P<body>.*?(?:passed|failed|error|errors|skipped|xfailed|xpassed).*?)\s+in\s+(?P<seconds>[0-9.]+)s(?:\s+\([^)]*\))?\s*=+"
)
COUNT_RE = re.compile(r"(?P<count>\d+)\s+(?P<kind>passed|failed|errors?|skipped|xfailed|xpassed)")


def _parse_pytest_summary(output: str) -> dict[str, Any]:
    matches = list(SUMMARY_RE.finditer(output))
    if not matches:
        return {
            "found": False,
            "summary_line": "",
            "duration_seconds": None,
            "counts": {},
        }
    match = matches[-1]
    body = match.group("body").strip()
    seconds = float(match.group("seconds"))
    counts: dict[str, int] = {}
    for count_match in COUNT_RE.finditer(body):
        kind = count_match.group("kind")
        if kind == "error":
            kind = "errors"
        counts[kind] = counts.get(kind, 0) + int(count_match.group("count"))
    return {
        "found": True,
        "summary_line": f"{body} in {seconds:.2f}s",
        "duration_seconds": seconds,
        "counts": counts,
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _run_pytest(command: list[str], timeout_seconds: int | None) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        timed_out = False
        return_code = int(proc.returncode)
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
    elapsed = time.time() - started
    combined = stdout + "\n" + stderr
    summary = _parse_pytest_summary(combined)
    return {
        "command": command,
        "return_code": return_code,
        "timed_out": timed_out,
        "elapsed_wall_seconds": elapsed,
        "stdout_tail": stdout[-8000:],
        "stderr_tail": stderr[-4000:],
        "pytest_summary": summary,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    run = payload["pytest_run"]
    summary = run["pytest_summary"]
    counts = summary.get("counts", {})
    claim = payload["claim_boundary"]
    gates = {
        "pytest_command_recorded": bool(run["command"]),
        "pytest_summary_found": summary.get("found") is True,
        "pytest_exit_zero": run["return_code"] == 0,
        "pytest_not_timed_out": run["timed_out"] is False,
        "passed_tests_positive": int(counts.get("passed", 0)) > 0,
        "no_failed_or_error_tests": int(counts.get("failed", 0)) == 0 and int(counts.get("errors", 0)) == 0,
        "runtime_recorded": run["elapsed_wall_seconds"] > 0,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "goal_kept_active": payload["goal_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_ar_full_suite_replay_pass" if passed == total else "stage43_ar_full_suite_replay_incomplete",
        "full_suite_replay_passed": passed == total,
        "goal_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_ar_gate"]
    run = payload["pytest_run"]
    summary = run["pytest_summary"]
    counts = summary.get("counts", {})
    lines = [
        "# Stage43-AR Full Test-Suite Replay Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- full suite replay passed: `{gate['full_suite_replay_passed']}`",
        f"- goal complete: `{gate['goal_complete']}`",
        "",
        "## Command",
        "",
        f"`{' '.join(run['command'])}`",
        "",
        "## Result",
        "",
        f"- return code: `{run['return_code']}`",
        f"- timed out: `{run['timed_out']}`",
        f"- wall seconds: `{run['elapsed_wall_seconds']:.2f}`",
        f"- pytest summary found: `{summary.get('found')}`",
        f"- pytest summary: `{summary.get('summary_line', '')}`",
        f"- pytest duration seconds: `{summary.get('duration_seconds')}`",
        f"- passed: `{counts.get('passed', 0)}`",
        f"- failed: `{counts.get('failed', 0)}`",
        f"- errors: `{counts.get('errors', 0)}`",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
        "",
        "## Claim Boundary",
        "",
        "- This audit is a software/reproducibility replay only.",
        "- It does not execute Stage5C.",
        "- It does not enable SMC.",
        "- It does not create a metric, seconds-level, true-3D, or foundation-model claim.",
    ]
    write_md(REPORT_MD, lines)
    write_md(GATE_MD, lines)
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ar_gate"]
    run = payload["pytest_run"]
    summary = run["pytest_summary"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"full_suite_replay_passed = `{gate['full_suite_replay_passed']}`",
        f"pytest_summary = `{summary.get('summary_line', '')}`",
        f"wall_seconds = `{run['elapsed_wall_seconds']:.2f}`",
        "",
        "Stage43-AR records a fresh full test-suite replay using the active arm64 Python runtime. It is a reproducibility/software health audit only; it does not change model claims, execute Stage5C, enable SMC, or create metric/seconds/true-3D/foundation evidence.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ar_full_suite_replay_audit"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "full_suite_replay_passed": gate["full_suite_replay_passed"],
        "pytest_summary": summary,
        "elapsed_wall_seconds": run["elapsed_wall_seconds"],
        "report": str(REPORT_MD),
        "goal_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ar_full_suite_replay_audit"
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
                        "stage": "Stage43-AR",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "pytest_summary": summary.get("summary_line", ""),
                        "goal_complete": False,
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    if args.reparse_existing:
        previous = read_json(REPORT_JSON, {})
        if not previous:
            raise FileNotFoundError(REPORT_JSON)
        pytest_run = dict(previous["pytest_run"])
        combined = str(pytest_run.get("stdout_tail", "")) + "\n" + str(pytest_run.get("stderr_tail", ""))
        pytest_run["pytest_summary"] = _parse_pytest_summary(combined)
        result_source = "fresh_full_test_suite_replay_reparsed_from_existing_capture"
    else:
        command = [sys.executable, "-m", "pytest", "tests"]
        pytest_run = _run_pytest(command, None if args.timeout_seconds <= 0 else int(args.timeout_seconds))
        result_source = "fresh_full_test_suite_replay"
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": result_source,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "pytest_run": pytest_run,
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "goal_complete": False,
    }
    payload["stage43_ar_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and record a fresh full pytest replay for Stage43.")
    parser.add_argument("--timeout-seconds", type=int, default=0, help="Optional timeout. 0 means no timeout.")
    parser.add_argument(
        "--reparse-existing",
        action="store_true",
        help="Do not rerun pytest; reparse the existing captured Stage43-AR stdout/stderr after parser fixes.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = _run(args)
    gate = result["stage43_ar_gate"]
    print(f"Stage43-AR: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(result["pytest_run"]["pytest_summary"].get("summary_line", "pytest summary unavailable"))
    return result


if __name__ == "__main__":
    main()
