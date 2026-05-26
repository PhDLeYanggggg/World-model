from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BL_JSON = OUT_DIR / "eth_person_xml_t100_conversion_stage42.json"
REPORT_JSON = OUT_DIR / "eth_person_terms_audit_stage42.json"
REPORT_MD = OUT_DIR / "eth_person_terms_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bm_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_eth_person_terms_stage42.md"

OPENTRAJ_ROOT = Path("external_data/OpenTraj")
OPENTRAJ_LICENSE = OPENTRAJ_ROOT / "LICENSE.txt"
OPENTRAJ_README = OPENTRAJ_ROOT / "README.md"
ETH_README = OPENTRAJ_ROOT / "datasets/ETH/README.md"
ETH_PERSON_ROOT = OPENTRAJ_ROOT / "datasets/ETH-Person"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BM 是 ETH-Person XML terms / official-use audit，不训练模型，不下载数据。",
    "Stage42-BL 的 ETH-Person XML t100 result 是 technical dry-run，terms 未确认前不能升级为 official converted/evaluated result。",
    "OpenTraj 根目录 MIT license 适用于 toolkit/software，不能自动覆盖第三方 trajectory datasets。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t100 仍是 raw-frame diagnostic，不能写成 seconds-level。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _find_terms_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    terms_names = {"readme", "license", "licence", "terms", "copying", "copyright", "notice"}
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stem = path.stem.lower()
        name = path.name.lower()
        if stem in terms_names or any(token in name for token in ["license", "licence", "terms", "copying", "copyright"]):
            hits.append(str(path.relative_to(root)))
    return hits


def _extract_eth_person_official_url(readme_text: str) -> str | None:
    for line in readme_text.splitlines():
        if "ETH-Person" not in line:
            continue
        match = re.search(r"\((https://data\.vision\.ee\.ethz\.ch/cvl/aess/)\)", line)
        if match:
            return match.group(1)
    return None


def _classify_opentraj_license(text: str) -> dict[str, Any]:
    normalized = text.lower()
    is_mit = "the mit license" in normalized and "permission is hereby granted" in normalized
    software_words = ["software", "associated documentation files"]
    dataset_words = ["dataset", "datasets", "trajectory", "annotations"]
    software_scoped = is_mit and any(word in normalized for word in software_words)
    mentions_dataset = any(word in normalized for word in dataset_words)
    return {
        "license_name": "MIT" if is_mit else "unknown",
        "path": str(OPENTRAJ_LICENSE),
        "scope_classification": "software_toolkit_only" if software_scoped and not mentions_dataset else "ambiguous_or_dataset_mentions_present",
        "can_cover_eth_person_dataset": False,
        "reason": "OpenTraj LICENSE uses software/toolkit language and is not accepted as underlying ETH-Person dataset terms.",
    }


def _inspect_local_metadata() -> dict[str, Any]:
    opentraj_license_text = _read_text(OPENTRAJ_LICENSE)
    opentraj_readme_text = _read_text(OPENTRAJ_README)
    eth_readme_text = _read_text(ETH_README)
    eth_person_terms_files = _find_terms_files(ETH_PERSON_ROOT)
    official_url = _extract_eth_person_official_url(opentraj_readme_text)
    eth_readme_has_no_license = "no license information is available with this dataset" in eth_readme_text.lower()
    eth_person_data_files = sorted(
        str(path.relative_to(ETH_PERSON_ROOT))
        for path in ETH_PERSON_ROOT.rglob("*")
        if path.is_file()
    ) if ETH_PERSON_ROOT.exists() else []
    return {
        "source": "fresh_local_terms_metadata_audit",
        "opentraj_license": _classify_opentraj_license(opentraj_license_text),
        "opentraj_readme_path": str(OPENTRAJ_README),
        "eth_readme_path": str(ETH_README),
        "eth_readme_has_no_license_statement": eth_readme_has_no_license,
        "eth_person_root": str(ETH_PERSON_ROOT),
        "eth_person_root_exists": ETH_PERSON_ROOT.exists(),
        "eth_person_terms_files_found": eth_person_terms_files,
        "eth_person_local_terms_found": len(eth_person_terms_files) > 0,
        "eth_person_data_files": eth_person_data_files,
        "official_url_from_opentraj_readme": official_url,
        "official_source_url_recorded": official_url is not None,
        "official_terms_verified": False,
        "official_terms_verification_method": "not_verified_from_local_files; user_or_official_terms_confirmation_required",
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bl = _load_json(BL_JSON)
    metadata = _inspect_local_metadata()
    bl_summary = bl.get("summary", {})
    claim_boundary = {
        "true_3d": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "t100_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
        "official_converted_dataset_claim_allowed": False,
        "deployable_t100_claim_allowed": False,
        "global_t100_positive_claim_allowed": False,
        "eth_person_terms_confirmed": False,
    }
    summary = {
        "source": "fresh_terms_audit_from_local_metadata",
        "bl_verdict": bl.get("stage42_bl_gate", {}).get("verdict"),
        "bl_technical_t100_all_folds_safe_positive": bool(bl_summary.get("technical_t100_all_folds_safe_positive", False)),
        "bl_technical_t100_mean_improvement_vs_fallback": bl_summary.get("technical_t100_mean_improvement_vs_fallback"),
        "bl_technical_t100_min_improvement_vs_fallback": bl_summary.get("technical_t100_min_improvement_vs_fallback"),
        "bl_technical_t100_max_easy_degradation": bl_summary.get("technical_t100_max_easy_degradation"),
        "opentraj_toolkit_license_is_not_dataset_license": not metadata["opentraj_license"]["can_cover_eth_person_dataset"],
        "eth_person_local_terms_found": metadata["eth_person_local_terms_found"],
        "official_source_url_recorded": metadata["official_source_url_recorded"],
        "official_terms_verified": metadata["official_terms_verified"],
        "license_terms_confirmed": False,
        "user_action_required": True,
        "auto_download_executed": False,
        "official_converted_dataset_claim_allowed": False,
        "deployable_t100_claim_allowed": False,
        "global_t100_positive_claim_allowed": False,
        "next_stage_official_conversion_allowed": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_eth_person_terms_audit",
        "stage": "Stage42-BM ETH-Person Terms Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BL_JSON), str(OPENTRAJ_LICENSE), str(OPENTRAJ_README), str(ETH_README)]),
        "current_facts": CURRENT_FACTS,
        "bl_input": {
            "path": str(BL_JSON),
            "verdict": bl.get("stage42_bl_gate", {}).get("verdict"),
            "source": bl.get("source"),
            "summary": bl_summary,
        },
        "local_metadata_audit": metadata,
        "summary": summary,
        "claim_boundary": claim_boundary,
    }
    payload["stage42_bm_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    metadata = payload["local_metadata_audit"]
    claim = payload["claim_boundary"]
    gates = {
        "bl_input_verified": summary["bl_verdict"] == "stage42_bl_eth_person_xml_t100_dry_run_pass",
        "technical_result_preserved": summary["bl_technical_t100_all_folds_safe_positive"] is True,
        "opentraj_license_inspected": metadata["opentraj_license"]["license_name"] == "MIT",
        "opentraj_license_not_treated_as_dataset_license": summary["opentraj_toolkit_license_is_not_dataset_license"] is True,
        "eth_person_local_terms_checked": metadata["eth_person_root_exists"] is True,
        "eth_person_terms_not_overclaimed": summary["license_terms_confirmed"] is False,
        "official_url_recorded_or_blocker": metadata["official_source_url_recorded"] is True,
        "official_conversion_blocked_without_terms": claim["official_converted_dataset_claim_allowed"] is False,
        "deployable_t100_blocked_without_terms": claim["deployable_t100_claim_allowed"] is False,
        "user_action_generated": summary["user_action_required"] is True,
        "no_auto_download": summary["auto_download_executed"] is False,
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False and claim["t100_seconds_claim"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_bm_eth_person_terms_audit_pass_claim_blocked" if passed == total else "stage42_bm_eth_person_terms_audit_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    metadata = payload["local_metadata_audit"]
    bl_summary = payload["bl_input"]["summary"]
    lines = [
        "# Stage42-BM ETH-Person Terms / Official-Use Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bm_gate']['passed']} / {payload['stage42_bm_gate']['total']}`",
        f"- verdict: `{payload['stage42_bm_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Stage42-BL Technical Result Preserved",
        "",
        f"- BL verdict: `{summary['bl_verdict']}`",
        f"- technical_t100_all_folds_safe_positive: `{summary['bl_technical_t100_all_folds_safe_positive']}`",
        f"- technical_t100_mean_improvement_vs_fallback: `{summary['bl_technical_t100_mean_improvement_vs_fallback']}`",
        f"- technical_t100_min_improvement_vs_fallback: `{summary['bl_technical_t100_min_improvement_vs_fallback']}`",
        f"- technical_t100_max_easy_degradation: `{summary['bl_technical_t100_max_easy_degradation']}`",
        f"- BL t100 windows total: `{bl_summary.get('t100_windows_total')}`",
        "",
        "## Local Terms / License Audit",
        "",
        f"- OpenTraj license path: `{metadata['opentraj_license']['path']}`",
        f"- OpenTraj license name: `{metadata['opentraj_license']['license_name']}`",
        f"- OpenTraj license scope classification: `{metadata['opentraj_license']['scope_classification']}`",
        f"- OpenTraj toolkit license can cover ETH-Person dataset: `{metadata['opentraj_license']['can_cover_eth_person_dataset']}`",
        f"- ETH README has no-license statement: `{metadata['eth_readme_has_no_license_statement']}`",
        f"- ETH-Person local terms files found: `{metadata['eth_person_terms_files_found']}`",
        f"- ETH-Person official URL from OpenTraj README: `{metadata['official_url_from_opentraj_readme']}`",
        f"- official terms verified: `{metadata['official_terms_verified']}`",
        "",
        "## Claim Boundary",
        "",
        f"- license_terms_confirmed: `{summary['license_terms_confirmed']}`",
        f"- official_converted_dataset_claim_allowed: `{summary['official_converted_dataset_claim_allowed']}`",
        f"- deployable_t100_claim_allowed: `{summary['deployable_t100_claim_allowed']}`",
        f"- global_t100_positive_claim_allowed: `{summary['global_t100_positive_claim_allowed']}`",
        f"- next_stage_official_conversion_allowed: `{summary['next_stage_official_conversion_allowed']}`",
        "",
        "## Interpretation",
        "",
        "- Stage42-BL remains useful technical evidence: the XML loader/source-CV path works and is strongly positive under dry-run conditions.",
        "- The local repository does not include ETH-Person-specific license or terms files.",
        "- The OpenTraj MIT license is treated as toolkit/software license only; it is not accepted as permission for the underlying ETH-Person dataset.",
        "- Therefore ETH-Person XML cannot be counted as official converted/evaluated data until the user confirms official terms or provides an official permission/terms link.",
        "- This audit intentionally blocks deployable/global t100 claims despite the positive technical result.",
    ]
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    metadata = payload["local_metadata_audit"]
    return [
        "# Stage42-BM User Action Required: ETH-Person Terms",
        "",
        f"- source: `{payload['source']}`",
        "",
        "## Required Before Official Use",
        "",
        "Please confirm ETH-Person dataset terms before it is used as official converted/evaluated evidence.",
        "",
        "Accepted confirmation examples:",
        "",
        "- Provide the official ETH-Person license / terms page.",
        "- Confirm that the official ETH-Person page permits this research use and derived feature/evaluation reports.",
        "- Provide a written permission statement or citation from the dataset owner/page.",
        "",
        "## Current Local Evidence",
        "",
        f"- local ETH-Person path: `{metadata['eth_person_root']}`",
        f"- local ETH-Person terms files found: `{metadata['eth_person_terms_files_found']}`",
        f"- official URL referenced by OpenTraj README: `{metadata['official_url_from_opentraj_readme']}`",
        "- OpenTraj root MIT license is software/toolkit-only for this audit and is not treated as ETH-Person dataset permission.",
        "",
        "## Until Confirmed",
        "",
        "- Keep Stage42-BL as `technical_dry_run_terms_unverified`.",
        "- Do not count ETH-Person XML as official converted/evaluated data.",
        "- Do not claim deployable/global t100 success from ETH-Person XML.",
        "- Do not report dataset-local/raw-frame horizons as metric or seconds-level.",
    ]


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bm_gate"]
    lines = [
        "# Stage42-BM Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def run_stage42_eth_person_terms_audit() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    run_stage42_eth_person_terms_audit()
