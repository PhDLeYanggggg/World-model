from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(".")
REPORT_DIR = Path("outputs/reports")
STATE_PATH = Path("research_state.json")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: str | Path) -> str:
    path = Path(path)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: str | Path, lines: Iterable[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()
    except Exception:
        return None


@dataclass
class ResearchState:
    current_stage: str = "unknown"
    current_verdict: str = "unknown"
    expert_audit_score: int = 0
    deterministic_ready: bool = False
    latent_generative_ready: bool = False
    smc_ready: bool = False
    pedestrian_long_horizon_ready: bool = False
    scene_annotation_ready: bool = False
    multi_agent_ready: bool = False
    strongest_causal_baselines: Dict[str, Any] = field(default_factory=dict)
    best_learned_models: Dict[str, Any] = field(default_factory=dict)
    datasets_converted: List[str] = field(default_factory=list)
    datasets_registry_only: List[str] = field(default_factory=list)
    datasets_failed: List[str] = field(default_factory=list)
    annotation_status: Dict[str, Any] = field(default_factory=dict)
    gates_passed: List[str] = field(default_factory=list)
    gates_failed: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    blockers_requiring_user: List[str] = field(default_factory=list)
    last_successful_command: str = ""
    last_failed_command: str = ""
    generated_reports: List[str] = field(default_factory=list)
    generated_figures: List[str] = field(default_factory=list)
    git_commit_hash_if_available: str | None = None

    @classmethod
    def load(cls, path: str | Path = STATE_PATH) -> "ResearchState":
        payload = read_json(path, default=None)
        if not isinstance(payload, dict):
            return cls()
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in known})

    def to_dict(self) -> Dict[str, Any]:
        payload = self.__dict__.copy()
        payload["git_commit_hash_if_available"] = self.git_commit_hash_if_available or git_commit_hash()
        return payload

    def save(self, path: str | Path = STATE_PATH) -> None:
        write_json(path, self.to_dict())


def write_research_state_markdown(state: ResearchState, path: str | Path = REPORT_DIR / "research_state.md") -> None:
    payload = state.to_dict()
    passed = [f"- {item}" for item in payload["gates_passed"]] or ["- none"]
    failed = [f"- {item}" for item in payload["gates_failed"]] or ["- none"]
    actions = [f"- {item}" for item in payload["next_actions"]] or ["- none"]
    blockers = [f"- {item}" for item in payload["blockers_requiring_user"]] or ["- none"]
    lines = [
        "# Research State",
        "",
        f"- current_stage: `{payload['current_stage']}`",
        f"- current_verdict: `{payload['current_verdict']}`",
        f"- expert_audit_score: `{payload['expert_audit_score']}`",
        f"- deterministic_ready: `{payload['deterministic_ready']}`",
        f"- latent_generative_ready: `{payload['latent_generative_ready']}`",
        f"- smc_ready: `{payload['smc_ready']}`",
        f"- pedestrian_long_horizon_ready: `{payload['pedestrian_long_horizon_ready']}`",
        f"- scene_annotation_ready: `{payload['scene_annotation_ready']}`",
        f"- multi_agent_ready: `{payload['multi_agent_ready']}`",
        f"- git_commit_hash_if_available: `{payload['git_commit_hash_if_available']}`",
        "",
        "## Gates Passed",
        "",
        *passed,
        "",
        "## Gates Failed",
        "",
        *failed,
        "",
        "## Next Actions",
        "",
        *actions,
        "",
        "## User Blockers",
        "",
        *blockers,
    ]
    write_md(path, lines)
