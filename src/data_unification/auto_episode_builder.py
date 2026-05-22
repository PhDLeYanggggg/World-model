from __future__ import annotations

from typing import Dict

from src.orchestrator.research_state import read_json, write_json, write_md


def summarize_auto_episodes() -> Dict:
    payload = read_json("outputs/reports/stage12_multiagent_episode_report.json", default={}) or {}
    return {
        "source": "stage12_multiagent_episode_report",
        "total_episodes": payload.get("total_episodes", 0),
        "episodes_ge2_agents": payload.get("episodes_ge2_agents", payload.get("episodes_ge2", 0)),
        "verified_t50_episodes": payload.get("verified_t50_episodes", 0),
        "verified_t100_episodes": payload.get("verified_t100_episodes", 0),
    }


def write_auto_episode_report() -> Dict:
    result = summarize_auto_episodes()
    write_json("outputs/reports/auto_episode_report.json", result)
    write_md(
        "outputs/reports/auto_episode_report.md",
        [
            "# Auto Episode Report",
            "",
            f"- total_episodes: `{result['total_episodes']}`",
            f"- episodes_ge2_agents: `{result['episodes_ge2_agents']}`",
            f"- verified_t50_episodes: `{result['verified_t50_episodes']}`",
            f"- verified_t100_episodes: `{result['verified_t100_episodes']}`",
        ],
    )
    return result

