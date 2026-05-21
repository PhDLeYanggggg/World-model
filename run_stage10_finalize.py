from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.stage10_common import REPORT_DIR, STAGE10_RESULTS_DIR, ensure_dir, read_json, write_json


def main() -> None:
    data = read_json(REPORT_DIR / "stage10_data_audit.json", [])
    gates = read_json(REPORT_DIR / "world_model_gate_stage10.json", {})
    ann = read_json(REPORT_DIR / "stage10_annotation_report.json", {"annotations": []})
    packs = read_json(REPORT_DIR / "stage10_scene_pack_report.json", {})
    eps = read_json(REPORT_DIR / "stage10_multiagent_episode_report.json", {"datasets": []})
    hard = read_json(REPORT_DIR / "stage10_hard_failure_report.json", {"summary": {}}).get("summary", {})
    goal = read_json(REPORT_DIR / "stage10_goalbench_v3_report.json", {})
    summary = {
        "pedestrian_drone_loaded": [r["dataset_name"] for r in data if r.get("eligible_for_stage10")],
        "verified_t50_t100_sources": [r["dataset_name"] for r in data if r.get("actual_verified_t50") or r.get("actual_verified_t100")],
        "human_confirmed_scenes": sum(r.get("annotation_quality") in {"gold_human", "silver_human_confirmed"} for r in ann.get("annotations", [])),
        "silver_rule_confirmed_scenes": sum(r.get("annotation_quality") == "silver_rule_confirmed" for r in ann.get("annotations", [])),
        "scene_packs_with_goals": packs.get("scenes_with_goals", 0),
        "multiagent_ge2": sum(r.get("episodes_ge2_agents", 0) for r in eps.get("datasets", [])),
        "hard_failure_total": hard.get("total_records", 0),
        "goalbench_official": goal.get("official_records_count", 0),
        "stage11_ready": gates.get("stage11_ready", False),
        "expert_audit_score": gates.get("expert_audit_score", 0),
        "verdict": gates.get("verdict", "unknown"),
    }
    write_reports(summary, gates)
    package_results()
    print(json.dumps({"stage10_results": str(STAGE10_RESULTS_DIR), **summary}, indent=2))


def write_reports(summary: dict, gates: dict) -> None:
    final = f"""# Stage 10 Final Report

Stage 10 is a data acquisition, human-in-the-loop annotation, and benchmark packaging stage. It does not train a new model, does not enable latent generative modeling, and does not enable SMC.

## Direct Answers

1. 是否接入真实 pedestrian/drone 数据：{'是' if summary['pedestrian_drone_loaded'] else '否'} ({summary['pedestrian_drone_loaded']})
2. 是否补上 verified t+50/t+100：{'是' if summary['verified_t50_t100_sources'] else '否'} ({summary['verified_t50_t100_sources']})
3. 是否建立 human-confirmed gold/silver annotations：{'是' if summary['human_confirmed_scenes'] >= 3 else '否'} (human_confirmed={summary['human_confirmed_scenes']})
4. 是否仍主要依赖 rule-confirmed silver：{'是' if summary['silver_rule_confirmed_scenes'] > summary['human_confirmed_scenes'] else '否'} (silver_rule_confirmed={summary['silver_rule_confirmed_scenes']})
5. 是否建立 usable scene packs：{'是' if summary['scene_packs_with_goals'] >= 3 else '否'} (scene_packs_with_goals={summary['scene_packs_with_goals']})
6. 是否扩展 multi-agent episodes：{'是' if summary['multiagent_ge2'] > 0 else '否'} (episodes_ge2={summary['multiagent_ge2']})
7. 是否扩展 hard/failure episodes：{'是' if summary['hard_failure_total'] > 0 else '否'} (records={summary['hard_failure_total']})
8. GoalBench v3 official records 是否足够：{'是' if summary['goalbench_official'] >= 500 else '否'} (official={summary['goalbench_official']})
9. 是否可以进入 Stage 11 training：{'是' if summary['stage11_ready'] else '否'}
10. 是否仍禁止 Stage 5C latent generative：是
11. 是否仍禁止 SMC：是

## Final Conclusion

项目是否跑通：是
pedestrian/drone 数据是否接入：{'是' if summary['pedestrian_drone_loaded'] else '否'}
verified pedestrian/drone t+50/t+100 是否补上：{'是' if summary['verified_t50_t100_sources'] else '否'}
human-confirmed annotation 是否建立：{'是' if summary['human_confirmed_scenes'] >= 3 else '否'}
scene packs 是否可用于 official training：部分
multi-agent episodes 是否足够：{'是' if summary['multiagent_ge2'] >= 500 else '部分'}
hard/failure episodes 是否足够：{'是' if summary['hard_failure_total'] >= 100 else '部分'}
GoalBench v3 是否足够：{'是' if summary['goalbench_official'] >= 500 else '否'}
是否可以进入 Stage 11：{'是' if summary['stage11_ready'] else '否'}
是否可以进入 Stage 5C latent generative：否
是否可以启用 SMC：否
当前 verdict：{summary['verdict']}
expert audit score：{summary['expert_audit_score']}

如果不能进入 Stage 11，下一步先修什么：

1. 人工确认至少 3 个 scenes，把 silver_rule_confirmed 升级为 silver_human_confirmed 或 gold_human。
2. 接入 SDD/OpenTraj 等真实 pedestrian/drone 长轨迹，补 verified t+50/t+100。
3. 扩展 multi-agent episodes 到 500+，并扩展 hard/failure records 到 100+ official human-confirmed records。
"""
    (REPORT_DIR / "report_stage10_final.md").write_text(final, encoding="utf-8")
    (REPORT_DIR / "data_card_stage10.md").write_text(
        f"""# Stage 10 Data Card

- Loaded pedestrian/drone sources: {summary['pedestrian_drone_loaded']}
- Verified pedestrian/drone t+50/t+100 sources: {summary['verified_t50_t100_sources']}
- Multi-agent episodes >=2 agents: {summary['multiagent_ge2']}
- Hard/failure records: {summary['hard_failure_total']}
- Metric warning: dataset-coordinate and pixel-space sources are not true metric world coordinates unless homography/scale is supplied.
- License warning: SDD remains non-commercial and is not downloaded automatically.
""",
        encoding="utf-8",
    )
    (REPORT_DIR / "annotation_card_stage10.md").write_text(
        f"""# Stage 10 Annotation Card

- Human-confirmed scenes: {summary['human_confirmed_scenes']}
- Rule-confirmed silver scenes: {summary['silver_rule_confirmed_scenes']}
- Rule-confirmed silver is not human gold.
- Annotation tasks are not completed annotations until reviewed by a person.
- Test endpoints are not used to construct candidate goals.
""",
        encoding="utf-8",
    )
    (REPORT_DIR / "stage10_next_steps.md").write_text(
        """# Stage 10 Next Steps

1. Review the top Batch A annotation tasks and promote valid scenes to silver_human_confirmed or gold_human.
2. Provide local SDD/OpenTraj path and run Stage 10 prepare/audit again.
3. Rebuild scene packs, GoalBench v3, hard/failure records, and gates before Stage 11.
""",
        encoding="utf-8",
    )
    write_json(REPORT_DIR / "stage10_final_summary.json", summary)


def package_results() -> None:
    ensure_dir(STAGE10_RESULTS_DIR)
    for name in ["reports", "annotations", "annotation_tasks", "scene_packs", "multiagent_episodes", "goalbench_v3", "hard_failure"]:
        ensure_dir(STAGE10_RESULTS_DIR / name)
    for report in Path("outputs/reports").glob("*stage10*"):
        shutil.copy2(report, STAGE10_RESULTS_DIR / "reports" / report.name)
    for src, dst in [
        ("data/stage10_annotations", "annotations"),
        ("data/stage10_annotation_tasks", "annotation_tasks"),
        ("data/stage10_scene_packs", "scene_packs"),
        ("data/stage10_multiagent_episodes", "multiagent_episodes"),
        ("data/stage10_goalbench_v3", "goalbench_v3"),
        ("data/stage10_hard_failure", "hard_failure"),
    ]:
        src_path = Path(src)
        if src_path.exists():
            dst_path = STAGE10_RESULTS_DIR / dst
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)


if __name__ == "__main__":
    main()
