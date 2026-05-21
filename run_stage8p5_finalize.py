from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict


REPORT_DIR = Path("outputs/reports")
PACKAGE = Path("outputs/world_model_stage8p5_results")


def load_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def main() -> None:
    gate = load_json(REPORT_DIR / "world_model_gate_stage8p5.json", {})
    data = load_json(REPORT_DIR / "stage8p5_data_audit.json", [])
    ann = load_json(REPORT_DIR / "stage8p5_annotation_report.json", {"annotations": []})
    packs = load_json(REPORT_DIR / "stage8p5_scene_gold_pack_report.json", {})
    eps = load_json(REPORT_DIR / "stage8p5_per_agent_episode_report.json", {"datasets": []})
    goal = load_json(REPORT_DIR / "stage8p5_goalbench_gold_v2_report.json", {})
    write_reports(gate, data, ann, packs, eps, goal)
    package_outputs()
    print(f"Stage 8.5 package: {PACKAGE.resolve()}")


def write_reports(gate: Dict, data, ann: Dict, packs: Dict, eps: Dict, goal: Dict) -> None:
    gold = int(packs.get("number_of_gold_scenes", 0))
    silver = int(packs.get("number_of_silver_scenes", 0))
    inferred = int(packs.get("number_of_inferred_only_scenes", 0))
    ge2 = sum(int(row.get("episodes_ge2_agents", 0)) for row in eps.get("datasets", []))
    ped_loaded = [r["dataset_name"] for r in data if r.get("eligible_for_stage8p5_gate")]
    long_loaded = [r["dataset_name"] for r in data if r.get("eligible_for_stage8p5_gate") and (r.get("actual_verified_t50") or r.get("actual_verified_t100"))]
    official = int(goal.get("official_gold_silver_records", 0))
    final = [
        "# Stage 8.5 Final Report",
        "",
        "Stage 8.5 is a data/annotation/per-agent preparation sprint. It does not train new residual models, does not enable latent generative modeling, and does not enable SMC.",
        "",
        "## Direct Answers",
        "",
        f"1. 是否接入真实 pedestrian/drone 数据：{'是' if ped_loaded else '否'} ({ped_loaded})",
        f"2. 是否有 verified t+50/t+100：{'是' if long_loaded else '否'} ({long_loaded})",
        f"3. 是否建立 gold/silver annotation：{'是' if gold + silver > 0 else '否'} (gold={gold}, silver={silver}, inferred_only={inferred})",
        f"4. 是否仍是 inferred-only：{'否' if gold + silver > 0 else '是'}",
        f"5. 是否建立 per-agent multi-agent episodes：{'是' if ge2 >= 50 else '部分'} ({ge2} episodes >=2 agents)",
        f"6. GoalBench-Gold v2 official records：{official}",
        f"7. 是否可进入 Stage 9：{'是' if gate.get('stage9_ready') else '否'}",
        "8. 是否仍禁止 Stage 5C latent generative：是",
        "9. 是否仍禁止 SMC：是",
        "",
        "## Final Conclusion",
        "",
        "项目是否跑通：是",
        f"pedestrian/drone 数据是否接入：{'是' if ped_loaded else '否'}",
        f"verified pedestrian/drone t+50/t+100 是否补上：{'是' if long_loaded else '否'}",
        f"gold/silver scene annotation 是否建立：{'是' if gold + silver > 0 else '否'}",
        f"per-agent multi-agent episodes 是否建立：{'是' if ge2 >= 50 else '部分'}",
        f"GoalBench-Gold official records 是否足够：{'是' if official >= 50 else '部分' if official > 0 else '否'}",
        f"是否可以进入 Stage 9：{'是' if gate.get('stage9_ready') else '否'}",
        "是否可以进入 Stage 5C latent generative：否",
        "是否可以启用 SMC：否",
        f"当前 verdict：{gate.get('verdict', 'unknown')}",
        f"expert audit score：{gate.get('expert_audit_score', 'unknown')}",
        "",
        "如果不能进入 Stage 9，下一步先修什么：",
        "",
        "1. 提供本地 SDD/OpenTraj 路径并转换更多真实 pedestrian/drone scenes。",
        "2. 把 rule-confirmed silver 升级为人工确认 gold/silver walkable/exit/goal annotation。",
        "3. 确保 GoalBench official records 超过 50，并保持 candidate goals train-only。",
    ]
    (REPORT_DIR / "report_stage8p5_final.md").write_text("\n".join(final) + "\n", encoding="utf-8")
    (REPORT_DIR / "data_card_stage8p5.md").write_text(data_card(data, eps), encoding="utf-8")
    (REPORT_DIR / "annotation_card_stage8p5.md").write_text(annotation_card(ann, packs), encoding="utf-8")
    (REPORT_DIR / "stage8p5_next_steps.md").write_text("\n".join(final[-5:]) + "\n", encoding="utf-8")


def data_card(data, eps) -> str:
    lines = ["# Stage 8.5 Data Card", ""]
    for row in data:
        lines.append(f"- {row.get('dataset_name')}: status={row.get('download_status')}, unit={row.get('coordinate_unit')}, t50={row.get('actual_verified_t50')}, t100={row.get('actual_verified_t100')}, license={row.get('license')}")
    lines.append("")
    for row in eps.get("datasets", []):
        lines.append(f"- episodes {row.get('dataset_name')}: total={row.get('total_episodes')}, ge2={row.get('episodes_ge2_agents')}, t100={row.get('verified_t100_episodes')}")
    return "\n".join(lines) + "\n"


def annotation_card(ann, packs) -> str:
    lines = ["# Stage 8.5 Annotation Card", "", "Silver means high-confidence rule-confirmed train-only endpoint regions, not true human-labelled goals.", ""]
    for row in ann.get("annotations", []):
        lines.append(f"- {row.get('dataset_name')}/{row.get('scene_id')}: {row.get('annotation_quality')}, goals={row.get('goal_count')}, preview={row.get('preview_image')}")
    lines += ["", f"gold={packs.get('number_of_gold_scenes', 0)} silver={packs.get('number_of_silver_scenes', 0)} inferred_only={packs.get('number_of_inferred_only_scenes', 0)}"]
    return "\n".join(lines) + "\n"


def package_outputs() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    (PACKAGE / "reports").mkdir(exist_ok=True)
    for path in REPORT_DIR.glob("*stage8p5*"):
        shutil.copy2(path, PACKAGE / "reports" / path.name)
    for name in ["report_stage8p5_final.md", "data_card_stage8p5.md", "annotation_card_stage8p5.md", "world_model_gate_stage8p5.md", "world_model_gate_stage8p5.json"]:
        src = REPORT_DIR / name
        if src.exists():
            shutil.copy2(src, PACKAGE / "reports" / src.name)
    for src_dir, dst_name in [
        (Path("data/stage8p5_annotations"), "annotations"),
        (Path("data/stage8p5_scene_gold_packs"), "scene_gold_packs"),
        (Path("data/stage8p5_per_agent_episodes"), "per_agent_episodes"),
        (Path("data/stage8p5_goalbench_gold_v2"), "goalbench_gold_v2"),
        (Path("outputs/figures/stage8p5_annotation_previews"), "annotation_previews"),
    ]:
        dst = PACKAGE / dst_name
        if dst.exists():
            shutil.rmtree(dst)
        if src_dir.exists():
            shutil.copytree(src_dir, dst)
    final = REPORT_DIR / "report_stage8p5_final.md"
    if final.exists():
        (PACKAGE / "STAGE8P5_EXECUTIVE_SUMMARY.md").write_text(final.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
