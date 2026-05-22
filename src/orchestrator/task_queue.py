from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class QueueTask:
    name: str
    priority: str
    command: str
    family: str = ""
    allow_failure: bool = True


def build_stage13_task_queue(max_trials_per_family: int = 2, allow_training: bool = True) -> List[QueueTask]:
    training_flag = "" if allow_training else " --no-training"
    return [
        QueueTask("read_current_state", "P0", "python run_auto_world_model_loop.py --mode quick --max-steps 1"),
        QueueTask("check_stage13_allowed", "P0", "python scripts/auto_run_gates.py"),
        QueueTask("run_stage13_data_audit", "P1", "python scripts/auto_convert_available_datasets.py"),
        QueueTask(
            "stage13_deterministic_search",
            "P2",
            f"python scripts/run_stage13_deterministic_search.py --max-trials-per-family {max_trials_per_family}{training_flag}",
            family="all",
        ),
        QueueTask("run_stage13_gates", "P3", "python run_stage13_gates.py"),
        QueueTask("run_stage13_failure_miner", "P4", "python run_stage13_failure_miner.py"),
        QueueTask("auto_data_annotation_fallback", "P5", "python scripts/auto_find_and_prepare_datasets.py --dry-run"),
        QueueTask("update_readme", "P6", "python scripts/auto_update_readme_results.py"),
    ]


def build_stage14_task_queue(
    max_trials_per_family: int = 1,
    allow_training: bool = True,
    allow_data_discovery: bool = True,
) -> List[QueueTask]:
    training_tasks: List[QueueTask] = []
    if allow_training:
        training_tasks.append(
            QueueTask(
                "stage14_deterministic_multimodal_train",
                "P2",
                f"python run_stage14_train_multimodal.py --max-trials-per-family {max_trials_per_family} --max-iterations 10",
                family="stage14",
            )
        )
    discovery_tasks: List[QueueTask] = []
    if allow_data_discovery:
        discovery_tasks.append(
            QueueTask("stage14_multimodal_data_dry_run", "P5", "python scripts/stage14_fetch_or_verify_multimodal_data.py --dry-run")
        )
    return [
        QueueTask("stage14_current_state", "P0", "python run_stage14_current_state.py"),
        QueueTask("stage14_ewap_t100_mask_audit", "P1", "python run_stage14_audit_ewap_t100_masks.py"),
        QueueTask("stage14_rebuild_ewap_t100_episodes", "P1", "python run_stage14_rebuild_ewap_t100_episodes.py --max-episodes 64"),
        *discovery_tasks,
        QueueTask("stage14_build_multimodal_scene_packs", "P1", "python run_stage14_build_multimodal_scene_packs.py --limit 64"),
        QueueTask("stage14_build_multimodal_episodes", "P1", "python run_stage14_build_multimodal_episodes.py --limit 256"),
        *training_tasks,
        QueueTask("stage14_multimodal_benchmark", "P3", "python run_stage14_multimodal_benchmark.py"),
        QueueTask("stage14_gates", "P3", "python run_stage14_gates.py"),
        QueueTask("stage13_failure_miner_refresh", "P4", "python run_stage13_failure_miner.py"),
        QueueTask("update_readme", "P6", "python scripts/auto_update_readme_results.py"),
    ]


def build_stage14_maintenance_queue(allow_data_discovery: bool = True) -> List[QueueTask]:
    tasks = [
        QueueTask("stage14_benchmark_refresh", "P3", "python run_stage14_multimodal_benchmark.py"),
        QueueTask("stage14_gate_refresh", "P3", "python run_stage14_gates.py"),
        QueueTask("stage14_failure_refresh", "P4", "python run_stage13_failure_miner.py"),
        QueueTask("stage14_py_compile", "P6", "python -m py_compile run_auto_world_model_loop.py run_stage14_current_state.py run_stage14_gates.py run_stage14_train_multimodal.py src/stage14_pipeline.py src/orchestrator/auto_loop.py src/orchestrator/overnight_runner.py"),
    ]
    if allow_data_discovery:
        tasks.insert(0, QueueTask("stage14_data_dry_run_refresh", "P5", "python scripts/stage14_fetch_or_verify_multimodal_data.py --dry-run"))
    return tasks


def build_stage15_task_queue(allow_training: bool = True, allow_data_discovery: bool = True) -> List[QueueTask]:
    tasks = [
        QueueTask("stage15_oracle_diagnostics", "P1", "python run_stage15_oracle_diagnostics.py", family="oracle"),
        QueueTask("stage15_expand_ewap_t100", "P1", "python run_stage15_expand_ewap_t100.py --max-t100 256 --max-t50 512", family="data"),
    ]
    if allow_data_discovery:
        tasks.append(QueueTask("stage15_data_verify", "P5", "python scripts/stage15_verify_or_fetch_data.py", family="data"))
    if allow_training:
        tasks.append(QueueTask("stage15_deterministic_search", "P2", "python run_stage15_deterministic_search.py --max-trials 12", family="training"))
    tasks.extend(
        [
            QueueTask("stage15_benchmark", "P3", "python run_stage15_benchmark.py", family="benchmark"),
            QueueTask("stage15_gates", "P3", "python run_stage15_gates.py", family="gates"),
            QueueTask("update_readme", "P6", "python scripts/auto_update_readme_results.py"),
        ]
    )
    return tasks


def build_stage15_maintenance_queue(allow_data_discovery: bool = True, allow_training: bool = True) -> List[QueueTask]:
    tasks = [
        QueueTask("stage15_oracle_refresh", "P1", "python run_stage15_oracle_diagnostics.py", family="oracle"),
        QueueTask("stage15_benchmark_refresh", "P3", "python run_stage15_benchmark.py", family="benchmark"),
        QueueTask("stage15_gate_refresh", "P3", "python run_stage15_gates.py", family="gates"),
        QueueTask("stage15_py_compile", "P6", "python -m py_compile run_auto_world_model_loop.py src/stage15_pipeline.py src/orchestrator/auto_loop.py src/orchestrator/overnight_runner.py"),
    ]
    if allow_data_discovery:
        tasks.insert(1, QueueTask("stage15_data_verify_refresh", "P5", "python scripts/stage15_verify_or_fetch_data.py", family="data"))
    if allow_training:
        tasks.insert(2, QueueTask("stage15_deterministic_search_refresh", "P2", "python run_stage15_deterministic_search.py --max-trials 12", family="training"))
    return tasks


def build_stage16_task_queue(allow_training: bool = True, allow_data_discovery: bool = True) -> List[QueueTask]:
    tasks = [
        QueueTask("stage16_current_state", "P0", "python run_stage16_current_state.py"),
        QueueTask("stage16_expand_ewap", "P1", "python run_stage16_expand_ewap.py", family="data"),
        QueueTask("stage16_oracle_distillation", "P1", "python run_stage16_build_oracle_distillation.py", family="oracle"),
        QueueTask("stage16_failure_type_predictor", "P2", "python run_stage16_train_failure_type_predictor.py", family="oracle"),
    ]
    if allow_training:
        tasks.append(QueueTask("stage16_correction_training", "P2", "python run_stage16_train_oracle_distilled_correction.py", family="training"))
    if allow_data_discovery:
        tasks.append(QueueTask("stage16_data_verify", "P5", "python scripts/stage16_verify_more_data.py", family="data"))
    tasks.extend(
        [
            QueueTask("stage16_annotation_tasks", "P5", "python run_stage16_annotation_tasks.py", family="annotation"),
            QueueTask("stage16_benchmark", "P3", "python run_stage16_benchmark.py", family="benchmark"),
            QueueTask("stage16_gates", "P3", "python run_stage16_gates.py", family="gates"),
            QueueTask("update_readme", "P6", "python scripts/auto_update_readme_results.py"),
        ]
    )
    return tasks


def build_stage16_maintenance_queue(allow_data_discovery: bool = True, allow_training: bool = True) -> List[QueueTask]:
    tasks = [
        QueueTask("stage16_oracle_refresh", "P1", "python run_stage16_build_oracle_distillation.py", family="oracle"),
        QueueTask("stage16_failure_predictor_refresh", "P2", "python run_stage16_train_failure_type_predictor.py", family="oracle"),
        QueueTask("stage16_benchmark_refresh", "P3", "python run_stage16_benchmark.py", family="benchmark"),
        QueueTask("stage16_gate_refresh", "P3", "python run_stage16_gates.py", family="gates"),
        QueueTask("stage16_py_compile", "P6", "python -m py_compile run_auto_world_model_loop.py src/stage16_pipeline.py src/orchestrator/auto_loop.py src/orchestrator/overnight_runner.py"),
    ]
    if allow_data_discovery:
        tasks.insert(2, QueueTask("stage16_data_verify_refresh", "P5", "python scripts/stage16_verify_more_data.py", family="data"))
        tasks.insert(3, QueueTask("stage16_annotation_refresh", "P5", "python run_stage16_annotation_tasks.py", family="annotation"))
    if allow_training:
        tasks.insert(2, QueueTask("stage16_correction_refresh", "P2", "python run_stage16_train_oracle_distilled_correction.py", family="training"))
    return tasks
