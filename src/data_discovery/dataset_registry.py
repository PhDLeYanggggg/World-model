from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List


REGISTRY_FIELDS = [
    "dataset_name",
    "domain",
    "official_url",
    "mirror_url_if_any",
    "download_status",
    "license",
    "commercial_use_allowed",
    "redistribution_allowed",
    "citation_required",
    "data_size_estimate",
    "file_format",
    "has_trajectories",
    "has_raw_video",
    "has_images",
    "has_scene_map",
    "has_homography",
    "has_ground_plane_coordinates",
    "has_metric_coordinates",
    "has_agent_type",
    "has_heading",
    "has_velocity",
    "has_acceleration",
    "has_action",
    "has_goal_or_destination",
    "has_lane_graph",
    "has_obstacle_geometry",
    "has_walkable_area",
    "has_interaction_labels",
    "has_semantic_event_labels",
    "frame_rate",
    "coordinate_unit",
    "average_track_length",
    "estimated_samples_t10",
    "estimated_samples_t25",
    "estimated_samples_t50",
    "estimated_samples_t100",
    "can_evaluate_t100",
    "expected_domain_gap",
    "priority_score",
    "reason_for_priority",
    "loader_status",
    "download_command",
    "preprocessing_command",
    "notes",
]


@dataclass
class DatasetRecord:
    dataset_name: str
    domain: str
    official_url: str
    mirror_url_if_any: str = ""
    download_status: str = "unknown"
    license: str = "unknown"
    commercial_use_allowed: str = "unknown"
    redistribution_allowed: str = "unknown"
    citation_required: str = "yes"
    data_size_estimate: str = "unknown"
    file_format: str = "unknown"
    has_trajectories: bool = True
    has_raw_video: bool = False
    has_images: bool = False
    has_scene_map: bool = False
    has_homography: bool = False
    has_ground_plane_coordinates: bool = False
    has_metric_coordinates: bool = False
    has_agent_type: bool = False
    has_heading: bool = False
    has_velocity: bool = False
    has_acceleration: bool = False
    has_action: bool = False
    has_goal_or_destination: bool = False
    has_lane_graph: bool = False
    has_obstacle_geometry: bool = False
    has_walkable_area: bool = False
    has_interaction_labels: bool = False
    has_semantic_event_labels: bool = False
    frame_rate: str = "unknown"
    coordinate_unit: str = "unknown"
    average_track_length: str = "unknown"
    estimated_samples_t10: str = "unknown"
    estimated_samples_t25: str = "unknown"
    estimated_samples_t50: str = "unknown"
    estimated_samples_t100: str = "unknown"
    can_evaluate_t100: bool = False
    expected_domain_gap: str = "unknown"
    priority_score: int = 0
    reason_for_priority: str = ""
    loader_status: str = "not_started"
    download_command: str = ""
    preprocessing_command: str = ""
    notes: str = ""

    def finalize(self) -> "DatasetRecord":
        self.priority_score = score_record(self)
        if not self.reason_for_priority:
            self.reason_for_priority = explain_priority(self)
        return self


def score_record(record: DatasetRecord) -> int:
    score = 0
    if record.download_status in {"downloaded", "downloadable"}:
        score += 20
    if record.has_metric_coordinates:
        score += 15
    if record.can_evaluate_t100:
        score += 15
    if record.has_scene_map or record.has_obstacle_geometry or record.has_walkable_area or record.has_lane_graph:
        score += 15
    if record.has_interaction_labels or record.domain in {"crowd", "driving", "traffic", "mixed", "drone"}:
        score += 10
    if record.has_agent_type:
        score += 5
    if record.has_heading or record.has_velocity:
        score += 5
    if record.has_goal_or_destination or record.has_lane_graph:
        score += 5
    if any(token in record.data_size_estimate.lower() for token in ["large", "gb", "million", "hours"]):
        score += 5
    if record.license and record.license != "unknown":
        score += 5
    return min(100, score)


def explain_priority(record: DatasetRecord) -> str:
    reasons = []
    if record.download_status in {"downloaded", "downloadable"}:
        reasons.append("legally accessible")
    if record.has_metric_coordinates:
        reasons.append("metric coordinates")
    if record.can_evaluate_t100:
        reasons.append("verified t+100 likely/available")
    if record.has_scene_map or record.has_lane_graph:
        reasons.append("scene/map context")
    if record.domain in {"traffic", "driving"}:
        reasons.append("traffic dynamics")
    if record.domain in {"pedestrian", "crowd", "drone"}:
        reasons.append("pedestrian/crowd dynamics")
    return ", ".join(reasons) or "catalog candidate"


def built_in_records() -> List[DatasetRecord]:
    records = [
        DatasetRecord(
            "TGSIM Foggy Bottom",
            "traffic",
            "https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-foggy-bottom-trajectories",
            download_status="downloaded",
            license="U.S. public data portal; verify resource terms before redistribution",
            commercial_use_allowed="likely_yes_verify",
            redistribution_allowed="likely_yes_verify",
            data_size_estimate="hundreds of MB for full CSV",
            file_format="CSV via Socrata/data.transportation.gov",
            has_scene_map=True,
            has_ground_plane_coordinates=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            has_acceleration=True,
            has_obstacle_geometry=True,
            has_walkable_area=True,
            frame_rate="10 Hz",
            coordinate_unit="meter",
            estimated_samples_t100="482 in current quick sample",
            can_evaluate_t100=True,
            expected_domain_gap="traffic, not pure pedestrian crowd",
            loader_status="working_quick",
            download_command="python scripts/download_stage5_datasets.py --dataset tgsim_foggy_bottom --max-gb 2",
            preprocessing_command="python run_stage4p5_dynamics_benchmark.py --dataset tgsim --data <csv-or-url> --quick",
            notes="Official quick endpoint has no scene polygons loaded yet; causal velocity is official.",
        ),
        DatasetRecord(
            "TGSIM other public corridors",
            "traffic",
            "https://data.transportation.gov/",
            download_status="downloadable",
            license="U.S. public data portal; verify each resource",
            commercial_use_allowed="likely_yes_verify",
            redistribution_allowed="likely_yes_verify",
            data_size_estimate="large CSV collections",
            file_format="CSV",
            has_ground_plane_coordinates=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            has_acceleration=True,
            has_scene_map=True,
            can_evaluate_t100=True,
            expected_domain_gap="traffic",
            loader_status="adapter_partial",
            download_command="python scripts/download_stage5_datasets.py --dataset tgsim_all --max-gb 20",
            notes="Discover individual Socrata resource ids before bulk download.",
        ),
        DatasetRecord(
            "Stanford Drone Dataset",
            "drone",
            "https://cvgl.stanford.edu/projects/uav_data/",
            download_status="downloadable",
            license="CC BY-NC-SA 3.0",
            commercial_use_allowed="no",
            redistribution_allowed="share_alike_noncommercial",
            data_size_estimate="about 69 GB",
            file_format="videos, annotations",
            has_raw_video=True,
            has_images=True,
            has_scene_map=True,
            has_agent_type=True,
            has_homography=False,
            has_metric_coordinates=False,
            has_ground_plane_coordinates=False,
            can_evaluate_t100=True,
            expected_domain_gap="mixed campus top-down; pixel to metric requires calibration",
            loader_status="planned",
            download_command="python scripts/download_stage5_datasets.py --dataset stanford_drone --max-gb 30",
            notes="Non-commercial license; do not use for commercial training without permission.",
        ),
        DatasetRecord(
            "TrajNet++",
            "pedestrian",
            "https://www.epfl.ch/labs/vita/datasets/",
            mirror_url_if_any="https://github.com/vita-epfl/trajnetplusplusdataset",
            download_status="downloadable",
            license="benchmark terms; verify individual files",
            commercial_use_allowed="unknown",
            redistribution_allowed="unknown",
            data_size_estimate="benchmark-sized",
            file_format="NDJSON",
            has_ground_plane_coordinates=True,
            has_metric_coordinates=True,
            has_interaction_labels=True,
            can_evaluate_t100=True,
            expected_domain_gap="pedestrian trajectories, sparse scene geometry",
            loader_status="partial_loader",
            download_command="python scripts/download_stage5_datasets.py --dataset trajnet --max-gb 5",
        ),
        DatasetRecord(
            "ETH Pedestrian",
            "pedestrian",
            "https://vision.ee.ethz.ch/datasets/",
            download_status="downloadable",
            license="academic dataset terms; verify before redistribution",
            commercial_use_allowed="unknown",
            redistribution_allowed="unknown",
            data_size_estimate="small",
            file_format="TXT/MAT/video depending release",
            has_images=True,
            has_homography=True,
            has_ground_plane_coordinates=True,
            has_metric_coordinates=True,
            can_evaluate_t100=True,
            expected_domain_gap="pedestrian fixed-camera",
            loader_status="partial_loader",
            download_command="python scripts/download_stage5_datasets.py --dataset eth_ucy --max-gb 5",
        ),
        DatasetRecord(
            "UCY Crowd",
            "crowd",
            "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
            download_status="downloadable",
            license="academic dataset terms; verify before redistribution",
            commercial_use_allowed="unknown",
            redistribution_allowed="unknown",
            data_size_estimate="small",
            file_format="TXT/video depending release",
            has_images=True,
            has_homography=True,
            has_ground_plane_coordinates=True,
            has_metric_coordinates=True,
            can_evaluate_t100=True,
            expected_domain_gap="pedestrian fixed-camera",
            loader_status="partial_loader",
        ),
        DatasetRecord(
            "AerialMPT",
            "drone",
            "local:data/aerialmpt/DLR_AerialMPT_Dataset.zip",
            download_status="downloaded",
            license="DLR dataset terms; verify before redistribution",
            commercial_use_allowed="unknown",
            redistribution_allowed="unknown",
            file_format="zip/video/annotations",
            has_images=True,
            has_trajectories=True,
            has_agent_type=True,
            has_metric_coordinates=False,
            can_evaluate_t100=False,
            expected_domain_gap="short local bauma3 slice has only t+12 verified",
            loader_status="project_specific",
        ),
        DatasetRecord(
            "OpenTraj Toolkit",
            "mixed",
            "https://github.com/crowdbotp/OpenTraj",
            download_status="downloadable",
            license="MIT toolkit; individual dataset licenses vary",
            commercial_use_allowed="toolkit_yes_data_varies",
            redistribution_allowed="toolkit_yes_data_varies",
            data_size_estimate="toolkit small, datasets vary",
            file_format="Python loaders",
            has_trajectories=True,
            has_metric_coordinates=True,
            can_evaluate_t100=True,
            expected_domain_gap="mixed datasets",
            loader_status="planned_external",
        ),
        DatasetRecord(
            "Argoverse 1 Motion Forecasting",
            "driving",
            "https://www.argoverse.org/av1.html",
            download_status="requires_application",
            license="Argoverse terms",
            commercial_use_allowed="unknown",
            redistribution_allowed="restricted",
            data_size_estimate="large",
            file_format="CSV/Parquet/maps",
            has_scene_map=True,
            has_lane_graph=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            can_evaluate_t100=True,
            expected_domain_gap="vehicle motion forecasting",
            loader_status="planned",
        ),
        DatasetRecord(
            "Argoverse 2 Motion Forecasting",
            "driving",
            "https://www.argoverse.org/av2.html",
            download_status="requires_application",
            license="CC BY-NC-SA 4.0 / Argoverse terms",
            commercial_use_allowed="no",
            redistribution_allowed="restricted_noncommercial",
            data_size_estimate="large",
            file_format="Parquet/maps",
            has_scene_map=True,
            has_lane_graph=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            can_evaluate_t100=True,
            expected_domain_gap="vehicle motion forecasting",
            loader_status="planned",
        ),
        DatasetRecord(
            "Waymo Open Motion Dataset",
            "driving",
            "https://waymo.com/open/data/motion/",
            download_status="gated",
            license="Waymo Open Dataset terms",
            commercial_use_allowed="restricted",
            redistribution_allowed="restricted",
            data_size_estimate="large",
            file_format="TFRecord/protobuf",
            has_scene_map=True,
            has_lane_graph=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            can_evaluate_t100=True,
            expected_domain_gap="large-scale autonomous driving",
            loader_status="planned",
        ),
        DatasetRecord(
            "nuScenes Prediction",
            "driving",
            "https://www.nuscenes.org/prediction",
            download_status="requires_application",
            license="nuScenes terms",
            commercial_use_allowed="restricted",
            redistribution_allowed="restricted",
            data_size_estimate="large",
            file_format="JSON/tables/maps",
            has_scene_map=True,
            has_lane_graph=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            can_evaluate_t100=True,
            expected_domain_gap="autonomous driving",
            loader_status="planned",
        ),
        DatasetRecord(
            "INTERACTION Dataset",
            "driving",
            "https://interaction-dataset.com/",
            download_status="requires_application",
            license="INTERACTION dataset terms",
            commercial_use_allowed="unknown",
            redistribution_allowed="restricted",
            data_size_estimate="large",
            file_format="CSV/maps",
            has_scene_map=True,
            has_lane_graph=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            can_evaluate_t100=True,
            expected_domain_gap="interactive driving scenes",
            loader_status="planned",
        ),
        DatasetRecord(
            "inD",
            "traffic",
            "https://www.ind-dataset.com/",
            download_status="requires_application",
            license="inD/leveLX terms",
            commercial_use_allowed="unknown",
            redistribution_allowed="restricted",
            data_size_estimate="moderate",
            file_format="CSV/maps",
            has_scene_map=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            can_evaluate_t100=True,
            expected_domain_gap="mixed intersection traffic",
            loader_status="planned",
        ),
        DatasetRecord(
            "rounD",
            "traffic",
            "https://www.round-dataset.com/",
            download_status="requires_application",
            license="rounD/leveLX terms",
            commercial_use_allowed="unknown",
            redistribution_allowed="restricted",
            data_size_estimate="moderate",
            file_format="CSV/maps",
            has_scene_map=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            can_evaluate_t100=True,
            expected_domain_gap="roundabout traffic",
            loader_status="planned",
        ),
        DatasetRecord(
            "highD",
            "traffic",
            "https://www.highd-dataset.com/",
            download_status="requires_application",
            license="highD/leveLX terms",
            commercial_use_allowed="unknown",
            redistribution_allowed="restricted",
            data_size_estimate="large",
            file_format="CSV",
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            can_evaluate_t100=True,
            expected_domain_gap="highway traffic",
            loader_status="planned",
        ),
        DatasetRecord(
            "exiD",
            "traffic",
            "https://www.exid-dataset.com/",
            download_status="requires_application",
            license="exiD/leveLX terms",
            commercial_use_allowed="unknown",
            redistribution_allowed="restricted",
            data_size_estimate="moderate",
            file_format="CSV/maps",
            has_scene_map=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            can_evaluate_t100=True,
            expected_domain_gap="highway exits",
            loader_status="planned",
        ),
        DatasetRecord(
            "uniD",
            "traffic",
            "https://www.unid-dataset.com/",
            download_status="requires_application",
            license="uniD/leveLX terms",
            commercial_use_allowed="unknown",
            redistribution_allowed="restricted",
            data_size_estimate="moderate",
            file_format="CSV/maps",
            has_scene_map=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            can_evaluate_t100=True,
            expected_domain_gap="university traffic",
            loader_status="planned",
        ),
        DatasetRecord(
            "OpenDD",
            "traffic",
            "https://l3pilot.eu/data/opendd.html",
            download_status="downloadable",
            license="CC BY-ND 4.0",
            commercial_use_allowed="yes_with_no_derivatives_constraint",
            redistribution_allowed="no_derivatives",
            data_size_estimate="large, 62+ hours",
            file_format="SQLite/CSV/shapefiles",
            has_scene_map=True,
            has_lane_graph=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            has_obstacle_geometry=True,
            can_evaluate_t100=True,
            expected_domain_gap="roundabout drone traffic",
            loader_status="planned",
        ),
        DatasetRecord(
            "NGSIM",
            "traffic",
            "https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm",
            download_status="downloadable",
            license="U.S. FHWA public research data; verify terms",
            commercial_use_allowed="likely_yes_verify",
            redistribution_allowed="likely_yes_verify",
            data_size_estimate="moderate",
            file_format="CSV",
            has_metric_coordinates=True,
            has_agent_type=True,
            has_velocity=True,
            has_acceleration=True,
            can_evaluate_t100=True,
            expected_domain_gap="highway traffic",
            loader_status="planned",
        ),
        DatasetRecord(
            "nuPlan",
            "driving",
            "https://www.nuscenes.org/nuplan",
            download_status="requires_application",
            license="nuPlan terms",
            commercial_use_allowed="restricted",
            redistribution_allowed="restricted",
            data_size_estimate="very large",
            file_format="DB/maps",
            has_scene_map=True,
            has_lane_graph=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            has_action=True,
            can_evaluate_t100=True,
            expected_domain_gap="autonomous driving planning",
            loader_status="planned",
        ),
        DatasetRecord(
            "Google Research Football",
            "multi_agent",
            "https://github.com/google-research/football",
            download_status="downloadable",
            license="Apache-2.0 code; generated data depends on simulator",
            commercial_use_allowed="code_yes_data_generated",
            redistribution_allowed="code_yes_data_generated",
            data_size_estimate="synthetic generated",
            file_format="sim logs",
            has_metric_coordinates=False,
            has_agent_type=True,
            has_action=True,
            has_goal_or_destination=True,
            can_evaluate_t100=True,
            expected_domain_gap="game/simulation, not real world",
            loader_status="planned",
        ),
        DatasetRecord(
            "SyntheticPhysicalCrowd2.5D",
            "synthetic",
            "local:src/data/synthetic_physical_crowd.py",
            download_status="downloaded",
            license="project-generated",
            commercial_use_allowed="yes",
            redistribution_allowed="yes",
            data_size_estimate="configurable",
            file_format="NPZ/JSON",
            has_scene_map=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            has_acceleration=True,
            has_goal_or_destination=True,
            has_obstacle_geometry=True,
            has_walkable_area=True,
            has_semantic_event_labels=True,
            can_evaluate_t100=True,
            expected_domain_gap="synthetic-to-real gap high",
            loader_status="working",
        ),
        DatasetRecord(
            "SyntheticTraffic2.5D",
            "synthetic",
            "local:planned",
            download_status="unavailable",
            license="project-generated when implemented",
            commercial_use_allowed="yes",
            redistribution_allowed="yes",
            data_size_estimate="configurable",
            file_format="planned NPZ/JSON",
            has_scene_map=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            has_acceleration=True,
            has_lane_graph=True,
            can_evaluate_t100=True,
            expected_domain_gap="synthetic-to-real gap",
            loader_status="planned",
        ),
        DatasetRecord(
            "SyntheticMixedAgents2.5D",
            "synthetic",
            "local:planned",
            download_status="unavailable",
            license="project-generated when implemented",
            commercial_use_allowed="yes",
            redistribution_allowed="yes",
            data_size_estimate="configurable",
            file_format="planned NPZ/JSON",
            has_scene_map=True,
            has_metric_coordinates=True,
            has_agent_type=True,
            has_heading=True,
            has_velocity=True,
            has_acceleration=True,
            has_goal_or_destination=True,
            can_evaluate_t100=True,
            expected_domain_gap="synthetic-to-real gap",
            loader_status="planned",
        ),
        DatasetRecord(
            "Multi-Agent Particle Environment",
            "synthetic",
            "https://github.com/openai/multiagent-particle-envs",
            download_status="downloadable",
            license="MIT",
            commercial_use_allowed="yes",
            redistribution_allowed="yes",
            data_size_estimate="generated",
            file_format="sim logs",
            has_metric_coordinates=False,
            has_agent_type=True,
            has_action=True,
            has_goal_or_destination=True,
            can_evaluate_t100=True,
            expected_domain_gap="toy simulation",
            loader_status="planned",
        ),
    ]
    return [record.finalize() for record in records]


def registry_as_dicts(records: Iterable[DatasetRecord]) -> List[Dict]:
    rows = []
    for record in records:
        row = asdict(record.finalize())
        rows.append({field: row.get(field, "") for field in REGISTRY_FIELDS})
    return sorted(rows, key=lambda row: (-int(row["priority_score"]), row["dataset_name"]))


def write_registry_outputs(records: Iterable[DatasetRecord], output_dir: str | Path = "outputs/data_registry") -> List[Dict]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = registry_as_dicts(records)
    (out / "dataset_registry_stage5.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out / "dataset_registry_stage5.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out / "dataset_registry_stage5.md").write_text(markdown_table(rows), encoding="utf-8")
    return rows


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No datasets registered._\n"
    fields = REGISTRY_FIELDS
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        cells = [str(row.get(field, "")).replace("|", "/") for field in fields]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
