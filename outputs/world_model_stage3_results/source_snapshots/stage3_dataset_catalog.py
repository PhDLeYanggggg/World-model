from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class DatasetSource:
    key: str
    name: str
    url: str
    license_note: str
    coordinate_quality: str
    scene_geometry: str
    agent_types: List[str]
    available_variables: List[str]
    t100_readiness: str
    size_note: str
    priority: int
    why_it_matters: str
    loader_status: str


def stage3_sources() -> List[DatasetSource]:
    return [
        DatasetSource(
            key="tgsim_foggy_bottom",
            name="TGSIM Foggy Bottom Trajectories",
            url="https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-foggy-bottom-trajectories",
            license_note="Public data.gov dataset; check resource terms before redistribution.",
            coordinate_quality="meter-scale coordinates; published conversion factor from pixels to meters",
            scene_geometry="aerial reference image plus 49 polygon boundaries for road/crosswalk/intersection regions",
            agent_types=["pedestrian", "bicycle", "scooter", "vehicle", "bus", "truck"],
            available_variables=[
                "position_m",
                "speed_mps",
                "acceleration_mps2",
                "width_m",
                "length_m",
                "road_user_type",
                "region_polygon_id",
                "aerial_reference_image",
            ],
            t100_readiness="strong: 0.1s records over a 2-hour window, enough for many t+100 windows",
            size_note="main CSV around 350 MB plus reference image and region annotations",
            priority=1,
            why_it_matters="Best immediate upgrade for physical world modeling because it includes acceleration, object size, meters, and scene polygons.",
            loader_status="stage3 adapter planned; no auto-download in this run",
        ),
        DatasetSource(
            key="stanford_drone_dataset",
            name="Stanford Drone Dataset",
            url="https://cvgl.stanford.edu/projects/uav_data/",
            license_note="Creative Commons BY-NC-SA 3.0 on official page.",
            coordinate_quality="image-space trajectories; scene-specific scale/homography must be estimated or supplied",
            scene_geometry="top-view videos/reference frames; no full metric scene graph by default",
            agent_types=["pedestrian", "bicyclist", "skateboarder", "cart", "car", "bus"],
            available_variables=["track_id", "bbox", "agent_type", "image_position", "scene_name", "video_id"],
            t100_readiness="medium-strong: long videos, but metric evaluation needs calibration",
            size_note="official Stanford Campus download is large, about 69 GB",
            priority=2,
            why_it_matters="Large top-view campus interactions with multiple agent classes; good for social navigation and mixed-agent priors.",
            loader_status="metadata/catalog only; avoid automatic 69 GB download",
        ),
        DatasetSource(
            key="opentraj_bundle",
            name="OpenTraj Dataset Toolkit",
            url="https://github.com/crowdbotp/OpenTraj",
            license_note="MIT toolkit; individual dataset licenses vary.",
            coordinate_quality="mixed; includes world-2D datasets such as ETH, inD, DUT, VRU, and image-space datasets",
            scene_geometry="mixed; some datasets include maps or context, many do not",
            agent_types=["pedestrian", "cyclist", "vehicle", "robot/person depending on dataset"],
            available_variables=["position", "track_id", "dataset_name", "fps", "coordinate_system"],
            t100_readiness="medium: unified loaders help, but every source needs horizon/scene validation",
            size_note="toolkit small; datasets vary",
            priority=3,
            why_it_matters="Fastest way to compare multiple pedestrian trajectory datasets under one API.",
            loader_status="adapter planned through optional external OpenTraj install",
        ),
        DatasetSource(
            key="eth_ucy",
            name="ETH / UCY Pedestrian Trajectories",
            url="https://vision.ee.ethz.ch/datsets.html and https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
            license_note="Academic benchmark; verify each archive before redistribution.",
            coordinate_quality="commonly used world/pixel pedestrian benchmark coordinates",
            scene_geometry="limited; homographies often available through benchmark preprocessing",
            agent_types=["pedestrian"],
            available_variables=["position", "track_id", "scene", "frame"],
            t100_readiness="medium: good benchmark, but trajectories are shorter and less scene-rich",
            size_note="small compared with SDD/TGSIM",
            priority=4,
            why_it_matters="Canonical social trajectory benchmark for sanity-checking learned dynamics.",
            loader_status="adapter planned",
        ),
        DatasetSource(
            key="trajnet_plus_plus",
            name="TrajNet++",
            url="https://www.epfl.ch/labs/vita/datasets/",
            license_note="Challenge/benchmark terms apply.",
            coordinate_quality="trajectory benchmark coordinates, interaction-centric",
            scene_geometry="usually trajectory-only; scene geometry sparse",
            agent_types=["pedestrian"],
            available_variables=["position", "track_id", "interaction_category"],
            t100_readiness="medium: interaction benchmark, but standard horizons may differ from t+100",
            size_note="benchmark-sized",
            priority=5,
            why_it_matters="Useful for stress-testing social interaction and collision-aware forecasting.",
            loader_status="adapter planned",
        ),
        DatasetSource(
            key="opendd",
            name="OpenDD Roundabout Drone Dataset",
            url="https://l3pilot.eu/data/opendd.html",
            license_note="CC BY-ND 4.0 on official page.",
            coordinate_quality="trajectory database with HD map information",
            scene_geometry="HD map, shapefiles, geo-referenced drone images",
            agent_types=["pedestrian", "vehicle", "cyclist"],
            available_variables=["trajectory", "bounding_box", "agent_type", "hd_map", "utm_coordinates"],
            t100_readiness="strong for mixed traffic; pedestrian-only subset must be checked",
            size_note="large multi-part dataset, 62+ hours according to official page",
            priority=6,
            why_it_matters="Excellent for map-aware constraints and mixed-agent physical interaction.",
            loader_status="later-stage adapter; license restricts derived redistribution",
        ),
        DatasetSource(
            key="ind",
            name="inD Intersection Drone Dataset",
            url="https://www.ind-dataset.com/",
            license_note="leveLX/inD terms; verify before download/use.",
            coordinate_quality="world-2D trajectories at German intersections",
            scene_geometry="intersection layouts and recording metadata",
            agent_types=["pedestrian", "bicyclist", "vehicle"],
            available_variables=["position", "velocity", "heading", "agent_type", "track_id"],
            t100_readiness="strong for mixed-agent intersections; pedestrian subset can evaluate long windows",
            size_note="moderate/large; access may require registration",
            priority=7,
            why_it_matters="Good mixed-agent interaction data but less immediately convenient than public TGSIM.",
            loader_status="later-stage adapter",
        ),
    ]


def write_stage3_data_reports(output_md: str | Path, output_json: str | Path, variable_md: str | Path) -> Dict:
    sources = sorted(stage3_sources(), key=lambda source: source.priority)
    output_md = Path(output_md)
    output_json = Path(output_json)
    variable_md = Path(variable_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sources": [asdict(source) for source in sources], "recommended_next": "tgsim_foggy_bottom"}
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(render_sources_markdown(sources), encoding="utf-8")
    variable_md.write_text(render_variable_schema_markdown(), encoding="utf-8")
    return payload


def render_sources_markdown(sources: List[DatasetSource]) -> str:
    lines = [
        "# Stage 3 Data Sources For 2.5D Crowd World Model",
        "",
        "## Recommendation",
        "",
        "Prioritize `TGSIM Foggy Bottom` first because it already provides meter-scale position, speed, acceleration, road-user type, dimensions, an aerial reference image, and region polygons. This directly attacks the current model's weak points: real t+100 evaluation, scene geometry, physical variables, and mixed-agent interactions.",
        "",
        "## Ranked Sources",
        "",
        "| Priority | Key | Dataset | Coordinate Quality | Scene Geometry | t+100 Readiness | Loader Status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for source in sources:
        lines.append(
            f"| {source.priority} | `{source.key}` | [{source.name}]({source.url}) | {source.coordinate_quality} | {source.scene_geometry} | {source.t100_readiness} | {source.loader_status} |"
        )
    lines.extend(["", "## Variables By Dataset", ""])
    for source in sources:
        lines.extend(
            [
                f"### {source.name}",
                "",
                f"- URL: {source.url}",
                f"- Agent types: {', '.join(source.agent_types)}",
                f"- Available variables: {', '.join(source.available_variables)}",
                f"- Size/access: {source.size_note}",
                f"- Why it matters: {source.why_it_matters}",
                f"- License note: {source.license_note}",
                "",
            ]
        )
    return "\n".join(lines)


def render_variable_schema_markdown() -> str:
    groups = world_model_stage3_variables()
    lines = [
        "# Stage 3 Variable Schema",
        "",
        "The model should stop treating trajectories as only `x/y/vx/vy`. Stage 3 expands each state into kinematic, interaction, scene, intent, observation, and uncertainty variables.",
        "",
    ]
    for group, variables in groups.items():
        lines.append(f"## {group}")
        lines.append("")
        lines.append("| Variable | Meaning | Source |")
        lines.append("| --- | --- | --- |")
        for variable, meaning, source in variables:
            lines.append(f"| `{variable}` | {meaning} | {source} |")
        lines.append("")
    return "\n".join(lines)


def world_model_stage3_variables() -> Dict[str, List[tuple[str, str, str]]]:
    return {
        "Kinematic": [
            ("speed_mps", "agent speed in meters per second", "observed or finite-difference"),
            ("acceleration_mps2", "agent acceleration in meters per second squared", "observed in TGSIM or finite-difference"),
            ("jerk_mps3", "temporal change in acceleration, used for smoothness", "derived"),
            ("heading_rate_radps", "turning rate", "derived"),
            ("stopping_probability_proxy", "short-horizon low-speed/stall tendency", "learned/derived"),
        ],
        "Interaction": [
            ("time_to_collision_s", "pairwise TTC under current relative velocity", "derived"),
            ("closing_speed_mps", "relative speed along neighbor normal", "derived"),
            ("front_density_people_per_m2", "density in front sector", "derived"),
            ("rear_density_people_per_m2", "density behind sector", "derived"),
            ("side_clearance_m", "minimum left/right clearance", "derived from neighbors/obstacles"),
        ],
        "Scene": [
            ("region_id", "polygon/semantic region assignment", "TGSIM polygons/OpenDD maps/manual scene geometry"),
            ("lane_or_area_type", "crosswalk, sidewalk, intersection, lane, plaza, obstacle", "scene geometry"),
            ("distance_to_crosswalk_m", "metric distance to nearest crosswalk region", "scene geometry"),
            ("distance_to_exit_m", "distance to candidate exit/goal", "scene geometry"),
            ("bottleneck_score", "local corridor/narrow-passage score", "derived from scene geometry"),
        ],
        "Intent": [
            ("goal_region_distribution", "probability over candidate exits/goals", "latent inferred"),
            ("velocity_goal_alignment", "alignment between velocity and sampled goal", "derived"),
            ("route_choice_entropy", "uncertainty over destination/route", "latent inferred"),
            ("dwell_time_frames", "how long an agent has been stopped or near-stopped", "derived"),
            ("intent_change_flag", "detected/sampled goal change", "latent/event"),
        ],
        "Uncertainty": [
            ("observation_noise_sigma_m", "measurement noise in world units", "dataset/calibration"),
            ("track_age_frames", "age of current identity track", "tracking metadata"),
            ("missing_observation_count", "recent missing frames", "tracking metadata"),
            ("calibration_quality_score", "whether metric scale/projection is trustworthy", "dataset metadata"),
            ("projection_cost", "how much physical correction was needed", "constraint layer"),
        ],
    }
