from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.m3w.token_schema import TOKEN_NAMES, TokenSchema, build_token_schema


@dataclass(frozen=True)
class M3WTokenRecord:
    token_type: str
    scene_id: str
    video_id: str
    frame_id: int
    agent_id: str | None
    feature_indices: List[int]
    valid: bool
    modality: str
    horizon: int | None
    split: str
    data_role: str


def build_m3w_token_schema(feature_names: List[str]) -> TokenSchema:
    return build_token_schema(feature_names)


def token_metadata_template(schema: TokenSchema, split: str, data_role: str) -> Dict[str, M3WTokenRecord]:
    return {
        token: M3WTokenRecord(
            token_type=token,
            scene_id="unknown",
            video_id="unknown",
            frame_id=-1,
            agent_id=None,
            feature_indices=list(schema.token_to_features[token]),
            valid=True,
            modality=_token_modality(token),
            horizon=None,
            split=split,
            data_role=data_role,
        )
        for token in TOKEN_NAMES
    }


def _token_modality(token: str) -> str:
    if token in {"scene_patch", "scene_sdf", "goal_region"}:
        return "scene"
    if token in {"agent_state", "agent_history", "interaction_edge"}:
        return "trajectory"
    if token == "baseline_rollout":
        return "baseline"
    return "metadata"
