from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np

from .final_agent_history_encoder import AgentHistoryEncoder
from .final_failure_heads import FailurePredictorHead
from .final_fallback import FallbackSelector
from .final_goal_encoder import GoalEncoder
from .final_interaction_encoder import InteractionFeatureEncoder
from .final_residual_decoder import BoundedResidualDecoder
from .final_scene_encoder import SceneEncoder


@dataclass
class BPSGMAWorldModel:
    """Baseline-Preserving Scene/Goal/Multi-Agent 2.5D World Model v1."""

    failure_threshold: float = 0.35
    residual_clip: float = 0.25
    force_dataset_baseline: bool = True

    def __post_init__(self) -> None:
        self.agent_encoder = AgentHistoryEncoder()
        self.scene_encoder = SceneEncoder()
        self.goal_encoder = GoalEncoder()
        self.interaction_encoder = InteractionFeatureEncoder()
        self.failure_head = FailurePredictorHead(threshold=self.failure_threshold)
        self.residual_decoder = BoundedResidualDecoder(residual_clip=self.residual_clip)
        self.fallback = FallbackSelector(force_dataset_baseline=self.force_dataset_baseline)

    def predict(
        self,
        all_agents_past_states: np.ndarray,
        valid_mask: np.ndarray,
        strongest_causal_baseline_rollout: np.ndarray,
        horizons: Iterable[int],
        scene_features: Dict[str, Any] | None = None,
        goal_features: Dict[str, Any] | List[Any] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        metadata = dict(metadata or {})
        baseline_rollout = np.asarray(strongest_causal_baseline_rollout, dtype=np.float64)
        agent_features = self.agent_encoder.encode(all_agents_past_states, valid_mask)
        scene = self.scene_encoder.encode(scene_features, baseline_rollout.shape[1])
        goal = self.goal_encoder.encode(agent_features["last_position"], goal_features)
        interaction = self.interaction_encoder.encode(all_agents_past_states, valid_mask)
        features = {**agent_features, **scene, **goal, **interaction}
        predictions = {}
        baselines = {}
        alphas = {}
        failure_probs = {}
        fallback_reasons = {}
        interventions = {}
        residual_norms = {}
        horizon_list = [int(h) for h in horizons]
        for horizon in horizon_list:
            idx = min(max(horizon - 1, 0), baseline_rollout.shape[0] - 1)
            baseline = baseline_rollout[idx]
            failure = self.failure_head.predict(features, horizon)
            residual = self.residual_decoder.decode(features, failure, horizon)
            alpha = failure["correction_needed_probability"]
            learned = baseline + alpha[:, None] * residual["bounded_residual"]
            selected = self.fallback.select(
                baseline,
                learned,
                alpha,
                failure["failure_probability"],
                residual["residual_norm"],
                metadata,
            )
            predictions[str(horizon)] = selected["final_prediction"]
            baselines[str(horizon)] = baseline
            alphas[str(horizon)] = alpha
            failure_probs[str(horizon)] = failure["failure_probability"]
            fallback_reasons[str(horizon)] = selected["fallback_reason"]
            interventions[str(horizon)] = selected["intervention_decision"]
            residual_norms[str(horizon)] = residual["residual_norm"]
        return {
            "predictions": predictions,
            "baseline_predictions": baselines,
            "alpha": alphas,
            "failure_probability": failure_probs,
            "fallback_reason": fallback_reasons,
            "intervention_decision": interventions,
            "residual_norm": residual_norms,
            "metadata": metadata,
        }

