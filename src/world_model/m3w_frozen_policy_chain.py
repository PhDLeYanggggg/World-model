"""Past-only composition of fixed forecasts, costs and joint diagnostic controls.

This numerical interface opens no datasets and grants no predictive admission.
Source-specific admission must precede constructing a reserved-source input.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform

if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError("Native arm64 runtime required before Torch import")

import joblib
import numpy as np
import torch

from src.data_unification.m3w_external_prefix_adapter import ExternalPrefixAdapter
from src.evaluation.m3w_recording_lineage import sha256
from src.evaluation.m3w_forecast_cost_bounds import disagreement
from src.training.m3w_external_cost_bank import predict_forest
from src.world_model.m3w_native_forecast import pack_geometry, predict as forecast
from src.world_model.m3w_native_gain_harm import cost_features
from src.world_model.m3w_bounded_cost_head import build, predict as cost_predict
from src.world_model.m3w_supervised_intervention import build_forecaster
from src.world_model.m3w_native_joint_controls import make_problem, compare

ARMS = ("floor", "uncontrolled", "strict") + tuple(
    f"{budget}_{arm}" for budget in ("full", "half")
    for arm in ("independent", "unary", "joint", "scene_uniform"))


def restore(local, origins, rotations, scales):
    return np.einsum("nti,nji->ntj", np.asarray(local, float)*scales[:, None, None], rotations)+origins[:, None]


@dataclass(frozen=True)
class SceneInput:
    recording_id: str
    frame_id: int
    frame_step: int
    agent_ids: np.ndarray
    target_ids: np.ndarray
    geometry: np.ndarray
    origins: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray
    current: np.ndarray
    baseline: np.ndarray
    cv_valid: np.ndarray

    def validate(self):
        ids, targets = np.asarray(self.agent_ids), np.asarray(self.target_ids)
        n, t = len(ids), len(targets)
        if (not isinstance(self.recording_id, str) or not self.recording_id
                or type(self.frame_id) is not int or self.frame_id < 0
                or type(self.frame_step) is not int or self.frame_step <= 0
                or ids.shape != (n,) or ids.dtype.kind not in "iu" or n == 0
                or targets.shape != (t,) or targets.dtype.kind not in "iu"
                or (ids < 0).any() or (ids >= 2**53).any() or (ids[1:] <= ids[:-1]).any()
                or (targets[1:] <= targets[:-1]).any() or not np.isin(targets, ids).all()):
            raise ValueError("Explicit query and sorted unique observed/target IDs required")
        for value, shape in ((self.geometry, (t, 476)), (self.origins, (t, 2)),
                (self.rotations, (t, 2, 2)), (self.scales, (t,)), (self.current, (n, 2)),
                (self.baseline, (n, 12, 2))):
            if np.shape(value) != shape or not np.isfinite(value).all():
                raise ValueError("Finite aligned causal geometry and common-frame rollouts required")
        if (np.shape(self.cv_valid) != (n,) or self.cv_valid.dtype != bool
                or (self.scales <= 0).any()):
            raise ValueError("Explicit CV support and positive past-only scales required")
        loc = np.searchsorted(ids, targets)
        if t:
            pack_geometry(self.geometry)
            if (not np.allclose(self.geometry[:, 16:24], np.arange(-7, 1)/12., rtol=0, atol=1e-7)
                    or not np.all(self.geometry[:, 306] == 12*self.frame_step)
                    or not np.all(self.geometry[:, 307] == self.frame_step)):
                raise ValueError("Metadata does not match the fixed eight/twelve native-step grid")
            if not np.allclose(self.rotations @ self.rotations.transpose(0, 2, 1), np.eye(2), atol=1e-7):
                raise ValueError("Invalid causal rotation")
            if (not np.allclose(np.linalg.det(self.rotations), 1., atol=1e-7)
                    or not self.cv_valid[loc].all()
                    or not np.allclose(self.origins, self.current[loc], rtol=0, atol=1e-7)):
                raise ValueError("Target common-frame identity or past support mismatch")
            b = restore(self.geometry[:, 332:356].reshape(-1, 12, 2), self.origins, self.rotations, self.scales)
            if np.any(np.abs(b-self.baseline[loc]) > 2e-5*(1+self.scales[:, None, None])):
                raise ValueError("CV rollouts do not use the target's causal transform")
        return loc


def scene_from_prefix(adapter):
    if not isinstance(adapter, ExternalPrefixAdapter):
        raise ValueError("Explicit input-only native-stride prefix adapter required")
    g, packed = adapter.geometry_batch()
    points, query = adapter.points, adapter.query_frame
    now = points[points[:, 0] == query]
    ids, current = now[:, 1].astype(np.int64), now[:, 2:4]
    b = np.repeat(current[:, None], 12, axis=1)
    valid = np.zeros(len(now), bool)
    for i, agent_id in enumerate(ids):
        old = points[(points[:, 1] == agent_id) & (points[:, 0] == query-1)]
        if len(old):
            b[i] += np.arange(1, 13)[:, None]*(current[i]-old[0, 2:4])
            valid[i] = True
    agents = packed["agents"]
    target_ids = np.array([a["agent_id"] for a in agents], dtype=np.int64)
    origins = np.array([a["coordinate_transform"]["origin_xy"] for a in agents]).reshape(-1, 2)
    rotations = np.array([a["coordinate_transform"]["rotation"] for a in agents]).reshape(-1, 2, 2)
    # Preserve the historical float32 stored feature scale used by fitted heads.
    scales = np.array([a["inputs"]["causal_features"][2] for a in agents], float)
    if len(agents):
        b[np.searchsorted(ids, target_ids)] = restore(g[:, 332:356].reshape(-1, 12, 2), origins, rotations, scales)
    result = SceneInput(packed["recording_id"], query, 1, ids, target_ids,
                       g, origins, rotations, scales, current.copy(), b, valid)
    result.validate()
    return result


def decide(scene, normalized_candidate, costs, *, cost_scale):
    loc = scene.validate()
    p, scores = np.asarray(normalized_candidate), np.asarray(costs, float)
    t, n = len(loc), len(scene.agent_ids)
    if (p.shape != (t, 12, 2) or scores.shape != (t, 2)
            or not np.isfinite(cost_scale) or cost_scale <= 0):
        raise ValueError("Aligned model outputs and frozen positive training cost scale required")
    b = scene.baseline.copy()
    choices = {name: np.zeros(n, bool) for name in ARMS}
    result = dict(agent_ids=scene.agent_ids.copy(), target_ids=scene.target_ids.copy(),
        baseline=b, candidate=b.copy(), forecast_valid=scene.cv_valid.copy(), choices=choices,
        eligible=np.zeros(n, bool), model_output_supported=np.zeros(t, bool), budgets={},
        geometry={}, cost_scale=float(cost_scale), costs=np.zeros((n, 2)),
        realized_risk_certified=False, physical_safety_certified=False,
        source_admission=False, deployment=False)
    if not t:
        return dict(result, reason="no_supported_neural_targets")
    local_cv = scene.geometry[:, 332:356].reshape(t, 12, 2)
    numerical = np.isfinite(p).all((1, 2))
    clean = np.where(numerical[:, None, None], p, local_cv)
    d = disagreement(clean, local_cv, scene.scales).mean(1)
    supported = (numerical & np.isfinite(scores).all(1) & (scores >= 0).all(1)
                 & (scores.sum(1) <= d+2e-6*(1+d)))
    clean = np.where(supported[:, None, None], clean, local_cv)
    safe_score = np.where(supported[:, None], scores, 0.)
    candidate = b.copy()
    candidate[loc] = restore(clean, scene.origins, scene.rotations, scene.scales)
    past = scene.geometry[:, :16].reshape(t, 8, 2)
    moving = ~np.all(past[:, -1] == past[:, -2], axis=1)
    eligible = np.zeros(n, bool)
    eligible[loc] = (supported & moving & (d > 0) & (safe_score[:, 0] > safe_score[:, 1])
                     & (safe_score[:, 1] <= .1*safe_score[:, 0]))
    target = np.zeros(n, bool); target[loc] = True
    all_costs = np.zeros((n, 2)); all_costs[loc] = safe_score
    problem, geometry = make_problem(baseline=b, candidate=candidate, current=scene.current,
        forecast_valid=scene.cv_valid, target_mask=target, eligible=eligible,
        benefit=all_costs[:, 0], harm=all_costs[:, 1], scale=cost_scale, past_target_scales=scene.scales)
    choices["strict"] = eligible.copy()
    choices["uncontrolled"][loc] = supported
    for budget, fraction in (("full", 1.), ("half", .5)):
        decisions, report = compare(problem, scene.agent_ids, target, fraction, 5.)
        choices.update({budget+"_"+arm: bits for arm, bits in decisions.items()})
        result["budgets"][budget] = report
    for name, bits in choices.items():
        if name != "uncontrolled" and np.any(bits & ~eligible):
            raise ArithmeticError("Unsupported intervention escaped the fixed guard")
        if np.any(bits & ~scene.cv_valid):
            raise ArithmeticError("Intervened on an unknown causal forecast")
    return dict(result, candidate=candidate, costs=all_costs, eligible=eligible,
        model_output_supported=supported, geometry=geometry, reason="fixed_uncalibrated_research_chain")


class FrozenView:
    """Load only hash-bound local fitted artifacts; never fit or select a model."""
    def __init__(self, root, predictor_record, head_record, architecture, seed):
        root = Path(root)
        if (predictor_record["seed"] != seed
                or head_record["view"] != f"{predictor_record['family']}_seed{seed}"):
            raise ValueError("Forecast and cost heads have different registered views")
        for record in (predictor_record, head_record):
            path = (root/record["checkpoint"]).resolve()
            if not path.is_relative_to(root.resolve()) or sha256(path) != record["checkpoint_sha256"]:
                raise ValueError("Changed or escaping frozen checkpoint")
        pc = torch.load(root/predictor_record["checkpoint"], map_location="cpu", weights_only=False)
        if architecture != pc["identity"]["identity"]["architectures"][predictor_record["family"]]:
            raise ValueError("Architecture does not match the frozen trained predictor")
        self.predictor = build_forecaster(architecture)
        self.predictor.load_state_dict(pc["model"]); self.predictor.eval()
        self.forest = head_record["head"] == "matched_fraction_forest"
        if not self.forest and head_record["head"] != "bounded_fraction":
            raise ValueError("Unregistered cost head")
        hc = (joblib.load(root/head_record["checkpoint"]) if self.forest else
              torch.load(root/head_record["checkpoint"], map_location="cpu", weights_only=False))
        if (pc["seed"] != seed or hc["seed"] != seed or pc["step"] != 4000
                or pc["identity"]["family"] != predictor_record["family"]
                or hc["identity"]["view"] != head_record["view"]):
            raise ValueError("Unmatched checkpoint seed or predictor endpoint")
        self.preprocess = hc["preprocess"]
        if self.forest:
            self.head = hc["model"]
            if len(self.head.estimators_) != 128:
                raise ValueError("Incomplete forest")
        else:
            if hc["step"] != 3000 or hc["arm"] != "bounded_fraction":
                raise ValueError("Incomplete or wrong neural cost endpoint")
            self.head = build(356, 64, seed)
            self.head.load_state_dict(hc["model"]); self.head.eval()

    def infer(self, scene):
        scene.validate()
        t = len(scene.target_ids)
        if not t:
            return decide(scene, np.empty((0, 12, 2)), np.empty((0, 2)), cost_scale=self.preprocess["cost_scale"])
        p = forecast(self.predictor, {"geometry": scene.geometry}, np.arange(t))
        b = scene.geometry[:, 332:356].reshape(-1, 12, 2)
        good = np.isfinite(p).all((1, 2))
        safe = np.where(good[:, None, None], p, b)
        x, _ = cost_features(scene.geometry, safe, scene.scales)
        d = disagreement(safe, b, scene.scales).mean(1)
        x = np.column_stack((x, np.log1p(d))).astype(np.float32)
        if self.forest:
            score = predict_forest(self.head, x, d, self.preprocess)
        else:
            score = cost_predict(self.head, x, d, self.preprocess, "bounded_fraction")
        return decide(scene, p, score, cost_scale=self.preprocess["cost_scale"])
