"""Label-separated scoring for the registered DUT calibration-domain readout."""
from __future__ import annotations

import numpy as np

from src.data_unification.m3w_causal_recordings import BASELINES, causal_baselines
from src.data_unification.m3w_external_prefix_adapter import ExternalPrefixAdapter
from src.world_model.m3w_frozen_policy_chain import ARMS, scene_from_prefix, decide
from src.world_model.m3w_native_forecast import predict as forecast
from src.world_model.m3w_native_gain_harm import cost_features
from src.evaluation.m3w_forecast_cost_bounds import disagreement
from src.training.m3w_external_cost_bank import predict_forest
from src.world_model.m3w_bounded_cost_head import predict as cost_predict


def batch_infer(view, scenes):
    """Same fixed learned functions; batch only across independent target examples.

    Graph decisions still run per original scene. No labels enter this interface.
    Fixed chunk/batch sizes are bound by the readout manifest for exact replay.
    """
    sizes = [len(s.target_ids) for s in scenes]
    for s in scenes:
        s.validate()
    if not sum(sizes):
        return [decide(s, np.empty((0, 12, 2)), np.empty((0, 2)),
                       cost_scale=view.preprocess["cost_scale"]) for s in scenes]
    g = np.concatenate([s.geometry for s in scenes])
    scales = np.concatenate([s.scales for s in scenes])
    p = forecast(view.predictor, {"geometry": g}, np.arange(len(g)), batch_size=128)
    b = g[:, 332:356].reshape(-1, 12, 2)
    safe = np.where(np.isfinite(p).all((1, 2))[:, None, None], p, b)
    x, _ = cost_features(g, safe, scales)
    d = disagreement(safe, b, scales).mean(1)
    x = np.column_stack((x, np.log1p(d))).astype(np.float32)
    score = (predict_forest(view.head, x, d, view.preprocess) if view.forest else
             cost_predict(view.head, x, d, view.preprocess, "bounded_fraction"))
    offsets = np.r_[0, np.cumsum(sizes)]
    return [decide(s, p[lo:hi], score[lo:hi], cost_scale=view.preprocess["cost_scale"])
            for s, lo, hi in zip(scenes, offsets[:-1], offsets[1:])]


class PrefixRecording:
    """Never use the future-conditioned window index to choose inference agents."""
    def __init__(self, points, frame_order, recording):
        self.points, self.order, self.recording = points, frame_order, recording
        self.frames = points[frame_order, 0]
        self.queries = np.unique(self.frames)
        self.queries = self.queries[self.queries >= self.queries[0]+7].astype(np.int64)

    def rows(self, begin, end):
        lo, hi = np.searchsorted(self.frames, [begin, end], side="left")
        return self.points[self.order[lo:hi]]

    def inputs(self, frame):
        prefix = self.rows(frame-7, frame+1)
        return scene_from_prefix(ExternalPrefixAdapter(prefix, query_frame=int(frame),
                                                      recording_id=self.recording))

    def labels(self, frame, target_ids):
        # Called only after decisions are produced. Partial labels stay explicit.
        target_ids = np.asarray(target_ids)
        y = np.zeros((len(target_ids), 12, 2), float)
        mask = np.zeros((len(target_ids), 12), bool)
        rows = self.rows(frame+1, frame+13)
        loc = {int(agent): i for i, agent in enumerate(target_ids)}
        for timestamp, agent, x, yy in rows:
            if int(agent) in loc:
                i, j = loc[int(agent)], int(timestamp)-int(frame)-1
                if mask[i, j]:
                    raise ValueError("Duplicate future agent/frame label")
                y[i, j], mask[i, j] = (x, yy), True
        return y, mask


def fixed_baselines(recording, scene):
    past = recording.rows(scene.frame_id-7, scene.frame_id+1)
    result = np.empty((len(BASELINES), len(scene.target_ids), 12, 2))
    for i, agent in enumerate(scene.target_ids):
        p = past[past[:, 1] == agent]
        p = p[np.argsort(p[:, 0])]
        result[:, i] = causal_baselines(p[:, 2:4], p[:, 0], np.arange(1, 13))
    return result


def score_arrays(baseline, prediction, labels, mask, scales, *, easy_cut, hard_cut):
    """Conditional complete-path errors plus label-free bounds on missing costs.

    Bounds use |error(B)-error(P)| <= ||B-P|| for each missing target step.
    They do not bound the unobserved absolute ADE or certify physical safety.
    """
    b, p, y, m, s = map(np.asarray, (baseline, prediction, labels, mask, scales))
    n = len(b)
    if (b.shape != (n, 12, 2) or p.shape != b.shape or y.shape != b.shape
            or m.shape != (n, 12) or m.dtype != bool or s.shape != (n,)
            or not np.isfinite(b).all() or not np.isfinite(p).all()
            or not np.isfinite(y[m]).all() or not np.isfinite(s).all() or (s <= 0).any()):
        raise ValueError("Aligned finite forecasts, supported labels and positive causal scales required")
    clean = np.where(m[:, :, None], y, 0.)
    be = np.linalg.norm(b-clean, axis=2)
    pe = np.linalg.norm(p-clean, axis=2)
    complete = m.all(1)
    ade_b, ade_p = be.mean(1), pe.mean(1)
    normalized = ade_b/s
    known_gain = np.where(m, be-pe, 0.).sum(1)/12
    unknown_radius = np.where(m, 0., np.linalg.norm(b-p, axis=2)).sum(1)/12
    return dict(complete=complete, baseline_ade=ade_b, ade=ade_p, fde=pe[:, -1],
        baseline_fde=be[:, -1], normalized_ade=ade_p/s,
        easy=complete & (normalized <= easy_cut),
        hard=complete & (normalized >= hard_cut), zero=complete & (ade_b == 0),
        gain_lower=known_gain-unknown_radius, gain_upper=known_gain+unknown_radius,
        known_steps=m.sum(1))


def empty_statistics():
    return dict(queries=0, targets=0, visible=0, unknown_cv_context=0, complete=0,
        known_steps=0, rejected_outputs=0, eligible=0, half_unmatched=0,
        half_joint_diff_agents=0, half_joint_diff_queries=0, arms={}, baseline_controls={})


def accumulate(stats, scene, decision, labels, mask, agent_table, *, easy_cut, hard_cut,
               controls=None):
    loc = np.searchsorted(scene.agent_ids, scene.target_ids)
    b = decision["baseline"][loc]
    types = np.array([agent_table[str(a)]["agent_type"] for a in scene.target_ids])
    stats["queries"] += 1
    stats["targets"] += len(loc)
    stats["visible"] += len(scene.agent_ids)
    stats["unknown_cv_context"] += int((~scene.cv_valid).sum())
    stats["complete"] += int(mask.all(1).sum())
    stats["known_steps"] += int(mask.sum())
    stats["rejected_outputs"] += int((~decision["model_output_supported"]).sum())
    stats["eligible"] += int(decision["eligible"].sum())
    if decision["budgets"]:
        report = decision["budgets"]["half"]
        stats["half_unmatched"] += int(not report["matched"])
    difference = decision["choices"]["half_joint"] != decision["choices"]["half_unary"]
    stats["half_joint_diff_agents"] += int(difference.sum())
    stats["half_joint_diff_queries"] += int(difference.any())

    def update(bucket, name, p, switched):
        s = score_arrays(b, p, labels, mask, scene.scales, easy_cut=easy_cut, hard_cut=hard_cut)
        item = bucket.setdefault(name, dict(switches=0, gain_lower_sum=0., gain_upper_sum=0., slices={}))
        item["switches"] += int(switched.sum())
        item["gain_lower_sum"] += float(s["gain_lower"].sum())
        item["gain_upper_sum"] += float(s["gain_upper"].sum())
        subsets = dict(all=s["complete"], easy=s["easy"], hard=s["hard"], zero=s["zero"],
                       pedestrian=s["complete"] & (types == "pedestrian"),
                       vehicle=s["complete"] & (types == "vehicle"))
        for subset, take in subsets.items():
            part = item["slices"].setdefault(subset, dict(n=0, ade_sum=0., fde_sum=0.,
                baseline_ade_sum=0., baseline_fde_sum=0., normalized_ade_sum=0., harm_count=0))
            part["n"] += int(take.sum())
            for target, key in (("ade_sum", "ade"), ("fde_sum", "fde"),
                    ("baseline_ade_sum", "baseline_ade"), ("baseline_fde_sum", "baseline_fde"),
                    ("normalized_ade_sum", "normalized_ade")):
                part[target] += float(s[key][take].sum())
            part["harm_count"] += int((s["ade"][take] > s["baseline_ade"][take]).sum())

    for name in ARMS:
        switch = decision["choices"][name][loc]
        p = np.where(switch[:, None, None], decision["candidate"][loc], b)
        update(stats["arms"], name, p, switch)
    if controls is not None:
        for name, p in zip(BASELINES, controls):
            update(stats["baseline_controls"], name, p, np.any(p != b, axis=(1, 2)))
    return stats
