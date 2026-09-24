"""Versioned unit-free input contract; existing pixel-trained heads stay frozen."""
from __future__ import annotations

import numpy as np

from src.data_unification.m3w_external_prefix_adapter import ExternalPrefixAdapter
from src.world_model.m3w_native_gain_harm import cost_features

SCHEMA = "past_extent_relative_cost_v1_requires_refit"


class UnitFreePrefixAdapter(ExternalPrefixAdapter):
    """Canonicalize a prefix, not a dataset, using no learned or future statistic.

    All agents share one reversible outer transform. A fully degenerate prefix
    is unsupported, rather than being given an arbitrary native-unit epsilon.
    Internal numerical tolerances of the existing encoder are now dimensionless.
    """

    def __init__(self, prefix_points, *, query_frame, recording_id):
        checked = ExternalPrefixAdapter(prefix_points, query_frame=query_frame,
                                        recording_id=recording_id)
        p = checked.points.copy()
        current = p[p[:, 0] == query_frame]
        origin = current[np.argmin(current[:, 1]), 2:4].copy()
        centered = p[:, 2:4] - origin
        extent = float(np.linalg.norm(centered, axis=1).max())
        if not np.isfinite(extent) or extent <= 0:
            raise ValueError("Zero-extent prefix has no identifiable scene unit")
        p[:, 2:4] = centered / extent
        super().__init__(p, query_frame=query_frame, recording_id=recording_id)
        self.outer_origin = origin
        self.outer_scale = extent
        self.metadata.update(feature_schema=SCHEMA, learned_head_requires_refit=True,
                             coordinate_claim="dimensionless_past_extent_not_metric")

    def get_scene_inputs(self):
        scene = super().get_scene_inputs()
        visible = self.points[self.points[:, 0] == self.query_frame]
        for agent in scene["agents"]:
            inputs, transform = agent["inputs"], agent["coordinate_transform"]
            center, scale = transform["origin_xy"], transform["scale"]
            other = visible[visible[:, 1] != agent["agent_id"]]
            distance = np.linalg.norm(other[:, 2:] - center, axis=1)
            # A fixed dimensionless tie bucket prevents unit-roundoff reordering.
            nearest = np.lexsort((other[:, 1], np.round(distance, 9)))[:8]
            inputs["neighbor_xy"][:] = 0
            inputs["neighbor_frame_offsets"][:] = 0
            inputs["neighbor_mask"][:] = False
            own = self.points[self.points[:, 1] == agent["agent_id"]]
            velocity = own[-1, 2:] - own[-2, 2:]
            ttcs, closings = [], []
            for slot, row in enumerate(other[nearest]):
                past = self.points[self.points[:, 1] == row[1]]
                n = len(past)
                inputs["neighbor_xy"][slot, -n:] = (past[:, 2:] - center) @ transform["rotation"] / scale
                inputs["neighbor_frame_offsets"][slot, -n:] = past[:, 0] - self.query_frame
                inputs["neighbor_mask"][slot, -n:] = True
                if n < 2 or past[-2, 0] != self.query_frame - 1:
                    continue
                relative = row[2:] - center
                v = past[-1, 2:] - past[-2, 2:] - velocity
                squared, dot = float(v @ v), float(relative @ v)
                ttc = -dot / squared if squared > 1e-12 and dot < 0 else 120.
                ttcs.append(min(ttc / 12., 10.))
                closings.append(max(0., -dot / max(float(np.linalg.norm(relative)), 1e-8)) / scale)
            inputs["causal_features"][9] = min(ttcs, default=10.)
            inputs["causal_features"][10] = max(closings, default=0.)
        return scene

    def restore_outer(self, canonical_xy):
        xy = np.asarray(canonical_xy)
        if xy.shape[-1:] != (2,) or not np.isfinite(xy).all():
            raise ValueError("Finite xy predictions required")
        return xy * self.outer_scale + self.outer_origin


def unit_free_cost_features(geometry, candidate):
    """354 normalized features plus log1p normalized disagreement; no native scale.

    Not a projection that may be fed to an old head. A new head must be fitted
    with this schema, normalized targets, and its own source-only preprocessing.
    """
    g, p = np.asarray(geometry), np.asarray(candidate)
    x, same = cost_features(g, p, np.ones(len(g)))
    d = np.linalg.norm(p.astype(float) - g[:, 332:356].reshape(-1, 12, 2), axis=-1).mean(1)
    result = np.column_stack((x[:, :-1], np.log1p(d))).astype(np.float32)
    if result.shape != (len(g), 355) or not np.isfinite(result).all():
        raise ValueError("Finite versioned 355-column unit-free cost schema required")
    return result, d, same


def require_unit_free_checkpoint(checkpoint):
    if (checkpoint.get("feature_schema") != SCHEMA
            or checkpoint.get("feature_width") != 355
            or checkpoint.get("target_unit") != "past_normalized"):
        raise ValueError("New source-only unit-free fit required; native cost heads are incompatible")
