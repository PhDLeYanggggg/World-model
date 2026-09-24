"""Versioned numerical precision for the engineering-only canonical prefix."""
import numpy as np
from src.data_unification.m3w_external_prefix_adapter import ExternalPrefixAdapter
from src.data_unification.m3w_unit_free_prefix import UnitFreePrefixAdapter


class QuantizedPrefixAdapter(UnitFreePrefixAdapter):
    """Apply fixed 1e-9 dimensionless precision before target-local normalization.

    This does not change source clocks or interpolate trajectories. It can remove
    sub-precision motion, which callers must measure rather than calling lossless.
    Frozen native models are engineering probes only; new deployment needs refit.
    """
    def __init__(self, prefix_points, *, query_frame, recording_id):
        checked = ExternalPrefixAdapter(prefix_points, query_frame=query_frame, recording_id=recording_id)
        points = checked.points.copy()
        current = points[points[:, 0] == query_frame]
        origin = current[np.argmin(current[:, 1]), 2:].copy()
        centered = points[:, 2:] - origin
        extent = float(np.linalg.norm(centered, axis=1).max())
        if not np.isfinite(extent) or extent <= 0:
            raise ValueError("Zero-extent prefix has no identifiable scene unit")
        points[:, 2:] = np.round(centered/extent, decimals=9)
        super().__init__(points, query_frame=query_frame, recording_id=recording_id)
        self.outer_origin = self.outer_origin*extent + origin
        self.outer_scale *= extent
        self.metadata.update(feature_schema="past_extent_precision9_v2_requires_refit",
                             dimensionless_coordinate_quantum=1e-9, lossless=False)
