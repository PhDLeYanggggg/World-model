"""Unit-consistent source-cutoff context; not an external calibration method."""
import numpy as np
from src.world_model.m3w_dimensionless_risk import features as previous_features


def features(raw, distance, scale, cutoff):
    if np.ndim(cutoff) != 0 or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("A positive frozen training-only scalar cutoff is required")
    x = previous_features(raw, distance, scale, "native")
    s, d = np.asarray(scale, float), np.asarray(distance, float)
    out = np.column_stack((x[:, :354], np.log(s/cutoff), np.log1p(d/cutoff))).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Cutoff-relative features overflowed")
    return out


def information_loss_witness():
    """Synthetic scale change at fixed units, not a measured-data causal claim."""
    raw = np.zeros((2, 356), dtype=np.float32)
    s, d, error, cutoff = np.array([1., 10.]), np.array([2., 20.]), np.array([.5, 5.]), 1.
    old = previous_features(raw, d, s, "dimensionless")
    new = features(raw, d, s, cutoff)
    return dict(kind="synthetic_information_loss_witness_not_empirical_causation",
        dimensionless_shape_features_identical=bool(np.array_equal(old[0], old[1])),
        fixed_cutoff_easy_labels=(error <= cutoff).tolist(),
        cutoff_relative_features_distinguish=bool(not np.array_equal(new[0], new[1])),
        unit_relabel_preserves_labels=bool(np.array_equal(error <= cutoff, error*100 <= cutoff*100)),
        future_errors_used_only_as_synthetic_labels=True)
