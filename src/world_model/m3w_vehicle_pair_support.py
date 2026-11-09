"""Exact matched two-actor choices with a conservative easy-support score."""

import numpy as np

ARMS = ('independent', 'uniform', 'joint')


def choose(value, probability, peak):
    """All arms choose two distinct recipients at the same radius."""
    if value.shape != (*peak.shape, 6) or probability.shape != (len(peak),):
        raise ValueError('Expected aligned native-unit costs and actor probabilities')
    if (not np.isfinite(value).all() or not np.isfinite(peak).all()
            or not ((probability >= 0.) & (probability <= 1.)).all()):
        raise ValueError('Invalid causal score')
    row, family = np.nonzero(peak > 0.)
    i, j = np.triu_indices(len(row), 1)
    keep = row[i] != row[j]
    i, j = i[keep], j[keep]
    same = family[i] == family[j]
    if not same.any():
        return {}
    radius = peak[row, family]
    np.testing.assert_allclose(radius, radius[0], atol=1e-9, rtol=1e-9)
    v = value[row, family]
    p = probability[row]
    gain = v[:, :2]
    excess = v[:, 2:4] - .02*v[:, 4:6]
    scalar = gain.min(1) - np.maximum(excess, 0.).sum(1) - (1.-p)*radius
    additive = scalar[i]+scalar[j]
    # P(no supported recipient) <= 1 - max marginal support, without independence.
    joint = ((gain[i]+gain[j]).min(1) - np.maximum(excess[i]+excess[j], 0.).sum(1)
             - (1.-np.maximum(p[i], p[j]))*(radius[i]+radius[j]))
    out = {}
    for arm, scores in zip(ARMS, (additive, np.where(same, additive, -np.inf), joint), strict=True):
        k = int(scores.argmax())
        a = np.array([i[k], j[k]])
        out[arm] = (row[a], family[a], float(scores[k]))
    return out
