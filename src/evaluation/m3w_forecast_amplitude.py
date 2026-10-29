"""Fixed causal forecast interpolation; outcome scoring remains separate."""
import numpy as np

from src.evaluation.m3w_cost_bank_factorization import factorized, validate
from src.evaluation.m3w_support_action_diagnostic import real_intervention_mask
from src.world_model.m3w_multiaction_cost import ARMS, select_actions


def coefficient(alpha):
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('One fixed finite coefficient in [0,1] required')
    return float(alpha)


def blend(floor, candidates, alpha):
    alpha = coefficient(alpha)
    floor, candidates = np.asarray(floor), np.asarray(candidates)
    if (floor.ndim != 3 or floor.shape[-1] != 2 or candidates.ndim != 4
            or candidates.shape[0] != len(floor) or candidates.shape[1] != 7
            or candidates.shape[2:] != floor.shape[1:]
            or not np.isfinite(floor).all() or not np.isfinite(candidates).all()):
        raise ValueError('Finite aligned floor and seven complete causal paths required')
    if alpha == 1:
        return candidates.copy()
    if alpha == 0:
        return np.broadcast_to(floor[:, None], candidates.shape).copy()
    return floor[:, None] + alpha * (candidates - floor[:, None])


def transported_bounds(prediction, alpha):
    """Convex-loss surrogate, NOT refitted expected moments or a risk certificate."""
    alpha = coefficient(alpha)
    out = validate(prediction).copy()
    out[..., (0, 1, 4)] *= alpha
    return out


def available(floor, candidates, allowed):
    floor, candidates, allowed = np.asarray(floor), np.asarray(candidates), np.asarray(allowed)
    blend(floor, candidates, 1)
    if allowed.shape != candidates.shape[:2] or allowed.dtype != bool:
        raise ValueError('Original past-only Boolean availability required')
    return allowed & np.stack([real_intervention_mask(p, floor) for p in candidates.swapaxes(0, 1)], 1)


def decisions(predictions, allowed, floor, candidates, alpha):
    if set(predictions) != set(ARMS):
        raise ValueError('All three original frozen cost banks required')
    prediction = {k: validate(v) for k, v in predictions.items()}
    prediction['factorized'] = factorized(prediction['physical_only'], prediction['neural_only'])
    paths = blend(floor, candidates, alpha)
    legal = available(floor, paths, allowed)
    choices, ranks, costs = {}, {}, {}
    for name, p in prediction.items():
        arm = 'physical_neural' if name == 'factorized' else name
        costs[name] = transported_bounds(p, alpha)
        choices[name], ranks[name] = select_actions(costs[name], legal, arm=arm)
        original, order = select_actions(p, allowed, arm=arm)
        take = original >= 0
        real = take & legal[np.arange(len(legal)), np.maximum(original, 0)]
        choices[name + '_same_set'] = np.where(real, original, -1)
        ranks[name + '_same_set'] = np.where(real, order, 0.)
    choices['floor'] = np.full(len(legal), -1, np.int64)
    ranks['floor'] = np.zeros(len(legal))
    return paths, legal, choices, ranks


def check_convex_labels(original, fractional, alpha):
    """Label-only numerical check. It cannot establish learned conditional calibration."""
    alpha = coefficient(alpha)
    a, b = np.asarray(original), np.asarray(fractional)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (7, 5):
        raise ValueError('Aligned seven-action evaluation labels required')
    known = np.isfinite(a).all((1, 2))
    np.testing.assert_array_equal(known, np.isfinite(b).all((1, 2)))
    np.testing.assert_array_equal(a[..., (2, 3)], b[..., (2, 3)])
    tolerance = 1e-8 * np.maximum(1., a[known, :, 2])
    excess = [alpha * a[known, :, 0] - b[known, :, 0],
              b[known, :, 1] - alpha * a[known, :, 1],
              b[known, :, 4] - alpha * a[known, :, 4]]
    if any(np.any(value > tolerance) for value in excess):
        raise ValueError('Fixed-mask convex loss inequality failed')
    return dict(known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        max_numerical_excess=max((float(x.max(initial=0.)) for x in excess), default=0.),
        future_labels_used_for_inference=False, learned_risk_certificate=False)
