"""Training-side source cross-fitting with an untouched outer-site boundary."""
import numpy as np

from src.world_model.m3w_source_start_probe import fit_normalizer, normalize


def nested_partition(sites, tracks, recordings, outer_site, inner_site):
    sites, tracks, recordings = map(np.asarray, (sites, tracks, recordings))
    if (sites.ndim != 1 or tracks.shape != sites.shape or recordings.shape != sites.shape
            or outer_site == inner_site or outer_site not in sites or inner_site not in sites):
        raise ValueError('Distinct supported outer/inner sites and aligned identities required')
    train = np.flatnonzero((sites != outer_site) & (sites != inner_site))
    held = np.flatnonzero(sites == inner_site)
    outer = np.flatnonzero(sites == outer_site)
    if not len(train):
        raise ValueError('Empty inner training complement')
    for identities in (tracks, recordings):
        groups = [set(identities[index]) for index in (train, held, outer)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError('Recording or scoped-track leakage across nested roles')
    return train, held, outer


def fold_normalizer(features, train_ids):
    ids = np.asarray(train_ids, dtype=int)
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Nonempty unique training identities required')
    weights = np.full(len(ids), 1 / len(ids))
    fitted = fit_normalizer(features[ids], weights)
    transformed, _ = normalize(features, fitted)
    return fitted, transformed, weights


def validate_query(ids, lower, upper, allowed):
    ids = np.asarray(ids, dtype=int)
    if (ids.ndim != 1 or not len(ids) or np.any(ids < lower) or np.any(ids >= upper)
            or not np.asarray(allowed)[ids].all()):
        raise ValueError('Query outside the registered training-side role')
    return ids


def assemble_oof(expected_ids, pieces):
    expected = np.asarray(expected_ids, dtype=int)
    ids = np.concatenate([np.asarray(p['ids'], dtype=int) for p in pieces])
    if len(np.unique(expected)) != len(expected) or len(np.unique(ids)) != len(ids):
        raise ValueError('Each source query must have exactly one OOF prediction')
    order = np.argsort(ids)
    if not np.array_equal(ids[order], expected):
        raise ValueError('OOF coverage or canonical ordering mismatch')
    prediction = np.concatenate([p['prediction'] for p in pieces])[order]
    scale = np.concatenate([p['cost_scale'] for p in pieces])[order]
    if (prediction.shape != (len(expected), 12, 2) or scale.shape != expected.shape
            or not np.isfinite(prediction).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError('Finite aligned OOF predictions and training-only scales required')
    return prediction, scale


def cost_labels(prediction, target, scale):
    prediction, target, scale = map(np.asarray, (prediction, target, scale))
    if (prediction.ndim != 3 or prediction.shape[1:] != (12, 2)
            or prediction.shape != target.shape or scale.shape != (len(target),)
            or not all(np.isfinite(a).all() for a in (prediction, target, scale))
            or np.any(scale <= 0)):
        raise ValueError('Aligned finite supervision and positive fold-training scales required')
    cv = np.linalg.norm(target.astype(float), axis=-1).mean(1)
    ade = np.linalg.norm(prediction.astype(float) - target, axis=-1).mean(1)
    fde = np.linalg.norm(prediction[:, -1].astype(float) - target[:, -1], axis=-1)
    return dict(cv_ade=cv, ade=ade, fde=fde, signed_gain=(cv - ade) / scale,
                beneficial=ade < cv, harmful=ade > cv)
