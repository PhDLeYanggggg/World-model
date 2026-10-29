"""Other-recording local cost estimates; query labels are never arguments."""
import hashlib

import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from src.world_model.m3w_completion_cost import completion_endpoints

MODES = ('local_known', 'local_bounds', 'global_bounds', 'shuffled_bounds')


def encode(observed, envelope, scales):
    endpoints, known = completion_endpoints(observed, envelope)
    scales = np.asarray(scales)
    if scales.shape != (len(endpoints),) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Positive fitting-only cost scales required')
    for col in (2, 3):
        np.testing.assert_allclose(endpoints[:, :, col], np.broadcast_to(endpoints[:, :1, col],
            endpoints.shape[:2]), rtol=1e-6, atol=1e-6)
    if (np.any(endpoints[..., 3] > endpoints[..., 2]+1e-6)
            or np.any(endpoints[..., 4] > endpoints[..., 1]+1e-6)):
        raise ValueError('Coupled easy moments required')
    out = endpoints.copy()
    for col in (0, 1, 4):
        out[..., col] = np.divide(endpoints[..., col], envelope,
            out=np.zeros_like(envelope, dtype=float), where=envelope > 0)
    out[..., 2:4] /= scales[:, None, None]
    return out, known


def shuffled_targets(target, known, recordings, seed=20261029):
    result = np.asarray(target).copy()
    for recording in sorted(set(recordings.tolist())):
        ix = np.flatnonzero(known & (recordings == recording))
        salt = int.from_bytes(hashlib.sha256(f'{seed}|{recording}'.encode()).digest()[:8], 'little')
        result[ix] = target[np.random.default_rng(salt).permutation(ix)]
    return result


def aggregate(target, known, real, weight, *, bounds):
    """One common reference and convex harm fractions, without reference offsets."""
    t, known, real, w = map(np.asarray, (target, known, real, weight))
    if (t.ndim != 4 or t.shape[2:] != (7, 5) or known.shape != t.shape[:2]
            or known.dtype != bool or real.shape != t.shape[:3] or real.dtype != bool
            or w.shape != known.shape or not np.isfinite(t).all() or (t < 0).any()
            or not np.isfinite(w).all() or (w <= 0).any()):
        raise ValueError('Aligned fitted donor moments and source weights required')
    mask = np.ones_like(known) if bounds else known
    qw = w*mask
    denominator = qw.sum(1)
    shared = (qw[..., None]*t[:, :, 0, 2:4]).sum(1)/denominator.clip(1e-300)[:, None]
    aw = qw[..., None]*real
    mass = aw.sum(1)
    fraction = (aw[..., None]*t[..., [0, 1, 4]]).sum(1)/mass.clip(1e-300)[..., None]
    supported = (mass > 0) & (denominator[:, None] > 0)
    return shared, fraction, supported


def decode(shared, fraction, query_envelope, query_scale):
    if (shared.shape != (len(query_scale), 2) or fraction.shape != (len(query_scale), 7, 3)
            or query_envelope.shape != (len(query_scale), 7)
            or not np.isfinite(query_scale).all() or (query_scale <= 0).any()
            or not np.isfinite(query_envelope).all() or (query_envelope < 0).any()):
        raise ValueError('Aligned moment averages and positive causal query scales required')
    out = np.zeros((len(query_scale), 7, 5), float)
    out[..., [0, 1, 4]] = fraction*query_envelope[..., None]
    out[..., 2:4] = shared[:, None]*query_scale[:, None, None]
    out[..., 0] = np.minimum(out[..., 0], out[..., 2])
    if (np.any(out[..., 0]+out[..., 1] > query_envelope+1e-5)
            or np.any(out[..., 4] > out[..., 1]+1e-5)
            or np.any(out[..., 3] > out[..., 2]+1e-5)):
        raise AssertionError('Convex coupled moment constraints changed')
    return out


def estimate(target, known, real, weight, query_envelope, query_scale, *, bounds):
    shared, fraction, supported = aggregate(target, known, real, weight, bounds=bounds)
    return decode(shared, fraction, query_envelope, query_scale), supported


def descriptor(x, scene, preprocess):
    """Fixed physical-bank mean of standardized causal fields, not a learned metric."""
    if x.ndim != 3 or x.shape[1:] != (7, 380) or scene.shape != (len(x), 7, 18):
        raise ValueError('Original 380 motion and18 static action fields required')
    joined = np.concatenate((x[:, :6], scene[:, :6]), -1)
    return np.clip((joined-preprocess['mean'])/preprocess['std'],
                   -preprocess['clip'], preprocess['clip']).mean(1).astype(np.float32)


def retrieve(z, recordings, causal, *, query_ids=None, k=32, chunk=256,
             heartbeat=lambda **_: None, cached=None, store=lambda *_: None):
    """Exact brute-force neighbors; exclude the entire original query recording."""
    z, recordings, causal = map(np.asarray, (z, recordings, causal))
    if (z.ndim != 2 or not np.isfinite(z).all() or recordings.shape != (len(z),)
            or causal.shape != (len(z),) or causal.dtype != bool or k != 32):
        raise ValueError('Finite causal descriptors, recording identities and fixed k32 required')
    queries = np.flatnonzero(causal) if query_ids is None else np.asarray(query_ids)
    if queries.ndim != 1 or len(np.unique(queries)) != len(queries) or not causal[queries].all():
        raise ValueError('Distinct past-eligible query rows required')
    result = np.empty((len(queries), k), np.int64)
    positions = np.full(len(z), -1, np.int64); positions[queries] = np.arange(len(queries))
    references = []
    with threadpool_limits(limits=4):
        for recording in sorted(set(recordings[queries].tolist())):
            ix = queries[recordings[queries] == recording]
            donors = np.flatnonzero(causal & (recordings != recording))
            if len(donors) < k:
                raise ValueError('Insufficient other-recording causal donors; no invented estimates')
            previous = None if cached is None else cached(recording, ix, donors)
            if previous is not None:
                chosen = previous
            else:
                index = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean', n_jobs=1).fit(z[donors])
                parts = []
                for begin in range(0, len(ix), chunk):
                    at = ix[begin:begin+chunk]
                    neighbor = index.kneighbors(z[at], return_distance=False)
                    parts.append(donors[neighbor])
                    heartbeat(state='neighbors', recording=str(recording), completed=min(begin+chunk, len(ix)), total=len(ix))
                chosen = np.concatenate(parts)
                store(recording, ix, donors, chosen)
            if (chosen.shape != (len(ix), k) or not causal[chosen].all()
                    or np.any(recordings[chosen] == recording)
                    or any(len(set(row)) != k for row in chosen)):
                raise ValueError('Distinct causal other-recording donor rule violated')
            result[positions[ix]] = chosen
            references.append(dict(recording=str(recording), queries=len(ix), donors=len(donors),
                donor_recordings=len(set(recordings[donors].tolist())), same_recording_donors=0))
    return queries, result, references


def predictions(target, known, allowed, sources, recordings, causal, queries, neighbors,
                query_envelope, query_scale, *, chunk=256):
    """Labels are fitted donor state only, not query-label inference arguments."""
    counts = {s: int((causal & (sources == s)).sum()) for s in sorted(set(sources.tolist()))}
    weight = np.array([1/counts[s] if counts[s] else 0. for s in sources])
    shuffled = shuffled_targets(target, known, recordings)
    output = {m: np.zeros((len(queries), 7, 5)) for m in MODES}
    support = {m: np.zeros((len(queries), 7), bool) for m in MODES}
    for begin in range(0, len(queries), chunk):
        ids = queries[begin:begin+chunk]; nn = neighbors[begin:begin+chunk]
        for mode in ('local_known', 'local_bounds', 'shuffled_bounds'):
            values = shuffled if mode == 'shuffled_bounds' else target
            p, yes = estimate(values[nn], known[nn], allowed[nn], weight[nn],
                              query_envelope[ids], query_scale[ids], bounds=mode != 'local_known')
            output[mode][begin:begin+len(ids)] = p; support[mode][begin:begin+len(ids)] = yes
    for recording in sorted(set(recordings[queries].tolist())):
        take = np.flatnonzero(recordings[queries] == recording)
        donors = np.flatnonzero(causal & (recordings != recording))
        # The global control has identical source weights and exclusions but no distance matching.
        shared, fraction, yes = aggregate(target[donors][None], known[donors][None],
                                          allowed[donors][None], weight[donors][None], bounds=True)
        for begin in range(0, len(take), chunk):
            pos = take[begin:begin+chunk]; ids = queries[pos]
            p = decode(np.broadcast_to(shared, (len(pos), 2)), np.broadcast_to(fraction, (len(pos), 7, 3)),
                       query_envelope[ids], query_scale[ids])
            output['global_bounds'][pos] = p; support['global_bounds'][pos] = yes
    return output, support
