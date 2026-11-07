"""Past-observed crowd motion and candidate-vs-floor proximity features."""
import numpy as np

OBSERVED = 6
ROLLOUT = 6


def features(history, scale, bank, floor, groups):
    """Forecast coordinates are offsets from the final observed position."""
    n, families = bank.shape[:2]
    if (history.shape != (n, 8, 2) or floor.shape != (n, 12, 2)
            or bank.shape != (n, families, 12, 2) or scale.shape != (n,)
            or not np.all(scale > 0)):
        raise ValueError('Require aligned eight-step histories and twelve-step paths')
    out = np.zeros((n, families, OBSERVED + ROLLOUT), dtype=np.float32)
    for q in groups:
        if len(q) < 2:
            continue
        h = history[q]
        diag = np.eye(len(q), dtype=bool)
        unit = (scale[q, None] + scale[None, q]) * .5
        d = np.linalg.norm(h[:, None] - h[None, :], axis=-1) / unit[..., None]
        current = np.where(diag, np.inf, d[:, :, -1])
        closest = np.argmin(current, axis=1)
        motion = h[:, -1] - h[:, -4]
        heading = motion / np.linalg.norm(motion, axis=-1).clip(1e-8)[:, None]
        closing = np.maximum(d[:, :, -4] - d[:, :, -1], 0.)
        out[q, :, :OBSERVED] = np.column_stack((
            np.log1p(current.min(1)),
            np.log1p(np.where(diag, np.inf, d.min(2)).min(1)),
            np.log1p(closing.max(1)),
            np.arcsinh((d[np.arange(len(q)), closest, -4] - current.min(1))),
            (heading * heading[closest]).sum(1),
            np.log1p(np.exp(-current).sum(1)),
        ))[:, None, :]
        abs_floor = floor[q] + h[:, -1, None]

        def proximity(paths):
            distances = np.linalg.norm(paths[:, None] - abs_floor[None, :], axis=-1)
            near = np.where(diag[..., None], 0., np.exp(-distances / unit[..., None]))
            return np.column_stack([v for part in (near[:, :, :4], near[:, :, 4:])
                for v in (part.max((1, 2)), part.max(1).mean(1))])

        ref = proximity(abs_floor)
        for a in range(families):
            alt = proximity(bank[q, a] + h[:, -1, None])
            out[q, a, OBSERVED:] = np.column_stack((ref[:, 0], ref[:, 2], alt - ref))
    return out


def augment(examples, descriptors, rollout):
    end = OBSERVED + ROLLOUT if rollout else OBSERVED
    return [(q, i, j, np.column_stack((x, descriptors[i, j, :end])))
            for q, i, j, x in examples]
