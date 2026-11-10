"""Past-derived map and simultaneous peer features for frozen exchange paths."""

import numpy as np


MAP, PEER = 7, 9
WIDTH = 2*(MAP+PEER)
MODES = ('motion', 'map', 'peer', 'aligned', 'misaligned')


def descriptors(h, cv, candidate, query, rows, raster):
    """Return map and peer descriptors by query and nominated slot."""
    unit = np.linalg.norm(np.diff(h, axis=1), axis=-1).mean(1).clip(1e-6)
    origin = h[:, -1:]
    hd, cd, pd = raster.sample(h), raster.sample(origin+cv), raster.sample(origin+candidate)
    ok = np.isfinite(hd).all(1) & np.isfinite(cd).all(1) & np.isfinite(pd).all(1)
    hd, cd, pd = (np.where(ok[:, None], z, 0.) for z in (hd, cd, pd))
    slope = np.clip((hd[:, -1]-hd[:, -4])/3., -unit, unit)
    trend = hd[:, -1:]+slope[:, None]*np.arange(1, cv.shape[1]+1)
    c_error, p_error = (np.abs(v-trend).mean(1) for v in (cd, pd))
    mapped = np.column_stack((ok, np.arcsinh(hd[:, -1]/unit), slope/unit,
                              np.arcsinh(c_error/unit), np.arcsinh(p_error/unit),
                              np.arcsinh((p_error-c_error)/unit), (hd.min(1) < 0.) & (hd.max(1) > 0.)))
    peer = np.zeros((len(rows), 2, PEER), dtype=float)
    road = np.zeros((len(rows), 2, MAP), dtype=float)
    for q, group in enumerate(rows):
        agents = np.flatnonzero(query == q)
        for slot, a in enumerate(group):
            if a < 0:
                continue
            if query[a] != q:
                raise ValueError('A nominated path is not in its simultaneous query')
            road[q, slot] = mapped[a]
            others = agents[agents != a]
            if not len(others):
                continue
            near = others[np.linalg.norm(origin[others]-origin[a], axis=-1).reshape(-1).argmin()]
            d = np.linalg.norm(h[a]-h[near], axis=-1)
            uc = np.linalg.norm(origin[a]+cv[a]-origin[near]-cv[near], axis=-1)
            up = np.linalg.norm(origin[a]+candidate[a]-origin[near]-cv[near], axis=-1)
            v, w = h[a, -1]-h[a, -2], h[near, -1]-h[near, -2]
            size = np.linalg.norm(v)*np.linalg.norm(w)
            alignment = float(v@w/size) if size > 0. else 0.
            peer[q, slot] = [1., np.log1p(d[-1]/unit[a]), np.arcsinh((d[-4]-d[-1])/(3.*unit[a])),
                np.arcsinh((d[-2]-d[-1])/unit[a]), alignment, np.log1p(uc.min()/unit[a]), np.log1p(up.min()/unit[a]),
                np.arcsinh((up.min()-uc.min())/unit[a]), np.arcsinh((up-uc).mean()/unit[a])]
    return road.astype(np.float32), peer.astype(np.float32)


def aligned_and_rotated(road, peer, recording):
    """Preserve recording and support; rotate geometry only in matched strata."""
    if (road.shape != (len(recording), 2, MAP) or peer.shape != (len(recording), 2, PEER)
            or not np.isfinite(road).all() or not np.isfinite(peer).all()):
        raise ValueError('Map or peer descriptors are not aligned')
    x = np.column_stack((road.reshape(-1, 2*MAP), peer.reshape(-1, 2*PEER)))
    support = np.column_stack((road[:, :, 0], peer[:, :, 0])) > 0.
    real, wrong = np.zeros_like(x), np.zeros_like(x)
    use = np.zeros(len(x), dtype=bool)
    for name in sorted(set(recording)):
        ix = np.flatnonzero(recording == name)
        groups = {}
        for i in ix:
            if support[i].any():
                groups.setdefault(tuple(support[i]), []).append(i)
        for g in groups.values():
            if len(g) > 1:
                use[g] = True
                real[g], wrong[g] = x[g], x[np.roll(g, 1)]
    return real, wrong, use


def mask(width, mode):
    """Use equal-width nets and keep non-modality features identical."""
    if mode not in MODES:
        raise ValueError('Unknown context arm')
    w = width + 2*WIDTH
    m = np.ones(4*w+9, dtype=np.float32)
    cols = np.zeros(2*WIDTH, dtype=np.float32)
    if mode in ('map', 'aligned', 'misaligned'):
        cols[:2*MAP] = 1.
    if mode in ('peer', 'aligned', 'misaligned'):
        cols[2*MAP:WIDTH] = 1.
    for start in (0, w, 2*w+6, 3*w+6):
        m[start+width:start+w] = cols
    return m
