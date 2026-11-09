"""Train causal choice on source-excluded donors and compare exact counts on A."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import resource
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 .venv-pytorch')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_joint_oof_bank as donor
from src.evaluation import m3w_cv_anchor_budget as utility
from src.evaluation import m3w_joint_path_audit as geometry
from src.world_model import m3w_cv_joint_selector as model

study, prior, prefix = donor.study, donor.prior, donor.prefix
np, torch = donor.np, donor.torch
read, sha, once, stable, artifact, checked = donor.read, donor.sha, donor.once, donor.stable, donor.artifact, donor.checked
OUT = donor.OUT.parent / 'moving_cv_joint_selector_v1'
PRIVATE = donor.PRIVATE.parent / OUT.name
SEEDS, STEPS, BATCH, CAP, PAIR_WEIGHT = donor.SEEDS, 1200, 128, 8, .25
ARMS = ('independent', 'uniform', 'additive_pair', 'joint')
CV, OLS = prefix.CV, prefix.OLS
PAIRS = {'joint_vs_additive': ('additive_pair', 'joint'), 'joint_vs_independent': ('independent', 'joint'),
         'joint_vs_uniform': ('uniform', 'joint'), **{f'{k}_vs_ols4': (OLS, k) for k in ARMS},
         **{f'{k}_vs_cv': (CV, k) for k in ARMS},
         'joint_vs_additive_matched': ('additive_pair_matched', 'joint_matched'),
         'joint_vs_independent_matched': ('independent_matched', 'joint_matched')}
GEOM = ('joint_vs_additive', 'joint_vs_independent', 'joint_vs_uniform', 'joint_vs_ols4', 'joint_vs_cv',
        'joint_vs_additive_matched', 'joint_vs_independent_matched')


def register():
    f = donor.OUT / 'freeze.json'
    d = read(f)
    if (d['registration'] != sha(donor.OUT / 'registration.json') or d['fits'] != 18
            or d['a_labels_accessed'] or d['held_source_draws'] or d['confirmation_access']):
        raise ValueError('Complete the source-excluded donor predictions first')
    refs = prior.parent.register()
    reg = OUT / 'registration.json'
    once(reg, stable(dict(donor=artifact(f), a_forecast_receipt=artifact(prior.parent.REG), files=[artifact(p)
        for p in (Path(__file__), Path(model.__file__), Path(utility.__file__), Path(geometry.__file__),
                  Path(prefix.trainer.__file__), OUT / 'protocol.md')],
        steps=STEPS, batch=BATCH, cap=CAP, pair_weight=PAIR_WEIGHT, seeds=SEEDS, arms=ARMS,
        independent_confirmation=False, original_2pct=True, uses_a_targets_only_for_scoring=True)))
    return sha(reg), d, refs


def layout(data, identity):
    p = donor.prior.prior
    home = p.bank.old.HOME
    adm = p.prep.load_admission(p.ROOT, p.ROOT / read(p.source.CONFIG)['admission'])
    lookup = {c['clip']: c for c in read(home / 'movingdronecrowd_intake_v1/analysis.json')['recordings']}
    hs, query, ids = [], [], []
    offset, slot = 0, 0
    for node, rec in sorted(adm['recordings'].items()):
        meta = lookup[rec['clip']]
        n = meta['window_support']['obs8_pred12_native_stride1']['history_complete_agent_instances']
        start = offset
        offset += n
        if rec['exclusion_group'] not in identity['normalizers'] or not n:
            continue
        file = p.ROOT / rec['source_annotation_path']
        p.prep.require_fit_record(adm, node, 'fit', file)
        stream = p.prep.mot_record(file.read_bytes(), rec['clip'], range(1, meta['image_frames']+1))
        total = 0
        for t in range(7, stream.valid.shape[1]-12):
            past = p.prep.past_inputs(stream, t, p.prep.SPEC)
            x = past['positions'][past['history_mask'].all(1)]
            if len(x):
                hs.append(x)
                query.extend([slot]*len(x))
                slot += 1
                total += len(x)
        if total != n:
            raise ValueError('Fitting history roster changed')
        ids.extend(range(start, offset))
    h = np.concatenate(hs)
    np.testing.assert_array_equal(ids, data['ids'])
    np.testing.assert_array_equal((h-h[:, -1:]).astype(np.float32).reshape(len(h), 16), data['geometry'][:, :16])
    return h, np.asarray(query)


def read_donor(d, seed, data):
    by = {}
    for fam in ('eqmotion', 'pct'):
        p = np.full((len(data['ids']), 12, 2), np.nan, np.float32)
        for c in d['cells']:
            info = c['identity']
            if info['family'] != fam or info['seed'] != seed:
                continue
            at = np.flatnonzero(np.isin(data['sites'], info['donor_held']))
            if (set(data['sites'][at]) & set(info['train_sites']) or np.isfinite(p[at]).any()
                    or len(at) != c['held_rows']):
                raise ValueError('OOF prediction source overlap or alignment failure')
            with np.load(checked(c['forecast']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], data['ids'][at])
                p[at] = z['prediction']
        if not np.isfinite(p).all():
            raise ValueError('Donor forecast bank is incomplete')
        by[fam] = p
    return by


def query_groups(q):
    order, offsets = utility.row_ranks(q)
    return [order[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]


def edges(h, groups):
    blocks = [geometry.local_edges(h[g]) for g in groups]
    at = np.concatenate([g[e] for g, e in zip(groups, blocks, strict=True)], axis=1)
    return at, blocks, np.r_[0, np.cumsum([e.shape[1] for e in blocks])]


def bank(h, p):
    raw = study.motion.bank(h)
    return raw[:, 0], raw[:, 1], np.concatenate((raw[:, 1:], np.stack((p['eqmotion'], p['pct']), axis=1)), axis=1)


def row_sample(h, cv, ols, paths, y, valid, sites):
    x = model.row_features(h, paths-cv[:, None])
    _, _, gain, _ = utility.candidate_terms(cv, ols, paths, y, valid)
    s = model.scale(h)[:, None, None]
    e, _ = study.prior.parent.readout.errors(cv, y, valid)
    easy = float(np.quantile(e[np.isfinite(e) & (e > 0.)], .25))
    risk = utility.risk_excess(cv, paths, y, valid, easy)
    target = np.concatenate((gain/(12*s), risk/s), axis=-1).clip(-8., 8.).astype(np.float32)
    groups = [np.flatnonzero(sites == k) for k in sorted(set(sites))]
    def sample(rng):
        q = draw(groups, rng)
        j = rng.integers(x.shape[1], size=len(q))
        return torch.from_numpy(x[q, j]), torch.from_numpy(target[q, j])
    return sample, x.shape[-1]


def draw(groups, rng):
    counts = rng.multinomial(BATCH, np.full(len(groups), 1./len(groups)))
    return np.concatenate([g[rng.integers(len(g), size=k)] for g, k in zip(groups, counts, strict=True)])


def edge_sample(h, cv, paths, y, valid, sites, group):
    i, j = edges(h, group)[0]
    if not np.array_equal(sites[i], sites[j]):
        raise ValueError('Pair crosses fitting source boundary')
    groups = [np.flatnonzero(sites[i] == k) for k in sorted(set(sites[i]))]
    delta = paths-cv[:, None]
    def actual(rng):
        at = draw(groups, rng)
        a, b = i[at], j[at]
        k, l = rng.integers(paths.shape[1], size=(2, len(at)))
        aa, bb = delta[a, k], delta[b, l]
        zero = np.zeros_like(aa)
        ctx = model.pair_context(h[a], h[b], cv[a], cv[b])
        z = model.pair_features(ctx, zero, zero)
        xx, yy = [], []
        v = valid[a] & valid[b]
        for da, db in ((aa, zero), (zero, bb), (aa, bb)):
            xx.append(model.pair_features(ctx, da, db))
            yy.append(model.pair_target(cv[a]-cv[b], y[a]-y[b], v, da, db, ctx[-1]))
        return (torch.from_numpy(np.concatenate(xx)), torch.from_numpy(np.tile(z, (3, 1))),
                torch.from_numpy(np.concatenate(yy).astype(np.float32)))
    return actual, 22


def row_loss(net, x, y):
    return torch.nn.functional.smooth_l1_loss(net(x), y)


def edge_loss(net, x, zero, y):
    return torch.nn.functional.smooth_l1_loss((net(x)-net(zero)).flatten(), y)


@torch.inference_mode()
def pick(row, pair, h, cv, paths, groups):
    delta = paths-cv[:, None]
    parts = []
    for start in range(0, len(h), 2048):
        sl = slice(start, start+2048)
        x = model.row_features(h[sl], delta[sl])
        parts.append(model.rank_output(row(torch.from_numpy(x)).numpy()))
    score = np.concatenate(parts)
    legal = np.linalg.norm(delta, axis=-1).max(-1) > 1e-8
    best = np.where(legal, score, -np.inf).argmax(1)
    diff = delta[np.arange(len(h)), best]
    diff[~legal.any(1)] = 0.
    (i, j), local, off = edges(h, groups)
    value = np.zeros((len(i), 3))
    for start in range(0, len(i), 4096):
        sl = slice(start, start+4096)
        a, b = i[sl], j[sl]
        d1, d2 = diff[a], diff[b]
        z = np.zeros_like(d1)
        ctx = model.pair_context(h[a], h[b], cv[a], cv[b])
        x0 = pair(torch.from_numpy(model.pair_features(ctx, z, z))).numpy().ravel()
        for k, (p, q) in enumerate(((d1, z), (z, d2), (d1, d2))):
            value[sl, k] = (pair(torch.from_numpy(model.pair_features(ctx, p, q))).numpy().ravel()-x0)
    result = {k: np.full(len(h), -1, dtype=np.int8) for k in ARMS}
    for ix, g in enumerate(groups):
        v = value[off[ix]:off[ix+1]]
        chosen = model.choose(score[g], legal[g], local[ix], *v.T, CAP, PAIR_WEIGHT)
        count = min(CAP, int(legal[g].any(1).sum()))
        for arm in ARMS:
            j = chosen[arm]
            at = np.flatnonzero(j >= 0)
            if len(at) != count or not legal[g[at], j[at]].all():
                raise ValueError('Matched query counts or legality changed')
            result[arm][g] = j
    return result


def a_population():
    a = study.parent.parent.a_inputs()
    q = np.unique(np.stack((a['recordings'].astype(str), a['queries'].astype(str)), axis=1), axis=0,
                  return_inverse=True)[1].reshape(-1)
    return a, q


def fit_freeze(reg, d, refs, replay, beat):
    _, jobs = study.parent.parent.register()
    fit = donor.prior.prior.fitting_only(jobs[0]['identity'], beat)
    h, q = layout(fit, jobs[0]['identity'])
    a, aq = a_population()
    if set(a['sites']) & set(fit['sites']):
        raise ValueError('Fit/evaluation sources overlap')
    cells, updates = [], 0
    for seed in SEEDS:
        cv, ols, p = bank(h, read_donor(d, seed, fit))
        row, width = row_sample(h, cv, ols, p, fit['target'], fit['valid'], fit['sites'])
        edge, width_edge = edge_sample(h, cv, p, fit['target'], fit['valid'], fit['sites'], query_groups(q))
        fits, nets = [], []
        for name, sampler, w, n, loss in (('row', row, width, 4, row_loss), ('edge', edge, width_edge, 1, edge_loss)):
            ident = dict(registration=reg, seed=seed, head=name)
            net, f, u = prefix.trainer.fit(ident, f'{name}_{seed}', seed, lambda: model.head(w, n),
                                          STEPS, sampler, loss, replay, beat)
            updates += u
            fits.append(f)
            nets.append(net)
        act_cv, _, act_paths = bank(a['history'], prior.parent.read_banks(refs, seed, a['global_ids'], a['sites']))
        act = pick(*nets, a['history'], act_cv, act_paths, query_groups(aq))
        choice = study.parent.parent.save_actions(PRIVATE / f'choice_{seed}.npz', dict(ids=a['global_ids'], **act))
        cells.append(dict(seed=seed, fits=fits, choices=choice))
        beat(phase='a_frozen', seed=seed, updates=updates)
    f = OUT / 'freeze.json'
    once(f, stable(dict(registration=reg, before_a_labels=True, cells=cells, training_updates=len(cells)*2*STEPS,
                        fit_rows=len(h), fit_queries=len(np.unique(q)))))
    return f, updates


def apply(cv, bank, action):
    p = cv.copy()
    use = action >= 0
    p[use] = bank[use, action[use]]
    return p


def pairs_for_record(h, paths, y, valid, groups):
    sums = np.zeros((len(GEOM), 4))
    names = tuple(paths)
    compare = [tuple(names.index(v) for v in PAIRS[k]) for k in GEOM]
    for g in groups:
        z = np.stack([paths[n][g] for n in names])
        _, d, _, (all_steps, known, _, _) = geometry.query(h[g], z, y[g], valid[g], compare)
        if all_steps:
            sums[:, 1] += d[:, 1]
            sums[:, 3] += 1
        if known:
            sums[:, 0] += d[:, 0]
            sums[:, 2] += 1
    return sums.tolist()


def pair_summary(rows):
    src = sorted({r['source'] for r in rows})
    draw = np.random.default_rng(73107).integers(len(src), size=(3000, len(src)))
    a = np.array([[np.sum([r['geometry'] for r in rows if (r['source'], r['seed']) == (s, k)], axis=0)
                   for k in SEEDS] for s in src])
    x = np.divide(a[..., :2], a[..., 2:], out=np.full_like(a[..., :2], np.nan), where=a[..., 2:] > 0.)
    return {name: {field: geometry.bootstrap(x[:, :, i, j], draw) for j, field in enumerate(('observed_pixel', 'lower_pixel'))}
            for i, name in enumerate(GEOM)}


def native_summary(rows):
    sources = sorted({r['source'] for r in rows})
    out = {}
    for name in rows[0]['metrics']:
        for label, metric in (('available_ade_pixel', 'known_ADE_pixel'),
                              ('available_fde_pixel', 'available_endpoint_FDE_diagnostic_pixel')):
            means = []
            for source in sources:
                v = [r['metrics'][name][metric] for r in rows if r['source'] == source
                     and r['metrics'][name][metric] is not None]
                if v:
                    means.append(float(np.mean(v)))
            out.setdefault(name, {})[label] = float(np.mean(means)) if len(means) == len(sources) else None
    return out


def score(reg, f, refs):
    frozen = read(f)
    if frozen['registration'] != reg or not frozen['before_a_labels'] or len(frozen['cells']) != 3:
        raise ValueError('Freeze all A choices first')
    a, q = a_population()
    old = prior.bank.old
    lab = old.prior.arrays(('target_eval', 'valid', 'origin'))
    y, mask = lab['target_eval']-lab['origin'][:, None], lab['valid']
    cuts = read(old.PARENTS[4])['cuts']
    cut = {s: cuts[str(v['job']['producer_fold'])] for v in read(old.PARENTS[3])['forecasts']
           for s in v['job']['donor_sites']}
    rows, roster, nedit = [], {}, {}
    for cell in frozen['cells']:
        seed = cell['seed']
        cv, ols, p = bank(a['history'], prior.parent.read_banks(refs, seed, a['global_ids'], a['sites']))
        with np.load(checked(cell['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], a['global_ids'])
            counts = np.bincount(q[z['independent'] >= 0], minlength=q.max()+1)
            for arm in ARMS:
                np.testing.assert_array_equal(counts, np.bincount(q[z[arm] >= 0], minlength=len(counts)))
            nedit[str(seed)] = int(counts.sum())
            paths = {CV: cv, OLS: ols, **{arm: apply(cv, p, z[arm]) for arm in ARMS},
                     'eqmotion': p[:, -2], 'pct': p[:, -1]}
            matched = model.matched_radius(cv, {k: paths[k] for k in ARMS}, q)
            paths.update({k+'_matched': v for k, v in matched.items()})
        for rec in np.unique(a['recordings']):
            ix = a['recordings'] == rec
            s, = np.unique(a['sites'][ix])
            roster[str(rec)] = str(s)
            local = {k: v[ix] for k, v in paths.items()}
            qq, c = q[ix], cut[s]
            r = prior.parent.readout.score_recording(local, {}, y[ix], mask[ix], qq, str(rec),
                                                     easy_cut=c['easy'], hard_cut=c['hard'])
            _, inv, cnt = np.unique(qq, return_inverse=True, return_counts=True)
            w = 1./cnt[inv]
            r['comparisons'] = {name: study.motion.full_masses(local[l], local[p], local[CV], y[ix], mask[ix], w)
                                for name, (l, p) in PAIRS.items()}
            geo = pairs_for_record(a['history'][ix], local, y[ix], mask[ix], query_groups(qq))
            rows.append(dict(r, source=str(s), seed=seed, geometry=geo))
    s = prefix.grouped.grouped_readout(rows, roster, SEEDS)
    if (s['views'], s['source_groups']) != (102, 14):
        raise ValueError('A roster changed')
    result = stable(dict(registration=reg, freeze=sha(f), agent=s, pair=pair_summary(rows), matched_counts=nedit,
                         native=native_summary(rows), matched_total_peak_per_query=True,
                         original_2pct=True, independent_confirmation=False, deployment_changed=False, submission_ready=False))
    once(OUT / 'readout.json', result)
    once(PRIVATE / 'recording_scores.json', stable(dict(readout=sha(OUT / 'readout.json'), rows=rows)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    prefix.trainer.PRIVATE = PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            x = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            prior.prior.source.atomic(PRIVATE / 'heartbeat.json', x)
            if v.get('step', STEPS) in (100, STEPS):
                print(x, flush=True)

        r, d, refs = register()
        f, u = fit_freeze(r, d, refs, args.replay, beat)
        score(r, f, refs)
        beat(phase='complete', updates=u, replay=args.replay, max_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == '__main__':
    main()
