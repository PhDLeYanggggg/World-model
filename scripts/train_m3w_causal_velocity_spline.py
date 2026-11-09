"""Train and score matched causal spline forecasters on exposed source rotations."""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch Python')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_fixed_budget_gain as parent
from src.world_model import m3w_causal_velocity_spline as model

old, base, prior, np, torch = parent.old, parent.base, parent.prior, parent.np, parent.torch
OUT = parent.OUT.parent / 'moving_causal_velocity_spline_v1'
PRIVATE = parent.PRIVATE.parent / OUT.name
STEPS = 1600
ARMS = (*model.MODES, 'selected', 'ols8', 'eqmotion', 'pct')


def draw_batch(inputs, y, v, sites):
    groups = [np.flatnonzero(sites == s) for s in sorted(set(sites))]
    if len(groups) != 3:
        raise ValueError('Require three fitting sources')

    def draw(rng):
        i = prior.draw(groups, rng)
        return tuple(torch.from_numpy(x[i]) for x in (*inputs, y, v))

    return draw


@torch.inference_mode()
def predict(net, inputs, mode):
    paths = [net(*(torch.from_numpy(a[i:i+2048]) for a in inputs), mode != 'ego').numpy()
             for i in range(0, len(inputs[0]), 2048)]
    p = np.concatenate(paths)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite forecast')
    return p


def calibrate(d, paths, cv, ols):
    full = prior.study.motion.full_masses
    by = {k: [] for k in paths}
    for site in sorted(set(d['sites'])):
        at = d['sites'] == site
        y, v, q = (d[k][at] for k in ('target', 'valid', 'query'))
        w = old.weight(q)
        for k, p in paths.items():
            m = full(ols[at], p[at], cv[at], y, v, w)
            if m['reference'] <= 0.:
                raise ValueError('Missing calibration reference')
            by[k].append(100.*m['known_delta']/m['reference'])
    scores = {k: float(np.mean(x)) for k, x in by.items()}
    return scores, max(model.MODES, key=scores.__getitem__)


def fit_cell(d, inputs, linear, c, reg, beat, replay):
    i = c['identity']
    roles = [set(i[k]) for k in ('train', 'calibration', 'held')]
    if any(len(s) != 3 for s in roles) or len(set.union(*roles)) != 9:
        raise ValueError('Expected disjoint source roles')
    tr, data = base.subset(d, i['train'])
    ca, cal = base.subset(d, i['calibration'])
    he, held = base.subset(d, i['held'])
    draw = draw_batch([x[tr] for x in inputs], data['target'].astype(np.float32), data['valid'], data['sites'])
    xc, xh = ([x[at] for x in inputs] for at in (ca, he))
    fits, predictions, calibration, state, updates = {}, {}, {}, None, 0
    for mode in model.MODES:
        def objective(net, history, peers, mask, frame, scale, velocity, y, v):
            p = net(history, peers, mask, frame, scale, velocity, mode != 'ego')
            return model.loss(p, velocity, y, v, scale, mode == 'completion')

        name = f'{mode}_{i["fold"]}_{i["seed"]}'
        net, fit, u = prior.prefix.trainer.fit(dict(i, arm=mode, registration=reg), name, i['seed'],
            model.Spline, STEPS, draw, objective, replay, beat)
        rng = torch.load(old.checked(fit['artifact']), weights_only=True, map_location='cpu')['rng']
        if state is not None and rng != state:
            raise ValueError('Matched fitting draws differ')
        state = rng
        updates += u
        fits[mode] = fit
        calibration[mode] = predict(net, xc, mode)
        predictions[mode] = predict(net, xh, mode)
    scores, choice = calibrate(cal, calibration, linear[ca, 0], linear[ca, 1])
    path = prior.study.parent.parent.save_actions(PRIVATE / f'paths_{i["fold"]}_{i["seed"]}.npz',
        dict(ids=held['ids'], **predictions))
    return dict(identity=i, cut=c['cut'], fits=fits, predictions=path,
                calibration_known_gain_pp=scores, selected=choice), updates


def absolute_masses(h, path):
    """Query-weighted known-step ADE and endpoint-available FDE."""
    y, v = h['target'], h['valid']
    e = np.linalg.norm(path-np.where(v[..., None], y, 0.), axis=-1)
    w = old.weight(h['query'])
    return np.array([w @ (np.where(v, e, 0.).sum(1)/v.sum(1).clip(1)),
        w @ v.any(1), w @ np.where(v[:, -1], e[:, -1], 0.), w @ v[:, -1]])


def score(d, donor, cells, linear, beat):
    rows, ab = [], {}
    pairs = parent.vs_references(ARMS)
    pairs.update(peers_vs_ego=('ego', 'peers'), completion_vs_peers=('peers', 'completion'),
                 selected_vs_eqmotion=('eqmotion', 'selected'), selected_vs_pct=('pct', 'selected'))
    bank = {seed: prior.read_donor(donor, seed, d) for seed in base.SEEDS}
    for c in cells:
        i = c['identity']
        ix, h = base.subset(d, i['held'])
        seed = i['seed']
        with np.load(old.checked(c['predictions']), allow_pickle=False) as p:
            np.testing.assert_array_equal(p['ids'], h['ids'])
            paths = {k: p[k] for k in model.MODES}
        paths.update({base.CV: linear[ix, 0], base.OLS: linear[ix, 1], 'ols8': linear[ix, 2],
                      'selected': paths[c['selected']], **{k: z[ix] for k, z in bank[seed].items()}})
        rows.extend(parent.rows_for(h, paths, paths[base.CV], c['cut'], pairs, i))
        for s in np.unique(h['sites']):
            at = h['sites'] == s
            per = {k: h[k][at] for k in ('target', 'valid', 'query')}
            ab[str(s), seed] = {k: absolute_masses(per, p[at]) for k, p in paths.items()}
        beat(phase='held_scored', fold=i['fold'], seed=seed)
    roster = {r['recording']: r['source'] for r in rows}
    summary = prior.prefix.grouped.grouped_readout(rows, roster, base.SEEDS)
    sites = sorted(set(roster.values()))
    forecast = {}
    for arm in (*ARMS, base.CV, base.OLS):
        n = np.array([ab[s, k][arm] for s in sites for k in base.SEEDS])
        forecast[arm] = dict(available_ade_pixel=float((n[:, 0]/n[:, 1]).mean()) if (n[:, 1] > 0.).all() else None,
            available_endpoint_fde_pixel=float((n[:, 2]/n[:, 3]).mean()) if (n[:, 3] > 0.).all() else None)
    e = summary['effects']['selected_vs_ols4']
    strong = all(e[k]['interval95'] is not None and e[k]['interval95'][0] > 0.
                 for k in ('known_pp', 'completion_lower_pp'))
    return dict(summary=summary, forecast=forecast, strong_direct_candidate=strong)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay', action='store_true', help='Do not train missing checkpoints')
    args = ap.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    prior.prefix.trainer.PRIVATE = PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE / 'train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 2), **v)
            prior.donor.prior.prior.source.atomic(PRIVATE / 'heartbeat.json', z)
            if v.get('phase') != 'fit' or v.get('step') in (100, STEPS):
                print(z, flush=True)

        cap = parent.OUT.parent / 'moving_ols_anchor_menu_v2' / 'readout.json'
        refs = (parent.OUT / 'freeze.json', parent.OUT / 'readout.json', cap)
        f, r, cap_readout = map(old.read, refs)
        if (r['freeze'] != old.sha(refs[0]) or not f['before_held_scoring']
                or any(x['independent_confirmation'] or x['a_labels_opened'] or not x['original_2pct']
                       for x in (r, cap_readout))):
            raise ValueError('Require development-only role freeze')
        _, donor, _ = base.register()
        reg = OUT / 'registration.json'
        old.once(reg, old.stable(dict(parents=[old.artifact(x) for x in
            (*refs, prior.donor.OUT / 'freeze.json', base.OUT / 'registration.json')],
            files=[old.artifact(x) for x in (Path(__file__), Path(model.__file__), OUT / 'protocol.md')],
            steps=STEPS, seeds=base.SEEDS, roles=[c['identity'] for c in f['cells']],
            original_2pct=True, no_external_or_confirmation_labels=True, full_roster=True)))
        d = base.fit_population(beat)
        groups = prior.query_groups(d['query'])
        if any(len(set(d['recording'][g])) != 1 for g in groups):
            raise ValueError('Query crosses recording')
        inputs = model.causal_inputs(d['history'], groups)
        linear = prior.study.motion.bank(d['history'])[:, :3]
        beat(phase='causal_inputs_ready', rows=len(d['ids']), queries=len(groups))
        cells, updates = [], 0
        for c in f['cells']:
            fit, n = fit_cell(d, inputs, linear, c, old.sha(reg), beat, args.replay)
            cells.append(fit)
            updates += n
            beat(phase='cell_frozen', fold=c['identity']['fold'], seed=c['identity']['seed'], updates=updates)
        fr = OUT / 'freeze.json'
        old.once(fr, old.stable(dict(registration=old.sha(reg), cells=cells,
            before_held_scoring=True, optimizer_steps=len(f['cells'])*len(model.MODES)*STEPS)))
        result = score(d, donor, old.read(fr)['cells'], linear, beat)
        old.once(OUT / 'readout.json', old.stable(dict(freeze=old.sha(fr), **result, original_2pct=True,
            independent_confirmation=False, a_labels_opened=False, deployment_changed=False)))
        beat(phase='complete', updates=updates, replay=args.replay)


if __name__ == '__main__':
    main()
