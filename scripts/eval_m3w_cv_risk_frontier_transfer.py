"""Test whether the frozen v141 scorer can exploit expanded causal menus."""
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_risk_frontier as audit

study, old, base, prior, np, torch = audit.study, audit.old, audit.base, audit.prior, audit.np, audit.torch
ARMS = ('original', 'expanded', 'expanded_matched', 'independent_matched', 'calibrated')
PAIRS = dict(expanded_vs_original=('original', 'expanded'),
    expansion_at_count=('original', 'expanded_matched'),
    joint_at_count=('independent_matched', 'expanded'))


def menu_packet(d, cv, ols, bank, packet, shape):
    gain, risk, legal, ref = study.parent.predictions({'shape': shape}, d, cv, ols, bank)['shape']
    h = d['history']
    scale = prior.model.scale(h)
    a = np.arange(legal.shape[1], dtype=np.int16)
    ray, amp = np.divmod(a, study.parent.model.M)
    out = []
    for p in packet:
        at = p['at']
        c, o, b, plans = cv[at], ols[at], bank[at], p['plans']
        g, r, l = gain[at], risk[at], legal[at]
        s = float(scale[at].mean())
        all_path = c[:, None] + np.asarray(study.parent.model.AMPLITUDES)[amp][None, :, None, None] * (b[:, ray] - c[:, None])
        agreement = np.linalg.norm(c - o, axis=-1).mean(1)[:, None] - np.linalg.norm(all_path - o[:, None], axis=-1).mean(2)
        active = plans >= 0
        forecast = study.paths(c, b, plans)
        projected = np.concatenate((g, r, agreement[..., None]), axis=-1)[np.arange(len(at)), plans]
        z = np.where(active[..., None], projected, 0.) / s
        disp = np.linalg.norm(forecast - c, axis=-1) / s
        edges = prior.geometry.local_edges(h[at])
        info = np.column_stack((z.mean(1), z.std(1), z.min(1), z.max(1),
            disp.mean((1, 2)), disp.max((1, 2)), active.mean(1),
            np.tile([np.log1p(len(at)), np.log1p(s), l.any(1).mean(), ref[at, 1].mean() / s], (len(plans), 1)),
            study.pair_inputs(h[at], c, forecast, active, s, edges)))
        x = np.arcsinh(info).astype(np.float32)
        score = np.where(l, g.min(-1) - 2. * r.clip(0.).sum(-1), -np.inf)
        action = score.argmax(1)
        order = np.argsort(-score[np.arange(len(at)), action], kind='stable')
        out.append(dict(p, x=x, counts=active.sum(1), order=order, action=action))
    return out


def restore(d, stem):
    path = audit.PRIVATE / f'menu_{stem}.npz'
    start = time.monotonic()
    while not path.exists():
        if time.monotonic()-start > 240.:
            raise TimeoutError(f'Menu {stem} not yet available from frontier audit')
        time.sleep(2.)
    groups = prior.query_groups(d['query'])
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], d['ids'])
        flat, lengths, before = (z[k] for k in ('plans', 'lengths', 'original'))
        widths = lengths * np.array([len(g) for g in groups])
        blocks = np.split(flat, np.cumsum(widths)[:-1])
        return [dict(at=at, original=int(n), plans=b.reshape(k, len(at)))
                for at, n, k, b in zip(groups, before, lengths, blocks, strict=True)], old.artifact(path)


def nets_for(cell, width):
    nets = {}
    for a, fit in cell['fits'].items():
        z = torch.load(old.checked(fit['artifact']), map_location='cpu', weights_only=True)
        if z['identity'] != fit['identity'] or z['step'] != study.STEPS:
            raise ValueError('Frozen scene scorer identity changed')
        net = prior.model.head(width, 6)
        net.load_state_dict(z['net'])
        nets[a] = net.eval()
    return nets


def choose(packet, est, old_choice):
    out = {k: np.full_like(old_choice, -1) for k in ARMS}
    new = 0
    for p, y in zip(packet, est, strict=True):
        n, counts, at = p['original'], p['counts'], p['at']
        i = study.model.select(y[:n], counts[:n], 2., .25, 0.)
        np.testing.assert_array_equal(p['plans'][i], old_choice[at])
        a = study.model.select(y, counts, 2., .25, 0.)
        b = study.model.select(y, counts, 2., .25, 0., counts[i])
        out['original'][at] = p['plans'][i]
        out['expanded'][at] = p['plans'][a]
        out['expanded_matched'][at] = p['plans'][b]
        take = p['order'][:counts[a]]
        out['independent_matched'][at[take]] = p['action'][take]
        new += a >= n
    return out, new


def evaluate(cells, d):
    scorer = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    pairs = {f'{a}_vs_{k}': (b, a) for a in ARMS for k, b in (('cv', base.CV), ('ols4', base.OLS))}
    pairs.update(PAIRS)
    rows, geo = [], {}
    for c in cells:
        ident = c['identity']
        _, h = base.subset(d, ident['held'])
        cv, ols, bank = study.parent.analytic(h['history'])
        with np.load(old.checked(c['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            paths = {base.CV: cv, base.OLS: ols, **{k: study.parent.model.apply(cv, bank, z[k]) for k in ARMS}}
        for rec in np.unique(h['recording']):
            at = h['recording'] == rec
            y, v, q, hist = (h[k][at] for k in ('target', 'valid', 'query', 'history'))
            source, = np.unique(h['sites'][at])
            z = {k: p[at] for k, p in paths.items()}
            r = scorer.score_recording(z, {}, y, v, q, str(rec), easy_cut=c['cut'][0], hard_cut=c['cut'][1])
            w = old.weight(q)
            terms = {k: full(z[l], z[r], cv[at], y, v, w) for k, (l, r) in pairs.items()}
            rows.append(dict(r, source=str(source), seed=ident['seed'], comparisons=terms))
            b = geo.setdefault((str(source), ident['seed']), np.zeros((len(PAIRS), 2)))
            for g in prior.query_groups(q):
                arr = np.stack([z[k][g] for pair in PAIRS.values() for k in pair])
                _, u, _, (steps, _, _, _) = prior.geometry.query(hist[g], arr, y[g], v[g],
                    tuple((2*j, 2*j+1) for j in range(len(PAIRS))))
                if steps:
                    b[:, 0] += u[:, 1]
                    b[:, 1] += 1
    roster = {r['recording']: r['source'] for r in rows}
    summary = prior.prefix.grouped.grouped_readout(rows, roster, base.SEEDS)
    sites = sorted(set(roster.values()))
    x = np.array([[geo[s, k] for k in base.SEEDS] for s in sites])
    a = np.divide(x[..., 0], x[..., 1], out=np.full_like(x[..., 0], np.nan), where=x[..., 1] > 0.)
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    return summary, {k: prior.geometry.bootstrap(a[:, :, i], draw) for i, k in enumerate(PAIRS)}


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    root = audit.PRIVATE / 'transfer'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            prior.donor.prior.prior.source.atomic(root / 'heartbeat.json', z)
            print(z, flush=True)

        rp, reg = audit.OUT / 'registration.json', audit.OUT / 'causal_registration.json'
        parents = old.read(rp)['parents']
        pp = [old.checked(p) for p in parents[:3]]
        cfg, freeze, result = map(old.read, pp)
        if (freeze['registration'] != old.sha(pp[0]) or result['freeze'] != old.sha(pp[1])
                or not result['original_2pct'] or result['a_labels_opened'] or result['independent_confirmation']):
            raise ValueError('Expected frozen development policy')
        shape = old.read(old.checked(cfg['parents'][1]))
        old.once(reg, old.stable(dict(parents=[old.artifact(p) for p in (rp, *pp)],
            files=[old.artifact(p) for p in (Path(__file__), audit.OUT / 'causal_protocol.md')],
            grid=study.GRID, original_2pct=True)))
        base.register()
        data = base.fit_population(lambda **_: None)
        cells = []
        for c, s in zip(freeze['cells'], shape['cells'], strict=True):
            ident, cut = c['identity'], c['cut']
            if s['identity'] != ident:
                raise ValueError('Parent role mismatch')
            stem = f'{ident["fold"]}_{ident["seed"]}'
            beat(phase='calibration', cell=stem)
            cp = root / f'cell_{stem}.json'
            binding = dict(registration=old.sha(reg), identity=ident)
            if cp.exists():
                z = old.read(cp)
                if z['binding'] != binding:
                    raise ValueError('Cached transfer roles changed')
                old.checked(z['choices'])
                old.checked(z['menu'])
                cells.append(z)
                continue
            _, cal = base.subset(data, ident['calibration'])
            past, _ = study.parent.parent.context.context(cal)
            shape_net = study.loader.load(s['fits'][1], past.shape[1]+37)
            cv, ols, bank, _, p = audit.prepare(cal, shape_net)
            p = menu_packet(cal, cv, ols, bank, p, shape_net)
            nets = nets_for(c, p[0]['x'].shape[1])
            table, sel = study.calibrate(cal, cv, ols, bank, p, study.predict(nets, p), cut)
            _, h = base.subset(data, ident['held'])
            before, menu = restore(h, stem)
            cv, ols, bank = study.parent.analytic(h['history'])
            p = menu_packet(h, cv, ols, bank, before, shape_net)
            est = study.predict(nets, p)
            with np.load(old.checked(c['choices']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], h['ids'])
                a, new = choose(p, est['joint'], z['joint'])
            if sel is not None:
                a['calibrated'] = study.choose(p, est, len(cv), setting=sel['setting'])['selected']
            path = prior.study.parent.parent.save_actions(root / f'choices_{stem}.npz', dict(ids=h['ids'], **a))
            z = dict(binding=binding, identity=ident, cut=cut, ladder=table, selected=sel,
                new_choices=int(new), choices=path, menu=menu)
            old.once(cp, old.stable(z))
            cells.append(z)
            beat(phase='cell_frozen', cell=stem, new_choices=int(new))
        fp = audit.OUT / 'causal_freeze.json'
        old.once(fp, old.stable(dict(registration=old.sha(reg), cells=cells, before_held_scoring=True)))
        summary, geometry = evaluate(cells, data)
        if summary['risk']['original'] != result['summary']['risk']['joint']:
            raise ValueError('Original scorer did not reproduce')
        old.once(audit.OUT / 'causal_readout.json', old.stable(dict(freeze=old.sha(fp), summary=summary,
            geometry=geometry, qualified=sum(c['selected'] is not None for c in cells),
            new_choices=sum(c['new_choices'] for c in cells), a_labels_opened=False,
            independent_confirmation=False, original_2pct=True)))
        beat(phase='complete')


if __name__ == '__main__':
    main()
