"""Train and calibrate causal shape substitutions without outer-site exposure."""
import argparse
from collections import Counter
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch Python')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_matched_spline as m
from src.world_model import m3w_donor_shape_choice as head
from src.world_model import m3w_spline_slot_choice as model

control, run, np, torch = m.control, m.run, m.np, m.torch
OUT = m.OUT.parent/'opendd_nested_spline_choice_v2'
PRIVATE = m.PRIVATE.parent/OUT.name
STEPS = 400
KINDS = ('gain', 'independent', 'joint')
ARMS = ('anchor', 'prefix_match', 'forced', *KINDS, 'joint_same_rule')
AMPLITUDES = np.array([.25, .5, 1.])


@torch.inference_mode()
def predict_head(net, x):
    return net(x).numpy()


def packet(d, bank, spline):
    cv, _, _, _, _, legal = bank
    row, arm = np.nonzero(legal)
    h = d['h'][row]
    empty = {k: np.zeros((len(row), 2, 8-k if k != 8 else 12, 2)) for k in (4, 5, 6, 8)}
    c, anchor, *_ = control.radius.choose(h, AMPLITUDES[arm], empty)
    np.testing.assert_allclose(c, cv[row], atol=1e-10)
    pred = control.radius.match_radius(c, anchor, (spline[row]-c)[:, None])[:, 0]
    _, _, scale = control.radius.motion_frame(h)
    x = head.features(h, c, anchor, pred, AMPLITUDES[arm], np.zeros(len(row)), np.zeros(len(row)))
    index = np.full(legal.shape, -1)
    index[row, arm] = np.arange(len(row))
    return dict(x=torch.from_numpy(x), scale=scale, row=row, arm=arm, index=index,
                query=d['query'][row], anchor=anchor, alternative=pred)


def targets(d, bank, p, cut):
    row = p['row']
    cv = bank[0][row]
    y, v = d['y'][row], d['valid'][row]
    a, b = p['anchor'], p['alternative']
    risk = [control.two.moments(cv, q, y, v, cut) for q in (a, b)]
    gain = control.paired(a, b, y, v)[1].sum(1)
    delta = np.column_stack((gain, *[12.*(risk[1][k]-risk[0][k])
                            for k in ('all_excess', 'easy_excess', 'easy_signed')]))
    mass = np.linalg.norm(bank[0]-d['y'], axis=-1)[d['valid']].sum()
    if mass <= 0. or not np.isfinite(delta).all():
        raise ValueError('Calibration reference or training targets invalid')
    return delta, float(mass)


def proxy(d, p):
    eligible = p['index'][:, [1, 0, 2]]
    found = eligible >= 0
    idx = np.flatnonzero(found.any(1))
    preferred = eligible[idx, found[idx].argmax(1)]
    if not len(idx):
        return preferred
    score = np.linalg.norm(np.diff(d['h'][idx], n=2, axis=1), axis=-1).mean(1)
    q = d['query'][idx]
    return np.concatenate([preferred[g[np.argsort(-score[g], kind='stable')[:2]]]
                           for g in m.donor_spline.prior.query_groups(q)])


def register():
    parent, source, _, _ = m.register()
    ff, rr, sf = (m.OUT/n for n in ('freeze.json', 'readout.json', 'slot_readout.json'))
    freeze, result, slots = map(run.read, (ff, rr, sf))
    if (freeze['registration'] != parent or result['freeze'] != run.sha(ff)
            or not freeze['before_held_scoring'] or any(x['confirmation_access']
            or x['independent_confirmation'] or not x['original_2pct'] for x in (result, slots))):
        raise ValueError('Expected frozen OpenDD development forecasts and slots')
    _, _, ph, h, anchor, _ = control.register()
    reg = OUT/'registration.json'
    run.once(reg, run.stable(dict(parents=[run.artifact(x) for x in (ff, rr, sf)],
        files=[run.artifact(x) for x in (Path(__file__), Path(model.__file__), Path(head.__file__),
            OUT/'protocol.md', OUT.parent/'opendd_nested_spline_choice_v1'/'protocol.md')],
        selector_steps=STEPS, producer_steps=m.STEPS, seeds=m.SEEDS, margins=model.MARGINS,
        penalties=model.PENALTIES, original_2pct=True, confirmation_access=False,
        primary='joint vs prefix at identical outer slots and peaks', nested_exclusion=True)))
    return run.sha(reg), source, freeze, slots, ph, h, anchor


def fit(data, source, reg, freeze, ph, h, anchor, replay, beat):
    sites = {s: a['d'] for s, a in data.items()}
    banks = {s: control.parent.prior.analytic(d) for s, d in sites.items()}
    producers, heads, cells, updates = {}, {}, [], Counter()

    def sorted_sites(z):
        return tuple(sorted(z))

    @lru_cache(maxsize=180)
    def producer(excluded, seed):
        fitting = [s for s in sites if s not in excluded]
        ident = dict(excluded=list(excluded), fit=fitting, seed=seed, registration=reg, arm='producer')
        name = 'producer_'+'_'.join(excluded)+f'_{seed}'

        def cost(net, *batch):
            return m.model.loss(net, batch[:6], *batch[6:])

        net, meta, n = control.two.fit.fit(ident, name, seed, m.model.Spline,
            m.STEPS, m.sampler(data, fitting, 'comoving_spline'), cost, replay, beat)
        producers[name] = meta
        updates['producer'] += n
        return net

    @lru_cache(maxsize=128)
    def pack(site, excluded, seed):
        if site not in excluded:
            raise ValueError('Producer must exclude the predicted site')
        if len(excluded) == 1:
            c = next(x for x in freeze['cells'] if x['held'] == site and x['seed'] == seed)
            if set(c['fit']) != set(sites)-set(excluded):
                raise ValueError('Frozen outer producer roles changed')
            with np.load(run.checked(c['paths']), allow_pickle=False) as z:
                p = z['comoving_spline']
        else:
            p = m.forecast(producer(excluded, seed), data[site]['inputs']['comoving_spline'])
        return packet(sites[site], banks[site], p)

    @lru_cache(maxsize=32)
    def cuts(use):
        return np.quantile(np.concatenate([sites[s]['cut_samples'] for s in use]), [.25, .75])

    @lru_cache(maxsize=96)
    def selector(excluded, seed):
        use = tuple(s for s in sites if s not in excluded)
        samples = []
        for s in use:
            p = pack(s, sorted_sites((*excluded, s)), seed)
            d, _ = targets(sites[s], banks[s], p, cuts(use)[0])
            y = (d/(12.*p['scale'][:, None])).clip(-8., 8.).astype(np.float32)
            samples.append((p['x'], torch.from_numpy(y)))
        ident = dict(excluded=list(excluded), fit=list(use), seed=seed, registration=reg, arm='selector')
        name = 'selector_'+'_'.join(excluded)+f'_{seed}'
        net, meta, n = control.two.fit.fit(ident, name, seed, lambda: head.network(samples[0][0].shape[1]),
            STEPS, control.two.batcher(samples), head.loss, replay, beat)
        heads[name] = meta
        updates['selector'] += n
        return net

    for held in sites:
        donors = tuple(s for s in sites if s != held)
        validation = {}
        for site in donors:
            e = sorted_sites((held, site))
            use = tuple(s for s in donors if s != site)
            validation[site] = []
            for seed in m.SEEDS:
                p = pack(site, e, seed)
                d, mass = targets(sites[site], banks[site], p, cuts(use)[0])
                at = proxy(sites[site], p)
                z = predict_head(selector(e, seed), p['x'][at])
                validation[site].append((z, p['query'][at], p['scale'][at], d[at], mass))
        calibrated = {k: model.calibrate(validation, k) for k in KINDS}
        rules = {k: v[0] for k, v in calibrated.items()}
        prefix = control.prior.prepare(sites, held, list(donors), banks[held])['paths']
        with np.load(source.PRIVATE/f'{held}_cohort.npz', allow_pickle=False) as z:
            rec, start = z['recording'], z['start']
        d, bank = sites[held], banks[held]
        for seed in m.SEEDS:
            p = pack(held, (held,), seed)
            menu, count = control.ledger.frozen_count(d, bank, rec, start,
                ph['heads'][f'{held}/{seed}/additive_rank'], h['heads'][f'{held}_{seed}_motion'])
            use = np.arange(2) < count[:, None]
            row, arm = menu['row'][use], menu['arm'][use]
            at = p['index'][row, arm]
            if (at < 0).any():
                raise ValueError('Original slots absent in spline bank')
            np.testing.assert_allclose(p['anchor'][at], prefix[row, arm, 0], atol=1e-10, rtol=0.)
            pred = predict_head(selector((held,), seed), p['x'][at])
            q, scale = p['query'][at], p['scale'][at]
            flags = {k: model.choose(pred, q, scale, rules[k], k == 'joint') for k in KINDS}
            flags['joint_same_rule'] = model.choose(pred, q, scale, rules['independent'], True)
            flags.update(anchor=np.zeros(len(at), bool), forced=np.ones(len(at), bool))
            paths = {k: bank[0].copy() for k in ARMS}
            paths['prefix_match'][row] = prefix[row, arm, 3]
            for k, flag in flags.items():
                paths[k][row] = np.where(flag[:, None, None], p['alternative'][at], p['anchor'][at])
            for z in paths.values():
                np.testing.assert_array_equal(np.flatnonzero((z != bank[0]).any((1, 2))), np.sort(row))
                np.testing.assert_allclose(control.radius.peak(z[row]-bank[0][row]),
                    control.radius.peak(paths['anchor'][row]-bank[0][row]), atol=1e-10)
            f = m.save(PRIVATE/f'{held}_{seed}_predictions.npz', paths)
            cells.append(dict(held=held, seed=seed, cut=cuts(donors).tolist(), paths=f,
                chosen={k:int(v.sum()) for k,v in flags.items()}, rules=rules))
        beat(phase='outer_frozen', held=held, updates=dict(updates), rules=rules)
        yield held, cells[-len(m.SEEDS):], calibrated, producers, heads, updates


def evaluate(data, cells, slots):
    refs = {(r['held'], r['seed']): r['policies'] for r in slots['cells']}
    out = []
    for c in cells:
        s, seed = c['held'], c['seed']
        d = data[s]['d']
        cv, ols, *_ = control.parent.prior.analytic(d)
        with np.load(run.checked(c['paths']), allow_pickle=False) as z:
            p = {k: z[k] for k in ARMS}
        policies = {k: control.parent.score(d, cv, ols, v, c['cut']) for k, v in p.items()}
        for a, b in (('anchor', 'anchor'), ('prefix_match', 'prefix_match'), ('forced', 'comoving_spline')):
            if policies[a] != refs[s, seed][b]:
                raise ValueError(f'Frozen matched control changed for {s}/{seed}/{a}')
        mass = np.linalg.norm(cv-d['y'], axis=-1)[d['valid']].sum()
        pairs = {f'{a}_vs_{b}': (b, a) for a in (*KINDS, 'joint_same_rule')
                 for b in ('anchor', 'prefix_match')}
        pairs.update(joint_vs_independent=('independent','joint'),
                     grouped_at_same_rule=('independent','joint_same_rule'))
        effects = {k: float(100.*control.paired(p[a], p[b], d['y'], d['valid'])[1].sum()/mass)
                   for k, (a, b) in pairs.items()}
        out.append(dict(site=s, seed=seed, policies=policies, effects=effects, chosen=c['chosen']))
    sites = list(data)
    draw = np.random.default_rng(817).integers(len(sites), size=(3000,len(sites)))

    def boot(fun):
        z = np.array([np.mean([fun(r) for r in out if r['site'] == s]) for s in sites])
        return dict(point=float(z.mean()), interval95=np.quantile(z[draw].mean(1), [.025, .975]).tolist())

    tables = {k: {f: sum(r['policies'][k][f] for r in out)
              for f in ('edits','views','undefined','risk_pass','dual_positive')} for k in ARMS}
    if len(out) != 21 or any((x['edits'],x['views']) != (1457,606) for x in tables.values()):
        raise ValueError('Prescribed fixed-slot roster changed')
    summary = dict(policies=tables,
        effects={k:boot(lambda r: r['effects'][k]) for k in out[0]['effects']},
        risk_pass_diff_joint_vs_prefix=boot(lambda r: r['policies']['joint']['risk_pass']
                                          -r['policies']['prefix_match']['risk_pass']),
        chosen={k:sum(x['chosen'][k] for x in out) for k in out[0]['chosen']})
    return out, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    control.two.fit.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE/'train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start,3), **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', z)
            if v.get('phase') != 'fit':
                print(z, flush=True)

        reg, source, freeze, slots, ph, h, anchor = register()
        data = m.load(source)
        cells, cal, fits, heads, updates = [], {}, {}, {}, {}
        for held, block, rules, fits, heads, updates in fit(data, source, reg, freeze, ph, h, anchor, args.replay, beat):
            cells.extend(block)
            cal[held] = rules
        fp = OUT/'freeze.json'
        run.once(fp, run.stable(dict(registration=reg, cells=cells, calibration=cal,
            producer_fits=fits, selector_fits=heads, before_held_scoring=True,
            configured_updates=dict(producer=m.STEPS*len(fits), selector=STEPS*len(heads)))))
        rows, summary = evaluate(data, run.read(fp)['cells'], slots)
        run.once(OUT/'readout.json', run.stable(dict(freeze=run.sha(fp), summary=summary, rows=rows,
            original_2pct=True, confirmation_access=False, independent_confirmation=False, deployment_changed=False)))
        beat(phase='complete', updates=dict(updates), replay=args.replay)


if __name__ == '__main__':
    main()
