"""Calibrate the frozen spline selector on double-excluded streaming slots."""
import argparse
from collections import Counter
import fcntl
from functools import lru_cache
import itertools
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_nested_spline_choice as v
from scripts.finalize_m3w_opendd_nested_spline_choice import portable_rules
from src.evaluation.m3w_selection_forecast_readout import policy
from src.evaluation import m3w_joint_path_audit as geometry
from src.world_model import m3w_stream_spline_calibration as cal

m, run, np, torch = v.m, v.run, v.np, v.torch
control = v.control
study, base = control.ledger.study, control.ledger.base
donor = control.ledger.donor_run
OUT = v.OUT.parent/'opendd_stream_spline_calibration_v1'
PRIVATE = v.PRIVATE.parent/OUT.name
ARMS = (*v.ARMS, 'recording')
GRID = (None, *itertools.product(v.model.MARGINS, v.model.PENALTIES))


def register():
    parent, source, forecasts, slots, _, _, _ = v.register()
    files = [v.OUT/k for k in ('registration.json', 'freeze.json', 'readout.json')]
    frozen, readout = map(run.read, files[1:])
    if (frozen['registration'] != parent or not frozen['before_held_scoring']
            or readout['freeze'] != run.sha(files[1]) or readout['confirmation_access']
            or readout['independent_confirmation'] or not readout['original_2pct']):
        raise ValueError('Require the frozen nested development experiment')
    ranks, triple = study.OUT/'head_freeze.json', donor.OUT/'head_freeze.json'
    dkey, *_, upstream = donor.register()
    nested = run.read(triple)
    if nested['registration'] != dkey or run.read(ranks)['rank_heads'] != upstream['rank_heads']:
        raise ValueError('Unbound rank lineage')
    f = OUT/'registration.json'
    run.once(f, run.stable(dict(parents=[run.artifact(p) for p in (*files, ranks, triple)],
        code=[run.artifact(p) for p in (Path(__file__), Path(cal.__file__), Path(base.__file__),
              Path(study.__file__), OUT/'protocol.md')], seeds=m.SEEDS, count_steps=base.STEPS,
        grid=GRID, threads=4, workers=0, original_2pct=True, confirmation_access=False,
        primary='stream joint vs proxy joint at frozen outer slots and peaks')))
    return run.sha(f), source, forecasts, slots, frozen, readout, upstream['rank_heads'], nested['triple_ranks']


def installed(cv, p, at, choose):
    q = cv.copy()
    q[p['row'][at]] = np.where(choose[:, None, None], p['alternative'][at], p['anchor'][at])
    return q


def fit(data, source, reg, forecasts, frozen, pairs, triples, replay, beat):
    sites = {s: a['d'] for s, a in data.items()}
    banks = {s: control.parent.prior.analytic(d) for s, d in sites.items()}
    count_heads, old_heads, updates = {}, {}, Counter()
    outer = {(c['held'], c['seed']): c for c in frozen['cells']}
    seen = {}

    @lru_cache(maxsize=32)
    def cuts(use):
        return np.quantile(np.concatenate([sites[s]['cut_samples'] for s in use]), [.25, .75])

    def included(ex):
        return tuple(s for s in sites if s not in ex)

    def suffix(ex, seed):
        return '_'.join(ex)+f'_{seed}'

    @lru_cache(maxsize=180)
    def rank(ex, seed):
        name = 'rank_'+suffix(ex, seed)
        ref = (pairs if len(ex) == 2 else triples)[name]
        cal.require_roles(ref['identity'], sites, ex, seed)
        seen[name] = ref['artifact']
        builder = lambda: study.prior.model.SwapValue(False)
        if len(ex) == 3:
            if donor.STEPS != study.RANK_STEPS:
                raise ValueError('Nested rank training budget changed')
            return control.ledger.load_pair(ref, builder)
        return study.load(ref, builder, study.RANK_STEPS)

    @lru_cache(maxsize=32)
    def rank_menu(s, ex, seed):
        if s not in ex:
            raise ValueError('Training menu ranker saw its target site')
        return study.menus(sites[s], banks[s], rank(ex, seed))

    @lru_cache(maxsize=63)
    def count(ex, seed):
        use = included(ex)
        inputs = []
        for s in use:
            excluded = tuple(sorted((*ex, s)))
            menu = rank_menu(s, excluded, seed)
            x, scale, _ = base.features(sites[s], banks[s], menu)
            y = base.labels(sites[s], banks[s], menu, cuts(included(excluded))[0], scale)
            on = menu['legal'][:, 1]
            inputs.append((dict(x=x['motion'][on], legal=menu['legal'][on]), y[on]))
        name = 'count_'+suffix(ex, seed)
        ident = dict(registration=reg, omitted=list(ex), fit=list(use), seed=seed)
        net, count_heads[name], u = study.fit(ident, name, seed,
            lambda: base.model.Value(inputs[0][0]['x'].shape[-1]), base.STEPS,
            base.batches(inputs), lambda net, x, legal, y: base.model.objective(net(x), y, legal), replay, beat)
        updates['count'] += u
        return net

    @lru_cache(maxsize=84)
    def selector(ex, seed, width):
        name = 'selector_'+suffix(ex, seed)
        ref = frozen['selector_fits'][name]
        cal.require_roles(dict(ref['identity'], omitted=ref['identity']['excluded']), sites, ex, seed)
        old_heads[name] = ref['artifact']
        return study.load(ref, lambda: v.head.network(width), v.STEPS)

    @lru_cache(maxsize=64)
    def packet(s, ex, seed):
        if len(ex) == 1:
            f = next(z for z in forecasts['cells'] if (z['held'], z['seed']) == (s, seed))
            if f['fit'] != list(included(ex)):
                raise ValueError('Outer forecast exposure changed')
            with np.load(run.checked(f['paths']), allow_pickle=False) as z:
                forecast = z['comoving_spline']
        else:
            f = frozen['producer_fits']['producer_'+suffix(ex, seed)]
            cal.require_roles(dict(f['identity'], omitted=f['identity']['excluded']), sites, ex, seed)
            forecast = m.forecast(study.load(f, m.model.Spline, m.STEPS), data[s]['inputs']['comoving_spline'])
        return v.packet(sites[s], banks[s], forecast)

    @lru_cache(maxsize=12)
    def timeline(s):
        with np.load(source.PRIVATE/f'{s}_cohort.npz', allow_pickle=False) as z:
            return z['recording'], z['start']

    cells, all_rules, roster = [], {}, {}
    for held in sites:
        donors = included((held,))
        validation = {}
        recording = {k: {} for k in GRID}
        roster[held] = {}
        for s in donors:
            ex = tuple(sorted((held, s)))
            cv, ols = banks[s][:2]
            cut = cuts(included(ex))
            rec, start = timeline(s)
            validation[s], roster[held][s] = [], []
            for seed in m.SEEDS:
                p, menu = packet(s, ex, seed), rank_menu(s, ex, seed)
                x, _, _ = base.features(sites[s], banks[s], menu)
                net = count(ex, seed)
                with torch.inference_mode():
                    score = base.model.score(net(torch.from_numpy(x['motion'])).numpy(), risk=False)
                counts = study.model.streaming_count(score, menu['legal'], rec, start)
                use = np.arange(2) < counts[:, None]
                at = p['index'][menu['row'][use], menu['arm'][use]]
                if (at < 0).any():
                    raise ValueError('Nonlegal streaming calibration slot')
                roster[held][s].append(dict(seed=seed, actual=len(at), proxy=len(v.proxy(sites[s], p)),
                    two_slot_queries=int((counts == 2).sum())))
                d, mass = v.targets(sites[s], banks[s], p, cut[0])
                z = v.predict_head(selector(ex, seed, p['x'].shape[1]), p['x'][at])
                q, scale = p['query'][at], p['scale'][at]
                validation[s].append((z, q, scale, d[at], mass))
                evaluated = {}
                for rule in GRID:
                    flag = v.model.choose(z, q, scale, rule, True)
                    key = flag.tobytes()
                    if key not in evaluated:
                        evaluated[key] = control.parent.score(sites[s], cv, ols, installed(cv, p, at, flag), cut)
                    recording[rule].setdefault(s, []).append(evaluated[key])
            beat(phase='donor_scored', held=held, donor=s, updates=dict(updates))
        calibration = {k: v.model.calibrate(validation, k) for k in v.KINDS}
        calibration['recording'] = cal.select_recording_rule(recording)
        all_rules[held] = {**portable_rules({k: calibration[k] for k in v.KINDS}),
                           'recording': calibration['recording']}
        rules = {k: z[0] for k, z in calibration.items()}
        for seed in m.SEEDS:
            c = outer[held, seed]
            with np.load(run.checked(c['paths']), allow_pickle=False) as z:
                previous = {k: z[k] for k in v.ARMS}
            cv = banks[held][0]
            p = packet(held, (held,), seed)
            rows = np.flatnonzero((previous['anchor'] != cv).any((1, 2)))
            peaks = control.radius.peak(previous['anchor'][rows]-cv[rows])
            option = p['index'][rows]
            candidates = control.radius.peak(p['anchor'][option]-cv[rows, None])
            match = (option >= 0) & (np.abs(candidates-peaks[:, None]) < 1e-10)
            if not (match.sum(1) == 1).all():
                raise ValueError('Cannot reconstruct exact outer slot radius')
            at = option[match]
            np.testing.assert_array_equal(previous['anchor'][rows], p['anchor'][at])
            np.testing.assert_array_equal(previous['forced'][rows], p['alternative'][at])
            pred = v.predict_head(selector((held,), seed, p['x'].shape[1]), p['x'][at])
            q, scale = p['query'][at], p['scale'][at]
            flags = {k: v.model.choose(pred, q, scale, rules[k], k in ('joint', 'recording')) for k in rules}
            flags['joint_same_rule'] = v.model.choose(pred, q, scale, rules['independent'], True)
            # Require identical outer decisions when replaying the proxy rule.
            old_rule = c['rules']['joint']
            np.testing.assert_array_equal(previous['joint'], installed(cv, p, at,
                v.model.choose(pred, q, scale, old_rule, True)))
            paths = {**previous, **{k: installed(cv, p, at, z) for k, z in flags.items()}}
            for path in paths.values():
                np.testing.assert_array_equal(np.flatnonzero((path != cv).any((1, 2))), rows)
                np.testing.assert_allclose(control.radius.peak(path[rows]-cv[rows]), peaks, atol=1e-10, rtol=0.)
            f = m.save(PRIVATE/f'{held}_{seed}_paths.npz', paths)
            cells.append(dict(held=held, seed=seed, cut=c['cut'], paths=f, rules=rules,
                chosen={**c['chosen'], **{k:int(z.sum()) for k,z in flags.items()}}, proxy_paths=c['paths']))
        beat(phase='outer_frozen', held=held, updates=dict(updates), rules=rules)
    return dict(cells=cells, calibration=all_rules, roster=roster, count_heads=count_heads,
                reused_rank=seen, reused_selector=old_heads, before_held_scoring=True), updates


def score(data, frozen, slots, readout):
    rows, summary = v.evaluate(data, frozen['cells'], slots)
    source = {(z['site'], z['seed']): z for z in readout['rows']}
    comparisons = dict(joint_vs_proxy=('proxy_joint', 'joint'), recording_vs_proxy=('proxy_joint', 'recording'),
        recording_vs_prefix=('prefix_match', 'recording'), joint_vs_independent=('independent', 'joint'))
    for r, c in zip(rows, frozen['cells'], strict=True):
        s, seed = r['site'], r['seed']
        d = data[s]['d']
        cv, ols = control.parent.prior.analytic(d)[:2]
        y, mask = d['y'], d['valid']
        with np.load(run.checked(c['paths']), allow_pickle=False) as z:
            p = {k: z[k] for k in ARMS}
        with np.load(run.checked(c['proxy_paths']), allow_pickle=False) as z:
            p['proxy_joint'] = z['joint']
        r['policies']['proxy_joint'] = source[s, seed]['policies']['joint']
        r['policies']['recording'] = control.parent.score(d, cv, ols, p['recording'], c['cut'])
        r['tail'], r['changed'], r['geometry'] = {}, {}, {}
        for k, path in p.items():
            z = policy(path, cv, cv, y, mask, d['rec'], easy_cut=c['cut'][0], hard_cut=c['cut'][1])
            easy = [u['known_easy_net_degradation'] for u in z['recordings'].values()
                    if u['known_easy_net_degradation'] is not None]
            err = np.linalg.norm(path-y, axis=-1)
            r['tail'][k] = dict(easy_signed_max_percent=100.*max(easy) if easy else None,
                easy_signed_over_2=sum(x > .02 for x in easy), hard_gain=z['subsets']['hard']['gain_percent'],
                complete_ade=float(err[mask.all(1)].mean()), observed_fde=float(err[mask[:, -1], -1].mean()),
                complete_ade_p95=float(np.quantile(err[mask.all(1)].mean(1), .95)))
        mass = np.linalg.norm(cv-y, axis=-1)[mask].sum()
        for k, (a, b) in comparisons.items():
            r['effects'][k] = float(100.*control.paired(p[a], p[b], y, mask)[1].sum()/mass)
            changed = (p[a] != p[b]).any((1,2))
            r['changed'][k] = int(changed.sum())
            lower, total = 0., 0
            for g in m.donor_spline.prior.query_groups(d['query']):
                _, delta, _, (steps, _, _, _) = geometry.query(d['h'][g], np.stack((p[a][g], p[b][g])),
                                                              y[g], mask[g], ((0, 1),))
                if steps:
                    lower += delta[0, 1]*steps
                    total += steps
            r['geometry'][k] = lower/total if total else None
    draw = np.random.default_rng(817).integers(len(data), size=(3000,len(data)))

    def ci(fn):
        return geometry.bootstrap([[fn(r) for r in rows if r['site'] == s] for s in data], draw)

    extra = dict(effects={k: ci(lambda r: r['effects'][k]) for k in comparisons},
        risk_count_delta={k: ci(lambda r: r['policies'][b]['risk_pass']-r['policies'][a]['risk_pass'])
                          for k,(a,b) in comparisons.items()},
        pair_ade_lower={k: ci(lambda r: r['geometry'][k]) for k in comparisons},
        changed={k:sum(r['changed'][k] for r in rows) for k in comparisons},
        tail={k:dict(worst_easy_percent=max(r['tail'][k]['easy_signed_max_percent'] for r in rows
                if r['tail'][k]['easy_signed_max_percent'] is not None),
                hard_gain=ci(lambda r: r['tail'][k]['hard_gain'])) for k in (*ARMS, 'proxy_joint')})
    for k in ('recording', 'proxy_joint'):
        summary['policies'][k] = {f:sum(r['policies'][k][f] for r in rows)
                                   for f in ('edits','views','undefined','risk_pass','dual_positive')}
    summary['chosen']['proxy_joint'] = readout['summary']['chosen']['joint']
    return rows, summary, extra


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    study.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE/'train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v0):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v0)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', z)
            if v0.get('phase') != 'fit':
                print(z, flush=True)

        reg, source, forecasts, slots, frozen, ref, pairs, triples = register()
        data = m.load(source)
        payload, updates = fit(data, source, reg, forecasts, frozen, pairs, triples, a.replay, beat)
        f = OUT/'freeze.json'
        run.once(f, run.stable(dict(registration=reg, **payload, configured_updates=base.STEPS*len(payload['count_heads']))))
        rows, summary, extra = score(data, run.read(f), slots, ref)
        run.once(OUT/'readout.json', run.stable(dict(freeze=run.sha(f), summary=summary, analysis=extra, rows=rows,
            original_2pct=True, confirmation_access=False, independent_confirmation=False, deployment_changed=False)))
        beat(phase='complete', updates=dict(updates), replay=a.replay)


if __name__ == '__main__':
    main()
