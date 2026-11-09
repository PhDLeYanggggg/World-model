"""Fit direct pair-effect heads on donor OpenDD locations, then replay fixed edit slots."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 Python')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_slate_shared_menu as prior
from src.world_model import m3w_pair_effect as model

exp, study, base, np, torch, shared = prior.exp, prior.study, prior.base, prior.np, prior.torch, prior.shared
SITES, SEEDS = prior.SITES, prior.SEEDS
OUT = prior.OUT.parent / 'opendd_direct_pair_effect_v1'
LOCAL = prior.LOCAL.parent / OUT.name
STEPS, BATCH, MIN_COMMON, MAX_PAIRS = 300, 128, 4, 16
RELATIVE_WEIGHT = .25


def register():
    slate = exp.OUT / 'registration.json'
    prior_reg = prior.OUT / 'registration.json'
    prior_freeze = prior.OUT / 'freeze.json'
    reg, parent, slots = (study.read(p) for p in (slate, exp.OUT / 'freeze.json', prior_freeze))
    if (parent['registration'] != base.sha(slate) or slots['registration'] != base.sha(prior_reg)
            or not reg['original_2pct'] or slots['outcomes_accessed'] or set(reg['cohorts']) != set(SITES)):
        raise ValueError('Require the original donor and fixed-budget freeze')
    bindings = (slate, exp.OUT / 'freeze.json', prior_reg, prior_freeze, Path(__file__),
                Path(model.__file__), Path(shared.__file__), Path(exp.slate.__file__))
    base.immutable(OUT / 'registration.json', dict(bindings=[study.artifact(p) for p in bindings],
        donor_geometry='sample a native edit pair from the observed prefix, then sample at its exact peaks',
        target='E10 + E01 - E00 - E11 on co-observed steps, divided by mean past motion scale',
        loss='Huber beta .1', steps=STEPS, batch=BATCH, min_common_steps=MIN_COMMON, max_pairs_per_query=MAX_PAIRS,
        score_weight_rule='0.25 * donor unary sum std / donor normalized effect std; no threshold search',
        unsupported_sources='zero co-observed legal pair labels: absent from fit batches; retained for policy scoring',
        seeds=SEEDS, fit_sites=SITES, worker_processes=0, compute_threads=4,
        independent_confirmation=False, original_2pct=True))
    return base.sha(OUT / 'registration.json'), reg['cohorts'], parent, slots


def examples(ref, index):
    h, q, cv, ols, bank = prior.source_inputs(ref)
    with np.load(study.checked(ref), allow_pickle=False) as z:
        y = z['future']
    valid = np.isfinite(y).all(axis=-1)
    y = np.where(valid[..., None], y, 0.)
    rng = np.random.default_rng(80131+index)
    bounds = np.r_[0, np.flatnonzero(np.diff(q))+1, len(q)]
    inputs, geometries, labels = [], [], []
    attempted, support = 0, 0
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if b-a < 3:
            continue
        g, c, o, p = h[a:b], cv[a:b], ols[a:b], bank[a:b]
        peak = np.linalg.norm(p-c[:, None], axis=-1).max(axis=-1)
        row, arm = np.nonzero(peak > 0)
        if len(np.unique(row)) < 2:
            continue
        i = rng.integers(len(row))
        j = rng.choice(np.flatnonzero(row != row[i]))
        radii = peak[row[[i, j]], arm[[i, j]]]
        paths, slots = shared.slot_menu(c, p, radii)
        options = shared.legal_assignments(slots[0][0], slots[1][0])
        if len(options) > MAX_PAIRS:
            options = options[rng.choice(len(options), MAX_PAIRS, replace=False)]
        features, scale, _, _ = shared.costs.features(g, c, o, paths)
        row = np.concatenate([s[0] for s in slots])
        arm = np.concatenate([s[1]+k*p.shape[1] for k, s in enumerate(slots)])
        offset = paths[row, arm]-c[row]
        r = row[options]
        target, common = model.observed_effect(c[r], offset[options], y[a+r], valid[a+r])
        good = common >= MIN_COMMON
        attempted += len(options)
        if good.any():
            x, geom, _ = exp.slate.pack(features[row, arm], g[row, -1], offset, scale[row], options[good])
            inputs.append(x)
            geometries.append(geom)
            labels.append((target[good]/scale[r[good]].mean(1)).astype(np.float32))
            support += 1
    result = (tuple(torch.from_numpy(np.concatenate(values)) for values in (inputs, geometries, labels))
              if inputs else None)
    return result, dict(sampled=attempted, observed=sum(map(len, labels)), queries_with_support=support)


def supported_donors(data, held):
    donors = tuple(s for s in SITES if s != held and data.get(s) is not None)
    if not donors:
        raise ValueError('No eligible donor source for pair-effect fit')
    return donors


def slate_for(parent, site, seed):
    fit = parent['fits'][f'{site}_{seed}']
    identity = fit['identity']
    if (identity['omitted_site'] != site or set(identity['fit_sites']) != set(SITES)-{site}
            or identity['arm'] != 'analytic' or identity['seed'] != seed):
        raise ValueError('Require donor-only frozen unary checkpoint')
    state = torch.load(study.checked(fit['checkpoint']), weights_only=True, map_location='cpu')
    if state['identity'] != identity or state['step'] != exp.STEPS:
        raise ValueError('Unverified unary fit')
    slate = exp.slate.Slate().eval()
    slate.load_state_dict(state['net'])
    return slate


def donor_encoding(net, groups):
    encoded, unary = [], []
    with torch.no_grad():
        for x, g, y in groups:
            encoded.append((torch.cat([net.encode(a, b) for a, b in zip(x.split(4096), g.split(4096), strict=True)]), y))
            unary.append(torch.cat([net.slate.unary(net.slate.action(a)).sum(1).flatten() for a in x.split(4096)]))
    delta_std = float(torch.cat([y for _, y in encoded]).std(unbiased=False))
    unary_std = float(torch.cat(unary).std(unbiased=False))
    if not np.isfinite(delta_std+unary_std) or delta_std <= 0.:
        raise ValueError('Undefined donor pair-score scale')
    return encoded, RELATIVE_WEIGHT*unary_std/delta_std


def heartbeat(**fields):
    t = LOCAL / 'heartbeat.pending'
    t.write_text(study.json.dumps(dict(pid=os.getpid(), **fields))+'\n')
    t.replace(LOCAL / 'heartbeat.json')


def fit(key, site, seed, parent, data):
    name = f'{site}_{seed}'
    donors = supported_donors(data, site)
    torch.manual_seed(seed)
    net = model.PairEffect(slate_for(parent, site, seed))
    encoded, net.weight = donor_encoding(net, [data[s] for s in donors])
    opt = torch.optim.AdamW(net.effect.parameters(), lr=.002)
    rng = np.random.default_rng(seed+1981)
    path = LOCAL / f'{name}.pt'
    identity = dict(registration=key, held=site, donors=donors, seed=seed,
                    unary_sha256=parent['fits'][name]['checkpoint']['sha256'], weight=net.weight)
    start, trace = 0, []
    if path.exists():
        old = torch.load(path, weights_only=True, map_location='cpu')
        if old['identity'] != identity or old['step'] not in range(50, STEPS+1, 50):
            raise ValueError('Resume checkpoint contract differs')
        net.effect.load_state_dict(old['effect'])
        opt.load_state_dict(old['optimizer'])
        start, trace, rng.bit_generator.state = old['step'], old['loss'], old['rng']
    for step in range(start+1, STEPS+1):
        n = rng.multinomial(BATCH, np.full(len(donors), 1./len(donors)))
        take = [rng.integers(len(y), size=k) for (_, y), k in zip(encoded, n, strict=True)]
        x, y = (torch.cat([g[i][a] for g, a in zip(encoded, take, strict=True)]) for i in (0, 1))
        p = net.effect(x).flatten()
        loss = torch.nn.functional.smooth_l1_loss(p, y, beta=.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.effect.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % 50 == 0:
            trace.append(float(loss.detach()))
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=identity, step=step, effect=net.effect.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, loss=trace), tmp)
            tmp.replace(path)
            heartbeat(phase='fit', fold=name, step=step, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(path), identity=identity, loss=trace, steps=STEPS)


def inference(net, h, q, cv, ols, bank, expected, native_joint):
    a, b = cv.copy(), cv.copy()
    peak = np.linalg.norm(native_joint-cv, axis=-1).max(axis=-1)
    changes = Counter()
    for j in np.unique(q[peak > 0]):
        ix = np.flatnonzero(q == j)
        radii = peak[ix][peak[ix] > 0]
        a[ix], b[ix], _ = shared.choose(net, h[ix], cv[ix], ols[ix], bank[ix], radii)
        for p in (a, b):
            slot = np.linalg.norm(p[ix]-cv[ix], axis=-1).max(axis=-1)
            np.testing.assert_allclose(np.sort(slot[slot > 0]), np.sort(radii), rtol=1e-9, atol=1e-9)
        diff = not np.allclose(a[ix], b[ix], atol=1e-9, rtol=1e-9)
        if diff and len(radii) != 2:
            raise ValueError('A pair head changed the singleton negative control')
        changes[f'queries_{len(radii)}_slots'] += 1
        changes['direct_differs_additive'] += int(diff)
        changes['direct_differs_imitation'] += int(not np.allclose(b[ix], expected['joint'][ix], atol=1e-9, rtol=1e-9))
    np.testing.assert_allclose(a, expected['matched_independent'], atol=1e-9, rtol=1e-9)
    return dict(q=q, cv=cv, ols4=ols, joint=b, imitation=expected['joint'], matched_independent=a), dict(changes)


def freeze(key, cohorts, parent, slots):
    # Keep each held source's labels out of its own training-example loader.
    cache, forecasts, fits, decisions, counts = {}, {}, {}, {}, {}
    for site in SITES:
        donors = [s for s in SITES if s != site]
        for s in donors:
            if s not in cache:
                cache[s], counts[s] = examples(cohorts[s], SITES.index(s))
        training = {s: cache[s] for s in donors}
        h, q, cv, ols, bank = prior.source_inputs(cohorts[site])
        for seed in SEEDS:
            name = f'{site}_{seed}'
            net, fits[name] = fit(key, site, seed, parent, training)
            with np.load(study.checked(slots['forecasts'][name]), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['q'], q)
                np.testing.assert_array_equal(z['cv'], cv)
                expected = {k: z[k] for k in ('joint', 'matched_independent')}
            with np.load(study.checked(parent['forecasts'][name]), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['q'], q)
                np.testing.assert_array_equal(z['cv'], cv)
                native_joint = z['joint']
            pred, decisions[name] = inference(net, h, q, cv, ols, bank, expected, native_joint)
            forecasts[name] = exp.previous.save_bank(LOCAL / f'{name}.npz', pred)
            print(name, decisions[name], fits[name]['loss'][-1], flush=True)
            heartbeat(phase='freeze', fold=name, completed=len(fits), total=21)
    base.immutable(OUT / 'freeze.json', dict(registration=key, decisions=decisions, fits=fits,
        forecasts=forecasts, examples=counts, updates=len(fits)*STEPS, fold_scoring_not_run=True))


def probe(parent, receipt, site, seed, x, g, y):
    net = model.PairEffect(slate_for(parent, site, seed))
    old = torch.load(study.checked(receipt['checkpoint']), weights_only=True, map_location='cpu')
    if old['identity'] != {**receipt['identity'], 'donors': tuple(receipt['identity']['donors'])}:
        raise ValueError('Probe checkpoint differs from training receipt')
    net.effect.load_state_dict(old['effect'])
    with torch.inference_mode():
        p = torch.cat([net.effect(net.encode(a, b)).flatten()
                       for a, b in zip(x.split(4096), g.split(4096), strict=True)])
    mse, null = float((p-y).square().mean()), float(y.square().mean())
    return dict(rmse=mse**.5, zero_rmse=null**.5, r2_vs_zero=1.-mse/null if null > 0 else None)


def score(key, cohorts, parent):
    frozen = OUT / 'freeze.json'
    f = study.read(frozen)
    if f['registration'] != key or not f['fold_scoring_not_run'] or len(f['fits']) != 21:
        raise ValueError('All folds must be frozen before evaluating held labels')
    data = exp.loader(cohorts)
    groups = {}
    names = ('joint', 'imitation', 'matched_independent')
    for site, d in data.items():
        rows, probes = [], {}
        cut = np.quantile(np.concatenate([data[s]['e'] for s in SITES if s != site]), [.25, .75])
        observed, support = examples(cohorts[site], SITES.index(site))
        for seed in SEEDS:
            name = f'{site}_{seed}'
            probes[str(seed)] = probe(parent, f['fits'][name], site, seed, *observed) if observed is not None else None
            with np.load(study.checked(f['forecasts'][name]), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['q'], d['q'])
                paths = {k: z[k] for k in ('cv', 'ols4', *names)}
            for r in np.unique(d['rec']):
                at = d['rec'] == r
                p = {k: z[at] for k, z in paths.items()}
                q = np.unique(d['q'][at], return_inverse=True)[1]
                m = exp.score_record(p, d['y'][at], d['v'][at], q, cut)
                if m['direct']['matched_independent'] is not None:
                    _, lower = study.paired(p['imitation'], p['joint'], d['y'][at], d['v'][at])
                    m['direct_imitation'] = float(100.*((1./np.bincount(q)[q]) @ lower.sum(1))/m['reference_mass'])
                rows.append(m)
        active = [r for r in rows if r['direct']['matched_independent'] is not None]
        def mean(values):
            return float(np.mean(values)) if values else None
        fields = dict(known_pp=mean([r['direct']['matched_independent'][0] for r in active]),
                      lower_pp=mean([r['direct']['matched_independent'][1] for r in active]),
                      lower_vs_imitation_pp=mean([r['direct_imitation'] for r in active]),
                      pair_error_gain_native=mean([r['pair_relative_error_px']['matched_independent']-
                                                  r['pair_relative_error_px']['joint'] for r in active
                                                  if all(r['pair_relative_error_px'][k] is not None for k in names)]))
        arms = {k: dict(edits=sum(r['arms'][k]['risk']['selected'] for r in rows),
                        risk_pass=sum(r['arms'][k]['risk']['supported'] for r in rows),
                        dual_pass=sum(r['arms'][k]['dual_pass'] for r in rows)) for k in names}
        groups[site] = dict(views=len(rows), estimable=len(active), probe_support=support, probes=probes, arms=arms, **fields)
    def interval(key):
        x = np.asarray([g[key] for g in groups.values() if g[key] is not None])
        draw = np.random.default_rng(3183).choice(x, size=(3000, len(x))).mean(1)
        return dict(sites=len(x), mean=float(x.mean()), ci95=np.quantile(draw, [.025, .975]).tolist())
    result = dict(result_source='fresh_run', freeze=study.artifact(frozen), groups=groups,
        totals=dict(sum(map(Counter, f['decisions'].values()), Counter())),
        contrasts={k: interval(k) for k in ('known_pp', 'lower_pp', 'lower_vs_imitation_pp', 'pair_error_gain_native')},
        independent_confirmation=False, reserved_outcomes_opened=False, original_2pct=True, stage5c=False, smc=False)
    base.immutable(OUT / 'readout.json', result)
    print(result['contrasts'], result['totals'], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=('freeze', 'score'))
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, parent, slots = register()
        if args.phase == 'freeze':
            freeze(key, cohorts, parent, slots)
        else:
            score(key, cohorts, parent)


if __name__ == '__main__':
    main()
