"""Compare fixed and reallocatable edit slots on the same OpenDD observations."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_m3w_opendd_expanded_costs as prior
from src.world_model import m3w_paired_window_budget as budget

menu, base, study, np, torch, exp = prior.menu, prior.base, prior.study, prior.np, prior.torch, prior.exp
SITES, SEEDS = prior.SITES, prior.SEEDS
OUT = prior.OUT.parent/'opendd_paired_window_budget_v1'
LOCAL = prior.LOCAL.parent/OUT.name
ARMS = ('fixed_additive', 'fixed_joint', 'pooled_additive', 'pooled_joint')


def register():
    key, cohorts, slate, _, _, weights, producer, _, _ = prior.register()
    reg, f, r = (prior.OUT/f'{k}.json' for k in ('registration', 'freeze', 'readout'))
    frozen, result = map(study.read, (f, r))
    if (frozen['registration'] != key or not frozen['held_scoring_not_run']
            or result['freeze'] != study.artifact(f) or not result['original_2pct']
            or result['reserved_outcomes_opened'] or result['independent_confirmation']):
        raise ValueError('Require the verified trained exact-slot controls')
    p = OUT/'registration.json'
    files = (Path(__file__), Path(budget.__file__), Path(prior.model.__file__), Path(prior.__file__), Path(menu.__file__))
    base.immutable(p, dict(parent=[study.artifact(x) for x in (reg, f, r)], code=[study.artifact(x) for x in files],
        arms=ARMS, sites=SITES, seeds=SEEDS, pairing='consecutive nonoverlapping queries within recording',
        preserve='per pair exact multiset of displacement slots; at most two edits per window, one CV recipient',
        objective='sum window predicted utility divided by visible-agent count',
        requires_later_observation_prefix=True, uses_future_labels_for_allocation=False, offline_diagnostic=True,
        original_2pct=True, threshold_tuning=False, new_training=False))
    return base.sha(p), cohorts, slate, frozen, result, weights, producer


def pairs(recording, start, stride):
    if len(recording) != len(start):
        raise ValueError('Unaligned query metadata')
    for rec in np.unique(recording):
        at = np.flatnonzero(recording == rec)
        if np.any(np.diff(start[at]) <= 19*stride):
            raise ValueError('Overlapping or out-of-order recording windows')
        yield from (at[i:i+2] for i in range(0, len(at), 2))


def predict(net, h, cv, ols, bank, q, origin, recording, starts):
    result = {k: cv.copy() for k in ARMS}
    peak = np.linalg.norm(origin-cv, axis=-1).max(-1)
    if not np.array_equal(np.unique(q), np.arange(len(recording))):
        raise ValueError('Query metadata does not match observation rows')
    stats = Counter()
    for window in pairs(recording, starts, study.donor.STRIDE):
        rows = [np.flatnonzero(q == t) for t in window]
        radii = [tuple(peak[ix][peak[ix] > 0.]) for ix in rows]
        if not any(radii):
            continue
        def choices(ix, radii):
            return budget.scored_window(net, h[ix], cv[ix], ols[ix], bank[ix], radii)
        if len(rows) == 1:
            pred, _ = choices(rows[0], radii[0])
            for k, p in zip(ARMS, pred*2, strict=True):
                result[k][rows[0]] = p
            stats['edited_tail_windows'] += 1
            continue
        p = budget.allocate(lambda a: choices(rows[0], a), lambda b: choices(rows[1], b), *radii)
        stats['edited_pairs'] += 1
        stats['feasible_partitions'] += p['feasible_partitions']
        flat = np.concatenate(rows)
        for k, forecast in zip(ARMS, p['fixed']+p['pooled'], strict=True):
            for ix, path in zip(rows, forecast, strict=True):
                result[k][ix] = path
            spent = np.linalg.norm(result[k][flat]-cv[flat], axis=-1).max(-1)
            np.testing.assert_allclose(np.sort(spent[spent > 0.]), np.sort(radii[0]+radii[1]), atol=1e-9, rtol=1e-9)
        for k, moved in zip(ARMS[2:], p['moved'], strict=True):
            stats[f'{k}_earlier'] += moved[0]
            stats[f'{k}_later'] += moved[1]
            stats[f'{k}_changed_pairs'] += bool(sum(moved))
        stats['pooled_decoder_disagreements'] += int(not np.allclose(result[ARMS[2]][flat], result[ARMS[3]][flat]))
    for k in ARMS:
        if np.count_nonzero(np.linalg.norm(result[k]-cv, axis=-1).max(-1)) != np.count_nonzero(peak):
            raise ValueError('A decoder changed the total edit count')
    return result, dict(stats)


def freeze(key, cohorts, slate, fitted, weights, producer):
    stats, banks = {}, {}
    for site in SITES:
        h, q, cv, ols, bank = menu.parent.source_inputs(cohorts[site])
        with np.load(study.checked(cohorts[site]), allow_pickle=False) as src:
            recording, start = src['recording'], src['start']
        for seed in SEEDS:
            name = f'{site}_{seed}'
            fit = fitted['fits'][name]
            state = torch.load(study.checked(fit['checkpoint']), weights_only=True, map_location='cpu')
            if (state['identity'] != fit['identity'] or state['identity']['held'] != site
                    or set(state['identity']['donors']) != set(SITES)-{site} or state['step'] != prior.STEPS):
                raise ValueError('The cost head does not exclude the scored location')
            net = prior.model.CostHead().eval()
            net.load_state_dict(state['net'])
            nn = menu.self_forecast(h, site, seed, weights[name], producer)
            with (np.load(study.checked(slate['forecasts'][name]), allow_pickle=False) as origin,
                  np.load(study.checked(fitted['banks'][name]), allow_pickle=False) as old):
                np.testing.assert_array_equal(q, origin['q'])
                np.testing.assert_array_equal(cv, origin['cv'])
                p, stats[name] = predict(net, h, cv, ols, np.concatenate((bank, nn[:, None]), axis=1), q,
                                         origin['joint'], recording, start)
                for k, b in zip(ARMS[:2], ('trained_additive', 'trained_joint'), strict=True):
                    np.testing.assert_array_equal(p[k], old[b])
            banks[name] = exp.previous.save_bank(LOCAL/f'{name}.npz', dict(q=q, cv=cv, ols4=ols, **p))
            tmp = LOCAL/'heartbeat.pending'
            tmp.write_text(study.json.dumps(dict(pid=os.getpid(), completed=len(banks), total=21, last=name))+'\n')
            tmp.replace(LOCAL/'heartbeat.json')
            print(name, stats[name], flush=True)
    base.immutable(OUT/'freeze.json', dict(registration=key, banks=banks, assignments=stats,
                                         outcomes_accessed=False, uses_later_prefix=True))


def score(key, cohorts, previous):
    p = OUT/'freeze.json'
    frozen = study.read(p)
    if frozen['registration'] != key or frozen['outcomes_accessed'] or len(frozen['banks']) != 21:
        raise ValueError('All allocations must freeze before scoring')
    sources = exp.loader(cohorts)
    names = dict(zip(menu.ARMS, ARMS, strict=True))
    sites = {}
    for site in SITES:
        d = sources[site]
        cut = np.quantile(np.concatenate([v['e'] for s, v in sources.items() if s != site]), [.25, .75])
        rows, effects = [], []
        for seed in SEEDS:
            with np.load(study.checked(frozen['banks'][f'{site}_{seed}']), allow_pickle=False) as z:
                np.testing.assert_array_equal(d['q'], z['q'])
                pths = {k: z[v] for k, v in names.items()}
                pths.update(cv=z['cv'], ols4=z['ols4'])
            for rec in np.unique(d['rec']):
                ix = d['rec'] == rec
                a, b = menu.recording({k: v[ix] for k, v in pths.items()}, d['y'][ix], d['v'][ix],
                                      np.unique(d['q'][ix], return_inverse=True)[1], cut)
                rows.append({names[k]: v for k, v in a.items()})
                effects.append({f'{names[l]}_vs_{names[r]}': b[f'{l}_vs_{r}'] for l, r in menu.CONTRASTS})
        counts = {k: {m: sum(r[k][m] for r in rows) for m in ('edits', 'risk', 'dual')} for k in ARMS}
        for k, b in zip(ARMS[:2], prior.ARMS[2:], strict=True):
            if counts[k] != previous['by_site'][site]['counts'][b]:
                raise ValueError('The fixed-budget control changed')
        vals = {k: [a[k] for a in effects if a[k] is not None] for k in effects[0]}
        sites[site] = dict(views=len(rows), counts=counts, estimable={k: len(v) for k, v in vals.items()},
            contrasts={k: float(np.mean(v)) if v else None for k, v in vals.items()})
    def interval(key):
        x = np.array([s['contrasts'][key] for s in sites.values() if s['contrasts'][key] is not None])
        d = np.random.default_rng(3183).choice(x, size=(3000, len(x))).mean(1)
        return dict(mean_pp=float(x.mean()), ci95_pp=np.quantile(d, [.025, .975]).tolist(), locations=len(x))
    result = dict(result_source='fresh_run', checkpoints='cached_verified', freeze=study.artifact(p),
        counts={k: dict(sum((Counter(s['counts'][k]) for s in sites.values()), Counter())) for k in ARMS},
        contrasts={k: interval(k) for k in next(iter(sites.values()))['contrasts']},
        assignments=dict(sum(map(Counter, frozen['assignments'].values()), Counter())), sites=sites,
        offline_diagnostic=True, original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False,
        deployment_changed=False)
    base.immutable(OUT/'readout.json', result)
    print(study.json.dumps({k: result[k] for k in ('counts', 'contrasts', 'assignments')}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'score'))
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, slate, fitted, result, weights, producer = register()
        if args.phase == 'freeze':
            freeze(key, cohorts, slate, fitted, weights, producer)
        else:
            score(key, cohorts, result)


if __name__ == '__main__':
    main()
