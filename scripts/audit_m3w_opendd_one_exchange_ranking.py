"""Exhaust the one-exchange menu and compare rankers to matched chance."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_one_exchange_ranker as study


run, base, prefix, parent, old, np, torch, model = (study.run, study.base, study.prefix, study.parent, study.old,
    study.np, study.torch, study.model)
ARMS = (study.PRIMARY, 'context_forced')


def matched_chance(dual, lower_mass, anchor, i):
    """Expected scores over the same feasible actions, only where the model acts."""
    if i is None:
        return dict(chosen_dual=anchor, random_dual=anchor, chosen_gain_mass=0., random_gain_mass=0.,
                    random_rescued=0., random_destroyed=0., rescues=0, destroys=0, exchanged=0)
    chance = float(dual.mean())
    return dict(chosen_dual=int(dual[i]), random_dual=chance, chosen_gain_mass=float(lower_mass[i]),
                random_gain_mass=float(lower_mass.mean()), random_rescued=(1-anchor)*chance,
                random_destroyed=anchor*(1-chance), rescues=(1-anchor)*int(dual[i]),
                destroys=anchor*(1-int(dual[i])), exchanged=1)


def evaluate(site, seed, motion_key, key, d, bank, menu, net, cut):
    x, legal, rows, path, _ = menu
    cv = bank[0]
    move = base.model.movement_prefix(cv, path, rows)
    totals = {k: Counter() for k in ARMS}
    exact = Counter({k: 0 for k in study.frontier.FLOORS})
    count = Counter()
    for i, name in enumerate(sorted(set(d['rec']))):
        ai, qi, r = study.indices(d, name, rows)
        tag = f'{site}_{seed}_{i:04d}.json'
        ident = dict(site=site, seed=seed, ordinal=i)
        frozen = run.read(base.PRIVATE / tag)
        f = run.read(study.frontier.PRIVATE / tag)
        if (frozen['identity'] != dict(registration=motion_key, **ident) or f['identity'] != dict(registration=key, **ident)
                or frozen['cap'] != f['cap'] or frozen['edits'] != f['edits']):
            raise ValueError('Unbound private plan or bound')
        a, cap = np.asarray(frozen['score_plan']), frozen['cap']
        if a.sum() != frozen['edits']:
            raise ValueError('Frozen matched edit count changed')
        p0, s0 = study.measure(d, ai, bank, path, r, a, cut)
        if s0['dual_positive'] != f['anchor_dual']:
            raise ValueError('Frozen outcome does not match the frontier')
        pairs = model.exchange_menu(a, legal[qi], move[qi], cap)
        choose = dict.fromkeys(ARMS)
        if len(pairs):
            feats, _ = model.features(x[qi], legal[qi], move[qi], a, cap, pairs)
            with torch.inference_mode():
                out = net(torch.from_numpy(feats))
            choose[study.PRIMARY] = model.rank(out)
            choose['context_forced'] = model.rank(out, margin=-np.inf, guard=False)
        y, valid = d['y'][ai], d['valid'][ai]
        best = min(s0['vs_cv'], s0['vs_ols4']) if s0['risk_pass'] else -np.inf
        dual, lower_mass = [], []
        for src, dst in pairs:
            c = model.exchanged(a, src, dst)
            p, score = study.measure(d, ai, bank, path, r, c, cut)
            dual.append(score['dual_positive'])
            lower_mass.append(float(parent.transfer.paired(p0, p, y, valid)[1].sum()))
            if score['risk_pass']:
                best = max(best, min(score['vs_cv'], score['vs_ols4']))
        for floor, value in study.frontier.FLOORS.items():
            exact[floor] += int(best > value)
            interval = f['results']['one']
            if ((interval['lower_pp'] is not None and interval['lower_pp'] > value and best <= value)
                    or (best > value and (interval['infeasible'] or (interval['upper_pp'] is not None and interval['upper_pp'] <= value)))):
                raise ValueError('Enumeration disagrees with a certified endpoint')
        count.update(views=1, edits=int(a.sum()), candidates=len(pairs), anchor_dual=s0['dual_positive'])
        dual, lower_mass = np.asarray(dual), np.asarray(lower_mass)
        for k in ARMS:
            totals[k].update(matched_chance(dual, lower_mass, s0['dual_positive'], choose[k]))
    mass = np.linalg.norm(cv - d['y'], axis=-1)[d['valid']].sum()
    for a in totals.values():
        for k in ('chosen', 'random'):
            a[k + '_gain_pp'] = 100. * a.pop(k + '_gain_mass') / mass
    return dict(**count, exact_frontier=dict(exact), comparisons={k: dict(v) for k, v in totals.items()})


def summarize(groups):
    flat = [r for v in groups.values() for r in v]
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))

    def compare(a, field):
        def diff(r):
            v = r['comparisons'][a]
            delta = v['chosen_' + field] - v['random_' + field]
            return 100. * delta/r['views'] if field == 'dual' else delta
        p = np.array([np.mean([diff(r) for r in rows]) for rows in groups.values()])
        return dict(point_pp=float(p.mean()), ci95_pp=np.quantile(p[draws].mean(1), [.025, .975]).tolist(),
                    locations_positive=int((p > 0.).sum()))

    totals = {k: sum(r[k] for r in flat) for k in ('views', 'edits', 'candidates', 'anchor_dual')}
    choices = {}
    for k in ARMS:
        a = {f: sum(r['comparisons'][k][f] for r in flat) for f in flat[0]['comparisons'][k] if not f.endswith('gain_pp')}
        a.update(dual_rate_minus_random_pp=compare(k, 'dual'), paired_gain_minus_random_pp=compare(k, 'gain_pp'))
        choices[k] = a
    return dict(**totals, exact_frontier={f: sum(r['exact_frontier'][f] for r in flat) for f in study.frontier.FLOORS},
                comparisons=choices)


def run_audit():
    reg, key, motion, cells, ranks, _, _, _ = study.register()
    path = study.OUT / 'readout.json'
    frozen, fitted = run.read(path), run.read(study.OUT / 'head_freeze.json')
    if (frozen['registration'] != reg or fitted['registration'] != reg or frozen['external_outcome_access']
            or not frozen['original_2pct'] or frozen['checkpoint_manifest'] != run.sha(study.OUT / 'head_freeze.json')):
        raise ValueError('Require the immutable completed ranker study')
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    groups = {}
    for s, d in sites.items():
        f = parent.prior.prior.old.source.prior.PRIVATE / f'{s}_cohort.npz'
        with np.load(f, allow_pickle=False) as z:
            d.update(recordings=z['recording'], starts=z['start'])
        bank = parent.prior.analytic(d)
        cut = np.quantile(np.concatenate([sites[t]['cut_samples'] for t in sites if t != s]), [.25, .75])
        groups[s] = []
        for cell in cells['by_site'][s]:
            seed = cell['seed']
            rank = study.study.load(ranks['heads'][f'{s}/{seed}/additive_rank'],
                                    lambda: study.study.prior.model.SwapValue(False), study.study.prior.STEPS)
            menu = prefix.menu_for(d, bank, rank)
            meta = fitted['checkpoints'][f'{s}_{seed}_context']
            if (meta['identity']['held'] != s or s in meta['identity']['train'] or meta['identity']['seed'] != seed):
                raise ValueError('Ranker reused held labels')
            net = study.study.load(meta, lambda: model.ExchangeRanker(4*menu[0].shape[1]+9), study.STEPS)
            row = evaluate(s, seed, motion, key, d, bank, menu, net, cut)
            ref = next(r for r in frozen['by_site'][s] if r['seed'] == seed)
            for k in ARMS:
                a, b = row['comparisons'][k], ref['transitions'][k]
                if (a['chosen_dual'], a['exchanged'], a['rescues'], a['destroys']) != (
                        ref['scores'][k]['dual_positive'], b['exchanged'], b['rescued'], b['destroyed']):
                    raise ValueError('Ranked choices did not reproduce the scored transitions')
            groups[s].append(dict(seed=seed, **row))
    s = summarize(groups)
    if (s['views'], s['edits'], s['candidates'], s['anchor_dual']) != (606, 780, 7198, 60):
        raise ValueError('Unexpected exhaustive-menu roster')
    out = dict(rank_readout_sha256=run.sha(path), frontier_registration_sha256=key, summary=s, by_site=groups,
               original_2pct=True, no_training=True, external_outcome_access=False, offline_matching=True)
    run.once(study.OUT / 'ranking_diagnostic.json', run.stable(out))
    print(s, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    run_audit()
