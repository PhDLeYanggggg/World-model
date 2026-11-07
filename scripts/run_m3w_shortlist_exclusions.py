"""Audit dropped early actions and replay a fixed six-plus-two repair."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_shortlist_portfolio as portfolio
from src.world_model import m3w_shortlist_exclusions as model

history, cost, run, np = portfolio.history, portfolio.cost, portfolio.run, portfolio.np
prior, anchor = portfolio.prior, portfolio.anchor
PUBLIC, PRIVATE = prior.PUBLIC/'shortlist_exclusions', prior.PRIVATE/'shortlist_exclusions'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_shortlist_exclusions.py')
ARMS = ('direct', 'base', 'hybrid', 'matched_base', 'matched_hybrid')
LISTS = ('family', 'family_distinct_agents', 'cost_no_agent_preference', 'portfolio', 'hybrid')


def register():
    analysis = portfolio.PUBLIC/'analysis.json'
    if run.read(portfolio.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(analysis):
        raise ValueError('Require verified earlier portfolio replay')
    for item in run.read(portfolio.REG)['parents']:
        run.checked(item)
    parents = (analysis, portfolio.REG, Path(portfolio.__file__), Path(portfolio.model.__file__),
               cost.PUBLIC/'analysis.json', cost.PUBLIC/'freeze.json')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], arms=ARMS, lists=LISTS, budget=.02,
        family_slots=6, learned_heads_refit=False, independent_outcomes_opened=False)))


def options(m, ix, g, co, rank, coef):
    dc, port, family = m.proposals(ix, g, co, rank, control=True)
    if len(port) != len(family):
        raise ValueError('Require same query roster')
    return dc, history.make_options(dc, [model.mix(f, p) for f, p in zip(family, port)], coef)


def freeze(m, controls, old, y, valid, beat):
    d, cuts = m.d, run.cuts_by_source()
    atoms = np.empty((*m.bank.shape[:2], 6))
    for s, ix in m.ix.items():
        _, atoms[ix], _ = anchor.model.labels(m.floor[ix], m.bank[ix], y[ix], valid[ix], cuts[s]['easy'])
    folds = []
    for item in controls:
        r = item['identity']
        s, fit, cal = r['held'], r['fit'], r['cal_a'] + r['cal_b']
        ident = dict(seed=m.seed, held=s, fit=fit, cal=cal, registration=run.sha(REG))
        if (len(fit), len(cal), len(set(fit+cal+[s]))) != (7, 6, 14):
            raise ValueError('Require disjoint source roles')
        if any(old[s]['identity'][k] != ident[k] for k in ('seed', 'held', 'fit', 'cal')):
            raise ValueError('Require the same roles as the old selector')
        mark = PRIVATE/f'{m.seed}_{s}.json'
        if not mark.exists():
            at = m.ix[s]
            base, coef = history.load_old(old[s], d, at)
            rank, co = m.rank(s), m.co[s]
            donors = []
            for src in cal:
                for ix, g in m.groups[src]:
                    dc, opts = options(m, ix, g, co, rank, coef)
                    z, k = atoms[ix], np.flatnonzero(dc >= 0)
                    donors.append(dict(mass=z[k, dc[k]].sum(0), atoms=z, crowd=d['crowd'][ix],
                        options=opts, easy=float(z[:, 0, 1].sum())))
            threshold, _ = anchor.model.calibrate(donors)
            ch = base[[0, 1, 0, 2, 0]].copy()
            for ix, g in m.groups[s]:
                dc, opts = options(m, ix, g, co, rank, coef)
                np.testing.assert_array_equal(dc, m.direct[ix])
                j = np.searchsorted(at, ix)
                for k, limit in ((2, threshold), (4, -np.inf)):
                    a = anchor.model.first(opts, limit)
                    if a is not None:
                        _, i, family, _ = a
                        ch[k, j[i]] = family
            cache = mark.with_suffix('.npz')
            history.feedback.write_array(cache, ids=d['global_ids'][at], choices=ch)
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache), threshold=threshold)))
        saved = run.read(mark)
        if saved['identity'] != ident:
            raise ValueError('Incompatible checkpoint')
        run.checked(saved['cache'])
        folds.append(saved)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def early(m, ix, g, site, target, valid, cut, coef):
    rank, co = m.rank(site), m.co[site]
    dc, port, family = m.proposals(ix, g, co, rank, control=True)
    x, scale, b, c = m.x[ix], m.scale[ix], m.bank[ix], m.floor[ix]
    j, _ = anchor.model.predict(x, rank)
    u, a, e, legal = run.model.decision_costs(x, co, b, c, scale)
    _, z, _ = anchor.model.labels(c, b, target, valid, cut)
    ref, easy, harm, _, _, utility = z.transpose(2, 0, 1)
    good = (easy > 0.) & (harm <= .02*ref) & (utility > 0.) & legal
    n = b.shape[1]
    hits = dict.fromkeys(LISTS, False)
    winner = dict(base=False, hybrid=False)
    dropped = Counter()
    for f, p in zip(family[:2], port[:2]):
        q = f[0]
        if (dc[q] >= 0).any():
            break
        unique, plain = model.alternatives(j[q], u[q]/scale[q, None],
            a[q]/scale[q, None], e[q]/scale[q, None], legal[q])
        mixed = model.mix(f, p)
        alts = ((f[1], f[2]), (q[unique//n], unique%n), (q[plain//n], plain%n),
                (p[1], p[2]), (mixed[1], mixed[2]))
        for key, (i, k) in zip(LISTS, alts):
            hits[key] |= bool(good[i, k].any())
        for key, example in (('base', f), ('hybrid', mixed)):
            _, i, k = cost.model.winners(example, coef)[1]
            winner[key] |= bool(good[i, k])
        excluded = set(zip(f[1], f[2])) - set(zip(p[1], p[2]))
        for i, k in excluded:
            if good[i, k]:
                dropped['all'] += 1
                dropped['agent_still_represented'] += int(i in p[1])
                dropped['family_still_represented'] += int(k in p[2])
                dropped[run.NAMES[k]] += 1
    return dict(hits=hits, winner=winner, dropped=dict(dropped))


def pairs():
    cv = run.parent.CV
    return dict(hybrid=('base', 'hybrid', cv), matched=('matched_base', 'matched_hybrid', cv))


def score(m, records, old, frozen, y, valid, rows, recalls):
    d, p, cuts = m.d, run.parent, run.cuts_by_source()
    ch = np.full((len(ARMS), len(m.floor)), -1, np.int16)
    coefs = {}
    for f in frozen['folds']:
        s = f['identity']['held']
        at = m.ix[s]
        base, coefs[s] = history.load_old(old[s], d, at)
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            ch[:, at] = z['choices']
        np.testing.assert_array_equal(ch[[0, 1, 3]][:, at], base)
    np.testing.assert_array_equal(ch[0], m.direct)
    for rec, (ix, g) in records.items():
        s = str(d['sites'][ix[0]])
        e, h = cuts[s]['easy'], cuts[s]['hard']
        z = ch[:, ix]
        dc, on = z[0] >= 0, z >= 0
        if (z[:, dc] != z[0, dc]).any() or (on & ~dc).sum(1).max() > 1:
            raise ValueError('Direct decisions changed or too many openings')
        if any(on[3, q].sum() != on[4, q].sum() for q in g):
            raise ValueError('Matched query switch counts differ')
        paths = {k: run.model.selected_paths(v, m.bank[ix], m.floor[ix]) for k, v in zip(ARMS, z)}
        out = p.readout.score_recording({p.CV: m.floor[ix], **paths}, pairs(), y[ix], valid[ix],
                                       d['queries'][ix], rec, easy_cut=e, hard_cut=h)
        rows.append(dict(seed=m.seed, source=s, comparisons=out['comparisons'],
                         risk={k: out['metrics'][k]['risk'] for k in ARMS}))
        recalls.append(dict(zero=not dc.any(), **early(m, ix, g, s, y[ix], valid[ix], e, coefs[s])))


def report(rows, recalls):
    sites = sorted({r['source'] for r in rows})
    zero = [r for r in recalls if r['zero']]
    if (len(sites), len(rows), len(zero)) != (14, 102, 18):
        raise ValueError('Incomplete roster')
    hits = {k: sum(r['hits'][k] for r in zero) for k in LISTS}
    lost = [r for r in zero if r['hits']['family'] and not r['hits']['portfolio']]
    prev = run.read(portfolio.PUBLIC/'analysis.json')
    if (hits['family'], hits['portfolio'], len(lost)) != (
            prev['first_two_recall']['base_list'], prev['first_two_recall']['portfolio'], prev['recall_lost']):
        raise ValueError('Original loss set does not replay')
    dropped = Counter()
    for r in lost:
        dropped.update(r['dropped'])
    causes = Counter(model.mechanism(r['hits']['family_distinct_agents'],
                     r['hits']['cost_no_agent_preference']) for r in lost)
    groups = [[r for r in rows if r['source'] == s and r['seed'] == k]
              for s in sites for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(14, size=(3000, 14))
    ci = {}
    for key in pairs():
        v = [run.parent.readout.merge_comparisons([r['comparisons'][key] for r in g]) for g in groups]
        x = np.asarray([a['completion_lower_pp'] for a in v], float).reshape(14, 3).mean(1)
        if not np.isfinite(x).all():
            raise ValueError('Require all paired contrasts')
        ci[key] = dict(point=float(x.mean()),
                       interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())
    risk = {}
    for key in ARMS:
        v = [r['risk'][key] for r in rows]
        risk[key] = dict(switches=sum(x['selected'] for x in v), passes=sum(x['supported'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None
                                and x['known_easy_net_degradation'] > .02 for x in v))
    if any(risk[k] != prev['risk'][k] for k in ('direct', 'base', 'matched_base')):
        raise ValueError('Control risk counts differ')
    win = {k: sum(r['winner'][k] for r in zero) for k in ('base', 'hybrid')}
    if win['base'] != prev['first_two_recall']['base']:
        raise ValueError('Control winner recall differs')
    run.once(PUBLIC/'analysis.json', run.stable(dict(views=102, source_groups=14, zero_views=18,
        first_two_zero_recall=hits, first_two_zero_winners=win,
        prefix_first_two_all_recall={k: sum(r['hits'][k] for r in recalls) for k in LISTS},
        lost_views=len(lost), lost_view_mechanisms=dict(causes), dropped_useful_actions=dict(dropped),
        lost_recovered_by_hybrid=sum(r['hits']['hybrid'] for r in lost),
        family_hybrid_overlap=sum(r['hits']['family'] and r['hits']['hybrid'] for r in zero),
        contrasts=ci, risk=risk, selector_heads_refit=0, original_rules_unchanged=True,
        independent_confirmation=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'freeze', 'score', 'replay'))
    phase = parser.parse_args().phase
    register()
    if phase == 'register':
        return
    for path in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))],
                                   cwd=ROOT) != path.read_bytes():
            raise ValueError('Commit experiment registration before execution')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic() - start > 43200:
                raise OSError('Resume from the last saved source checkpoint')
            message = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic() - start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', message)
            print(message, flush=True)

        beat(state='start')
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = prior.blocks(d, lab['frames'])
        y, valid = lab['target_eval'] - d['origin'][:, None], lab['valid']
        parent, controls, ranks = cost.banks()
        old = {c['seed']: {f['identity']['held']: f for f in c['folds']}
               for c in run.read(cost.PUBLIC/'freeze.json')['cells']}
        out = PUBLIC/'freeze.json'
        if phase != 'freeze':
            if run.read(PUBLIC/'freeze_execution.json')['artifact']['sha256'] != run.sha(out):
                raise ValueError('Require frozen decisions before score')
            saved = {c['seed']: c for c in run.read(out)['cells']}
        cells, rows, recalls = [], [], []
        for cell in parent:
            m = portfolio.Seed(d, refs, records, cell, controls, ranks)
            seed = m.seed
            if phase == 'freeze':
                cells.append(freeze(m, controls[seed], old[seed], y, valid, beat))
            else:
                score(m, records, old[seed], saved[seed], y, valid, rows, recalls)
            del m
            beat(state='seed_complete', seed=seed)
        if phase == 'freeze':
            run.once(out, run.stable(dict(cells=cells, held_excluded=True)))
        else:
            report(rows, recalls)
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            artifact=run.artifact(out if phase == 'freeze' else PUBLIC/'analysis.json')))


if __name__ == '__main__':
    main()
