"""Refit two opening decisions on an eight-slot causal retrieval portfolio."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_shortlist_history as history
from src.world_model import m3w_shortlist_portfolio as model

cost, run, np = history.cost, history.run, history.np
prior, anchor, timing = history.prior, history.anchor, history.timing
PUBLIC, PRIVATE = prior.PUBLIC/'shortlist_portfolio', prior.PRIVATE/'shortlist_portfolio'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_shortlist_portfolio.py')
ARMS = ('direct', 'base', 'risk', 'event', 'matched_base', 'matched_risk', 'matched_event')


def register():
    previous = prior.PUBLIC/'shortlist_interaction'
    for path in (history.PUBLIC, previous):
        if run.read(path/'replay_execution.json')['artifact']['sha256'] != run.sha(path/'analysis.json'):
            raise ValueError('Require replayed controls')
    for item in run.read(cost.REG)['parents']:
        run.checked(item)
    parents = (cost.REG, cost.PUBLIC/'freeze.json', cost.PUBLIC/'analysis.json', Path(cost.__file__),
               Path(cost.model.__file__), Path(history.__file__), history.PUBLIC/'analysis.json',
               previous/'analysis.json', previous/'registration.json')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], arms=ARMS, budget=.02, shortlist=8,
        independent_outcomes_opened=False)))


class Seed:
    def __init__(self, d, refs, records, cell, controls, old_rank):
        self.d = d
        self.seed = seed = cell['seed']
        self.old_rank = old_rank
        self.floor, self.bank = run.candidates(d, refs, seed)
        self.direct, self.co = timing.parents(d, cell, controls)
        self.x, self.scale = run.model.descriptors(d['history'], self.bank, self.floor, d['crowd'])
        self.ix = {str(s): np.flatnonzero(d['sites'] == s) for s in np.unique(d['sites'])}
        self.groups = {s: [] for s in self.ix}
        for ix, g in records.values():
            self.groups[str(d['sites'][ix[0]])].append((ix, g))

    def rank(self, site):
        with np.load(run.checked(self.old_rank[self.seed, site]), allow_pickle=False) as z:
            return z['coefficients']

    def proposals(self, ix, groups, co, rank, control=False):
        x, s, b, c = self.x[ix], self.scale[ix], self.bank[ix], self.floor[ix]
        u, a, e, ok = run.model.decision_costs(x, co, b, c, s)
        w = prior.nested.base.select(groups, u, a, e, s, ok)
        direct, _ = prior.model.stream(groups, u/s[:, None], a, e, ok,
                            counts=[int((w[j] >= 0).sum()) for j in groups], witness=w)
        new = model.features(x, s, u, a, e, ok, rank, b, c, groups)
        old = cost.model.features(x, s, u, a, e, ok, rank, b, c, groups) if control else None
        return direct, new, old


def fit(m, controls, y, valid, old, beat):
    d, cuts = m.d, run.cuts_by_source()
    original, ranking = {}, {}
    atoms = np.empty((*m.bank.shape[:2], 6))
    for s, ix in m.ix.items():
        x, scale, c, b = m.x[ix], m.scale[ix], m.floor[ix], m.bank[ix]
        label, atoms[ix], _ = anchor.model.labels(c, b, y[ix], valid[ix], cuts[s]['easy'])
        legal = np.linalg.norm(b - c[:, None], axis=-1).max(2) > 0.
        ranking[s] = anchor.model.fitting(x, label, legal, d['crowd'][ix])
        original[s] = run.model.fitting_moments(x, b, c, y[ix], valid[ix], scale,
                                              1./d['crowd'][ix], cuts[s]['easy'])
    folds = []
    for item in controls:
        r = item['identity']
        s, learn, cal = r['held'], r['fit'], r['cal_a'] + r['cal_b']
        ident = dict(seed=m.seed, held=s, fit=learn, cal=cal, registration=run.sha(REG))
        if (len(learn), len(cal), len(set(learn+cal+[s]))) != (7, 6, 14):
            raise ValueError('Invalid source roles')
        if any(old[s]['identity'][k] != ident[k] for k in ('seed', 'held', 'fit', 'cal')):
            raise ValueError('Require identical control roles')
        mark = PRIVATE/f'{m.seed}_{s}.json'
        if not mark.exists():
            stats = []
            for src in learn:
                donors = [t for t in learn if t != src]
                co = run.model.coefficients([original[t] for t in donors])
                rank = anchor.model.fit([ranking[t] for t in donors])
                local = []
                for ix, g in m.groups[src]:
                    dc, a, _ = m.proposals(ix, g, co, rank)
                    target = cost.model.targets(atoms[ix], m.scale[ix], a)
                    local.append(cost.model.moments(a, target, dc))
                stats.append(local)
            weights = cost.model.fit(stats)
            co, rank = m.co[s], m.rank(s)
            donors = [[], []]
            for src in cal:
                for ix, g in m.groups[src]:
                    dc, a, _ = m.proposals(ix, g, co, rank)
                    z, k = atoms[ix], np.flatnonzero(dc >= 0)
                    for bucket, opts in zip(donors, model.options(dc, a, weights)):
                        bucket.append(dict(mass=z[k, dc[k]].sum(0), atoms=z, crowd=d['crowd'][ix],
                            options=opts, easy=float(z[:, 0, 1].sum())))
            thresholds = [anchor.model.calibrate(v)[0] for v in donors]
            at = m.ix[s]
            base, _ = history.load_old(old[s], d, at)
            np.testing.assert_array_equal(base[0], m.direct[at])
            ch = base[[0, 1, 0, 0, 2, 0, 0]].copy()
            for ix, g in m.groups[s]:
                dc, a, _ = m.proposals(ix, g, co, rank)
                np.testing.assert_array_equal(dc, m.direct[ix])
                j = np.searchsorted(at, ix)
                for n, (opts, limit) in enumerate(zip(model.options(dc, a, weights), thresholds)):
                    for row, threshold in ((2+n, limit), (5+n, -np.inf)):
                        result = anchor.model.first(opts, threshold)
                        if result is not None:
                            _, i, family, _ = result
                            ch[row, j[i]] = family
            cache = mark.with_suffix('.npz')
            history.feedback.write_array(cache, ids=d['global_ids'][at], choices=ch, coefficients=weights)
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache), thresholds=thresholds)))
        fold = run.read(mark)
        if fold['identity'] != ident:
            raise ValueError('Incompatible fold checkpoint')
        run.checked(fold['cache'])
        folds.append(fold)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def pairs():
    cv = run.parent.CV
    return {a: ('matched_base' if a.startswith('matched') else 'base', a, cv)
            for a in ('risk', 'event', 'matched_risk', 'matched_event')}


def score(m, records, y, valid, old, freeze, rows, recall):
    d, p, cuts = m.d, run.parent, run.cuts_by_source()
    ch = np.full((len(ARMS), len(m.floor)), -1, np.int16)
    weights, base_weights = {}, {}
    for f in freeze['folds']:
        s = f['identity']['held']
        at = m.ix[s]
        base, base_weights[s] = history.load_old(old[s], d, at)
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            ch[:, at] = z['choices']
            weights[s] = z['coefficients']
        np.testing.assert_array_equal(ch[[0, 1, 4]][:, at], base)
    np.testing.assert_array_equal(ch[0], m.direct)
    for rec, (ix, g) in records.items():
        s = str(d['sites'][ix[0]])
        e, h = cuts[s]['easy'], cuts[s]['hard']
        z = ch[:, ix]
        dc, on = z[0] >= 0, z >= 0
        if (z[:, dc] != z[0, dc]).any() or (on & ~dc).sum(1).max() > 1:
            raise ValueError('Direct decisions changed or too many openings')
        for q in g:
            count = on[4:, q].sum(1)
            if not (count == count[0]).all():
                raise ValueError('Diagnostic coverage differs')
        paths = {k: run.model.selected_paths(v, m.bank[ix], m.floor[ix]) for k, v in zip(ARMS, z)}
        out = p.readout.score_recording({p.CV: m.floor[ix], **paths}, pairs(), y[ix], valid[ix],
                                       d['queries'][ix], rec, easy_cut=e, hard_cut=h)
        rows.append(dict(seed=m.seed, source=s, comparisons=out['comparisons'],
                         risk={k: out['metrics'][k]['risk'] for k in ARMS}))
        if dc.any():
            continue
        _, a, original = m.proposals(ix, g, m.co[s], m.rank(s), control=True)
        _, z, _ = anchor.model.labels(m.floor[ix], m.bank[ix], y[ix], valid[ix], e)
        v, w = [cost.model.targets(z, m.scale[ix], src[:2]) for src in (a, original)]
        hits = dict(base_list=False, portfolio=False, base=False, risk=False, event=False)
        for t, (example, before) in enumerate(zip(a[:2], original[:2])):
            np.testing.assert_array_equal(example[0], before[0])
            if len(example[1]) != len(before[1]):
                raise ValueError('Candidate count differs')
            for k, sample, label in (('base_list', before, w[t]), ('portfolio', example, v[t])):
                hits[k] |= bool((label[:, 4] > 0.).any())
            for k, sample, label, (_, i, j) in (
                ('base', before, w[t], cost.model.winners(before, base_weights[s])[1]),
                *[(k, example, v[t], p) for k, p in zip(('risk', 'event'), model.winners(example, weights[s]))],
            ):
                hits[k] |= bool((label[(sample[1] == i) & (sample[2] == j), 4] > 0.).any())
        recall.append(hits)


def report(rows, recall):
    sites = sorted({r['source'] for r in rows})
    if (len(sites), len(rows), len(recall)) != (14, 102, 18):
        raise ValueError('Require complete development roster')
    hits = {k: sum(r[k] for r in recall) for k in recall[0]}
    previous = run.read(history.PUBLIC/'analysis.json')
    if (hits['base_list'], hits['base']) != tuple(previous['recall']['first_two'][k]
                                                for k in ('shortlist', 'base')):
        raise ValueError('Require exact unaugmented recall')
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
    if any(risk[k] != previous['risk'][k] for k in ('direct', 'base', 'matched_base')):
        raise ValueError('Require exact control risk counts')
    run.once(PUBLIC/'analysis.json', run.stable(dict(views=102, source_groups=14, zero_views=18,
        fit_heads=42, first_two_recall=hits, contrasts=ci, risk=risk,
        recall_retained=sum(r['base_list'] and r['portfolio'] for r in recall),
        recall_added=sum(not r['base_list'] and r['portfolio'] for r in recall),
        recall_lost=sum(r['base_list'] and not r['portfolio'] for r in recall),
        original_rules_unchanged=True, independent_confirmation=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
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
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resume from the last saved source checkpoint')
            message = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', message)
            print(message, flush=True)

        beat(state='start')
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = prior.blocks(d, lab['frames'])
        y, valid = lab['target_eval'] - d['origin'][:, None], lab['valid']
        parent, controls, old_rank = cost.banks()
        control = {c['seed']: {f['identity']['held']: f for f in c['folds']}
                   for c in run.read(cost.PUBLIC/'freeze.json')['cells']}
        frozen = PUBLIC/'freeze.json'
        if phase != 'fit':
            if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(frozen):
                raise ValueError('Require a complete fit before scoring')
            saved = {c['seed']: c for c in run.read(frozen)['cells']}
        cells, rows, recall = [], [], []
        for cell in parent:
            m = Seed(d, refs, records, cell, controls, old_rank)
            seed = m.seed
            if phase == 'fit':
                cells.append(fit(m, controls[seed], y, valid, control[seed], beat))
            else:
                score(m, records, y, valid, control[seed], saved[seed], rows, recall)
            del m
            beat(state='seed_complete', seed=seed)
        if phase == 'fit':
            run.once(frozen, run.stable(dict(cells=cells, held_excluded=True)))
        else:
            report(rows, recall)
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            artifact=run.artifact(frozen if phase == 'fit' else PUBLIC/'analysis.json')))


if __name__ == '__main__':
    main()
