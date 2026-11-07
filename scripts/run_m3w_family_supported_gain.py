"""Fit supported-gain retrieval without dropping any forecast family."""
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
from scripts import run_m3w_shortlist_exclusions as previous
from src.world_model import m3w_family_supported_gain as model

portfolio, history, cost, run, np = previous.portfolio, previous.history, previous.cost, previous.run, previous.np
prior, anchor = previous.prior, previous.anchor
PUBLIC, PRIVATE = prior.PUBLIC/'family_supported_gain', prior.PRIVATE/'family_supported_gain'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_family_supported_gain.py')
ARMS = ('direct', 'base', 'family_gain', 'matched_base', 'matched_family_gain')


def register():
    a = previous.PUBLIC/'analysis.json'
    if run.read(previous.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(a):
        raise ValueError('Require exact previous replay')
    for item in run.read(portfolio.REG)['parents']:
        run.checked(item)
    parents = (a, previous.REG, portfolio.REG, Path(portfolio.__file__),
               cost.REG, cost.PUBLIC/'freeze.json', Path(cost.model.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], arms=ARMS, budget=.02,
        shortlist_family_slots_fixed=True, independent_outcomes_opened=False)))


def proposals(m, ix, g, co, old_rank, new_rank):
    x, scale, b, c = m.x[ix], m.scale[ix], m.bank[ix], m.floor[ix]
    u, a, e, legal = run.model.decision_costs(x, co, b, c, scale)
    w = prior.nested.base.select(g, u, a, e, scale, legal)
    dc, _ = prior.model.stream(g, u/scale[:, None], a, e, legal,
                            counts=[int((w[j] >= 0).sum()) for j in g], witness=w)
    before = cost.model.features(x, scale, u, a, e, legal, old_rank, b, c, g)
    after = model.replace(before, x, scale, u, a, e, legal, old_rank, new_rank, b, c)
    return dc, before, after


def fit(m, controls, old, y, valid, beat):
    cuts, d = run.cuts_by_source(), m.d
    atoms = np.empty((*m.bank.shape[:2], 6))
    moments = {}
    for s, ix in m.ix.items():
        b, c = m.bank[ix], m.floor[ix]
        _, atoms[ix], _ = anchor.model.labels(c, b, y[ix], valid[ix], cuts[s]['easy'])
        legal = np.linalg.norm(b - c[:, None], axis=-1).max(2) > 0.
        moments[s] = model.moments(m.x[ix], atoms[ix], legal, m.scale[ix], d['crowd'][ix])
    folds = []
    for item in controls:
        r = item['identity']
        s, train, cal = r['held'], r['fit'], r['cal_a'] + r['cal_b']
        ident = dict(seed=m.seed, held=s, fit=train, cal=cal, registration=run.sha(REG))
        if (len(train), len(cal), len(set(train+cal+[s]))) != (7, 6, 14):
            raise ValueError('Require disjoint source roles')
        if any(old[s]['identity'][k] != ident[k] for k in ('seed', 'held', 'fit', 'cal')):
            raise ValueError('Require the same roles as the old selector')
        mark = PRIVATE/f'{m.seed}_{s}.json'
        if not mark.exists():
            at = m.ix[s]
            base, coef = history.load_old(old[s], d, at)
            co, old_rank = m.co[s], m.rank(s)
            new_rank = model.coefficients([moments[t] for t in train])
            donors = []
            for src in cal:
                for ix, g in m.groups[src]:
                    dc, _, a = proposals(m, ix, g, co, old_rank, new_rank)
                    z, k = atoms[ix], np.flatnonzero(dc >= 0)
                    donors.append(dict(mass=z[k, dc[k]].sum(0), atoms=z, crowd=d['crowd'][ix],
                        options=history.make_options(dc, a, coef), easy=float(z[:, 0, 1].sum())))
            threshold, _ = anchor.model.calibrate(donors)
            ch = base[[0, 1, 0, 2, 0]].copy()
            for ix, g in m.groups[s]:
                dc, _, a = proposals(m, ix, g, co, old_rank, new_rank)
                np.testing.assert_array_equal(dc, m.direct[ix])
                j = np.searchsorted(at, ix)
                opts = history.make_options(dc, a, coef)
                for k, limit in ((2, threshold), (4, -np.inf)):
                    z = anchor.model.first(opts, limit)
                    if z is not None:
                        _, i, f, _ = z
                        ch[k, j[i]] = f
            cache = mark.with_suffix('.npz')
            history.feedback.write_array(cache, ids=d['global_ids'][at], choices=ch, ranking=new_rank)
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache), threshold=threshold)))
        f = run.read(mark)
        if f['identity'] != ident:
            raise ValueError('Incompatible checkpoint')
        run.checked(f['cache'])
        folds.append(f)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def pairs():
    cv = run.parent.CV
    return dict(calibrated=('base', 'family_gain', cv),
                matched=('matched_base', 'matched_family_gain', cv))


def score(m, records, old, frozen, y, valid, rows, recall):
    d, p, cuts = m.d, run.parent, run.cuts_by_source()
    ch = np.full((len(ARMS), len(m.floor)), -1, np.int16)
    coefs, ranks = {}, {}
    for f in frozen['folds']:
        s = f['identity']['held']
        at = m.ix[s]
        base, coefs[s] = history.load_old(old[s], d, at)
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            ch[:, at] = z['choices']
            ranks[s] = z['ranking']
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
        prefix = []
        for q in g:
            if dc[q].any():
                break
            prefix.append(q)
        _, before, after = proposals(m, ix, prefix, m.co[s], m.rank(s), ranks[s])
        _, atoms, _ = anchor.model.labels(m.floor[ix], m.bank[ix], y[ix], valid[ix], e)
        hit = dict(zero=not dc.any())
        counts = {}
        for key, src in (('base', before), ('family_gain', after)):
            a = src[:2]
            target = cost.model.targets(atoms, m.scale[ix], a)
            hit[key] = any((v[:, 4] > 0.).any() for v in target)
            counts[key] = Counter()
            hit[key+'_winner'] = False
            for example, v in zip(a, target):
                good = v[:, 4] > 0.
                _, i, f, _ = example
                counts[key].update(run.NAMES[k] for k in f[good])
                _, j, k = cost.model.winners(example, coefs[s])[1]
                hit[key+'_winner'] |= bool((good & (i == j) & (f == k)).any())
        recall.append(dict(**hit, counts=counts))


def report(rows, recall):
    sites = sorted({r['source'] for r in rows})
    zero = [r for r in recall if r['zero']]
    if (len(sites), len(rows), len(zero)) != (14, 102, 18):
        raise ValueError('Incomplete roster')
    prev = run.read(previous.PUBLIC/'analysis.json')
    keys = ('base', 'family_gain', 'base_winner', 'family_gain_winner')
    hits = {k: sum(r[k] for r in zero) for k in keys}
    full = {k: sum(r[k] for r in recall) for k in keys}
    if (hits['base'], hits['base_winner'], full['base']) != (
            prev['first_two_zero_recall']['family'], prev['first_two_zero_winners']['base'],
            prev['prefix_first_two_all_recall']['family']):
        raise ValueError('Prior control recall differs')
    families = {key: dict(sum((r['counts'][key] for r in zero), Counter()))
                for key in ('base', 'family_gain')}
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
    run.once(PUBLIC/'analysis.json', run.stable(dict(views=102, source_groups=14, zero_views=18,
        first_two_zero_recall=hits, prefix_first_two_all_recall=full, zero_useful_actions_by_family=families,
        retained=sum(r['base'] and r['family_gain'] for r in zero),
        added=sum(not r['base'] and r['family_gain'] for r in zero),
        lost=sum(r['base'] and not r['family_gain'] for r in zero),
        contrasts=ci, risk=risk, fitted_rank_heads=42, original_rules_unchanged=True,
        independent_confirmation=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)))


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
            raise ValueError('Commit registration before execution')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resume from the last saved source checkpoint')
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            print(msg, flush=True)

        beat(state='start')
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = prior.blocks(d, lab['frames'])
        y, valid = lab['target_eval'] - d['origin'][:, None], lab['valid']
        parent, controls, old_ranks = cost.banks()
        old = {c['seed']: {f['identity']['held']: f for f in c['folds']}
               for c in run.read(cost.PUBLIC/'freeze.json')['cells']}
        out = PUBLIC/'freeze.json'
        if phase != 'fit':
            if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(out):
                raise ValueError('Require frozen decisions before score')
            saved = {c['seed']: c for c in run.read(out)['cells']}
        cells, rows, recalls = [], [], []
        for cell in parent:
            m = portfolio.Seed(d, refs, records, cell, controls, old_ranks)
            seed = m.seed
            if phase == 'fit':
                cells.append(fit(m, controls[seed], old[seed], y, valid, beat))
            else:
                score(m, records, old[seed], saved[seed], y, valid, rows, recalls)
            del m
            beat(state='seed_complete', seed=seed)
        if phase == 'fit':
            run.once(out, run.stable(dict(cells=cells, held_excluded=True)))
        else:
            report(rows, recalls)
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            artifact=run.artifact(out if phase == 'fit' else PUBLIC/'analysis.json')))


if __name__ == '__main__':
    main()
