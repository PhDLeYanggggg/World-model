"""Test support-specific retrieval and joint edits with frozen component heads."""
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
from scripts import run_m3w_support_acquisition as capacity
from src.world_model import m3w_support_joint as model

run, np, cost, prune, prior = capacity.run, capacity.np, capacity.cost, capacity.base, capacity.prior
anchor = cost.anchor
PUBLIC, PRIVATE = prior.PUBLIC/'support_joint', prior.PRIVATE/'support_joint'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_support_joint.py')
ARMS = ('direct', *model.ARMS)


def register():
    a = capacity.PUBLIC/'analysis.json'
    if run.read(capacity.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(a):
        raise ValueError('Require exact previous capacity replay')
    for r in (capacity.REG, prune.REG):
        for p in run.read(r)['parents']:
            run.checked(p)
    parents = (a, capacity.REG, Path(capacity.model.__file__), prune.REG, prune.PUBLIC/'freeze.json',
               Path(prune.__file__), cost.REG, Path(cost.model.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], arms=ARMS, cuts=model.CUTS,
        budget=.02, slots=8, independent_outcomes_opened=False)))


def prepare(m, ix, groups, co, rank, moment):
    x, s, b, c = m.x[ix], m.scale[ix], m.bank[ix], m.floor[ix]
    dc, i, _, features, _, _ = prune.frozen_direct(m, ix, groups, co)
    u, a, e, legal = run.model.decision_costs(x, co, b, c, s)
    legal &= dc[:, None] < 0
    predicted = np.einsum('nfd,fdt->nft', x, co)
    ref = np.minimum(predicted[..., 3], predicted[..., 4]).clip(0.)
    joint, _ = anchor.model.predict(x, rank)
    old = {tuple(q): np.column_stack((i, j)) for q, i, j, _ in
           cost.model.features(x, s, u, a, e, legal, rank, b, c, groups)}
    empty = np.empty((0, 2), dtype=int)
    new, n = [], b.shape[1]
    for q in groups:
        family = old.get(tuple(q), empty)
        j = model.retrieve(legal[q], -e[q]/s[q, None], ref[q], u[q]/s[q, None], joint[q]) if len(q)>1 else []
        j = np.asarray(j, dtype=int)
        support = np.column_stack((q[j//n], j%n))
        if len(family) != len(support):
            raise ValueError('Both methods must have the same eligible shortlist count')
        new.append((q, family, support))
    removal = np.zeros((len(ix), 4))
    z = features @ moment
    removal[i, :3] = z[:, :3] * s[i, None]
    removal[i, 3] = z[:, 4].clip(0.)
    return dc, new, (u, a, e, ref, joint), removal


def calibrate(rows, arm):
    best = None
    for t in model.CUTS:
        gained = lost = added_easy = changes = 0
        utility = 0.
        for arguments, atoms, full, w, known, old_risk, orig in rows:
            dc = arguments[0]
            choice = model.select(*arguments, arm, t)
            i = np.flatnonzero((choice >= 0) & (dc < 0))
            j = np.flatnonzero((choice < 0) & (dc >= 0))
            added = atoms[i, choice[i]]
            removed = atoms[j, dc[j]]
            after = anchor.model.risk(orig + added.sum(0) - removed.sum(0), full)
            gained += int(after['supported'] and not old_risk['supported'])
            lost += int(old_risk['supported'] and not after['supported'])
            added_easy += int(after['easy_net_over'] and not old_risk['easy_net_over'])
            # Undoing unknown trajectories pays their original disagreement.
            old_utility = removed[:, -1]
            delta = np.where(known[j], -old_utility, -removed[:, 2])
            utility += float(w[i] @ added[:, -1] + w[j] @ delta)
            changes += len(i) + len(j)
        if gained and not lost and not added_easy and utility >= 0.:
            option = gained, utility, -changes, t
            best = option if best is None else max(best, option)
    return None if best is None else best[-1]


def freeze(m, controls, old, y, valid, beat):
    d, cuts = m.d, run.cuts_by_source()
    atoms = np.empty((*m.bank.shape[:2], 6))
    for s, ix in m.ix.items():
        _, atoms[ix], _ = anchor.model.labels(m.floor[ix], m.bank[ix], y[ix], valid[ix], cuts[s]['easy'])
    folds = []
    for item in controls:
        r = item['identity']
        site, fit, cal = r['held'], r['fit'], r['cal_a'] + r['cal_b']
        identity = dict(seed=m.seed, held=site, fit=fit, cal=cal, registration=run.sha(REG))
        if (len(fit), len(cal), len(set(fit+cal+[site]))) != (7, 6, 14):
            raise ValueError('Require disjoint source roles')
        if any(old[site]['identity'][k] != identity[k] for k in ('seed', 'held', 'fit', 'cal')):
            raise ValueError('Require matching frozen component roles')
        mark = PRIVATE/f'{m.seed}_{site}.json'
        if not mark.exists():
            at = m.ix[site]
            with np.load(run.checked(old[site]['cache']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
                moment = z['coefficients']
                np.testing.assert_array_equal(z['choices'][0], m.direct[at])
            co, rank = m.co[site], m.rank(site)
            donors = []
            for s in cal:
                for ix, g in m.groups[s]:
                    args = prepare(m, ix, g, co, rank, moment)
                    dc, a = args[0], atoms[ix]
                    i = np.flatnonzero(dc >= 0)
                    orig, full = a[i, dc[i]].sum(0), float(a[:, 0, 1].sum())
                    donors.append((args, a, full, 1./d['crowd'][ix], valid[ix].any(1),
                                   anchor.model.risk(orig, full), orig))
            thresholds = [calibrate(donors, k) for k in model.ARMS]
            ch = np.broadcast_to(m.direct[at], (len(ARMS), len(at))).copy()
            for ix, g in m.groups[site]:
                args = prepare(m, ix, g, co, rank, moment)
                np.testing.assert_array_equal(args[0], m.direct[ix])
                j = np.searchsorted(at, ix)
                for k, (arm, cut) in enumerate(zip(model.ARMS, thresholds), 1):
                    ch[k, j] = model.select(*args, arm, cut)
            cache = mark.with_suffix('.npz')
            prune.previous.history.feedback.write_array(cache, ids=d['global_ids'][at], choices=ch)
            run.once(mark, run.stable(dict(identity=identity, cache=run.artifact(cache), thresholds=thresholds)))
        f = run.read(mark)
        if f['identity'] != identity:
            raise ValueError('Incompatible checkpoint')
        run.checked(f['cache'])
        folds.append(f)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def pairs():
    cv = run.parent.CV
    return dict(**{a: ('direct', a, cv) for a in model.ARMS},
                retrieval=('family_joint', 'support_joint', cv),
                joint=('support_insert', 'support_joint', cv))


def shortlist_capacity(m, rec, ix, g, y, valid, easy, hard, args):
    p, d = run.parent, m.d
    c, b = m.floor[ix], m.bank[ix]
    direct = args[0]
    original = run.model.selected_paths(direct, b, c)
    stats = {}
    for k, name in enumerate(('eligible_family', 'support'), 1):
        legal = np.zeros(b.shape[:2], bool)
        for q in args[1]:
            i, j = q[k].T
            legal[i, j] = True
        problem = capacity.model.pool(c, b, y[ix], valid[ix], easy, direct, g, legal)
        sol, status, _ = capacity.model.solve(problem, 'shortlist_joint')
        if sol is not None:
            drop, add = sol
            chosen = direct.copy()
            chosen[problem['selected'][drop]] = -1
            chosen[problem['row'][add]] = problem['action'][add]
            if any((chosen[q] >= 0).sum() > min(2, len(q)-1) for q in g) or (
                    problem['selected_query'][drop] < problem['query'][add]).any():
                raise ValueError('Invalid shortlist witness chronology or cap')
            out = p.readout.score_recording({p.CV: c, 'direct': original,
                    'witness': run.model.selected_paths(chosen, b, c)},
                {'paired': ('direct', 'witness', p.CV)}, y[ix], valid[ix], d['queries'][ix],
                rec, easy_cut=easy, hard_cut=hard)
            v = out['comparisons']['paired']
            if not out['metrics']['witness']['risk']['supported'] or v['known_delta']-v['unknown_radius'] < -1e-8:
                status = 'unresolved_postcheck'
        stats[name] = status
    return stats


def score(m, records, old, frozen, y, valid, rows):
    d, p, cuts = m.d, run.parent, run.cuts_by_source()
    ch = np.full((len(ARMS), len(m.floor)), -1, np.int16)
    coefs = {}
    for f in frozen['folds']:
        s = f['identity']['held']
        at = m.ix[s]
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            ch[:, at] = z['choices']
        with np.load(run.checked(old[s]['cache']), allow_pickle=False) as z:
            coefs[s] = z['coefficients']
    np.testing.assert_array_equal(ch[0], m.direct)
    for rec, (ix, g) in records.items():
        s = str(d['sites'][ix[0]])
        e, h = cuts[s]['easy'], cuts[s]['hard']
        z = ch[:, ix]
        add, remove = (z >= 0) & (z[0] < 0), (z < 0) & (z[0] >= 0)
        if ((z >= 0) & (z[0] >= 0) & (z != z[0])).any() or (add.sum(1) > 1).any() or remove[2].any():
            raise ValueError('Invalid edit type')
        inserted = np.zeros(len(ARMS), int)
        for q in g:
            inserted += add[:, q].sum(1)
            if (remove[:, q].sum(1)>inserted).any() or (remove[:, q].sum(1)>1).any() or (
                    (z[:, q] >= 0).sum(1)>min(2, len(q)-1)).any():
                raise ValueError('Invalid edit chronology or cap')
        c, b = m.floor[ix], m.bank[ix]
        paths = {k: run.model.selected_paths(v, b, c) for k, v in zip(ARMS, z)}
        out = p.readout.score_recording({p.CV: c, **paths}, pairs(), y[ix], valid[ix],
                    d['queries'][ix], rec, easy_cut=e, hard_cut=h)
        risk = {k: out['metrics'][k]['risk'] for k in ARMS}
        status = None
        if risk['direct']['completion_easy_risk_upper'] is None:
            args = prepare(m, ix, g, m.co[s], m.rank(s), coefs[s])
            status = shortlist_capacity(m, rec, ix, g, y, valid, e, h, args)
        rows.append(dict(seed=m.seed, source=s, contrasts=out['comparisons'], risk=risk,
            capacity=status, added=add.sum(1).tolist(), removed=remove.sum(1).tolist()))


def report(rows, frozen):
    sites = sorted({r['source'] for r in rows})
    missing = [r for r in rows if r['capacity'] is not None]
    if (len(sites), len(rows), len(missing)) != (14, 102, 35):
        raise ValueError('Unexpected cohort')
    groups = [[r for r in rows if r['source'] == s and r['seed'] == k]
              for s in sites for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(14, size=(3000, 14))
    ci = {}
    for key in pairs():
        v = [run.parent.readout.merge_comparisons([r['contrasts'][key] for r in g]) for g in groups]
        x = np.asarray([a['completion_lower_pp'] for a in v], float).reshape(14, 3).mean(1)
        if not np.isfinite(x).all():
            raise ValueError('Require all paired contrasts')
        ci[key] = dict(point=float(x.mean()),
                       interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())
    risk = {}
    for i, k in enumerate(ARMS):
        v = [r['risk'][k] for r in rows]
        risk[k] = dict(switches=sum(x['selected'] for x in v), passes=sum(x['supported'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None
                                and x['known_easy_net_degradation'] > .02 for x in v),
            insertions=sum(r['added'][i] for r in rows), deletions=sum(r['removed'][i] for r in rows),
            new_passes=sum(r['risk'][k]['supported'] and not r['risk']['direct']['supported'] for r in rows),
            lost_passes=sum(not r['risk'][k]['supported'] and r['risk']['direct']['supported'] for r in rows))
    old = run.read(prune.PUBLIC/'analysis.json')['risk']['direct']
    if {k: risk['direct'][k] for k in old} != old:
        raise ValueError('Require unchanged direct score')
    f = [v for c in frozen['cells'] for v in c['folds']]
    run.once(PUBLIC/'analysis.json', run.stable(dict(views=102, source_groups=14, missing_views=35,
        risk=risk, contrasts=ci, capacity_status={k: dict(Counter(r['capacity'][k] for r in missing))
            for k in ('eligible_family', 'support')},
        calibrated_folds={k: sum(x['thresholds'][j] is not None for x in f)
                          for j, k in enumerate(model.ARMS)},
        new_heads_fitted=0, original_rules_unchanged=True, oracle_choices_exported=False,
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
            raise ValueError('Commit registration before execution')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resume from the last completed source checkpoint')
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            print(msg, flush=True)

        beat(state='start')
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = prior.blocks(d, lab['frames'])
        y, valid = lab['target_eval'] - d['origin'][:, None], lab['valid']
        parent, controls, ranks = cost.banks()
        old = {c['seed']: {f['identity']['held']: f for f in c['folds']}
               for c in run.read(prune.PUBLIC/'freeze.json')['cells']}
        out = PUBLIC/'freeze.json'
        if phase != 'freeze':
            if run.read(PUBLIC/'freeze_execution.json')['artifact']['sha256'] != run.sha(out):
                raise ValueError('Require frozen decisions before score')
            frozen = run.read(out)
            saved = {c['seed']: c for c in frozen['cells']}
        cells, rows = [], []
        for cell in parent:
            m = prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
            if phase == 'freeze':
                cells.append(freeze(m, controls[m.seed], old[m.seed], y, valid, beat))
            else:
                score(m, records, old[m.seed], saved[m.seed], y, valid, rows)
            beat(state='seed_complete', seed=m.seed)
            del m
        if phase == 'freeze':
            run.once(out, run.stable(dict(cells=cells, held_excluded=True)))
        else:
            report(rows, frozen)
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            artifact=run.artifact(out if phase == 'freeze' else PUBLIC/'analysis.json')))


if __name__ == '__main__':
    main()
