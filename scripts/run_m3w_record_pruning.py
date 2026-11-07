"""Test deletion-only recording repair against an exact hindsight capacity diagnostic."""
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
from scripts import run_m3w_family_supported_gain as previous
from src.world_model import m3w_record_pruning as model

cost, run, np, prior = previous.cost, previous.run, previous.np, previous.prior
PUBLIC, PRIVATE = prior.PUBLIC/'record_pruning', prior.PRIVATE/'record_pruning'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_record_pruning.py')
ARMS = ('direct', 'prune', 'matched_cost')


def register():
    a, diagnostic = (previous.PUBLIC/k for k in ('analysis.json', 'opening_diagnostic.json'))
    if run.read(diagnostic)['analysis_sha256'] != run.sha(a):
        raise ValueError('Require verified local-to-record gap')
    for item in run.read(cost.REG)['parents']:
        run.checked(item)
    parents = (a, diagnostic, previous.REG, cost.REG, cost.PUBLIC/'freeze.json',
               Path(run.model.__file__), Path(run.parent.readout.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], arms=ARMS, budget=.02, max_deletions_per_query=1,
        independent_outcomes_opened=False)))


def frozen_direct(m, ix, groups, co):
    x, s = m.x[ix], m.scale[ix]
    u, a, e, ok = run.model.decision_costs(x, co, m.bank[ix], m.floor[ix], s)
    w = prior.nested.base.select(groups, u, a, e, s, ok)
    z, _ = prior.model.stream(groups, u/s[:, None], a, e, ok,
                                counts=[int((w[g] >= 0).sum()) for g in groups], witness=w)
    i = np.flatnonzero(z >= 0)
    f = z[i]
    x = np.column_stack((x[i, f], np.eye(ok.shape[1])[f],
                         *[np.arcsinh(v[i, f]/s[i]) for v in (u, a, e)]))
    q = m.d['queries'][ix[i]]
    group = [np.flatnonzero(q == v) for v in np.unique(q)]
    return z, i, f, x, group, (a[i, f]+e[i, f])/s[i]


def targets(m, ix, i, f, y, valid, cut):
    floor = m.floor[ix]
    v = model.realized(floor[i], m.bank[ix[i], f], y[ix[i]], valid[ix[i]], cut)
    full = model.moments(floor, floor, y[ix], valid[ix], cut)['ref_easy'].sum()
    return v, float(full), 1./m.d['crowd'][ix[i]]


def fit(m, controls, y, valid, beat):
    cuts = run.cuts_by_source()
    original = {}
    for s, ix in m.ix.items():
        original[s] = run.model.fitting_moments(m.x[ix], m.bank[ix], m.floor[ix], y[ix], valid[ix],
                       m.scale[ix], 1./m.d['crowd'][ix], cuts[s]['easy'])
    folds = []
    for control in controls:
        r = control['identity']
        site, learn, cal = r['held'], r['fit'], r['cal_a'] + r['cal_b']
        ident = dict(seed=m.seed, held=site, fit=learn, cal=cal, registration=run.sha(REG))
        if (len(learn), len(cal), len(set(learn+cal+[site]))) != (7, 6, 14):
            raise ValueError('Require disjoint source roles')
        mark = PRIVATE/f'{m.seed}_{site}.json'
        if not mark.exists():
            stats = []
            for s in learn:
                co = run.model.coefficients([original[t] for t in learn if t != s])
                local = []
                for ix, g in m.groups[s]:
                    _, i, f, x, _, _ = frozen_direct(m, ix, g, co)
                    if len(i):
                        v, _, w = targets(m, ix, i, f, y, valid, cuts[s]['easy'])
                        local.append(model.fit_stats(x, v, m.scale[ix[i]], w))
                if local:
                    stats.append(tuple(np.mean([v[k] for v in local], axis=0) for k in (0, 1)))
            coeff = model.coefficients(stats)
            co = m.co[site]
            donors = []
            for s in cal:
                for ix, g in m.groups[s]:
                    _, i, f, x, group, _ = frozen_direct(m, ix, g, co)
                    v, full, w = targets(m, ix, i, f, y, valid, cuts[s]['easy'])
                    donors.append((group, v, model.deletion_rank(x, coeff), full, w))
            threshold = model.calibrate(donors)
            at = m.ix[site]
            ch = np.broadcast_to(m.direct[at], (len(ARMS), len(at))).copy()
            for ix, g in m.groups[site]:
                dc, i, _, x, group, fallback_rank = frozen_direct(m, ix, g, co)
                np.testing.assert_array_equal(dc, m.direct[ix])
                drop = model.choose(group, model.deletion_rank(x, coeff), threshold)
                j = np.searchsorted(at, ix[i])
                ch[1, j[drop]] = -1
                for q in group:
                    if drop[q].any():
                        ch[2, j[q[fallback_rank[q].argmax()]]] = -1
            cache = mark.with_suffix('.npz')
            previous.history.feedback.write_array(cache, ids=m.d['global_ids'][at], choices=ch,
                                                  coefficients=coeff)
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache), threshold=threshold)))
        fold = run.read(mark)
        if fold['identity'] != ident:
            raise ValueError('Incompatible checkpoint')
        run.checked(fold['cache'])
        folds.append(fold)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def pairs():
    p = run.parent
    return dict(prune_vs_direct=('direct', 'prune', p.CV),
                prune_vs_count_control=('matched_cost', 'prune', p.CV))


def score(m, records, frozen, y, valid, rows, beat):
    d, p, cuts = m.d, run.parent, run.cuts_by_source()
    ch = np.full((len(ARMS), len(m.floor)), -1, np.int16)
    for f in frozen['folds']:
        at = m.ix[f['identity']['held']]
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            ch[:, at] = z['choices']
    np.testing.assert_array_equal(ch[0], m.direct)
    for j, (rec, (ix, groups)) in enumerate(records.items()):
        s = str(d['sites'][ix[0]])
        e, h = cuts[s]['easy'], cuts[s]['hard']
        z = ch[:, ix]
        if ((z >= 0) & (z != z[0])).any():
            raise ValueError('A deletion must not change or add any forecast family')
        for g in groups:
            n = (z[:, g] >= 0).sum(1)
            if n[1] != n[2] or n[0] - n[1] > 1:
                raise ValueError('Require same-query matched counts and cap')
        c, b = m.floor[ix], m.bank[ix]
        path = {k: run.model.selected_paths(v, b, c) for k, v in zip(ARMS, z)}
        scored = p.readout.score_recording({p.CV: c, **path}, pairs(), y[ix], valid[ix],
                         d['queries'][ix], rec, easy_cut=e, hard_cut=h)
        risk = {k: scored['metrics'][k]['risk'] for k in ARMS}
        i = np.flatnonzero(z[0] >= 0)
        v, full, w = targets(m, ix, i, z[0, i], y, valid, e)
        proofs = {}
        mark = PRIVATE/f'oracle_{m.seed}_{j:03d}.json'
        ident = dict(registration=run.sha(REG), seed=m.seed, recording=rec, source=s)
        if not mark.exists():
            for k, protect in (('risk_only', False), ('non_loss', True)):
                drop, status = model.oracle(v, d['queries'][ix[i]], w, full, protect)
                n, delta = None, None
                if drop is not None:
                    a = z[0].copy()
                    a[i[drop]] = -1
                    pred = run.model.selected_paths(a, b, c)
                    witness = p.readout.score_recording({p.CV: c, 'direct': path['direct'], 'oracle': pred},
                              {'pair': ('direct', 'oracle', p.CV)}, y[ix], valid[ix],
                              d['queries'][ix], rec, easy_cut=e, hard_cut=h)
                    pair = witness['comparisons']['pair']
                    delta = pair['known_delta'] - pair['unknown_radius']
                    if not witness['metrics']['oracle']['risk']['supported'] or (protect and delta < -1e-8):
                        status, delta = 'unresolved_postcheck', None
                    else:
                        n = int(drop.sum())
                proofs[k] = dict(status=status, removed=n, paired_lower_mass=delta)
            run.once(mark, run.stable(dict(identity=ident, results=proofs)))
        packet = run.read(mark)
        if packet['identity'] != ident:
            raise ValueError('Incompatible oracle receipt')
        rows.append(dict(seed=m.seed, source=s, contrasts=scored['comparisons'], risk=risk,
                         oracle=packet['results']))
        beat(state='view_scored', seed=m.seed, count=j+1)


def report(rows, frozen):
    sites = sorted({r['source'] for r in rows})
    if (len(sites), len(rows)) != (14, 102):
        raise ValueError('Incomplete roster')
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
    for a in ARMS:
        v = [r['risk'][a] for r in rows]
        risk[a] = dict(switches=sum(x['selected'] for x in v), passes=sum(x['supported'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None
                                and x['known_easy_net_degradation'] > .02 for x in v))
    if risk['direct'] != run.read(previous.PUBLIC/'analysis.json')['risk']['direct']:
        raise ValueError('Require exact original direct risk counts')
    oracle = {}
    for a in ('risk_only', 'non_loss'):
        z = [r['oracle'][a] for r in rows]
        feasible = [v for v in z if v['removed'] is not None]
        oracle[a] = dict(status=dict(Counter(v['status'] for v in z)), feasible=len(feasible),
            total_removals_across_feasible=sum(v['removed'] for v in feasible),
            positive_paired_gain=sum(v['paired_lower_mass'] > 0. for v in feasible))
    cut = Counter(str(f['threshold']) for c in frozen['cells'] for f in c['folds'])
    run.once(PUBLIC/'analysis.json', run.stable(dict(views=102, source_groups=14, prune_heads=42,
        risk=risk, thresholds=dict(cut), contrasts=ci, oracle=oracle,
        original_rules_unchanged=True, oracle_choices_exported=False,
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
            if v.get('count', 0) % 5 == 0 or v.get('count') == 34:
                print(msg, flush=True)

        beat(state='start')
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = prior.blocks(d, lab['frames'])
        y, valid = lab['target_eval'] - d['origin'][:, None], lab['valid']
        parent, controls, ranks = cost.banks()
        out = PUBLIC/'freeze.json'
        if phase != 'fit':
            if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(out):
                raise ValueError('Require frozen choices')
            frozen = run.read(out)
            saved = {c['seed']: c for c in frozen['cells']}
        cells, rows = [], []
        for cell in parent:
            m = previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
            if phase == 'fit':
                cells.append(fit(m, controls[m.seed], y, valid, beat))
            else:
                score(m, records, saved[m.seed], y, valid, rows, beat)
            del m
        if phase == 'fit':
            run.once(out, run.stable(dict(cells=cells, held_excluded=True)))
        else:
            report(rows, frozen)
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            artifact=run.artifact(out if phase == 'fit' else PUBLIC/'analysis.json')))


if __name__ == '__main__':
    main()
