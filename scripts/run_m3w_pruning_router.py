"""Test causal repairability routing and prefix support protection."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_pruning_teacher as teacher
from src.world_model import m3w_pruning_router as model

base, run, np, cost, prior = teacher.base, teacher.run, teacher.np, teacher.cost, teacher.prior
PUBLIC, PRIVATE = prior.PUBLIC/'pruning_router', prior.PRIVATE/'pruning_router'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_pruning_router.py')
ARMS = ('direct', 'imitation', 'gain_guard', 'route', 'route_reserve')


def register():
    a, b = (teacher.PUBLIC/k for k in ('analysis.json', 'transition_diagnostic.json'))
    if run.read(teacher.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(a) or (
            run.read(b)['analysis_sha256'] != run.sha(a)):
        raise ValueError('Require exact transition study')
    for item in run.read(teacher.REG)['parents']:
        run.checked(item)
    parents = (a, b, teacher.REG, teacher.PUBLIC/'freeze.json', base.PUBLIC/'analysis.json',
               base.PUBLIC/'freeze.json', Path(base.__file__), Path(base.model.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], arms=ARMS, budget=.02,
        reserve=model.RESERVE, independent_outcomes_opened=False)))


def inputs(m, ix, g, co):
    dc, i, family, x, _, _ = base.frozen_direct(m, ix, g, co)
    groups = [np.searchsorted(i, q[dc[q] >= 0]) for q in g if (dc[q] >= 0).any()]
    scale = m.scale[ix[i]]
    raw = m.x[ix[i], family]
    ref, easy = [np.einsum('nd,nd->n', raw, co[family, :, j]).clip(0.) for j in (3, 4)]
    easy = np.minimum(ref, easy)
    features = model.prefix(groups, np.column_stack((x[:, -3:], np.log1p(easy))))
    return dc, i, family, x, groups, easy*scale, features


def fit(m, controls, old, y, valid, beat):
    cuts, d = run.cuts_by_source(), m.d
    moments = {s: run.model.fitting_moments(m.x[ix], m.bank[ix], m.floor[ix], y[ix], valid[ix],
        m.scale[ix], 1./d['crowd'][ix], cuts[s]['easy']) for s, ix in m.ix.items()}
    folds = []
    for item in controls:
        r = item['identity']
        site, learn, cal = r['held'], r['fit'], r['cal_a'] + r['cal_b']
        ident = dict(seed=m.seed, held=site, fit=learn, cal=cal, registration=run.sha(REG))
        if (len(learn), len(cal), len(set(learn+cal+[site]))) != (7, 6, 14):
            raise ValueError('Require disjoint source roles')
        if any(old[site]['identity'][k] != ident[k] for k in ('seed', 'held', 'fit', 'cal')):
            raise ValueError('Require exact comparison roles')
        mark = PRIVATE/f'{m.seed}_{site}.json'
        if not mark.exists():
            stats, label_counts = [], Counter()
            for s in learn:
                co = run.model.coefficients([moments[t] for t in learn if t != s])
                local = []
                for ix, g in m.groups[s]:
                    _, i, f, _, _, _, v = inputs(m, ix, g, co)
                    if not len(i):
                        continue
                    z, full, w = base.targets(m, ix, i, f, y, valid, cuts[s]['easy'])
                    old_pass, _ = base.model.observe(z, np.zeros(len(i), bool), full)
                    if old_pass:
                        label = False
                    else:
                        drop, status = base.model.oracle(z, d['queries'][ix[i]], w, full, True)
                        if status not in ('optimal', 'infeasible', 'no_original_support'):
                            label_counts['excluded_solver'] += 1
                            continue
                        label = drop is not None
                        if label and (not base.model.observe(z, drop, full)[0] or
                                      w[drop] @ z['remove_lower'][drop] < -1e-8):
                            label_counts['excluded_postcheck'] += 1
                            continue
                    local.append(model.moments(v, float(label)))
                    label_counts['positive' if label else 'negative'] += 1
                if local:
                    stats.append(tuple(np.mean([v[k] for v in local], axis=0) for k in (0, 1)))
            coef = base.model.coefficients(stats)
            at, co = m.ix[site], m.co[site]
            with np.load(run.checked(old[site]['cache']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
                old_ch = z['choices'][:3]
                imit = z['coefficients']
            np.testing.assert_array_equal(old_ch[0], m.direct[at])
            if not len(imit):
                imit = None
            donors = []
            for s in cal:
                for ix, g in m.groups[s]:
                    _, i, f, x, groups, support, v = inputs(m, ix, g, co)
                    z, full, w = base.targets(m, ix, i, f, y, valid, cuts[s]['easy'])
                    donors.append((groups, teacher.model.scores(x, imit)[0], support,
                                   (v @ coef).clip(0., 1.), z, full, w))
            results = [model.calibrate(donors, reserve) for reserve in (False, True)]
            ch = old_ch[[0, 1, 2, 0, 0]].copy()
            probs = np.full(len(at), -1.)
            for ix, g in m.groups[site]:
                dc, i, f, x, groups, support, v = inputs(m, ix, g, co)
                np.testing.assert_array_equal(dc, m.direct[ix])
                route = (v @ coef).clip(0., 1.)
                score = teacher.model.scores(x, imit)[0]
                j = np.searchsorted(at, ix[i])
                for q, p in zip(groups, route):
                    probs[j[q]] = p
                for k, (reserve, ((r, a), _)) in enumerate(zip((False, True), results), 3):
                    ch[k, j[model.choose(groups, score, support, route, a, r, reserve)]] = -1
            cache = mark.with_suffix('.npz')
            base.previous.history.feedback.write_array(cache, ids=d['global_ids'][at], choices=ch,
                                                       probabilities=probs, router=coef)
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache),
                calibration=results, labels=dict(label_counts))))
        f = run.read(mark)
        if f['identity'] != ident:
            raise ValueError('Incompatible checkpoint')
        run.checked(f['cache'])
        folds.append(f)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def pairs():
    cv = run.parent.CV
    return dict(route=('direct', 'route', cv), route_reserve=('direct', 'route_reserve', cv),
                reserve_vs_route=('route', 'route_reserve', cv))


def score(m, records, old, frozen, y, valid, rows):
    d, p, cuts = m.d, run.parent, run.cuts_by_source()
    ch = np.full((len(ARMS), len(m.floor)), -1, np.int16)
    prob = np.full(len(m.floor), -1.)
    for f in frozen['folds']:
        s = f['identity']['held']
        at = m.ix[s]
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            ch[:, at] = z['choices']
            prob[at] = z['probabilities']
        with np.load(run.checked(old[s]['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(ch[:3, at], z['choices'][:3])
    np.testing.assert_array_equal(ch[0], m.direct)
    for j, (rec, (ix, groups)) in enumerate(records.items()):
        s = str(d['sites'][ix[0]])
        e, h = cuts[s]['easy'], cuts[s]['hard']
        z = ch[:, ix]
        if ((z >= 0) & (z != z[0])).any():
            raise ValueError('Only deletions are permitted')
        first = None
        for q in groups:
            count = (z[:, q] >= 0).sum(1)
            if (count[0]-count > 1).any():
                raise ValueError('At most one deletion per query')
            if first is None and count[0]:
                first = float(prob[ix[q[z[0, q] >= 0][0]]])
                if not 0. <= first <= 1.:
                    raise ValueError('Missing first causal score')
        c, b = m.floor[ix], m.bank[ix]
        paths = {k: run.model.selected_paths(v, b, c) for k, v in zip(ARMS, z)}
        out = p.readout.score_recording({p.CV: c, **paths}, pairs(), y[ix], valid[ix],
                         d['queries'][ix], rec, easy_cut=e, hard_cut=h)
        risk = {k: out['metrics'][k]['risk'] for k in ARMS}
        oracle = run.read(base.PRIVATE/f'oracle_{m.seed}_{j:03d}.json')
        if oracle['identity'] != dict(registration=run.sha(base.REG), seed=m.seed, recording=rec, source=s):
            raise ValueError('Unaligned independent-of-fit capacity receipt')
        label = oracle['results']['non_loss']['removed'] is not None and not risk['direct']['supported']
        rows.append(dict(seed=m.seed, source=s, contrasts=out['comparisons'], risk=risk,
                         first=first, repairable_failure=label))


def detection(rows, sites, draw):
    active = [r for r in rows if r['first'] is not None]
    y = np.asarray([r['repairable_failure'] for r in active], bool)
    p = np.asarray([r['first'] for r in active])
    sources = np.asarray([sites.index(r['source']) for r in active])
    by_site = np.eye(len(sites), dtype=int)[draw].sum(1)
    metrics = [roc_auc_score, average_precision_score]
    values = []
    for fn in metrics:
        reps = [fn(y, p, sample_weight=w) for w in by_site[:, sources] if w[y].sum() and w[~y].sum()]
        values.append(dict(point=float(fn(y, p)),
                           interval95=np.quantile(reps, [.025, .975]).tolist(), resamples=len(reps)))
    return dict(active_views=len(y), positive_views=int(y.sum()), positive_rate=float(y.mean()),
                auc=values[0], average_precision=values[1])


def report(rows, frozen):
    sites = sorted({r['source'] for r in rows})
    if (len(sites), len(rows), sum(r['repairable_failure'] for r in rows)) != (14, 102, 28):
        raise ValueError('Incomplete cohort')
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
    for k in ARMS:
        v = [r['risk'][k] for r in rows]
        risk[k] = dict(switches=sum(x['selected'] for x in v), passes=sum(x['supported'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None
                                and x['known_easy_net_degradation'] > .02 for x in v))
    prev = run.read(teacher.PUBLIC/'analysis.json')
    if any(risk[k] != prev['risk'][k] for k in ARMS[:3]):
        raise ValueError('Require exact original control risk counts')
    transitions = {}
    for k in ARMS[1:]:
        z = Counter()
        for r in rows:
            old, v = r['risk']['direct'], r['risk'][k]
            active = old['selected'] != v['selected']
            z.update(dict(changed_views=int(active), changed_repairable=int(active and r['repairable_failure']),
                new_passes=int(v['supported'] and not old['supported']),
                lost_passes=int(old['supported'] and not v['supported']),
                lost_easy_reference=int(old['completion_easy_risk_upper'] is not None
                                        and v['completion_easy_risk_upper'] is None)))
        transitions[k] = dict(z)
    f = [v for c in frozen['cells'] for v in c['folds']]
    counts = sum((Counter(v['labels']) for v in f), Counter())
    run.once(PUBLIC/'analysis.json', run.stable(dict(views=102, source_groups=14, heads_fitted=len(f),
        active_first_query_detection=detection(rows, sites, draw),
        donor_training_label_occurrences=dict(counts), risk=risk, contrasts=ci, transitions=transitions,
        calibrated_folds={k: sum(v['calibration'][j][1]>0 for v in f)
                          for j, k in enumerate(ARMS[3:])},
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
        parent, controls, ranks = cost.banks()
        old = {c['seed']: {f['identity']['held']: f for f in c['folds']}
               for c in run.read(teacher.PUBLIC/'freeze.json')['cells']}
        out = PUBLIC/'freeze.json'
        if phase != 'fit':
            if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(out):
                raise ValueError('Require frozen choices')
            frozen = run.read(out)
            saved = {c['seed']: c for c in frozen['cells']}
        cells, rows = [], []
        for cell in parent:
            m = base.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
            if phase == 'fit':
                cells.append(fit(m, controls[m.seed], old[m.seed], y, valid, beat))
            else:
                score(m, records, old[m.seed], saved[m.seed], y, valid, rows)
            beat(state='seed_complete', seed=m.seed)
            del m
        if phase == 'fit':
            run.once(out, run.stable(dict(cells=cells, held_excluded=True)))
        else:
            report(rows, frozen)
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            artifact=run.artifact(out if phase == 'fit' else PUBLIC/'analysis.json')))


if __name__ == '__main__':
    main()
