"""Test source-excluded feasible-set imitation on the fixed direct stream."""
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
from scripts import run_m3w_record_pruning as base
from src.world_model import m3w_pruning_teacher as model

run, np, cost, prior = base.run, base.np, base.cost, base.prior
PUBLIC, PRIVATE = prior.PUBLIC/'pruning_teacher', prior.PRIVATE/'pruning_teacher'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_pruning_teacher.py')
ARMS = ('direct', 'imitation', 'gain_guard', 'probe_imitation', 'probe_gain_guard',
        'probe_moment_for_imitation', 'probe_moment_for_gain_guard')
PROBE = .10


def register():
    a = base.PUBLIC/'analysis.json'
    if run.read(base.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(a):
        raise ValueError('Require verified deletion capacity')
    for item in run.read(base.REG)['parents']:
        run.checked(item)
    parents = (a, base.PUBLIC/'freeze.json', base.PUBLIC/'calibration_diagnostic.json', base.REG,
               Path(base.__file__), Path(base.model.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], arms=ARMS, budget=.02,
        diagnostic_cut=PROBE, independent_outcomes_opened=False)))


def teaching_record(m, ix, groups, co, y, valid, cut, hard):
    d, p = m.d, run.parent
    dc, i, a, x, _, _ = base.frozen_direct(m, ix, groups, co)
    v, full, w = base.targets(m, ix, i, a, y, valid, cut)
    drop, status = base.model.oracle(v, d['queries'][ix[i]], w, full, True)
    if status != 'optimal':
        return None, status, 0
    # Verify the solver's label with the original metric before fitting.
    ch = dc.copy()
    ch[i[drop]] = -1
    c, b = m.floor[ix], m.bank[ix]
    paths = {p.CV: c, 'direct': run.model.selected_paths(dc, b, c),
             'teacher': run.model.selected_paths(ch, b, c)}
    rec = str(d['recordings'][ix[0]])
    out = p.readout.score_recording(paths, {'gain': ('direct', 'teacher', p.CV)},
        y[ix], valid[ix], d['queries'][ix], rec, easy_cut=cut, hard_cut=hard)
    lower = out['comparisons']['gain']
    if not out['metrics']['teacher']['risk']['supported'] or (
            lower['known_delta']-lower['unknown_radius'] < -1e-8):
        return None, 'unresolved_postcheck', 0
    return model.moments(x, drop, v['remove_lower'], m.scale[ix[i]], w), 'verified', int(drop.sum())


def fit(m, controls, old, y, valid, beat):
    cuts, d = run.cuts_by_source(), m.d
    original = {s: run.model.fitting_moments(m.x[ix], m.bank[ix], m.floor[ix], y[ix], valid[ix],
                m.scale[ix], 1./d['crowd'][ix], cuts[s]['easy']) for s, ix in m.ix.items()}
    folds = []
    for item in controls:
        r = item['identity']
        site, learn, cal = r['held'], r['fit'], r['cal_a'] + r['cal_b']
        ident = dict(seed=m.seed, held=site, fit=learn, cal=cal, registration=run.sha(REG))
        if (len(learn), len(cal), len(set(learn+cal+[site]))) != (7, 6, 14):
            raise ValueError('Require disjoint source roles')
        if any(old[site]['identity'][k] != ident[k] for k in ('seed', 'held', 'fit', 'cal')):
            raise ValueError('Require identical comparison roles')
        mark = PRIVATE/f'{m.seed}_{site}.json'
        if not mark.exists():
            stats, witness = [], Counter()
            for s in learn:
                co = run.model.coefficients([original[t] for t in learn if t != s])
                local = []
                for ix, g in m.groups[s]:
                    stats0, status, count = teaching_record(m, ix, g, co, y, valid,
                                              cuts[s]['easy'], cuts[s]['hard'])
                    witness[status] += 1
                    if stats0 is not None:
                        local.append(stats0)
                        witness['positive_deletions'] += count
                if local:
                    stats.append(tuple(np.mean([v[k] for v in local], axis=0) for k in (0, 1)))
            coef = base.model.coefficients(stats) if witness['positive_deletions'] else None
            co = m.co[site]
            donors = []
            for s in cal:
                for ix, g in m.groups[s]:
                    _, i, a, x, group, _ = base.frozen_direct(m, ix, g, co)
                    v, full, w = base.targets(m, ix, i, a, y, valid, cuts[s]['easy'])
                    donors.append((group, v, model.scores(x, coef), full, w))
            result = [model.calibrate(donors, k) for k in range(2)]
            thresholds = [v[0] for v in result]
            at = m.ix[site]
            ch = np.broadcast_to(m.direct[at], (len(ARMS), len(at))).copy()
            with np.load(run.checked(old[site]['cache']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
                np.testing.assert_array_equal(z['choices'][0], ch[0])
                moment = z['coefficients']
            for ix, g in m.groups[site]:
                dc, i, _, x, group, _ = base.frozen_direct(m, ix, g, co)
                np.testing.assert_array_equal(dc, m.direct[ix])
                j = np.searchsorted(at, ix[i])
                scores = model.scores(x, coef)
                reference = base.model.deletion_rank(x, moment)
                for k, s in enumerate(scores):
                    ch[1+k, j[base.model.choose(group, s, thresholds[k])]] = -1
                    probe = base.model.choose(group, s, PROBE)
                    ch[3+k, j[probe]] = -1
                    for q in group:
                        if probe[q].any():
                            ch[5+k, j[q[reference[q].argmax()]]] = -1
            cache = mark.with_suffix('.npz')
            base.previous.history.feedback.write_array(cache, ids=d['global_ids'][at], choices=ch,
                                                       coefficients=coef if coef is not None else np.empty((0, 2)))
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache),
                thresholds=thresholds, witnesses=dict(witness),
                calibration=[v[1] for v in result])))
        fold = run.read(mark)
        if fold['identity'] != ident:
            raise ValueError('Incompatible checkpoint')
        run.checked(fold['cache'])
        folds.append(fold)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def pairs():
    cv = run.parent.CV
    return dict(imitation=('direct', 'imitation', cv), gain_guard=('direct', 'gain_guard', cv),
        probe_imitation=('probe_moment_for_imitation', 'probe_imitation', cv),
        probe_gain_guard=('probe_moment_for_gain_guard', 'probe_gain_guard', cv))


def score(m, records, frozen, y, valid, rows):
    d, p, cuts = m.d, run.parent, run.cuts_by_source()
    ch = np.full((len(ARMS), len(m.floor)), -1, np.int16)
    for f in frozen['folds']:
        at = m.ix[f['identity']['held']]
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            ch[:, at] = z['choices']
    np.testing.assert_array_equal(ch[0], m.direct)
    for rec, (ix, groups) in records.items():
        s = str(d['sites'][ix[0]])
        e, h = cuts[s]['easy'], cuts[s]['hard']
        z = ch[:, ix]
        if ((z >= 0) & (z != z[0])).any():
            raise ValueError('Only deletions are permitted')
        for q in groups:
            count = (z[:, q] >= 0).sum(1)
            if count[3] != count[5] or count[4] != count[6] or (count[0]-count > 1).any():
                raise ValueError('Unmatched deletion counts')
        c, b = m.floor[ix], m.bank[ix]
        paths = {k: run.model.selected_paths(v, b, c) for k, v in zip(ARMS, z)}
        out = p.readout.score_recording({p.CV: c, **paths}, pairs(), y[ix], valid[ix],
                         d['queries'][ix], rec, easy_cut=e, hard_cut=h)
        rows.append(dict(seed=m.seed, source=s, contrasts=out['comparisons'],
                         risk={k: out['metrics'][k]['risk'] for k in ARMS}))


def report(rows, frozen):
    sites = sorted({r['source'] for r in rows})
    if (len(sites), len(rows)) != (14, 102):
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
    for a in ARMS:
        v = [r['risk'][a] for r in rows]
        risk[a] = dict(switches=sum(x['selected'] for x in v), passes=sum(x['supported'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None
                                and x['known_easy_net_degradation'] > .02 for x in v))
    if risk['direct'] != run.read(base.PUBLIC/'analysis.json')['risk']['direct']:
        raise ValueError('Require exact original control')
    f = [v for c in frozen['cells'] for v in c['folds']]
    counts = sum((Counter(v['witnesses']) for v in f), Counter())
    stat = {a: dict(thresholds=dict(Counter(str(v['thresholds'][j]) for v in f)),
                   folds_with_recovery_preserving_risk=sum(v['calibration'][j]['recover_and_preserve_risk']>0 for v in f),
                   probe_deletions=risk['direct']['switches']-risk['probe_'+a]['switches'])
            for j, a in enumerate(('imitation', 'gain_guard'))}
    run.once(PUBLIC/'analysis.json', run.stable(dict(views=102, source_groups=14, folds=42,
        heads_fitted=sum(v['witnesses'].get('positive_deletions', 0)>0 for v in f),
        repeated_fit_witness_counts=dict(counts), arm_details=stat, risk=risk, contrasts=ci,
        original_rules_unchanged=True, teacher_choices_exported=False,
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
        parent, controls, ranks = cost.banks()
        old = {c['seed']: {f['identity']['held']: f for f in c['folds']}
               for c in run.read(base.PUBLIC/'freeze.json')['cells']}
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
                score(m, records, saved[m.seed], y, valid, rows)
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
