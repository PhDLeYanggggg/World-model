"""Compare two past-error inputs within the fixed risk-aware shortlist."""
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
from scripts import run_m3w_shortlist_cost as cost
from scripts import run_m3w_resolved_motion_feedback as feedback
from src.world_model import m3w_shortlist_history as history

run, np, prior, anchor, timing = cost.run, cost.np, cost.prior, cost.anchor, cost.timing
PUBLIC, PRIVATE = prior.PUBLIC/'shortlist_history', prior.PRIVATE/'shortlist_history'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(history.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_shortlist_history.py')
INPUTS = ('backcast', 'resolved')
ARMS = ('direct', 'base', *INPUTS, 'matched_base', *(f'matched_{a}' for a in INPUTS))


def register():
    analysis = cost.PUBLIC/'analysis.json'
    if run.read(cost.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(analysis):
        raise ValueError('Require exact replay of the unaugmented control')
    parents = (analysis, cost.REG, cost.PUBLIC/'freeze.json', Path(cost.__file__),
               Path(cost.model.__file__), feedback.REG, feedback.PUBLIC/'features.json',
               Path(feedback.model.__file__), Path(run.parent.readout.__file__))
    for item in run.read(cost.REG)['parents']:
        run.checked(item)
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], arms=ARMS, risk_budget=.02,
        independent_outcomes_opened=False)))


def read_feedback(d, seed):
    frozen = run.read(feedback.PUBLIC/'features.json')
    if not frozen['no_target_arrays_opened']:
        raise ValueError('Require geometry-only feedback')
    entry, = [v for v in frozen['cells'] if v['seed'] == seed]
    if entry['registration'] != run.sha(feedback.REG):
        raise ValueError('Feedback registration mismatch')
    with np.load(run.checked(entry['cache']), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], d['global_ids'])
        return z['feedback']


def load_old(fold, d, ix):
    with np.load(run.checked(fold['cache']), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], d['global_ids'][ix])
        return z['choices'][[0, 2, 5]], z['coefficients']


def make_options(direct, examples, coef):
    return cost.options(direct, examples, coef)[1]


def fit(d, refs, records, y, valid, beat):
    cuts = run.cuts_by_source()
    parent, controls, old_rank = cost.banks()
    previous = {(c['seed'], f['identity']['held']): f
                for c in run.read(cost.PUBLIC/'freeze.json')['cells'] for f in c['folds']}
    source_ix = {str(s): np.flatnonzero(d['sites'] == s) for s in np.unique(d['sites'])}
    groups = {s: [] for s in source_ix}
    for ix, g in records.values():
        s = str(d['sites'][ix[0]])
        groups[s].append((ix, g, np.searchsorted(source_ix[s], ix)))
    cells = []
    for cell in parent:
        seed = cell['seed']
        c, b = run.candidates(d, refs, seed)
        resolved = read_feedback(d, seed)
        direct, coefs = timing.parents(d, cell, controls)
        raw, scale, atoms, original, ranking = {}, {}, {}, {}, {}
        for s, ix in source_ix.items():
            x, z = run.model.descriptors(d['history'][ix], b[ix], c[ix], d['crowd'][ix])
            raw[s], scale[s] = x, z
            e = cuts[s]['easy']
            label, atoms[s], _ = anchor.model.labels(c[ix], b[ix], y[ix], valid[ix], e)
            legal = np.linalg.norm(b[ix] - c[ix, None], axis=-1).max(2) > 0.
            ranking[s] = anchor.model.fitting(x, label, legal, d['crowd'][ix])
            original[s] = run.model.fitting_moments(
                x, b[ix], c[ix], y[ix], valid[ix], z, 1./d['crowd'][ix], e)

        def inputs(s, ix, g, j, co, rank):
            dc, examples = cost.proposals(raw[s][j], scale[s][j], b[ix], c[ix], g, co, rank)
            return dc, examples, [history.augment(examples, a, d['history'][ix], scale[s][j],
                b[ix], c[ix], resolved[ix]) for a in INPUTS]

        folds = []
        for control in controls[seed]:
            identity = control['identity']
            held, learn, cal = identity['held'], identity['fit'], identity['cal_a'] + identity['cal_b']
            if (len(learn), len(cal), len(set(learn + cal + [held]))) != (7, 6, 14):
                raise ValueError('Require separated source roles')
            ident = dict(seed=seed, held=held, fit=learn, cal=cal, registration=run.sha(REG))
            base = previous[seed, held]
            if any(base['identity'][k] != ident[k] for k in ('seed', 'held', 'fit', 'cal')):
                raise ValueError('Require identical roles to unaugmented comparison')
            mark = PRIVATE/f'{seed}_{held}.json'
            if not mark.exists():
                stats = [[] for _ in INPUTS]
                for s in learn:
                    donors = [t for t in learn if t != s]
                    co = run.model.coefficients([original[t] for t in donors])
                    rank = anchor.model.fit([ranking[t] for t in donors])
                    own = [[] for _ in INPUTS]
                    for ix, g, j in groups[s]:
                        dc, examples, augmented = inputs(s, ix, g, j, co, rank)
                        target = cost.model.targets(atoms[s][j], scale[s][j], examples)
                        for store, a in zip(own, augmented):
                            store.append(cost.model.moments(a, target, dc))
                    for store, values in zip(stats, own):
                        store.append(values)
                weights = [cost.model.fit(v) for v in stats]
                with np.load(run.checked(old_rank[seed, held]), allow_pickle=False) as z:
                    rank = z['coefficients']
                co = coefs[held]
                rows = [[] for _ in INPUTS]
                for s in cal:
                    for ix, g, j in groups[s]:
                        dc, _, augmented = inputs(s, ix, g, j, co, rank)
                        z = atoms[s][j]
                        k = np.flatnonzero(dc >= 0)
                        for store, a, w in zip(rows, augmented, weights):
                            store.append(dict(mass=z[k, dc[k]].sum(0), options=make_options(dc, a, w),
                                atoms=z, easy=float(z[:, 0, 1].sum()), crowd=d['crowd'][ix]))
                thresholds = [anchor.model.calibrate(v)[0] for v in rows]
                old, _ = load_old(base, d, source_ix[held])
                np.testing.assert_array_equal(old[0], direct[source_ix[held]])
                ch = old[[0, 1, 0, 0, 2, 0, 0]].copy()
                for ix, g, j in groups[held]:
                    dc, _, augmented = inputs(held, ix, g, j, co, rank)
                    np.testing.assert_array_equal(dc, direct[ix])
                    for n, (a, w, cutoff) in enumerate(zip(augmented, weights, thresholds)):
                        opts = make_options(dc, a, w)
                        for dest, limit in ((2+n, cutoff), (5+n, -np.inf)):
                            result = anchor.model.first(opts, limit)
                            if result is not None:
                                _, row, family, _ = result
                                ch[dest, j[row]] = family
                cache = mark.with_suffix('.npz')
                feedback.write_array(cache, ids=d['global_ids'][source_ix[held]], choices=ch,
                                     **dict(zip(INPUTS, weights)))
                run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache),
                                              thresholds=thresholds)))
            fold = run.read(mark)
            if fold['identity'] != ident:
                raise ValueError('Incompatible fold checkpoint')
            run.checked(fold['cache'])
            folds.append(fold)
            beat(state='fold_frozen', seed=seed, complete=len(folds))
        cells.append(dict(seed=seed, folds=folds))
    run.once(PUBLIC/'freeze.json', run.stable(dict(cells=cells, held_excluded=True)))


def score(d, refs, records, y, valid, beat):
    f = PUBLIC/'freeze.json'
    frozen = run.read(f)
    if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(f):
        raise ValueError('Require complete decision freeze')
    p, cuts = run.parent, run.cuts_by_source()
    parent, controls, old_rank = cost.banks()
    parent = {c['seed']: c for c in parent}
    previous = {(c['seed'], f['identity']['held']): f
                for c in run.read(cost.PUBLIC/'freeze.json')['cells'] for f in c['folds']}
    pairs = {f'{a}_vs_base': ('base', a, p.CV) for a in INPUTS}
    pairs.update({f'matched_{a}_vs_base': ('matched_base', f'matched_{a}', p.CV) for a in INPUTS})
    rows, recall = [], []
    for cell in frozen['cells']:
        seed = cell['seed']
        c, b = run.candidates(d, refs, seed)
        resolved = read_feedback(d, seed)
        direct, co = timing.parents(d, parent[seed], controls)
        ch = np.full((len(ARMS), len(c)), -1, np.int16)
        weights = {}
        for fold in cell['folds']:
            s = fold['identity']['held']
            at = np.flatnonzero(d['sites'] == s)
            old, w = load_old(previous[seed, s], d, at)
            with np.load(run.checked(fold['cache']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
                ch[:, at] = z['choices']
                weights[s] = dict(base=w, **{a: z[a] for a in INPUTS})
            np.testing.assert_array_equal(ch[[0, 1, 4]][:, at], old)
        np.testing.assert_array_equal(ch[0], direct)
        for rec, (ix, g) in records.items():
            s = str(d['sites'][ix[0]])
            e, h = cuts[s]['easy'], cuts[s]['hard']
            z = ch[:, ix]
            dc = z[0] >= 0
            on = z >= 0
            if (z[:, dc] != z[0, dc]).any() or (on & ~dc).sum(1).max() > 1:
                raise ValueError('Opening may not modify direct decisions')
            for q in g:
                counts = on[4:, q].sum(1)
                if not (counts == counts[0]).all():
                    raise ValueError('Require identical matched query counts')
            paths = {k: run.model.selected_paths(v, b[ix], c[ix]) for k, v in zip(ARMS, z)}
            out = p.readout.score_recording({p.CV: c[ix], **paths}, pairs, y[ix], valid[ix],
                                           d['queries'][ix], rec, easy_cut=e, hard_cut=h)
            rows.append(dict(seed=seed, source=s, contrasts=out['comparisons'],
                             risk={k: out['metrics'][k]['risk'] for k in ARMS}))
            if dc.any():
                continue
            x, scale = run.model.descriptors(d['history'][ix], b[ix], c[ix], d['crowd'][ix])
            with np.load(run.checked(old_rank[seed, s]), allow_pickle=False) as cache:
                rank = cache['coefficients']
            _, examples = cost.proposals(x, scale, b[ix], c[ix], g, co[s], rank)
            examples = examples[:2]
            _, atoms, _ = anchor.model.labels(c[ix], b[ix], y[ix], valid[ix], e)
            targets = cost.model.targets(atoms, scale, examples)
            augmented = {a: history.augment(examples, a, d['history'][ix], scale, b[ix],
                                             c[ix], resolved[ix]) for a in INPUTS}
            support = resolved[ix]
            cases = {}
            for label, k in (('first', 1), ('first_two', 2)):
                hits = dict(shortlist=False, base=False, backcast=False, resolved=False,
                            resolved_own_available=False, resolved_scene_available=False,
                            useful_with_resolved_own=False, useful_with_resolved_scene=False)
                for t, target in enumerate(targets[:k]):
                    _, i, j, _ = examples[t]
                    u = target[:, 4] > 0.
                    hits['shortlist'] |= bool(u.any())
                    for kind, col in (('own', 0), ('scene', feedback.model.OWN)):
                        has = support[i, j, col] > 0.
                        hits[f'resolved_{kind}_available'] |= bool(has.any())
                        hits[f'useful_with_resolved_{kind}'] |= bool((has & u).any())
                    for a in ('base', *INPUTS):
                        sample = examples[t] if a == 'base' else augmented[a][t]
                        _, row, family = cost.model.winners(sample, weights[s][a])[1]
                        hits[a] |= bool(u[(i == row) & (j == family)].any())
                cases[label] = hits
            recall.append(cases)
        beat(state='seed_scored', seed=seed)
    sites = sorted({r['source'] for r in rows})
    if len(rows) != 102 or len(sites) != 14 or len(recall) != 18:
        raise ValueError('Incomplete registered development roster')
    hits = {k: {n: sum(r[k][n] for r in recall) for n in recall[0][k]} for k in recall[0]}
    old = run.read(cost.PUBLIC/'analysis.json')
    for k in hits:
        if (hits[k]['shortlist'], hits[k]['base']) != (old['recall'][k]['shortlist'], old['recall'][k]['risk']):
            raise ValueError('Unaugmented shortlist or winner recall differs')
    groups = [[r for r in rows if r['source'] == s and r['seed'] == k]
              for s in sites for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(14, size=(3000, 14))
    ci = {}
    for name in pairs:
        v = [p.readout.merge_comparisons([r['contrasts'][name] for r in g]) for g in groups]
        x = np.asarray([r['completion_lower_pp'] for r in v], float).reshape(14, 3).mean(1)
        if not np.isfinite(x).all():
            raise ValueError('All source contrasts required')
        ci[name] = dict(point=float(x.mean()),
                        interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())
    risk = {}
    for a in ARMS:
        v = [r['risk'][a] for r in rows]
        risk[a] = dict(switches=sum(x['selected'] for x in v), passes=sum(x['supported'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None
                                and x['known_easy_net_degradation'] > .02 for x in v))
    for a, before in (('direct', 'direct'), ('base', 'risk'), ('matched_base', 'diagnostic_risk')):
        if risk[a] != {k: old['risk'][before][k] for k in risk[a]}:
            raise ValueError('Unaugmented risk counts must replay exactly')
    run.once(PUBLIC/'analysis.json', run.stable(dict(views=102, sources=14, zero_views=18,
        recall=hits, contrasts=ci, risk=risk, fit_heads=84, original_rules_unchanged=True,
        independent_confirmation=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = parser.parse_args().phase
    register()
    if phase == 'register':
        return
    for path in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT) != path.read_bytes():
            raise ValueError('Commit the design before execution')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic() - start > 43200:
                raise OSError('Resume from saved source checkpoints')
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            print(msg, flush=True)
        beat(state='start')
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = prior.blocks(d, lab['frames'])
        (fit if phase == 'fit' else score)(d, refs, records,
                                          lab['target_eval'] - d['origin'][:, None], lab['valid'], beat)
        path = PUBLIC/('freeze.json' if phase == 'fit' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(),
            seconds=time.monotonic()-start, artifact=run.artifact(path)))


if __name__ == '__main__':
    main()
