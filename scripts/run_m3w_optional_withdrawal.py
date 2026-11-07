"""Learn an insertion that can be followed by an optional causal withdrawal."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_atomic_swap as previous
from src.world_model import m3w_optional_withdrawal as model

h, run, np, anchor = previous.h, previous.run, previous.np, previous.anchor
PUBLIC, PRIVATE = previous.PUBLIC.parent/'optional_withdrawal', previous.PRIVATE.parent/'optional_withdrawal'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_optional_withdrawal.py')
ARMS = ('direct', 'insert_only', 'matched_insert', 'optional')
GRID = [(i, j) for i in model.INSERT_CUTS for j in model.DROP_CUTS]


def register():
    a = previous.PUBLIC/'analysis.json'
    if run.read(previous.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(a):
        raise ValueError('Require exact preceding replay')
    for artifact in (*run.read(previous.REG)['parents'], *run.read(previous.REG)['files']):
        run.checked(artifact)
    paths = (a, previous.REG, previous.PUBLIC/'ordered_capacity.json', Path(previous.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in paths], budget=.02, max_add=1, max_drop=1,
        steps=model.STEPS, batch=model.BATCH, grid=GRID, independent_outcomes_opened=False)))


def design(m, ix, groups, co, rank, motion):
    dc, legal, context, _, _, u, e = h.previous.features(m, ix, groups, co, rank, motion)
    ordinal = np.empty(len(ix), dtype=int)
    for j, q in enumerate(groups):
        ordinal[q] = j
        if (dc[q] >= 0).sum() >= min(2, len(q)-1):
            legal[q] = False
    i, k = np.where(legal)
    j = np.flatnonzero(dc >= 0)
    order = np.argsort(ordinal[i], kind='stable')
    i, k = i[order], k[order]
    j = j[np.argsort(ordinal[j], kind='stable')]
    raw, s = m.x[ix], m.scale[ix]

    def side(row, family):
        return np.column_stack((context[row], raw[row, family],
            np.arcsinh(u[row, family]/s[row]), np.arcsinh(e[row, family]/s[row])))

    return dc, (i, k, ordinal[i], side(i, k)), (j, ordinal[j], side(j, dc[j]))


def routes(insert, withdraw, predict_add, predict_drop):
    i, k, q, x = insert
    j, v, z = withdraw
    first = model.first
    scores = predict_add(x)
    cache, out = {}, {}
    for cut in model.INSERT_CUTS:
        a = first(np.arange(len(i)), q, scores, cut)
        if a is None:
            continue
        if a not in cache:
            at = v >= q[a]
            prob = predict_drop(model.pair_features(x[a], z[at], v[at]-q[a]))
            cache[a] = {r: first(j[at], v[at], prob, r) for r in model.DROP_CUTS}
        for d, removed in cache[a].items():
            out[cut, d] = int(i[a]), int(k[a]), -1 if removed is None else removed
    return out


def observe(move, atoms, known, dc, original, full, crowd):
    i, k, j = move
    a = atoms[i, k]
    mass = original + a
    gain = a[-1]/crowd[i]
    if j >= 0:
        d = atoms[j, dc[j]]
        mass = mass-d
        gain += (-d[-1] if known[j] else d[-1])/crowd[j]
    return anchor.model.risk(mass, full), float(gain)


def calibrate(records):
    best = [None, None]
    for ix, key in enumerate(GRID):
        new = lost = bad = edits = 0
        gain = 0.
        for table, old, *args in records:
            move = table.get(key)
            if move is None:
                continue
            after, utility = observe(move, *args)
            new += int(after['supported'] and not old['supported'])
            lost += int(old['supported'] and not after['supported'])
            bad += int(after['easy_net_over'] and not old['easy_net_over'])
            edits += 1 + int(move[-1] >= 0)
            gain += utility
        if new and not lost and not bad and gain >= 0.:
            value = (new, gain, -edits, -ix)
            for j in range(2):
                if (j == 1 or key[1] is None) and (best[j] is None or value > best[j]):
                    best[j] = value
    return [GRID[-v[-1]] if v is not None else None for v in best]


def fit(m, controls, motion, y, valid, beat):
    d, cuts, known = m.d, run.cuts_by_source(), valid.any(1)
    moments, stats = {}, {}
    atoms = np.empty((*m.bank.shape[:2], 6))
    for s, ix in m.ix.items():
        x, b, c, scale = m.x[ix], m.bank[ix], m.floor[ix], m.scale[ix]
        events, atoms[ix], _ = anchor.model.labels(c, b, y[ix], valid[ix], cuts[s]['easy'])
        legal = np.linalg.norm(b-c[:, None], axis=-1).max(2) > 0.
        moments[s] = run.model.fitting_moments(x, b, c, y[ix], valid[ix], scale,
                                               1./d['crowd'][ix], cuts[s]['easy'])
        stats[s] = anchor.model.fitting(x, events, legal, d['crowd'][ix])
    folds = []
    for item in controls:
        r = item['identity']
        site, learn, cal = r['held'], r['fit'], r['cal_a'] + r['cal_b']
        ident = dict(seed=m.seed, held=site, fit=learn, cal=cal, registration=run.sha(REG))
        if (len(learn), len(cal), len(set(learn+cal+[site]))) != (7, 6, 14):
            raise ValueError('Require disjoint source roles')
        mark = PRIVATE/f'{m.seed}_{site}.json'
        if not mark.exists():
            seed = int.from_bytes(hashlib.sha256(f'optional:{m.seed}:{site}'.encode()).digest()[:4], 'little')
            rng = np.random.default_rng(seed)
            add_blocks, drop_blocks = [], []
            for s in learn:
                co = run.model.coefficients([moments[t] for t in learn if s != t])
                rank = anchor.model.fit([stats[t] for t in learn if s != t])
                adds, drops = [], []
                for ix, g in m.groups[s]:
                    dc, (i, k, q, x), (j, v, z) = design(m, ix, g, co, rank, motion)
                    if not len(i):
                        continue
                    a, crowd = atoms[ix], d['crowd'][ix]
                    original, full = previous.mass(a, dc)
                    dt = a[j, dc[j]]
                    dg = np.where(known[ix[j]], -dt[:, -1], dt[:, -1])
                    reach, pairs, label = model.teach(a[i, k], dt, dg, q, v,
                        crowd[i], crowd[j], original, full, rng)
                    w = 1./(crowd[i]*np.bincount(i, minlength=len(ix))[i])
                    adds.append((x, reach, w/w.sum()))
                    if len(pairs):
                        u, b = pairs.T
                        ww = w[u]/np.bincount(u, minlength=len(i))[u]
                        xpair = np.column_stack((x[u], z[b], np.log1p(v[b]-q[u])))
                        drops.append((xpair, label, ww/ww.sum()))
                add_blocks.extend((x, t, w/len(adds)) for x, t, w in adds)
                drop_blocks.extend((x, t, w/len(drops)) for x, t, w in drops)
            predictors, scalers, checkpoints, counts = [], [], [], []
            for name, blocks in (('insert', add_blocks), ('withdraw', drop_blocks)):
                x, target, weight = [np.concatenate(v) for v in zip(*blocks)]
                center = np.average(x, weights=weight, axis=0)
                spread = np.sqrt(np.average((x-center)**2, weights=weight, axis=0)).clip(.01)
                ck = mark.with_name(f'{mark.stem}_{name}.pt')
                net = model.train(np.clip((x-center)/spread, -8., 8.).astype(np.float32),
                    target, weight, seed, ck,
                    lambda **v: beat(state='training', seed=m.seed, source=site, head=name, **v))
                predictors.append(lambda x, net=net, c=center, s=spread:
                    model.predict(net, np.clip((x-c)/s, -8., 8.).astype(np.float32)))
                scalers.extend((center, spread))
                checkpoints.append(run.artifact(ck))
                counts.append([len(target), int(target.sum())])
            co, rank = m.co[site], m.rank(site)

            def infer(ix, g):
                dc, add, drop = design(m, ix, g, co, rank, motion)
                return dc, routes(add, drop, *predictors)

            donors = []
            for s in cal:
                for ix, g in m.groups[s]:
                    dc, table = infer(ix, g)
                    a = atoms[ix]
                    original, full = previous.mass(a, dc)
                    donors.append((table, anchor.model.risk(original, full), a, known[ix],
                                   dc, original, full, d['crowd'][ix]))
            insertion, optional = calibrate(donors)
            matched = None if optional is None else (optional[0], None)
            at = m.ix[site]
            choice = np.broadcast_to(m.direct[at], (len(ARMS), len(at))).copy()
            for ix, g in m.groups[site]:
                dc, table = infer(ix, g)
                np.testing.assert_array_equal(dc, m.direct[ix])
                for k, key in enumerate((insertion, matched, optional), 1):
                    move = table.get(key)
                    if move is not None:
                        i, f, j = move
                        choice[k, np.searchsorted(at, ix[i])] = f
                        if j >= 0:
                            choice[k, np.searchsorted(at, ix[j])] = -1
            cache = mark.with_suffix('.npz')
            h.prune.previous.history.feedback.write_array(cache, ids=d['global_ids'][at], choices=choice,
                insert_center=scalers[0], insert_spread=scalers[1],
                withdraw_center=scalers[2], withdraw_spread=scalers[3])
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache),
                checkpoints=checkpoints, thresholds=[insertion, matched, optional], fit_counts=counts)))
        f = run.read(mark)
        if f['identity'] != ident:
            raise ValueError('Incompatible fold checkpoint')
        for artifact in (f['cache'], *f['checkpoints']):
            run.checked(artifact)
        folds.append(f)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def score(m, records, frozen, y, valid):
    d, p, cuts = m.d, run.parent, run.cuts_by_source()
    choice = np.full((len(ARMS), len(m.floor)), -1, np.int16)
    for f in frozen['folds']:
        at = m.ix[f['identity']['held']]
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            choice[:, at] = z['choices']
    np.testing.assert_array_equal(choice[0], m.direct)
    rows = []
    for rec, (ix, groups) in records.items():
        s = str(d['sites'][ix[0]])
        e, hard = cuts[s]['easy'], cuts[s]['hard']
        z, c, b = choice[:, ix], m.floor[ix], m.bank[ix]
        add = (z >= 0) & (z[0] < 0)
        drop = (z < 0) & (z[0] >= 0)
        if (not np.array_equal(add[2], add[3]) or not np.array_equal(z[2, add[2]], z[3, add[3]])
                or (drop[:3]).any() or (drop.sum(1) > add.sum(1)).any()
                or (add.sum(1) > 1).any() or ((z >= 0) & (z[0] >= 0) & (z != z[0])).any()):
            raise ValueError('Invalid matched insertion or edit types')
        seen = False
        for q in groups:
            seen |= bool(add[3, q].any())
            if ((not seen and drop[3, q].any()) or ((z[:, q] >= 0).sum(1) > min(2, len(q)-1)).any()
                    or ((z[0, q] >= 0).sum() >= min(2, len(q)-1) and add[:, q].any())):
                raise ValueError('Invalid chronology or insertion cap')
        paths = {k: run.model.selected_paths(v, b, c) for k, v in zip(ARMS, z)}
        contrasts = {a: ('direct', a, p.CV) for a in ARMS[1:]}
        contrasts['optional_vs_matched'] = ('matched_insert', 'optional', p.CV)
        out = p.readout.score_recording({p.CV: c, **paths}, contrasts,
            y[ix], valid[ix], d['queries'][ix], rec, easy_cut=e, hard_cut=hard)
        rows.append(dict(source=s, seed=m.seed, comparisons=out['comparisons'],
            risk={k: out['metrics'][k]['risk'] for k in ARMS},
            insertions=add.sum(1).tolist(), withdrawals=drop.sum(1).tolist()))
    return rows


def report(rows, freeze):
    sites = sorted({r['source'] for r in rows})
    missing = [r for r in rows if r['risk']['direct']['completion_easy_risk_upper'] is None]
    if (len(sites), len(rows), len(missing)) != (14, 102, 35):
        raise ValueError('Require original cohort')
    groups = [[r for r in rows if r['source'] == s and r['seed'] == k]
              for s in sites for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(14, size=(3000, 14))

    def summary(x):
        x = np.asarray(x, float).reshape(14, 3).mean(1)
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    risk = {}
    for j, a in enumerate(ARMS):
        v = [r['risk'][a] for r in rows]
        risk[a] = dict(passes=sum(x['supported'] for x in v), switches=sum(x['selected'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None
                                and x['known_easy_net_degradation'] > .02 for x in v),
            insertions=sum(r['insertions'][j] for r in rows), withdrawals=sum(r['withdrawals'][j] for r in rows),
            new_passes=sum(r['risk'][a]['supported'] and not r['risk']['direct']['supported'] for r in rows),
            lost_passes=sum(not r['risk'][a]['supported'] and r['risk']['direct']['supported'] for r in rows))
    if any(risk['direct'][k] != v for k, v in run.read(previous.PUBLIC/'analysis.json')['risk']['direct'].items()):
        raise ValueError('Require unchanged direct score')
    f = [v for c in freeze['cells'] for v in c['folds']]
    utility = {a: summary([run.parent.readout.merge_comparisons([r['comparisons'][a] for r in g])
                          ['completion_lower_pp'] for g in groups])
               for a in (*ARMS[1:], 'optional_vs_matched')}
    run.once(PUBLIC/'analysis.json', run.stable(dict(views=102, missing_views=35, source_groups=14,
        risk=risk, paired_lower_pp=utility, torch_models=2*len(f),
        fit_counts_insert_withdraw=np.sum([v['fit_counts'] for v in f], axis=0).tolist(),
        calibrated_folds={a: sum(v['thresholds'][j] is not None for v in f) for j, a in enumerate(ARMS[1:])},
        withdrawal_enabled_folds=sum(v['thresholds'][2] is not None and v['thresholds'][2][1] is not None for v in f),
        original_rules_unchanged=True, independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)))


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
    model.torch.set_num_threads(4)
    model.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resume from the last saved update')
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            if v.get('state') != 'training' or v.get('step') == model.STEPS:
                print(msg, flush=True)

        beat(state='start')
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = h.previous.prior.blocks(d, lab['frames'])
        y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
        motion = h.previous.model.motion(d['history'], d['groups']) if phase == 'fit' else None
        parent, controls, ranks = h.cost.banks()
        out = PUBLIC/'freeze.json'
        if phase != 'fit':
            if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(out):
                raise ValueError('Require frozen choices')
            frozen = run.read(out)
            saved = {c['seed']: c for c in frozen['cells']}
        cells, rows = [], []
        for cell in parent:
            m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
            if phase == 'fit':
                cells.append(fit(m, controls[m.seed], motion, y, valid, beat))
            else:
                rows.extend(score(m, records, saved[m.seed], y, valid))
            beat(state='seed_complete', seed=m.seed)
            del m
        if phase == 'fit':
            run.once(out, run.stable(dict(cells=cells, held_excluded=True)))
        else:
            report(rows, frozen)
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            artifact=run.artifact(out if phase == 'fit' else PUBLIC/'analysis.json'),
            python_architecture=platform.machine(), torch_threads=model.torch.get_num_threads()))


if __name__ == '__main__':
    main()
