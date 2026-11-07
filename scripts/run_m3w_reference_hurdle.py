"""Fit, freeze and replay a source-separated support/value insertion model."""
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
from scripts import run_m3w_easy_reference_rank as previous
from src.world_model import m3w_reference_hurdle as model

run, np, cost, anchor, prune = previous.run, previous.np, previous.cost, previous.anchor, previous.prune
PUBLIC, PRIVATE = previous.prior.PUBLIC/'reference_hurdle', previous.prior.PRIVATE/'reference_hurdle'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_reference_hurdle.py')
ARMS = ('direct', *model.ARMS)


def register():
    a = previous.previous.PUBLIC/'analysis.json'
    if run.read(previous.previous.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(a):
        raise ValueError('Require exact support-joint replay')
    for parent in run.read(previous.REG)['parents']:
        run.checked(parent)
    paths = (a, previous.REG, Path(previous.__file__), cost.REG,
             previous.previous.REG, Path(previous.previous.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in paths], steps=model.STEPS, batch=model.BATCH,
        arms=ARMS, cuts=model.CUTS, budget=.02, independent_outcomes_opened=False)))


def features(m, ix, g, co, rank, motion):
    dc, legal, context, _, _, utility, easy = previous.features(m, ix, g, co, rank, motion)
    for q in g:
        if (dc[q] >= 0).sum() >= min(2, len(q)-1):
            legal[q] = False
    row, family = np.where(legal)
    scale = m.scale[ix[row]]
    x = np.column_stack((context[row], m.x[ix[row], family],
                         np.arcsinh(utility[row, family]/scale),
                         np.arcsinh(easy[row, family]/scale)))
    return dc, legal, row, family, x


def calibrate(records):
    best = None
    for cut in model.CUTS:
        new = lost = violations = edits = 0
        utility = 0.
        for options, atoms, mass, full, crowd, old in records:
            choice = model.first(options, cut)
            if choice is None:
                continue
            i, k = choice
            after = anchor.model.risk(mass+atoms[i, k], full)
            new += int(after['supported'] and not old['supported'])
            lost += int(old['supported'] and not after['supported'])
            violations += int(after['easy_net_over'] and not old['easy_net_over'])
            utility += float(atoms[i, k, -1]/crowd[i])
            edits += 1
        if new and not lost and not violations and utility >= 0.:
            candidate = (new, utility, -edits, cut)
            best = candidate if best is None else max(best, candidate)
    return None if best is None else best[-1]


def fit(m, controls, motion, y, valid, beat):
    d, cuts = m.d, run.cuts_by_source()
    orig, stats = {}, {}
    atoms = np.empty((*m.bank.shape[:2], 6))
    for s, ix in m.ix.items():
        x, b, c, z = m.x[ix], m.bank[ix], m.floor[ix], m.scale[ix]
        events, atoms[ix], _ = anchor.model.labels(c, b, y[ix], valid[ix], cuts[s]['easy'])
        legal = np.linalg.norm(b-c[:, None], axis=-1).max(2) > 0.
        orig[s] = run.model.fitting_moments(x, b, c, y[ix], valid[ix], z, 1./d['crowd'][ix], cuts[s]['easy'])
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
            blocks = []
            for s in learn:
                co = run.model.coefficients([orig[t] for t in learn if s != t])
                rank = anchor.model.fit([stats[t] for t in learn if s != t])
                local = []
                for ix, g in m.groups[s]:
                    _, legal, i, k, x = features(m, ix, g, co, rank, motion)
                    if len(i):
                        w = 1./(d['crowd'][ix[i]]*legal.sum(1)[i])
                        local.append((x, model.targets(atoms[ix], m.scale[ix])[i, k], w/w.sum()))
                blocks.extend((x, t, w/len(local)) for x, t, w in local)
            x, t, w = [np.concatenate(v) for v in zip(*blocks)]
            if not 0 < t[:, 0].sum() < len(t):
                raise ValueError('Require observed and unobserved support targets')
            center = np.average(x, weights=w, axis=0)
            scale = np.sqrt(np.average((x-center)**2, weights=w, axis=0)).clip(.01)

            def normal(x):
                return np.clip((x-center)/scale, -8., 8.).astype(np.float32)

            seed = int.from_bytes(hashlib.sha256(f'hurdle:{m.seed}:{site}'.encode()).digest()[:4], 'little')
            ck = mark.with_suffix('.pt')
            net = model.train(normal(x), t, w, seed, ck,
                lambda **v: beat(state='training', seed=m.seed, source=site, **v))
            co, rank = m.co[site], m.rank(site)

            def options(ix, g):
                dc, legal, row, family, x = features(m, ix, g, co, rank, motion)
                scores = np.zeros((*legal.shape, len(model.ARMS)), np.float32)
                if len(row):
                    scores[row, family] = model.predict(net, normal(x))
                return dc, [list(model.options(dc, g, legal, scores, arm)) for arm in model.ARMS]

            donors = [[] for _ in model.ARMS]
            for s in cal:
                for ix, g in m.groups[s]:
                    dc, candidates = options(ix, g)
                    a, i = atoms[ix], np.flatnonzero(dc >= 0)
                    mass, full = a[i, dc[i]].sum(0), float(a[:, 0, 1].sum())
                    old = anchor.model.risk(mass, full)
                    for dest, opts in zip(donors, candidates):
                        dest.append((opts, a, mass, full, d['crowd'][ix], old))
            thresholds = [calibrate(donor) for donor in donors]
            at = m.ix[site]
            ch = np.broadcast_to(m.direct[at], (len(ARMS), len(at))).copy()
            top = np.full((len(model.ARMS), len(at)), -1, dtype=np.int16)
            for ix, g in m.groups[site]:
                dc, candidates = options(ix, g)
                np.testing.assert_array_equal(dc, m.direct[ix])
                for j, (opts, cut) in enumerate(zip(candidates, thresholds)):
                    if opts:
                        i, k, *_ = opts[0]
                        top[j, np.searchsorted(at, ix[i])] = k
                    found = model.first(opts, cut)
                    if found is not None:
                        i, k = found
                        ch[j+1, np.searchsorted(at, ix[i])] = k
            cache = mark.with_suffix('.npz')
            prune.previous.history.feedback.write_array(cache, ids=d['global_ids'][at], choices=ch,
                top=top, center=center, scale=scale)
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache),
                checkpoint=run.artifact(ck), thresholds=thresholds, fitting_actions=len(t))))
        f = run.read(mark)
        if f['identity'] != ident:
            raise ValueError('Incompatible fold checkpoint')
        run.checked(f['cache'])
        run.checked(f['checkpoint'])
        folds.append(f)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def score(m, records, frozen, y, valid):
    d, p, cuts = m.d, run.parent, run.cuts_by_source()
    ch = np.full((len(ARMS), len(m.floor)), -1, np.int16)
    top = np.full((len(model.ARMS), len(m.floor)), -1, np.int16)
    for f in frozen['folds']:
        at = m.ix[f['identity']['held']]
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            ch[:, at], top[:, at] = z['choices'], z['top']
    np.testing.assert_array_equal(ch[0], m.direct)
    rows = []
    for rec, (ix, g) in records.items():
        s = str(d['sites'][ix[0]])
        e, h = cuts[s]['easy'], cuts[s]['hard']
        z, c, b = ch[:, ix], m.floor[ix], m.bank[ix]
        changed = z != z[0]
        if (changed[:, z[0] >= 0].any() or (changed.sum(1) > 1).any()
                or any(((z[:, q] >= 0).sum(1) > min(2, len(q)-1)).any() for q in g)):
            raise ValueError('Expected one addition and the unchanged per-query limit')
        paths = {k: run.model.selected_paths(v, b, c) for k, v in zip(ARMS, z)}
        out = p.readout.score_recording({p.CV: c, **paths},
            {a: ('direct', a, p.CV) for a in model.ARMS},
            y[ix], valid[ix], d['queries'][ix], rec, easy_cut=e, hard_cut=h)
        _, a, _ = anchor.model.labels(c, b, y[ix], valid[ix], e)
        hits, useful, added_support = [], [], []
        for opts, chosen in zip(top[:, ix], z[1:]):
            i = np.flatnonzero(opts >= 0)
            if len(i) > 1:
                raise ValueError('Expected one first-query probe')
            v = a[i, opts[i]]
            hits.append(bool((v[:, 1] > 0).any()))
            useful.append(bool(((v[:, 1] > 0) & (v[:, -1] >= 0)).any()))
            i = np.flatnonzero((chosen >= 0) & (z[0] < 0))
            added_support.append(bool((a[i, chosen[i], 1] > 0).any()))
        rows.append(dict(source=s, seed=m.seed, comparison=out['comparisons'],
            risk={k: out['metrics'][k]['risk'] for k in ARMS}, insertions=changed.sum(1).tolist(),
            support=hits, useful=useful, added_support=added_support))
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
    for i, k in enumerate(ARMS):
        v = [r['risk'][k] for r in rows]
        risk[k] = dict(passes=sum(x['supported'] for x in v), switches=sum(x['selected'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None
                                and x['known_easy_net_degradation'] > .02 for x in v),
            insertions=sum(r['insertions'][i] for r in rows),
            new_passes=sum(r['risk'][k]['supported'] and not r['risk']['direct']['supported'] for r in rows),
            lost_passes=sum(not r['risk'][k]['supported'] and r['risk']['direct']['supported'] for r in rows))
    base = run.read(previous.previous.PUBLIC/'analysis.json')['risk']['direct']
    if any(v != base[k] for k, v in risk['direct'].items()):
        raise ValueError('Require unchanged direct score')
    f = [x for c in freeze['cells'] for x in c['folds']]
    data = dict(views=len(rows), missing_views=len(missing), source_groups=len(sites),
        torch_folds=len(f), fitting_actions=sum(x['fitting_actions'] for x in f),
        risk=risk, calibrated_folds={}, probe={}, paired_lower_pp={},
        independent_confirmation=False, original_rules_unchanged=True,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    for j, a in enumerate(model.ARMS):
        data['calibrated_folds'][a] = sum(x['thresholds'][j] is not None for x in f)
        data['paired_lower_pp'][a] = summary([run.parent.readout.merge_comparisons(
            [r['comparison'][a] for r in group])['completion_lower_pp'] for group in groups])
        data['probe'][a] = dict(first_support=sum(r['support'][j] for r in rows),
            first_nonnegative_supported_gain=sum(r['useful'][j] for r in rows),
            missing_first_support=sum(r['support'][j] for r in missing),
            missing_first_nonnegative_supported_gain=sum(r['useful'][j] for r in missing),
            added_support=sum(r['added_support'][j] for r in rows))
    data['hurdle_minus_direct_first_nonnegative_supported_gain'] = summary([
        np.mean([int(r['useful'][2])-int(r['useful'][1]) for r in g]) for g in groups])
    run.once(PUBLIC/'analysis.json', run.stable(data))


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
        records = previous.prior.blocks(d, lab['frames'])
        y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
        motion = previous.model.motion(d['history'], d['groups']) if phase == 'fit' else None
        parent, controls, ranks = cost.banks()
        out = PUBLIC/'freeze.json'
        if phase != 'fit':
            if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(out):
                raise ValueError('Require frozen decisions')
            frozen = run.read(out)
            saved = {c['seed']: c for c in frozen['cells']}
        cells, rows = [], []
        for cell in parent:
            m = prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
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
