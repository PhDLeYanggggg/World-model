"""Train and score a single count-preserving, same-query forecast exchange."""
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
from scripts import run_m3w_record_rescue as previous
from src.world_model import m3w_atomic_swap as model

h, run, np, anchor = previous.h, previous.run, previous.np, previous.anchor
PUBLIC, PRIVATE = previous.PUBLIC.parent/'atomic_swap', previous.PRIVATE.parent/'atomic_swap'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_atomic_swap.py')
ARMS = ('direct', 'additive', 'joint')


def register():
    a = previous.PUBLIC/'analysis.json'
    if run.read(previous.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(a):
        raise ValueError('Require exact prior replay')
    for artifact in (*run.read(previous.REG)['parents'], *run.read(previous.REG)['files']):
        run.checked(artifact)
    paths = (a, previous.REG, previous.PUBLIC/'transition_diagnostic.json', Path(previous.__file__),
             Path(previous.model.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in paths], budget=.02, max_pairs=1, cuts=model.CUTS,
        steps=model.STEPS, batch=model.BATCH, independent_outcomes_opened=False)))


def design(m, ix, groups, co, rank, motion):
    direct, legal, context, _, _, u, e = h.previous.features(m, ix, groups, co, rank, motion)
    queries = list(model.roster(groups, direct, legal))
    i, f, j = np.concatenate(queries).T
    raw, s = m.x[ix], m.scale[ix]

    def side(row, family):
        return np.column_stack((context[row], raw[row, family],
            np.arcsinh(u[row, family]/s[row]), np.arcsinh(e[row, family]/s[row])))

    return direct, queries, np.stack((side(i, f), side(j, direct[j])), axis=1)


def mass(atoms, direct):
    i = np.flatnonzero(direct >= 0)
    return atoms[i, direct[i]].sum(0), float(atoms[:, 0, 1].sum())


def calibrate(records):
    best = None
    for cut in model.CUTS:
        new = lost = violations = edits = 0
        gain = 0.
        for options, atoms, known, direct, original, full, crowd, old in records:
            pick = model.first(options, cut)
            if pick is None:
                continue
            delta, lower = model.edits(atoms, direct, known, np.asarray([pick]))
            after = anchor.model.risk(original+delta[0], full)
            new += int(after['supported'] and not old['supported'])
            lost += int(old['supported'] and not after['supported'])
            violations += int(after['easy_net_over'] and not old['easy_net_over'])
            gain += float(lower[0]/crowd[pick[0]])
            edits += 1
        if new and not lost and not violations and gain >= 0.:
            value = new, gain, -edits, cut
            best = value if best is None else max(best, value)
    return None if best is None else best[-1]


def fit(m, controls, motion, y, valid, beat):
    d, cuts, known = m.d, run.cuts_by_source(), valid.any(1)
    orig, stats = {}, {}
    atoms = np.empty((*m.bank.shape[:2], 6))
    for s, ix in m.ix.items():
        x, b, c, scale = m.x[ix], m.bank[ix], m.floor[ix], m.scale[ix]
        events, atoms[ix], _ = anchor.model.labels(c, b, y[ix], valid[ix], cuts[s]['easy'])
        legal = np.linalg.norm(b-c[:, None], axis=-1).max(2) > 0.
        orig[s] = run.model.fitting_moments(x, b, c, y[ix], valid[ix], scale,
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
            blocks = []
            for s in learn:
                co = run.model.coefficients([orig[t] for t in learn if s != t])
                rank = anchor.model.fit([stats[t] for t in learn if s != t])
                local = []
                for ix, groups in m.groups[s]:
                    dc, queries, x = design(m, ix, groups, co, rank, motion)
                    if not len(x):
                        continue
                    a = atoms[ix]
                    original, full = mass(a, dc)
                    delta, lower = model.edits(a, dc, known[ix], np.concatenate(queries))
                    passes = previous.model.supported(original+delta, full)
                    rescue = passes & (lower >= 0.) & ~previous.model.supported(original, full)
                    sizes = np.asarray([len(q) for q in queries])
                    w = np.repeat(1./sizes[sizes > 0], sizes[sizes > 0])
                    local.append((x, rescue, w/w.sum()))
                blocks.extend((x, t, w/len(local)) for x, t, w in local)
            x, t, w = [np.concatenate(v) for v in zip(*blocks)]
            center = np.average(x, weights=w, axis=0)
            spread = np.sqrt(np.average((x-center)**2, weights=w, axis=0)).clip(.01)

            def normal(x):
                return np.clip((x-center)/spread, -8., 8.).astype(np.float32)

            seed = int.from_bytes(hashlib.sha256(f'swap:{m.seed}:{site}'.encode()).digest()[:4], 'little')
            checkpoint = mark.with_suffix('.pt')
            net = model.train(normal(x), t, w, seed, checkpoint,
                lambda **v: beat(state='training', seed=m.seed, source=site, **v))
            co, rank = m.co[site], m.rank(site)

            def options(ix, groups):
                dc, queries, x = design(m, ix, groups, co, rank, motion)
                scores = model.predict(net, normal(x))
                return dc, [list(model.ranked(queries, scores[:, j])) for j in range(2)]

            donors = [[], []]
            for s in cal:
                for ix, groups in m.groups[s]:
                    dc, picks = options(ix, groups)
                    a = atoms[ix]
                    original, full = mass(a, dc)
                    old = anchor.model.risk(original, full)
                    for dest, opts in zip(donors, picks):
                        dest.append((opts, a, known[ix], dc, original, full, d['crowd'][ix], old))
            thresholds = [calibrate(donor) for donor in donors]
            at = m.ix[site]
            choice = np.broadcast_to(m.direct[at], (len(ARMS), len(at))).copy()
            probe = choice[1:].copy()
            for ix, groups in m.groups[site]:
                dc, picks = options(ix, groups)
                np.testing.assert_array_equal(dc, m.direct[ix])
                for j, (opts, cut) in enumerate(zip(picks, thresholds)):
                    if opts:
                        i, f, k = opts[0][0]
                        probe[j, np.searchsorted(at, ix[[i, k]])] = [f, -1]
                    pick = model.first(opts, cut)
                    if pick is not None:
                        i, f, k = pick
                        choice[j+1, np.searchsorted(at, ix[[i, k]])] = [f, -1]
            cache = mark.with_suffix('.npz')
            h.prune.previous.history.feedback.write_array(cache, ids=d['global_ids'][at], choices=choice,
                probe=probe, center=center, spread=spread)
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache),
                checkpoint=run.artifact(checkpoint), thresholds=thresholds,
                fitting_pairs=len(t), positive_pairs=int(t.sum()))))
        f = run.read(mark)
        if f['identity'] != ident:
            raise ValueError('Incompatible fold checkpoint')
        for artifact in (f['cache'], f['checkpoint']):
            run.checked(artifact)
        folds.append(f)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def score(m, records, frozen, y, valid):
    d, p, cuts = m.d, run.parent, run.cuts_by_source()
    choice = np.broadcast_to(m.direct, (len(ARMS), len(m.direct))).copy()
    probe = choice[1:].copy()
    if frozen is not None:
        for f in frozen['folds']:
            at = m.ix[f['identity']['held']]
            with np.load(run.checked(f['cache']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
                choice[:, at], probe[:, at] = z['choices'], z['probe']
    np.testing.assert_array_equal(choice[0], m.direct)
    rows = []
    for rec, (ix, groups) in records.items():
        s = str(d['sites'][ix[0]])
        easy, hard = cuts[s]['easy'], cuts[s]['hard']
        dc, c, b = m.direct[ix], m.floor[ix], m.bank[ix]
        a = anchor.model.labels(c, b, y[ix], valid[ix], easy)[1]
        original, full = mass(a, dc)
        old = previous.model.supported(original, full)
        legal = np.linalg.norm(b-c[:, None], axis=-1).max(2) > 0.
        queries = list(model.roster(groups, dc, legal))
        pool = np.concatenate(queries)
        delta, lower = model.edits(a, dc, valid[ix].any(1), pool)
        rescue = previous.model.supported(original+delta, full) & (lower >= 0.) & ~old
        paths = {cv: run.model.selected_paths(z[ix], b, c) for cv, z in zip(ARMS, choice)}
        contrasts = dict(additive=('direct', 'additive', p.CV), joint=('direct', 'joint', p.CV),
                         joint_vs_additive=('additive', 'joint', p.CV))
        out = p.readout.score_recording({p.CV: c, **paths}, contrasts,
            y[ix], valid[ix], d['queries'][ix], rec, easy_cut=easy, hard_cut=hard)
        risk = {k: out['metrics'][k]['risk'] for k in ARMS}
        if bool(old) != risk['direct']['supported']:
            raise ValueError('Teacher/scorer mismatch')
        first = next((q for q in queries if len(q)), np.empty((0, 3), int))
        first_hits, edits = [], []
        for k, (z, t) in enumerate(zip(choice[1:, ix], probe[:, ix]), 1):
            changed = z != dc
            i, j = np.flatnonzero(changed & (z >= 0)), np.flatnonzero(changed & (z < 0))
            if (len(i) != len(j) or len(i) > 1 or (changed & (dc >= 0) & (z >= 0)).any()
                    or any((z[q] >= 0).sum() != (dc[q] >= 0).sum() for q in groups)):
                raise ValueError('Require a single count-preserving pair')
            edits.append(int(len(i)))
            ti, tj = np.flatnonzero((t >= 0) & (dc < 0)), np.flatnonzero((t < 0) & (dc >= 0))
            if frozen is not None and len(first):
                if len(ti) != 1 or len(tj) != 1:
                    raise ValueError('Require one first-query pair in both arms')
                match = (pool == [ti[0], t[ti[0]], tj[0]]).all(1)
                if not match[:len(first)].any():
                    raise ValueError('First-query pair absent from legal roster')
                first_hits.append(bool((match & rescue).any()))
            else:
                first_hits.append(False)
            if len(i):
                move = np.array([[i[0], z[i[0]], j[0]]])
                change, _ = model.edits(a, dc, valid[ix].any(1), move)
                if bool(previous.model.supported(original+change[0], full)) != risk[ARMS[k]]['supported']:
                    raise ValueError('Atomic teacher/scorer mismatch')
        if rescue.any():
            i, f, j = pool[np.flatnonzero(rescue)[0]]
            witness = dc.copy()
            witness[[i, j]] = [f, -1]
            exact = p.readout.score_recording({p.CV: c, 'direct': paths['direct'],
                'witness': run.model.selected_paths(witness, b, c)},
                {'pair': ('direct', 'witness', p.CV)}, y[ix], valid[ix],
                d['queries'][ix], rec, easy_cut=easy, hard_cut=hard)
            pair = exact['comparisons']['pair']
            if not exact['metrics']['witness']['risk']['supported'] or (
                    pair['known_delta']-pair['unknown_radius'] < -1e-8):
                raise ValueError('Atomic capacity witness failed exact scorer')
        rows.append(dict(source=s, seed=m.seed, risk=risk, comparisons=out['comparisons'],
            edits=edits, first_hits=first_hits, legal_pairs=len(pool), first_available=bool(len(first)),
            capacity=bool(rescue.any()), first_capacity=bool(rescue[:len(first)].any())))
    return rows


def report(rows, frozen):
    sites = sorted({r['source'] for r in rows})
    missing = [r for r in rows if r['risk']['direct']['completion_easy_risk_upper'] is None]
    if (len(sites), len(rows), len(missing)) != (14, 102, 35):
        raise ValueError('Require original cohort')
    scope = {}
    for name, group in (('all', rows), ('missing_reference', missing)):
        scope[name] = dict(views=len(group), with_legal_pair=sum(r['first_available'] for r in group),
            with_feasible_pair=sum(r['capacity'] for r in group),
            with_first_feasible_pair=sum(r['first_capacity'] for r in group),
            legal_pairs=sum(r['legal_pairs'] for r in group))
    if frozen is None:
        run.once(PUBLIC/'capacity.json', run.stable(dict(scope=scope, source_groups=len(sites),
            oracle_only=True, independent_confirmation=False, oracle_choices_exported=False)))
        return
    if scope != run.read(PUBLIC/'capacity.json')['scope']:
        raise ValueError('Frozen score changed action capacity')
    groups = [[r for r in rows if r['source'] == s and r['seed'] == k]
              for s in sites for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(14, size=(3000, 14))

    def summary(x):
        x = np.asarray(x, float).reshape(14, 3).mean(1)
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    risk, probe = {}, {}
    for i, a in enumerate(ARMS):
        v = [r['risk'][a] for r in rows]
        risk[a] = dict(passes=sum(x['supported'] for x in v), switches=sum(x['selected'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None
                                and x['known_easy_net_degradation'] > .02 for x in v),
            new_passes=sum(r['risk'][a]['supported'] and not r['risk']['direct']['supported'] for r in rows),
            lost_passes=sum(not r['risk'][a]['supported'] and r['risk']['direct']['supported'] for r in rows))
        if i:
            probe[a] = dict(edits=sum(r['edits'][i-1] for r in rows),
                first_rescues=sum(r['first_hits'][i-1] for r in rows),
                missing_first_rescues=sum(r['first_hits'][i-1] for r in missing))
        if risk[a]['switches'] != risk['direct']['switches']:
            raise ValueError('Count-matched policy changed aggregate switch count')
    baseline = run.read(previous.PUBLIC/'analysis.json')['risk']['direct']
    if any(baseline[k] != v for k, v in risk['direct'].items()):
        raise ValueError('Require unchanged direct score')
    utility = {a: summary([run.parent.readout.merge_comparisons([r['comparisons'][a] for r in group])
                          ['completion_lower_pp'] for group in groups])
               for a in ('additive', 'joint', 'joint_vs_additive')}
    folds = [x for c in frozen['cells'] for x in c['folds']]
    run.once(PUBLIC/'analysis.json', run.stable(dict(scope=scope, source_groups=len(sites), risk=risk,
        probe=probe, paired_lower_pp=utility,
        joint_minus_additive_first_rescue=summary([
            np.mean([int(r['first_hits'][1])-int(r['first_hits'][0]) for r in g]) for g in groups]),
        torch_models=len(folds), fitting_pairs=sum(x['fitting_pairs'] for x in folds),
        positive_pairs=sum(x['positive_pairs'] for x in folds),
        calibrated_folds={a: sum(x['thresholds'][i] is not None for x in folds)
                          for i, a in enumerate(ARMS[1:])},
        original_rules_unchanged=True, independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'capacity', 'fit', 'score', 'replay'))
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
        frozen, saved = None, {}
        if phase in ('score', 'replay'):
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
                rows.extend(score(m, records, saved.get(m.seed), y, valid))
            beat(state='seed_complete', seed=m.seed)
            del m
        if phase == 'fit':
            run.once(out, run.stable(dict(cells=cells, held_excluded=True)))
        else:
            report(rows, frozen)
            out = PUBLIC/('capacity.json' if phase == 'capacity' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            artifact=run.artifact(out), python_architecture=platform.machine(),
            torch_threads=model.torch.get_num_threads()))


if __name__ == '__main__':
    main()
