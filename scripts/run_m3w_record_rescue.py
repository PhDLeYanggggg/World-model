"""Train whole-record rescue labels and compare with matched local supervision."""
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
from scripts import run_m3w_reference_hurdle as h
from src.world_model import m3w_record_rescue as model

run, np, anchor = h.run, h.np, h.anchor
PUBLIC, PRIVATE = h.PUBLIC.parent/'record_rescue', h.PRIVATE.parent/'record_rescue'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_record_rescue.py')
ARMS = ('direct', *model.ARMS)


def register():
    parent = h.PUBLIC/'analysis.json'
    if run.read(h.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(parent):
        raise ValueError('Require exact preceding replay')
    for p in (*run.read(h.REG)['parents'], *run.read(h.REG)['files']):
        run.checked(p)
    paths = (parent, h.REG, h.PUBLIC/'operating_diagnostic.json',
             Path(h.__file__), Path(h.model.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in paths], steps=model.STEPS, batch=model.BATCH,
        arms=ARMS, cuts=h.model.CUTS, budget=.02, independent_outcomes_opened=False)))


def features(m, ix, groups, co, rank, motion):
    dc, legal, i, k, x = h.features(m, ix, groups, co, rank, motion)
    scale = m.scale[ix]
    costs = run.model.decision_costs(m.x[ix], co, m.bank[ix], m.floor[ix], scale)[:3]
    estimate = np.stack(costs, axis=-1)/scale[:, None, None]
    context = model.prefix(groups, dc, estimate)
    return dc, legal, i, k, x, context[i]


def fit(m, controls, motion, y, valid, beat):
    d, cuts = m.d, run.cuts_by_source()
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
                for ix, g in m.groups[s]:
                    dc, legal, i, k, x, prefix = features(m, ix, g, co, rank, motion)
                    if len(i):
                        w = 1./(d['crowd'][ix[i]]*legal.sum(1)[i])
                        target, _, _ = model.supervision(atoms[ix], dc)
                        local.append((np.column_stack((x, prefix)), target[i, k], w/w.sum()))
                blocks.extend((x, t, w/len(local)) for x, t, w in local)
            x, t, w = [np.concatenate(v) for v in zip(*blocks)]
            if (t.sum(0) == 0).any():
                raise ValueError('Missing fitting support for a supervision head')
            center = np.average(x, weights=w, axis=0)
            spread = np.sqrt(np.average((x-center)**2, weights=w, axis=0)).clip(.01)

            def normal(x):
                return np.clip((x-center)/spread, -8., 8.).astype(np.float32)

            seed = int.from_bytes(hashlib.sha256(f'rescue:{m.seed}:{site}'.encode()).digest()[:4], 'little')
            xx = normal(x)
            nets, checkpoints = [], []
            for name, x_train in (('local', xx[:, :-5]), ('prefix', xx)):
                ck = mark.with_name(f'{mark.stem}_{name}.pt')
                nets.append(model.train(x_train, t, w, seed, ck,
                    lambda **v: beat(state='training', seed=m.seed, source=site, head=name, **v)))
                checkpoints.append(run.artifact(ck))
            co, rank = m.co[site], m.rank(site)

            def options(ix, g):
                dc, legal, i, k, x, pref = features(m, ix, g, co, rank, motion)
                scores = np.zeros((*legal.shape, len(model.ARMS)), np.float32)
                if len(i):
                    x = normal(np.column_stack((x, pref)))
                    scores[i, k] = np.column_stack((model.predict(nets[0], x[:, :-5]),
                                                    model.predict(nets[1], x)[:, 1]))
                return dc, [list(model.options(g, legal, scores, arm)) for arm in range(len(model.ARMS))]

            donors = [[] for _ in model.ARMS]
            for s in cal:
                for ix, g in m.groups[s]:
                    dc, candidates = options(ix, g)
                    a, i = atoms[ix], np.flatnonzero(dc >= 0)
                    mass, full = a[i, dc[i]].sum(0), float(a[:, 0, 1].sum())
                    old = anchor.model.risk(mass, full)
                    for dest, opts in zip(donors, candidates):
                        dest.append((opts, a, mass, full, d['crowd'][ix], old))
            thresholds = [h.calibrate(donor) for donor in donors]
            at = m.ix[site]
            choice = np.broadcast_to(m.direct[at], (len(ARMS), len(at))).copy()
            top = np.full((len(model.ARMS), len(at)), -1, dtype=np.int16)
            for ix, g in m.groups[site]:
                dc, candidates = options(ix, g)
                np.testing.assert_array_equal(dc, m.direct[ix])
                for j, (opts, cut) in enumerate(zip(candidates, thresholds)):
                    if opts:
                        i, k, *_ = opts[0]
                        top[j, np.searchsorted(at, ix[i])] = k
                    found = h.model.first(opts, cut)
                    if found is not None:
                        i, k = found
                        choice[j+1, np.searchsorted(at, ix[i])] = k
            cache = mark.with_suffix('.npz')
            h.prune.previous.history.feedback.write_array(cache, ids=d['global_ids'][at], choices=choice,
                top=top, center=center, scale=spread)
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache),
                checkpoints=checkpoints, thresholds=thresholds, fitting_actions=len(t),
                positive_labels=t.sum(0).tolist())))
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
    top = np.full((len(model.ARMS), len(m.floor)), -1, np.int16)
    for f in frozen['folds']:
        at = m.ix[f['identity']['held']]
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            choice[:, at], top[:, at] = z['choices'], z['top']
    np.testing.assert_array_equal(choice[0], m.direct)
    rows = []
    for rec, (ix, g) in records.items():
        s = str(d['sites'][ix[0]])
        e, hard = cuts[s]['easy'], cuts[s]['hard']
        z, c, b = choice[:, ix], m.floor[ix], m.bank[ix]
        changed = z != z[0]
        if (changed[:, z[0] >= 0].any() or (changed.sum(1) > 1).any()
                or any(((z[:, q] >= 0).sum(1) > min(2, len(q)-1)).any() for q in g)):
            raise ValueError('Expected one insertion and the original query cap')
        paths = {k: run.model.selected_paths(v, b, c) for k, v in zip(ARMS, z)}
        out = p.readout.score_recording({p.CV: c, **paths},
            {a: ('direct', a, p.CV) for a in model.ARMS},
            y[ix], valid[ix], d['queries'][ix], rec, easy_cut=e, hard_cut=hard)
        _, a, _ = anchor.model.labels(c, b, y[ix], valid[ix], e)
        target, mass, full = model.supervision(a, z[0])
        risk = {k: out['metrics'][k]['risk'] for k in ARMS}
        if bool(model.supported(mass, full)) != risk['direct']['supported']:
            raise ValueError('Teacher/scorer mismatch')
        probes, successful = [], []
        for arm, opts, chosen, add in zip(model.ARMS, top[:, ix], z[1:], changed[1:]):
            i = np.flatnonzero(opts >= 0)
            if len(i) > 1:
                raise ValueError('Expected a single first-query probe')
            probes.append(target[i, opts[i]].any(0).tolist())
            i = np.flatnonzero(add)
            successful.append(bool(target[i, chosen[i], 1].any()))
            if bool(model.supported(mass+a[i, chosen[i]].sum(0), full)) != risk[arm]['supported']:
                raise ValueError('Insertion teacher/scorer mismatch')
        rows.append(dict(source=s, seed=m.seed, comparison=out['comparisons'], risk=risk,
            insertions=changed.sum(1).tolist(), probes=probes, nonloss_rescues=successful))
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
            insertions=sum(r['insertions'][j] for r in rows),
            new_passes=sum(r['risk'][a]['supported'] and not r['risk']['direct']['supported'] for r in rows),
            lost_passes=sum(not r['risk'][a]['supported'] and r['risk']['direct']['supported'] for r in rows))
    if risk['direct'] != run.read(h.PUBLIC/'analysis.json')['risk']['direct']:
        raise ValueError('Require unchanged direct score')
    folds = [x for c in freeze['cells'] for x in c['folds']]
    probe, utility = {}, {}
    for j, a in enumerate(model.ARMS):
        probe[a] = dict(first_local=sum(r['probes'][j][0] for r in rows),
            first_rescue=sum(r['probes'][j][1] for r in rows),
            first_rescue_missing=sum(r['probes'][j][1] for r in missing),
            nonloss_rescues=sum(r['nonloss_rescues'][j] for r in rows),
            calibrated_folds=sum(f['thresholds'][j] is not None for f in folds))
        utility[a] = summary([run.parent.readout.merge_comparisons(
            [r['comparison'][a] for r in group])['completion_lower_pp'] for group in groups])
    contrasts = {}
    for left, right in ((1, 0), (2, 1)):
        contrasts[f'{model.ARMS[left]}_minus_{model.ARMS[right]}_first_rescue'] = summary([
            np.mean([int(r['probes'][left][1])-int(r['probes'][right][1]) for r in g]) for g in groups])
    run.once(PUBLIC/'analysis.json', run.stable(dict(views=102, missing_views=35, source_groups=14,
        torch_models=2*len(folds), fitting_actions=sum(f['fitting_actions'] for f in folds),
        positive_labels=np.sum([f['positive_labels'] for f in folds], axis=0).tolist(),
        risk=risk, probe=probe, contrasts=contrasts, paired_lower_pp=utility,
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
                raise ValueError('Require frozen decisions')
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
