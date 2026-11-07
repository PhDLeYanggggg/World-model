"""Fit and score source-stable veto costs against exact frozen controls."""
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
from scripts import run_m3w_calibration_size as size
from src.world_model import m3w_veto_source_stability as model

base, run, np = size.base, size.run, size.np
PUBLIC, PRIVATE = base.PUBLIC/'stability', base.PRIVATE/'stability'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_veto_source_stability.py')
ARMS = ('base', 'pooled', 'stable')


def register():
    parents = (size.REG, size.PUBLIC/'freeze.json', size.PUBLIC/'analysis.json',
               size.PUBLIC/'replay_execution.json', Path(size.__file__),
               Path(base.__file__), Path(base.parent.model.__file__),
               Path(run.model.__file__), Path(run.parent.readout.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], arms=ARMS, paired_draws=3000,
        margin_jackknife_sd=1, original_budget=.02, independent_outcomes_opened=False)))


def fit(d, refs, records, beat):
    ids, sites = d['global_ids'], d['sites']
    splits = {str(s): np.flatnonzero(sites == s) for s in np.unique(sites)}
    by_source = {s: [] for s in splits}
    for ri, g in records.values():
        s = str(sites[ri[0]])
        by_source[s].append((ri, g, np.searchsorted(splits[s], ri)))
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval', 'valid'))
    y, valid = labels['target_eval'] - d['origin'][:, None], labels['valid']
    writer = base.parent.base.nested.base.parent.base.write_array
    cells = []
    for seed_cell in run.read(size.PUBLIC/'freeze.json')['cells']:
        seed = seed_cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        data, main_moments = {}, {}
        for s, ix in splits.items():
            c, b = cv[ix], bank[ix]
            x, scale = run.model.descriptors(d['history'][ix], b, c, d['crowd'][ix])
            _, a, e = base.parent.base.nested.base.teacher.actual(b, c, y[ix], valid[ix], cuts[s]['easy'])
            data[s] = b, c, x, scale, a, e
            main_moments[s] = run.model.fitting_moments(x, b, c, y[ix], valid[ix], scale,
                                                        1./d['crowd'][ix], cuts[s]['easy'])
        folds = []
        for control in seed_cell['selection']:
            i = control['identity']
            held, learn = i['held'], i['fit']
            ix = splits[held]
            ident = dict(held=held, seed=seed, learn=learn, registration=run.sha(REG))
            mark = PRIVATE/f'{seed}_{held}.json'
            if not mark.exists():
                moments = []
                for donor in learn:
                    b, c, x, scale, ta, te = data[donor]
                    co = run.model.coefficients([main_moments[k] for k in learn if k != donor])
                    z = np.empty((*x.shape[:2], x.shape[2]+4))
                    vetoed = np.zeros(x.shape[:2], bool)
                    residual = np.empty((*x.shape[:2], 2))
                    for _, g, j in by_source[donor]:
                        _, a, e, _, _, _, _, z[j], vetoed[j] = base.original_context(
                            g, x[j], scale[j], b[j], c[j], co)
                        residual[j] = np.stack((ta[j]-a, te[j]-e), axis=-1)/scale[j, None, None]
                    moments.append(base.parent.model.moments(z, residual, vetoed, 1./d['crowd'][splits[donor]]))
                full, deleted = model.fit(moments)
                with np.load(run.checked(control['cache']), allow_pickle=False) as cache:
                    np.testing.assert_array_equal(cache['ids'], ids[ix])
                    np.testing.assert_array_equal(cache['residual_coefficients'], full)
                    old = cache['choices'][:2].copy()
                    co = cache['baseline_coefficients'].copy()
                b, c, x, scale, _, _ = data[held]
                choice = np.vstack((old, np.full(len(ix), -1, np.int16)))
                reopened = np.zeros(2, int)
                for _, g, j in by_source[held]:
                    u, a, e, legal, w, ch, before, z, vetoed = base.original_context(
                        g, x[j], scale[j], b[j], c[j], co)
                    radius = np.linalg.norm(b[j]-c[j, None], axis=-1).max(2)
                    point, cautious = model.correction(z, full, deleted)
                    np.testing.assert_array_equal(ch, old[0, j])
                    for k, delta in enumerate((point, cautious)):
                        p, release = model.allocate(g, u, a, e, legal, w, before, vetoed, delta, scale[j], radius)
                        if k == 0:
                            np.testing.assert_array_equal(p, old[1, j])
                        else:
                            choice[2, j] = p
                        rows = np.flatnonzero(p >= 0)
                        reopened[k] += release[rows, p[rows]].sum()
                path = mark.with_suffix('.npz')
                writer(path, ids=ids[ix], choices=choice, full=full, deleted=deleted)
                run.once(mark, dict(identity=ident, cache=run.artifact(path), reopened=reopened.tolist()))
            cell = run.read(mark)
            if cell['identity'] != ident:
                raise ValueError('Saved fold roles differ')
            folds.append(cell)
            beat(state='held_frozen', seed=seed, fold=len(folds))
        cells.append(dict(seed=seed, folds=folds))
    run.once(PUBLIC/'freeze.json', run.stable(dict(cells=cells, decisions_frozen_before_scoring=True)))


def score(d, refs, records, beat):
    freeze = run.read(PUBLIC/'freeze.json')
    p = run.parent
    cuts = run.cuts_by_source()
    labels = run.old.prior.arrays(('target_eval', 'valid'))
    y, valid = labels['target_eval'] - d['origin'][:, None], labels['valid']
    pairs = dict(stable_vs_pooled=('pooled', 'stable', p.CV),
                 pooled_vs_base=('base', 'pooled', p.CV), stable_vs_base=('base', 'stable', p.CV))
    previous = {(r['seed'], r['recording']): r for r in run.read(size.PUBLIC/'analysis.json')['records']}
    results = []
    for cell in freeze['cells']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        ch = np.full((3, len(cv)), -1, np.int16)
        for fold in cell['folds']:
            ix = np.flatnonzero(d['sites'] == fold['identity']['held'])
            with np.load(run.checked(fold['cache']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'][ix])
                ch[:, ix] = z['choices']
        for r, (ix, g) in records.items():
            s = str(d['sites'][ix[0]])
            v = ch[:, ix]
            if any(not (n == n[0]).all() for n in ((v[:, j] >= 0).sum(1) for j in g)):
                raise ValueError('Every query must preserve its intervention count')
            c, b = cv[ix], bank[ix]
            paths = {a: run.model.selected_paths(z, b, c) for a, z in zip(ARMS, v)}
            out = p.readout.score_recording({p.CV:c, **paths}, pairs, y[ix], valid[ix], d['queries'][ix],
                                            r, easy_cut=cuts[s]['easy'], hard_cut=cuts[s]['hard'])
            for a, prev in zip(ARMS[:2], ('base', 'unguarded')):
                if out['metrics'][a] != previous[seed, r]['metrics'][prev]:
                    raise ValueError('Frozen control metric differs')
            results.append(dict(seed=seed, source=s, changed=int((v[1] != v[2]).sum()), **out))
        beat(state='seed_scored', seed=seed)
    sites = sorted({r['source'] for r in results})
    if len(results) != 102 or len(sites) != 14:
        raise ValueError('Require the full frozen roster')
    groups = [[r for r in results if r['source'] == s and r['seed'] == k]
              for s in sites for k in run.kin.prior.SEEDS]
    draws = np.random.default_rng(73107).integers(14, size=(3000, 14))
    comparisons = {}
    for name in pairs:
        v = [p.readout.merge_comparisons([r['comparisons'][name] for r in g]) for g in groups]
        z = np.asarray([r['completion_lower_pp'] for r in v], dtype=float).reshape(14, 3).mean(1)
        if not np.isfinite(z).all():
            raise ValueError('Paired utility must be estimable in all source-seed cells')
        comparisons[name] = dict(point=float(z.mean()), interval95=np.quantile(z[draws].mean(1), [.025,.975]).tolist())
    risk = {}
    for a in ARMS:
        v = [r['metrics'][a]['risk'] for r in results]
        risk[a] = dict(switches=sum(x['selected'] for x in v), passes=sum(x['supported'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_over_2=sum(x['completion_easy_risk_upper'] is not None and x['completion_easy_risk_upper'] > .02 for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and x['known_easy_net_degradation'] > .02 for x in v))
    run.once(PUBLIC/'analysis.json', run.stable(dict(records=results, risk=risk, comparisons=comparisons,
        changed=sum(r['changed'] for r in results), controls_exact=True, original_rules_unchanged=True,
        independent_confirmation=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register','fit','score','replay'))
    phase = p.parse_args().phase
    register()
    if phase == 'register':
        return
    for path in (*FILES, REG):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT) != path.read_bytes():
            raise ValueError('Commit design and registration before fitting')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resume from held-source checkpoints')
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            print(msg, flush=True)
        beat(state='start')
        d, refs = run.population(), run.register()
        records = size.prior.blocks(d, run.old.prior.arrays(('frames',))['frames'])
        (fit if phase == 'fit' else score)(d, refs, records, beat)
        artifact = PUBLIC/('freeze.json' if phase == 'fit' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
                                                        artifact=run.artifact(artifact)))


if __name__ == '__main__':
    main()
