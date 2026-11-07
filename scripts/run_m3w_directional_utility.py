"""Test directional ADE utility with fixed chronological risk costs and counts."""
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
from src.world_model import m3w_directional_utility as model

run, np, prior = size.run, size.np, size.prior
PUBLIC, PRIVATE = prior.PUBLIC/'directional_utility', prior.PRIVATE/'directional_utility'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_directional_utility.py')
ARMS = ('direct', 'tangent', 'directional')


def register():
    parents = (size.REG, size.PUBLIC/'freeze.json', size.PUBLIC/'analysis.json',
               size.PUBLIC/'replay_execution.json', Path(size.__file__),
               Path(size.base.__file__), Path(prior.model.__file__),
               Path(run.model.__file__), Path(run.parent.readout.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], arms=ARMS, matched_counts=True,
        original_budget=.02, draws=3000, independent_outcomes_opened=False)))


def fit(d, refs, records, beat):
    ids, sites = d['global_ids'], d['sites']
    splits = {str(s): np.flatnonzero(sites == s) for s in np.unique(sites)}
    groups = {s: [] for s in splits}
    for ix, g in records.values():
        s = str(sites[ix[0]])
        local = np.searchsorted(splits[s], ix)
        groups[s].append([local[j] for j in g])
    labels = run.old.prior.arrays(('target_eval', 'valid'))
    y, valid = labels['target_eval'] - d['origin'][:, None], labels['valid']
    writer = size.base.parent.base.nested.base.parent.base.write_array
    cells = []
    for parent in run.read(size.PUBLIC/'freeze.json')['cells']:
        seed = parent['seed']
        cv, bank = run.candidates(d, refs, seed)
        source_moments = {}
        for s, ix in splits.items():
            h, b, c, crowd = d['history'][ix], bank[ix], cv[ix], d['crowd'][ix]
            x, scale = run.model.descriptors(h, b, c, crowd)
            z, change, axes = model.features(h, scale, crowd, b, c)
            t, q = model.targets(c, y[ix], valid[ix], axes, change, scale)
            source_moments[s] = model.moments(z, x, t, q, valid[ix], crowd)
        folds = []
        for control in parent['selection']:
            ident = control['identity']
            held, learn = ident['held'], ident['fit']
            if len(set(learn)) != 7 or held in learn:
                raise ValueError('Require seven distinct fitting sources excluding held')
            ix = splits[held]
            mark = PRIVATE/f'{seed}_{held}.json'
            identity = dict(seed=seed, held=held, fit=learn, registration=run.sha(REG))
            if not mark.exists():
                weights = model.fit([source_moments[s] for s in learn])
                with np.load(run.checked(control['cache']), allow_pickle=False) as data:
                    np.testing.assert_array_equal(ids[ix], data['ids'])
                    co = data['baseline_coefficients'].copy()
                    direct = data['choices'][0].copy()
                h, b, c, crowd = d['history'][ix], bank[ix], cv[ix], d['crowd'][ix]
                x, scale = run.model.descriptors(h, b, c, crowd)
                z, change, _ = model.features(h, scale, crowd, b, c)
                u, a, e, legal = run.model.decision_costs(x, co, b, c, scale)
                values = [u/scale[:, None], *[model.coherent_value(p, x, co, scale, b-c[:, None])
                    for p in model.predict(z, change, x, scale, weights)]]
                choices = np.full((len(ARMS), len(ix)), -1, np.int16)
                for g in groups[held]:
                    w = prior.nested.base.select(g, u, a, e, scale, legal)
                    count = [int((w[j] >= 0).sum()) for j in g]
                    for row, value in enumerate(values):
                        ch, _ = prior.model.stream(g, value, a, e, legal, counts=count, witness=w)
                        for j in g:
                            choices[row, j] = ch[j]
                np.testing.assert_array_equal(direct, choices[0])
                path = mark.with_suffix('.npz')
                writer(path, ids=ids[ix], choices=choices, direction_weights=weights[0],
                       curvature_weights=weights[1])
                run.once(mark, dict(identity=identity, cache=run.artifact(path)))
            fold = run.read(mark)
            if fold['identity'] != identity:
                raise ValueError('Saved fold identity differs')
            folds.append(fold)
            beat(state='held_frozen', seed=seed, fold=len(folds))
        cells.append(dict(seed=seed, folds=folds))
    run.once(PUBLIC/'freeze.json', run.stable(dict(cells=cells, decisions_frozen_before_scoring=True)))


def score(d, refs, records, beat):
    p = run.parent
    freeze = run.read(PUBLIC/'freeze.json')
    labels = run.old.prior.arrays(('target_eval', 'valid'))
    y, valid = labels['target_eval'] - d['origin'][:, None], labels['valid']
    cuts = run.cuts_by_source()
    previous = {(r['seed'], r['recording']): r for r in run.read(size.PUBLIC/'analysis.json')['records']}
    pairs = dict(directional_vs_direct=('direct', 'directional', p.CV),
                 tangent_vs_direct=('direct', 'tangent', p.CV),
                 directional_vs_tangent=('tangent', 'directional', p.CV))
    rows = []
    for cell in freeze['cells']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        all_choices = np.full((len(ARMS), len(cv)), -1, np.int16)
        for fold in cell['folds']:
            ix = np.flatnonzero(d['sites'] == fold['identity']['held'])
            with np.load(run.checked(fold['cache']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'][ix])
                all_choices[:, ix] = z['choices']
        for rec, (ix, groups) in records.items():
            s, v = str(d['sites'][ix[0]]), all_choices[:, ix]
            if any(not (n == n[0]).all() for n in ((v[:, j] >= 0).sum(1) for j in groups)):
                raise ValueError('Switch counts must match within every query')
            c, b = cv[ix], bank[ix]
            pred = {a: run.model.selected_paths(z, b, c) for a, z in zip(ARMS, v)}
            out = p.readout.score_recording({p.CV: c, **pred}, pairs, y[ix], valid[ix],
                d['queries'][ix], rec, easy_cut=cuts[s]['easy'], hard_cut=cuts[s]['hard'])
            if out['metrics']['direct'] != previous[seed, rec]['metrics']['base']:
                raise ValueError('The direct control must replay exactly')
            rows.append(dict(source=s, seed=seed, **out,
                reassigned={a: int((v[j] != v[0]).sum()) for j, a in enumerate(ARMS[1:], 1)}))
        beat(state='seed_scored', seed=seed)
    sources = sorted({r['source'] for r in rows})
    if len(rows) != 102 or len(sources) != 14:
        raise ValueError('Require all registered views')
    grouped = [[r for r in rows if r['source'] == s and r['seed'] == k]
               for s in sources for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(14, size=(3000, 14))
    comparisons = {}
    for name in pairs:
        vals = [p.readout.merge_comparisons([r['comparisons'][name] for r in g]) for g in grouped]
        comparisons[name] = {}
        for metric in ('known_pp', 'completion_lower_pp', 'completion_upper_pp'):
            z = np.asarray([r[metric] for r in vals], dtype=float).reshape(14, 3).mean(1)
            if not np.isfinite(z).all():
                raise ValueError('All source-seed utility contrasts must be estimable')
            comparisons[name][metric] = dict(point=float(z.mean()),
                interval95=np.quantile(z[draw].mean(1), [.025, .975]).tolist())
    risk = {}
    for a in ARMS:
        v = [r['metrics'][a]['risk'] for r in rows]
        risk[a] = dict(switches=sum(x['selected'] for x in v), passes=sum(x['supported'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None and
                                x['known_easy_net_degradation'] > .02 for x in v),
            easy_over_2=sum(x['completion_easy_risk_upper'] is not None and
                            x['completion_easy_risk_upper'] > .02 for x in v))
    run.once(PUBLIC/'analysis.json', run.stable(dict(records=rows, comparisons=comparisons, risk=risk,
        reassigned={a: sum(r['reassigned'][a] for r in rows) for a in ARMS[1:]},
        controls_exact=True, original_rules_unchanged=True, independent_confirmation=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = p.parse_args().phase
    register()
    if phase == 'register':
        return
    for path in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))],
                                   cwd=ROOT) != path.read_bytes():
            raise ValueError('Commit the design before running it')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resume from saved source checkpoints')
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            print(msg, flush=True)
        beat(state='start')
        d, refs = run.population(), run.register()
        records = prior.blocks(d, run.old.prior.arrays(('frames',))['frames'])
        (fit if phase == 'fit' else score)(d, refs, records, beat)
        path = PUBLIC/('freeze.json' if phase == 'fit' else 'analysis.json')
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
                                                        artifact=run.artifact(path)))


if __name__ == '__main__':
    main()
