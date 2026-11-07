"""Fit and compare first-query weighted listwise ranking on native CPU."""
import argparse
from collections import Counter
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
    raise RuntimeError('Run with native arm64 .venv-pytorch, not an Intel/Rosetta interpreter')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_easy_reference_rank as previous
from src.world_model import m3w_early_reference_listwise as model

run, np, cost, anchor, prune = previous.run, previous.np, previous.cost, previous.anchor, previous.prune
PUBLIC, PRIVATE = previous.prior.PUBLIC/'early_reference_listwise', previous.prior.PRIVATE/'early_reference_listwise'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_early_reference_listwise.py')
ARMS = ('old_joint', 'boosted', 'listwise', 'early_listwise')


def register():
    a = previous.PUBLIC/'analysis.json'
    if run.read(previous.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(a):
        raise ValueError('Require exact pointwise baseline replay')
    for item in run.read(previous.REG)['parents']:
        run.checked(item)
    parents = (a, previous.PUBLIC/'freeze.json', previous.PUBLIC/'first_query_diagnostic.json',
               previous.REG, Path(previous.__file__), Path(previous.model.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], steps=model.STEPS, batch=model.BATCH,
        early_multiplier=model.EARLY_WEIGHT, budget=.02, independent_outcomes_opened=False)))


def fit(m, controls, old, motion, y, valid, beat):
    d, cuts = m.d, run.cuts_by_source()
    orig, stats = {}, {}
    for s, ix in m.ix.items():
        x, b, c = m.x[ix], m.bank[ix], m.floor[ix]
        z = m.scale[ix]
        event, _, _ = anchor.model.labels(c, b, y[ix], valid[ix], cuts[s]['easy'])
        orig[s] = run.model.fitting_moments(x, b, c, y[ix], valid[ix], z, 1./d['crowd'][ix], cuts[s]['easy'])
        legal = np.linalg.norm(b-c[:, None], axis=-1).max(2) > 0.
        stats[s] = anchor.model.fitting(x, event, legal, d['crowd'][ix])
    folds = []
    for item in controls:
        r = item['identity']
        s, learn, cal = r['held'], r['fit'], r['cal_a'] + r['cal_b']
        ident = dict(seed=m.seed, held=s, fit=learn, cal=cal, registration=run.sha(REG))
        if (len(learn), len(cal), len(set(learn+cal+[s]))) != (7, 6, 14):
            raise ValueError('Require disjoint source roles')
        if any(old[s]['identity'][k] != ident[k] for k in ('seed', 'held', 'fit', 'cal')):
            raise ValueError('Require same pointwise control roles')
        mark = PRIVATE/f'{m.seed}_{s}.json'
        if not mark.exists():
            sources = []
            for src in learn:
                donors = [t for t in learn if t != src]
                co = run.model.coefficients([orig[t] for t in donors])
                rank = anchor.model.fit([stats[t] for t in donors])
                recs = []
                for ix, g in m.groups[src]:
                    _, legal, x, *_ = previous.features(m, ix, g, co, rank, motion)
                    ok = legal.any(1)
                    truth = previous.labels(m, ix, y, valid, cuts[src]['easy'])
                    group = [q[ok[q]] for q in g if ok[q].any()]
                    mixed = [(x[q], truth[q], j < 2) for j, q in enumerate(group)
                             if truth[q].any() and not truth[q].all()]
                    if mixed:
                        recs.append(mixed)
                if recs:
                    sources.append(recs)
            x = np.concatenate([q[0] for rs in sources for r in rs for q in r])
            center, scale = x.mean(0), x.std(0).clip(.01)
            sources = [[[(np.clip((x-center)/scale, -8., 8.).astype(np.float32), t, e)
                         for x, t, e in rec] for rec in rs] for rs in sources]
            seed = int.from_bytes(hashlib.sha256(f'{m.seed}:{s}'.encode()).digest()[:4], 'little')
            plan = model.sample_plan(sources, seed)
            nets, checkpoints = [], []
            for name, weight in (('listwise', 1.), ('early', model.EARLY_WEIGHT)):
                ck = PRIVATE/f'{m.seed}_{s}_{name}.pt'
                def progress(**v):
                    beat(state='training', seed=m.seed, source=s, arm=name, **v)
                nets.append(model.train(sources, plan, len(center), seed, weight, ck, progress))
                checkpoints.append(run.artifact(ck))
            at = m.ix[s]
            with np.load(run.checked(old[s]['cache']), allow_pickle=False) as cache:
                np.testing.assert_array_equal(cache['ids'], d['global_ids'][at])
                eligible = cache['eligible']
                score = np.column_stack((cache['scores'][:, [1, 3]], np.zeros((len(at), 2)))).astype(np.float32)
            rank, co = m.rank(s), m.co[s]
            for ix, g in m.groups[s]:
                dc, legal, x, *_ = previous.features(m, ix, g, co, rank, motion)
                np.testing.assert_array_equal(dc, m.direct[ix])
                j = np.searchsorted(at, ix)
                np.testing.assert_array_equal(eligible[j], legal.any(1))
                xn = np.clip((x-center)/scale, -8., 8.)
                score[j, 2:] = np.column_stack([model.predict(net, xn) for net in nets])
            cache = mark.with_suffix('.npz')
            prune.previous.history.feedback.write_array(cache, ids=d['global_ids'][at],
                scores=score, eligible=eligible, normalization_center=center, normalization_scale=scale)
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache),
                checkpoints=checkpoints, mixed_queries=sum(len(r) for rs in sources for r in rs),
                early_mixed_queries=sum(v[2] for rs in sources for r in rs for v in r))))
        f = run.read(mark)
        if f['identity'] != ident:
            raise ValueError('Incompatible fold checkpoint')
        for a in [f['cache'], *f['checkpoints']]:
            run.checked(a)
        folds.append(f)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def score(m, records, frozen, y, valid, rows):
    d, cuts = m.d, run.cuts_by_source()
    scores = np.empty((len(m.floor), len(ARMS)), np.float32)
    mask = np.zeros(len(m.floor), bool)
    for f in frozen['folds']:
        at = m.ix[f['identity']['held']]
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            scores[at], mask[at] = z['scores'], z['eligible']
    for ix, g in records.values():
        s = str(d['sites'][ix[0]])
        target = previous.labels(m, ix, y, valid, cuts[s]['easy'])
        active = [q[mask[ix[q]]] for q in g if mask[ix[q]].any()]
        if not active:
            raise ValueError('Require full eligible recording roster')
        p = scores[ix]
        hits = np.stack([target[q[p[q].argmax(0)]] for q in active])
        rows.append(dict(seed=m.seed, source=s, mean=hits.mean(0), first=hits[0],
            first_two=hits[:2].mean(0), missing=not target[m.direct[ix] >= 0].any(),
            first_available=bool(target[active[0]].any())))


def report(rows, frozen):
    sites = sorted({r['source'] for r in rows})
    missing = [r for r in rows if r['missing']]
    if (len(sites), len(rows), len(missing)) != (14, 102, 35):
        raise ValueError('Require original cohort')
    draw = np.random.default_rng(73107).integers(14, size=(3000, 14))
    readout = {}
    old = run.read(previous.PUBLIC/'analysis.json')
    for key in ('first', 'first_two', 'mean'):
        x = np.stack([np.mean([r[key] for r in rows if r['source'] == s and r['seed'] == k], axis=0)
                      for s in sites for k in run.kin.prior.SEEDS]).reshape(14, 3, len(ARMS)).mean(1)
        readout[key] = dict(zip(ARMS, x.mean(0).tolist()))
        if key in ('first', 'mean'):
            for k in ARMS[:2]:
                if readout[key][k] != old['top_one'][key][k]:
                    raise ValueError('Require exact old control top-one rates')
        for j, before in enumerate(ARMS[:3]):
            z = x[:, -1]-x[:, j]
            readout[f'{key}_early_minus_{before}'] = dict(point=float(z.mean()),
                interval95=np.quantile(z[draw].mean(1), [.025, .975]).tolist())
    f = [v for c in frozen['cells'] for v in c['folds']]
    run.once(PUBLIC/'analysis.json', run.stable(dict(views=102, source_groups=14, missing_views=35,
        trained_heads=2*len(f), updates_per_head=model.STEPS, runtime='native_torch_cpu_4_threads',
        mixed_query_occurrences=sum(v['mixed_queries'] for v in f),
        early_mixed_query_occurrences=sum(v['early_mixed_queries'] for v in f),
        top_one=readout, missing_first_hits=dict(zip(ARMS, np.stack([r['first'] for r in missing]).sum(0).tolist())),
        missing_first_available=sum(r['first_available'] for r in missing), operating_policy_evaluated=False,
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
    model.torch.set_num_threads(4)
    model.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True)
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
        y, valid = lab['target_eval'] - d['origin'][:, None], lab['valid']
        motion = previous.model.motion(d['history'], d['groups']) if phase == 'fit' else None
        parent, controls, ranks = cost.banks()
        old = {c['seed']: {f['identity']['held']: f for f in c['folds']}
               for c in run.read(previous.PUBLIC/'freeze.json')['cells']}
        out = PUBLIC/'freeze.json'
        if phase != 'fit':
            if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(out):
                raise ValueError('Require frozen scores')
            frozen = run.read(out)
            saved = {c['seed']: c for c in frozen['cells']}
        cells, rows = [], []
        for cell in parent:
            m = prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
            if phase == 'fit':
                cells.append(fit(m, controls[m.seed], old[m.seed], motion, y, valid, beat))
            else:
                score(m, records, saved[m.seed], y, valid, rows)
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
