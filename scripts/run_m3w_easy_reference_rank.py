"""Compare learned easy-reference ranking on the identical eligible agents."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_support_joint as previous
from src.world_model import m3w_easy_reference_rank as model

run, np, cost, prune, prior, anchor = (
    previous.run, previous.np, previous.cost, previous.prune, previous.prior, previous.anchor)
PUBLIC, PRIVATE = prior.PUBLIC/'easy_reference_rank', prior.PRIVATE/'easy_reference_rank'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_easy_reference_rank.py')
ARMS = ('old_reference', 'old_joint', 'ridge', 'boosted')
TREE = dict(max_iter=60, max_depth=3, min_samples_leaf=32, learning_rate=.07,
            early_stopping=False, random_state=73107)


def register():
    a = previous.PUBLIC/'analysis.json'
    if run.read(previous.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(a):
        raise ValueError('Require exact preceding replay')
    for v in run.read(previous.REG)['parents']:
        run.checked(v)
    parents = (a, previous.REG, previous.PUBLIC/'calibration_diagnostic.json',
               Path(previous.__file__), cost.REG, Path(cost.model.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], arms=ARMS, tree=TREE, budget=.02,
        independent_outcomes_opened=False)))


def features(m, ix, g, co, rank, motion):
    raw, scale, b, c = m.x[ix], m.scale[ix], m.bank[ix], m.floor[ix]
    u, a, e, legal = run.model.decision_costs(raw, co, b, c, scale)
    w = prior.nested.base.select(g, u, a, e, scale, legal)
    dc, _ = prior.model.stream(g, u/scale[:, None], a, e, legal,
                               counts=[int((w[q] >= 0).sum()) for q in g], witness=w)
    legal &= dc[:, None] < 0
    legal &= m.d['crowd'][ix, None] >= 2
    p = np.einsum('nfd,fdt->nft', raw, co)
    ref = np.minimum(p[..., 3], p[..., 4]).clip(0.)
    joint, _ = anchor.model.predict(raw, rank)
    def best(v):
        return np.where(legal.any(1), np.max(np.where(legal, v, -np.inf), axis=1), -1.)
    ref_best, joint_best = best(ref), best(joint)
    x = np.column_stack((motion[ix], np.log1p(scale), np.log1p(m.d['crowd'][ix]),
                         np.log1p(ref_best.clip(0.)), joint_best,
                         best(np.arcsinh(u/scale[:, None])), best(np.arcsinh(-e/scale[:, None])),
                         np.log1p(legal.sum(1))))
    return dc, legal, x, ref_best, joint_best, u, e


def labels(m, ix, y, valid, cut):
    c = m.floor[ix]
    return prune.model.moments(c, c, y[ix], valid[ix], cut)['ref_easy'] > 0.


def fit(m, controls, motion, y, valid, beat):
    d, cuts = m.d, run.cuts_by_source()
    orig, rank_stats = {}, {}
    for s, ix in m.ix.items():
        x, b, c = m.x[ix], m.bank[ix], m.floor[ix]
        z = m.scale[ix]
        event, _, _ = anchor.model.labels(c, b, y[ix], valid[ix], cuts[s]['easy'])
        legal = np.linalg.norm(b-c[:, None], axis=-1).max(2) > 0.
        orig[s] = run.model.fitting_moments(x, b, c, y[ix], valid[ix], z, 1./d['crowd'][ix], cuts[s]['easy'])
        rank_stats[s] = anchor.model.fitting(x, event, legal, d['crowd'][ix])
    folds = []
    for item in controls:
        r = item['identity']
        s, learn, cal = r['held'], r['fit'], r['cal_a'] + r['cal_b']
        identity = dict(seed=m.seed, held=s, fit=learn, cal=cal, registration=run.sha(REG))
        if (len(learn), len(cal), len(set(learn+cal+[s]))) != (7, 6, 14):
            raise ValueError('Require disjoint source roles')
        mark = PRIVATE/f'{m.seed}_{s}.json'
        if not mark.exists():
            blocks = []
            for source in learn:
                donors = [t for t in learn if t != source]
                co = run.model.coefficients([orig[t] for t in donors])
                rank = anchor.model.fit([rank_stats[t] for t in donors])
                records = []
                for ix, g in m.groups[source]:
                    _, legal, x, *_ = features(m, ix, g, co, rank, motion)
                    ok = legal.any(1)
                    if ok.any():
                        weight = 1./d['crowd'][ix[ok]]
                        records.append((x[ok], labels(m, ix, y, valid, cuts[source]['easy'])[ok],
                                        weight/weight.sum()))
                blocks.extend((x, z, w/len(records)) for x, z, w in records)
            x, target, weight = [np.concatenate([v[j] for v in blocks]) for j in range(3)]
            if not target.any() or target.all():
                raise ValueError('Require both observed support outcomes in fit sources')
            coef = model.ridge(x, target, weight)
            tree = HistGradientBoostingClassifier(**TREE).fit(x, target,
                                                  sample_weight=weight/weight.mean())
            at = m.ix[s]
            scores = np.full((len(at), len(ARMS)), -2., dtype=np.float32)
            mask = np.zeros(len(at), bool)
            rank, co = m.rank(s), m.co[s]
            for ix, g in m.groups[s]:
                dc, legal, x, ref, joint, *_ = features(m, ix, g, co, rank, motion)
                np.testing.assert_array_equal(dc, m.direct[ix])
                j = np.searchsorted(at, ix)
                mask[j] = legal.any(1)
                scores[j] = np.column_stack((ref, joint, np.column_stack((np.ones(len(x)), x)) @ coef,
                                              tree.predict_proba(x)[:, 1]))
            cache = mark.with_suffix('.npz')
            fitted = mark.with_suffix('.joblib')
            partial = mark.with_suffix('.tmp')
            dump(dict(ridge=coef, boosted=tree), partial)
            os.replace(partial, fitted)
            prune.previous.history.feedback.write_array(cache, ids=d['global_ids'][at],
                                                       scores=scores, eligible=mask)
            run.once(mark, run.stable(dict(identity=identity, cache=run.artifact(cache),
                model=run.artifact(fitted), fitting_rows=len(target),
                positive_fitting_rows=int(target.sum()))))
        f = run.read(mark)
        if f['identity'] != identity:
            raise ValueError('Incompatible checkpoint')
        run.checked(f['cache'])
        run.checked(f['model'])
        folds.append(f)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def score(m, records, frozen, motion, y, valid, rows, agent):
    d, cuts = m.d, run.cuts_by_source()
    scores = np.empty((len(m.floor), len(ARMS)), np.float32)
    mask = np.zeros(len(m.floor), bool)
    for f in frozen['folds']:
        at = m.ix[f['identity']['held']]
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            scores[at], mask[at] = z['scores'], z['eligible']
    for rec, (ix, g) in records.items():
        s = str(d['sites'][ix[0]])
        e, h = cuts[s]['easy'], cuts[s]['hard']
        target = labels(m, ix, y, valid, e)
        active = [q[mask[ix[q]]] for q in g if mask[ix[q]].any()]
        if not active:
            raise ValueError('Require full eligible recording roster')
        p = scores[ix]
        hit = np.stack([target[q[p[q].argmax(0)]] for q in active])
        missing = not target[m.direct[ix] >= 0].any()
        status = {}
        if missing:
            dc, legal, _, _, _, u, excess = features(m, ix, g, m.co[s], m.rank(s), motion)
            np.testing.assert_array_equal(mask[ix], legal.any(1))
            x, scale, c, b = m.x[ix], m.scale[ix], m.floor[ix], m.bank[ix]
            _, a, _, _ = run.model.decision_costs(x, m.co[s], b, c, scale)
            empty = np.empty((0, 2), int)
            old = {tuple(q): np.column_stack((i, j)) for q, i, j, _ in
                   cost.model.features(x, scale, u, a, excess, legal, m.rank(s), b, c, g)}
            for k, name in enumerate(ARMS[2:], 2):
                new = model.retrieve(g, legal, p[:, k], excess, u)
                joined = [(q, old.get(tuple(q), empty), v) for q, v in zip(g, new)]
                if any(len(t) != len(v) for _, t, v in joined):
                    raise ValueError('Require equal query candidate budgets')
                status[name] = previous.shortlist_capacity(m, rec, ix, g, y, valid, e, h, (dc, joined))['support']
        rows.append(dict(seed=m.seed, source=s, mean=hit.mean(0), first=hit[0], missing=missing,
                         capacity=status, query_count=len(active)))
        at = np.concatenate(active)
        weight = np.concatenate([np.full(len(q), 1./len(q)) for q in active])
        agent.append((target[at], p[at], weight/weight.sum()))


def report(rows, agent):
    sites = sorted({r['source'] for r in rows})
    missing = [r for r in rows if r['missing']]
    if (len(sites), len(rows), len(missing)) != (14, 102, 35):
        raise ValueError('Require original roster')
    draw = np.random.default_rng(73107).integers(14, size=(3000, 14))
    stats = {}
    for key in ('mean', 'first'):
        x = np.stack([np.mean([r[key] for r in rows if r['source'] == s and r['seed'] == k], axis=0)
                      for s in sites for k in run.kin.prior.SEEDS]).reshape(14, 3, len(ARMS)).mean(1)
        stats[key] = dict(zip(ARMS, x.mean(0).tolist()))
        for j, name in enumerate(ARMS[2:], 2):
            for k, before in enumerate(ARMS[:2]):
                z = x[:, j]-x[:, k]
                stats[f'{key}_{name}_minus_{before}'] = dict(point=float(z.mean()),
                    interval95=np.quantile(z[draw].mean(1), [.025, .975]).tolist())
    y, p, w = [np.concatenate([v[j] for v in agent]) for j in range(3)]
    pooled = {a: dict(auc=float(roc_auc_score(y, p[:, k], sample_weight=w)),
                       ap=float(average_precision_score(y, p[:, k], sample_weight=w)))
              for k, a in enumerate(ARMS)}
    run.once(PUBLIC/'analysis.json', run.stable(dict(views=102, source_groups=14, missing_views=35,
        fitted_heads=84, eligible_agent_query_occurrences=len(y), query_occurrences=sum(r['query_count'] for r in rows),
        pooled_record_query_weighted_prevalence=float(w@y/w.sum()), pooled_metrics=pooled,
        top_one=stats, missing_first_query_hits=dict(zip(ARMS, np.stack([r['first'] for r in missing]).sum(0).tolist())),
        missing_capacity_status={k: dict(Counter(r['capacity'][k] for r in missing)) for k in ARMS[2:]},
        operating_policy_evaluated=False, original_rules_unchanged=True,
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
    with (PRIVATE/'process.lock').open('a') as lock, threadpool_limits(limits=4):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resume from the last completed source checkpoint')
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            print(msg, flush=True)

        beat(state='start')
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = prior.blocks(d, lab['frames'])
        y, valid = lab['target_eval'] - d['origin'][:, None], lab['valid']
        motion = model.motion(d['history'], d['groups'])
        parent, controls, ranks = cost.banks()
        out = PUBLIC/'freeze.json'
        if phase != 'fit':
            if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(out):
                raise ValueError('Require frozen scores')
            frozen = {c['seed']: c for c in run.read(out)['cells']}
        cells, rows, agent = [], [], []
        for cell in parent:
            m = prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
            if phase == 'fit':
                cells.append(fit(m, controls[m.seed], motion, y, valid, beat))
            else:
                score(m, records, frozen[m.seed], motion, y, valid, rows, agent)
            beat(state='seed_complete', seed=m.seed)
            del m
        if phase == 'fit':
            run.once(out, run.stable(dict(cells=cells, held_excluded=True)))
        else:
            report(rows, agent)
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            artifact=run.artifact(out if phase == 'fit' else PUBLIC/'analysis.json')))


if __name__ == '__main__':
    main()
