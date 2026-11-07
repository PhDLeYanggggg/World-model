"""Compare point and exact recording-mass fits with chronological decisions."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_postsearch_risk as prior
from src.world_model import m3w_postsearch_mass as model

run, np, cal = prior.run, prior.np, prior.cal
retrieval = prior.prior.prev.prior
PUBLIC = prior.PUBLIC.parent/'native_postsearch_mass_v1'
PRIVATE = prior.PRIVATE.parent/'native_postsearch_mass_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_postsearch_mass.py')
ARMS = ('screen_first', 'point_mass', 'record_mass', 'matched_point')
BASE, OLS, SEEDS = prior.BASE, prior.OLS, prior.SEEDS


def register():
    cut = prior.register()
    run.checked(run.read(prior.PUBLIC/'replay_execution.json')['artifact'])
    parents = (prior.REG, prior.PUBLIC/'analysis.json', prior.PUBLIC/'replay_execution.json',
               Path(prior.model.__file__), Path(prior.risk.__file__), *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in parents], cut=cut, seeds=SEEDS,
        roles=[7, 6, 1], factors=model.FACTORS, original_scoring_budget=.02,
        primary='record_mass_vs_matched_point', secondary='recording_excess_error_on_fixed_search',
        original_rule=True, zero_initial_expected_credit=True, external_access=False)))
    return cut


def features(rows, frames, scale):
    return (np.stack([r[2] for r in rows]), np.array([scale[r[0]].sum() for r in rows]),
            np.array([r[4] for r in rows]), np.array([frames[r[0][0]] for r in rows]))


def targets(rows, cv, bank, y, valid, cut):
    return np.stack([model.targets(cv[i], bank[i, a], y[i], valid[i], cut['easy'])
                     for i, a, *_ in rows])


def choose_factor(rows, predicted, rec, frame, donor_records, d, cv, ols, bank, y, valid, cut):
    read = run.parent.readout
    candidates = []
    for f in model.FACTORS:
        z = prior.render(rows, model.ledger(predicted, rec, frame, f), len(cv))
        used = int((z >= 0).sum())
        if not used:
            continue
        pass_count = violations = 0
        gains = []
        for r, ix in donor_records:
            p = run.model.selected_paths(z[ix], bank[ix], cv[ix])
            g = read.policy(p, cv[ix], cv[ix], y[ix], valid[ix], np.full(len(ix), r),
                            easy_cut=cut['easy'], hard_cut=cut['hard'])['recordings'][r]
            pass_count += int(g['supported'])
            risk = g['completion_all_risk_upper']
            violations += int(risk is not None and risk > .02)+int(g['completion_easy_risk_upper'] is None)
            gains.append(read.comparison_masses(ols[ix], p, cv[ix], y[ix], valid[ix], 1./d['crowd'][ix]))
        gain = read.merge_comparisons(gains)['completion_lower_pp']
        candidates.append(((-pass_count, violations, -gain, used, f), f,
                           dict(passes=pass_count, required=len(donor_records), switches=used)))
    return min(candidates)[1:] if candidates else (None, dict(passes=0, required=len(donor_records), switches=0))


def fit_seed(d, refs, lab, records, cut, seed, specs, beat):
    y, valid, frames = (lab['target_eval']-d['origin'][:, None], lab['valid'], lab['frames'])
    cv, ols, bank = retrieval.bank_for(d, refs, seed)
    site, crowd, h = (d[k] for k in ('sites', 'crowd', 'history'))
    g = {s: [] for s in sorted(set(site))}
    for q in d['groups']:
        g[site[q[0]]].append(q)
    x = retrieval.model.features(h, bank, cv, ols, d['groups'], crowd)
    scale = 1.+6.5*np.median(np.linalg.norm(np.diff(h, axis=1), axis=-1), axis=1)
    gain = retrieval.model.opportunity(cv, bank, y, valid, scale, d['groups'],
                                       cut['easy'], np.zeros(len(d['groups']), int))[:, 0]
    witness = np.zeros(len(site))
    for r, (ix, groups) in records.items():
        loc = ix[np.concatenate(groups)]
        z = run.read(prior.prior.prev.PRIVATE/f'teacher_{seed}_{r}.json')
        if z['registration'] != run.sha(prior.prior.prev.REG) or z['count'] != len(loc):
            raise ValueError('Stale source-local witness')
        witness[loc[np.asarray(z['selected'], int)]] = 1.
    m, cost = {}, {}
    for s in g:
        i = np.flatnonzero(site == s)
        m[s] = retrieval.model.moments(x[i], np.column_stack((gain[i], witness[i])), 1./crowd[i])
        z, sc = run.model.descriptors(h[i], bank[i], cv[i], crowd[i])
        cost[s] = run.model.fitting_moments(z, bank[i], cv[i], y[i], valid[i], sc, 1./crowd[i], cut['easy'])

    def propose(train, apply):
        if set(train)&set(apply):
            raise ValueError('A source may not fit its own search')
        cb = retrieval.model.coefficients([m[s] for s in train])
        co = run.model.coefficients([cost[s] for s in train])
        return prior.choices([q for s in apply for q in g[s]], d, cv, ols, bank, x, cb, co)

    packets = []
    for cell in specs:
        p = cell['identity']
        held, learn, donor = p['held'], p['fit'], p['cal_a']+p['cal_b']
        if (len(learn), len(donor), len(set(learn+donor+[held]))) != (7, 6, 14):
            raise ValueError('Source roles must be disjoint')
        mark = PRIVATE/f'{seed}_{held}.json'
        ident = dict(registration=run.sha(REG), seed=seed, held=held, fit=learn, donors=donor)
        if not mark.exists():
            inner = [r for s in learn for r in propose([k for k in learn if k != s], [s])]
            tx, ts, tr, _ = features(inner, frames, scale)
            coef = model.fit(tx, targets(inner, cv, bank, y, valid, cut), ts, tr, np.array([r[-1] for r in inner]))
            rows = propose(learn, donor)
            dr = [(r, i) for r, (i, _) in records.items() if site[i[0]] in donor]
            if rows:
                vx, vs, vr, vf = features(rows, frames, scale)
                selected = [choose_factor(rows, model.predict(vx, vs, b), vr, vf, dr,
                                           d, cv, ols, bank, y, valid, cut) for b in coef]
            else:
                selected = [(None, dict(passes=0, required=len(dr), switches=0))]*2
            rows = propose(learn, [held])
            at = np.flatnonzero(site == held)
            chosen = np.full((len(ARMS), len(at)), -1, dtype=np.int16)
            pred, sc, anchor = np.empty((2, 0, 4)), np.empty(0), np.empty(0, int)
            if rows:
                hx, sc, hr, hf = features(rows, frames, scale)
                pred = np.stack([model.predict(hx, sc, b) for b in coef])
                mask = [model.ledger(a, hr, hf, f) if f is not None else np.zeros(len(rows), bool)
                        for a, (f, _) in zip(pred, selected)]
                pscore = -model.excess(pred[0]).max(1)/sc
                match = prior.model.rank_count(pscore, mask[1], hr, np.array([len(r[0]) for r in rows]))
                chosen = np.stack([prior.render(rows, u, len(site))[at] for u in (np.ones(len(rows), bool), *mask, match)])
                anchor = np.array([r[0][0] for r in rows])
            out, tmp = mark.with_suffix('.npz'), mark.with_suffix('.pending.npz')
            np.savez(tmp, ids=at, choices=chosen, masses=pred, anchor=anchor, scale=sc, coef=coef)
            tmp.replace(out)
            run.once(mark, run.stable(dict(identity=ident, artifact=run.artifact(out),
                factors=[f for f, _ in selected], donor=[s for _, s in selected], fitting_queries=len(inner),
                held_search_queries=len(rows))))
        saved = run.read(mark)
        if saved['identity'] != ident:
            raise ValueError('Incompatible source checkpoint')
        run.checked(saved['artifact'])
        packets.append(saved)
        beat(seed=seed, completed=len(packets), state='outer_fold_frozen')
    return packets


def score(d, refs, lab, records, cut, cells, beat):
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    out, roster, errors = [], {}, {}
    pairs = {f'{a}_vs_{b}': (b, a, BASE) for a in ARMS for b in (BASE, OLS)}
    pairs.update(record_mass_vs_matched_point=('matched_point', 'record_mass', BASE),
                 record_mass_vs_point_mass=('point_mass', 'record_mass', BASE))
    for seed in SEEDS:
        cv, ols, bank = retrieval.bank_for(d, refs, seed)
        qmap = {i: q for q in d['groups'] for i in q}
        z = np.full((len(ARMS), len(cv)), -1, dtype=np.int16)
        for c in (v for v in cells if v['identity']['seed'] == seed):
            s = c['identity']['held']
            at = np.flatnonzero(d['sites'] == s)
            with np.load(run.checked(c['artifact']), allow_pickle=False) as f:
                np.testing.assert_array_equal(at, f['ids'])
                z[:, at] = f['choices']
                anchors = f['anchor']
                if not len(anchors):
                    continue
                meta = []
                for i in anchors:
                    q = qmap[i]
                    rows = q[z[0, q] >= 0]
                    meta.append((rows, z[0, rows]))
                true = targets(meta, cv, bank, y, valid, cut)
                group = d['recordings'][anchors]
                for r in np.unique(group):
                    j = group == r
                    m = f['masses'][:, j].sum(1)
                    err = (model.excess(m)-model.excess(true[j].sum(0, keepdims=True)))/f['scale'][j].sum()
                    errors.setdefault((s, seed), []).append(err**2)
        for r, (ix, _) in records.items():
            s = str(d['sites'][ix[0]])
            roster[r] = s
            if (z[2, ix] >= 0).sum() != (z[3, ix] >= 0).sum():
                raise ValueError('Matched switch count changed')
            p = {k: run.model.selected_paths(v[ix], bank[ix], cv[ix]) for k, v in zip(ARMS, z)}
            data = run.parent.readout.score_recording({BASE: cv[ix], OLS: ols[ix], **p}, pairs,
                y[ix], valid[ix], d['queries'][ix], r, easy_cut=cut['easy'], hard_cut=cut['hard'])
            out.append(dict(data, seed=seed, source=s))
        beat(seed=seed, state='scored')
    result = retrieval.grouped.grouped_readout(out, roster, SEEDS)
    if result['risk']['screen_first'] != run.read(prior.PUBLIC/'analysis.json')['grouped']['risk']['screen_first']:
        raise ValueError('The same nested screen-first control did not replay')
    src = sorted({s for s, _ in errors})
    vals = np.stack([np.mean([np.mean(v, axis=0) for (a, _), v in errors.items() if a == s], axis=0)
                     for s in src])
    draw = np.random.default_rng(73107).integers(len(src), size=(3000, len(src)))
    change = vals[:, 1]-vals[:, 0]
    return dict(grouped=result, fit_cells=len(cells), normalized_excess_mse_point_record=vals.mean(0).tolist(),
        excess_mse_change_record_vs_point=change.mean(0).tolist(),
        excess_mse_change_interval95=np.quantile(change[draw].mean(1), [.025, .975], axis=0).T.tolist(),
        excess_mse_source_seed_cells=len(errors), excess_mse_sources=len(src),
        excess_mse_recording_views=sum(map(len, errors.values())),
        factor_counts={n: dict(Counter(str(c['factors'][i]) for c in cells)) for i, n in enumerate(ARMS[1:3])},
        donor_all_pass={n: sum(c['donor'][i]['passes'] == c['donor'][i]['required']
                               and c['donor'][i]['switches'] > 0 for c in cells) for i, n in enumerate(ARMS[1:3])},
        external_data_opened=False, independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = p.parse_args().phase
    cut = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit mass comparison before executing')
    start = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            print(row, flush=True)

        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = cal.prior.h.previous.prior.blocks(d, lab['frames'])
        cache = PRIVATE/'freeze.json'
        if phase == 'fit':
            _, specs, _ = prior.roles.h.cost.banks()
            cells = [c for seed in SEEDS for c in fit_seed(d, refs, lab, records, cut, seed, specs[seed], beat)]
            if len(cells) != 42:
                raise ValueError('Require all nested folds')
            run.once(cache, run.stable(dict(cells=cells)))
            out = cache
        else:
            run.checked(run.read(PUBLIC/'fit_execution.json')['artifact'])
            if phase == 'replay':
                run.checked(run.read(PUBLIC/'score_execution.json')['artifact'])
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(d, refs, lab, records, cut, run.read(cache)['cells'], beat)))
        receipt = PUBLIC/f'{phase}_execution.json'
        if not receipt.exists():
            run.once(receipt, dict(artifact=run.artifact(out), seconds=time.monotonic()-start, pid=os.getpid()))
        elif run.read(receipt)['artifact'] != run.artifact(out):
            raise ValueError('Replayed mass result differs')


if __name__ == '__main__':
    main()
