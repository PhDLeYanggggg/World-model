"""Fit post-search risk on inner-fold choices and select on disjoint donors."""
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
from scripts import run_m3w_joint_path_portfolio as prior
from scripts import run_m3w_common_motion as roles
from src.evaluation import m3w_recording_switch_ceiling as risk
from src.world_model import m3w_postsearch_risk as model

run, np, cal = prior.run, prior.np, prior.cal
PUBLIC = prior.PUBLIC.parent/'native_postsearch_risk_v1'
PRIVATE = prior.PRIVATE.parent/'native_postsearch_risk_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_postsearch_risk.py')
ARMS = ('screen_first', 'margin', 'learned', 'matched_margin')
BASE, OLS, SEEDS = prior.BASE, prior.OLS, prior.SEEDS


def register():
    cut, _, _ = prior.register()
    run.checked(run.read(prior.PUBLIC/'replay_execution.json')['artifact'])
    parents = (prior.REG, prior.PUBLIC/'analysis.json', prior.PUBLIC/'replay_execution.json',
               roles.h.cost.anchor.SIZE.PUBLIC/'freeze.json', Path(risk.__file__), *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in parents], cut=cut, seeds=SEEDS,
        roles=[7, 6, 1], quantiles=model.QUANTILES, primary='learned_vs_matched_margin',
        candidate_fitting='inner_six_of_seven_then_refit_seven',
        threshold_rule='donor_supported_then_violation_count_then_ols4_lower_then_smaller_switch_count',
        postsearch_targets=['local_all_and_easy_excess_nonpositive', 'easy_reference_present'],
        external_access=False, original_risk_rule=True)))
    return cut


def choices(queries, d, cv, ols, bank, x, cb, costs):
    h, crowd, rec, src = (d[k] for k in ('history', 'crowd', 'recordings', 'sites'))
    rows = []
    for q in queries:
        f, z = run.model.descriptors(h[q], bank[q], cv[q], crowd[q])
        u, a, e, legal = run.model.decision_costs(f, costs, bank[q], cv[q], z)
        cap = min(2, len(q)-1, int(legal[:, :-1].sum(0).max()))
        if not cap:
            continue
        keep = prior.prev.prior.model.lists(x[q]@cb, legal)[1]
        mask = np.isin(np.arange(len(q)), keep)
        p = np.einsum('nad,adc->nac', f, costs)
        support = np.minimum(p[..., 4].clip(0.), p[..., 3].clip(0.))*z[:, None]
        val = model.search(h[q], cv[q], ols[q], bank[q], z, u, a, e, support, legal & mask[:, None], cap)
        if val is not None:
            i, k, xq, slack = val
            rows.append((q[i], k, xq, slack, rec[q[0]], src[q[0]]))
    return rows


def local_targets(choices, cv, bank, y, valid, cut):
    result = []
    for i, a, *_ in choices:
        m = risk.moments(cv[i], bank[i, a], y[i], valid[i], cut['easy'])
        result.append((m['all_excess'].sum() <= 0. and m['easy_excess'].sum() <= 0.,
                       m['ref_easy'].sum() > 0.))
    return np.asarray(result, dtype=float).reshape(-1, 2)


def render(rows, mask, n):
    out = np.full(n, -1, dtype=np.int16)
    for use, (i, a, *_) in zip(mask, rows, strict=True):
        if use:
            out[i] = a
    return out


def select_threshold(rows, score, ladder, donor_records, d, cv, ols, bank, y, valid, cut):
    readout = run.parent.readout
    standings = []
    for j, threshold in enumerate(ladder):
        choice = render(rows, score >= threshold, len(cv))
        switches = int((choice >= 0).sum())
        if switches == 0:
            continue
        counts = np.zeros(3, int)
        gain = []
        for r, ix in donor_records:
            p = run.model.selected_paths(choice[ix], bank[ix], cv[ix])
            m = readout.policy(p, cv[ix], cv[ix], y[ix], valid[ix], np.full(len(ix), r),
                               easy_cut=cut['easy'], hard_cut=cut['hard'])['recordings'][r]
            g = readout.comparison_masses(ols[ix], p, cv[ix], y[ix], valid[ix], 1./d['crowd'][ix])
            counts += [int(m['supported']), int(m['completion_all_risk_upper'] is not None
                       and m['completion_all_risk_upper'] > .02), int(m['completion_easy_risk_upper'] is None)]
            gain.append(g)
        value = readout.merge_comparisons(gain)['completion_lower_pp']
        standings.append(((-counts[0], counts[1]+counts[2], -value, switches, j), j,
                          dict(supported=int(counts[0]), views=len(donor_records), switches=switches)))
    if not standings:
        raise ValueError('Nonempty control should appear on the threshold ladder')
    _, j, m = min(standings)
    return j, m


def fit_seed(d, refs, labels, records, cut, seed, specs, beat):
    cv, ols, bank = prior.prev.prior.bank_for(d, refs, seed)
    y, valid = labels
    site, crowd, h = (d[k] for k in ('sites', 'crowd', 'history'))
    g = {s: [] for s in sorted(set(site))}
    for q in d['groups']:
        g[site[q[0]]].append(q)
    feature = prior.prev.prior.model.features(h, bank, cv, ols, d['groups'], crowd)
    witness = np.zeros(len(site))
    for r, (ix, queries) in records.items():
        loc = ix[np.concatenate(queries)]
        z = run.read(prior.prev.PRIVATE/f'teacher_{seed}_{r}.json')
        if z['registration'] != run.sha(prior.prev.REG) or z['count'] != len(loc):
            raise ValueError('Stale source-local teacher')
        witness[loc[np.asarray(z['selected'], int)]] = 1.
    scale = 1.+6.5*np.median(np.linalg.norm(np.diff(h, axis=1), axis=-1), axis=1)
    plain = prior.prev.prior.model.opportunity(cv, bank, y, valid, scale, d['groups'],
                                               cut['easy'], np.zeros(len(d['groups']), int))[:, 0]
    m, cost = {}, {}
    for s in g:
        i = np.flatnonzero(site == s)
        m[s] = prior.prev.prior.model.moments(feature[i], np.column_stack((plain[i], witness[i])), 1./crowd[i])
        z, sc = run.model.descriptors(h[i], bank[i], cv[i], crowd[i])
        cost[s] = run.model.fitting_moments(z, bank[i], cv[i], y[i], valid[i], sc, 1./crowd[i], cut['easy'])

    def propose(train, test):
        if set(train)&set(test):
            raise ValueError('A search cannot fit on its scoring source')
        cb = prior.prev.prior.model.coefficients([m[s] for s in train])
        costs = run.model.coefficients([cost[s] for s in train])
        return choices([q for s in test for q in g[s]], d, cv, ols, bank, feature, cb, costs)

    packets = []
    for cell in specs:
        x = cell['identity']
        held, learn, donor = x['held'], x['fit'], x['cal_a']+x['cal_b']
        if (len(learn), len(donor), len(set([held]+learn+donor))) != (7, 6, 14):
            raise ValueError('Require disjoint 7/6/1 source roles')
        out = PRIVATE/f'{seed}_{held}.json'
        ident = dict(registration=run.sha(REG), seed=seed, held=held, train=learn, donor=donor)
        if not out.exists():
            inner = [z for s in learn for z in propose([k for k in learn if k != s], [s])]
            ix, iy = np.stack([z[2] for z in inner]), local_targets(inner, cv, bank, y, valid, cut)
            sources = np.array([z[-1] for z in inner])
            coef = model.fit(ix, iy, sources)
            prior_rate = np.stack([iy[sources == s].mean(0) for s in sorted(set(sources))]).mean(0)
            thresholds = (model.levels(np.array([z[3] for z in inner])),
                          model.levels(model.learned_score(ix, coef)[0]))
            cal_rows = propose(learn, donor)
            if cal_rows:
                cal_x = np.stack([z[2] for z in cal_rows])
                v = (np.array([z[3] for z in cal_rows]), model.learned_score(cal_x, coef)[0])
                rec = [(r, i) for r, (i, _) in records.items() if site[i[0]] in donor]
                found = [select_threshold(cal_rows, a, b, rec, d, cv, ols, bank, y, valid, cut)
                         for a, b in zip(v, thresholds)]
            else:
                found = [(0, dict(supported=0, switches=0)), (0, dict(supported=0, switches=0))]
            held_rows = propose(learn, [held])
            held_idx = np.flatnonzero(site == held)
            if held_rows:
                xs = np.stack([z[2] for z in held_rows])
                learning, prob = model.learned_score(xs, coef)
                margin = np.array([z[3] for z in held_rows])
                selected = [s >= t[j] for s, t, (j, _) in zip((margin, learning), thresholds, found)]
                match = model.rank_count(margin, selected[1], np.array([z[4] for z in held_rows]),
                                         np.array([len(z[0]) for z in held_rows]))
                keep = (np.ones(len(margin), bool), *selected, match)
                chosen = np.stack([render(held_rows, k, len(site))[held_idx] for k in keep])
                anchors = np.array([z[0][0] for z in held_rows])
            else:
                chosen = np.full((len(ARMS), len(held_idx)), -1, dtype=np.int16)
                anchors, prob = np.empty(0, int), np.empty((0, 2))
            dest, tmp = out.with_suffix('.npz'), out.with_suffix('.pending.npz')
            np.savez(tmp, ids=held_idx, choices=chosen, anchor=anchors, prob=prob, prior_rate=prior_rate, coef=coef)
            tmp.replace(dest)
            run.once(out, run.stable(dict(identity=ident, artifact=run.artifact(dest),
                threshold_indices=[j for j, _ in found], donor=[v for _, v in found],
                fit_queries=len(inner), validation_queries=len(cal_rows), held_queries=len(held_rows))))
        saved = run.read(out)
        if saved['identity'] != ident:
            raise ValueError('Stale source fold')
        run.checked(saved['artifact'])
        packets.append(saved)
        beat(seed=seed, completed=len(packets), state='nested_fold_frozen')
    return packets


def score(d, refs, labels, records, cut, cells, beat):
    rows, roster, calib = [], {}, {}
    pairs = {f'{n}_vs_{a}': (a, n, BASE) for n in ARMS for a in (BASE, OLS)}
    pairs.update(learned_vs_margin=('margin', 'learned', BASE),
                 learned_vs_matched_margin=('matched_margin', 'learned', BASE))
    y, valid = labels
    for seed in SEEDS:
        cv, ols, bank = prior.prev.prior.bank_for(d, refs, seed)
        chosen = np.full((len(ARMS), len(cv)), -1, dtype=np.int16)
        qmap = {i: q for q in d['groups'] for i in q}
        for c in (v for v in cells if v['identity']['seed'] == seed):
            held = c['identity']['held']
            with np.load(run.checked(c['artifact']), allow_pickle=False) as f:
                ix = np.flatnonzero(d['sites'] == held)
                np.testing.assert_array_equal(f['ids'], ix)
                chosen[:, ix] = f['choices']
                target_rows = []
                for i in f['anchor']:
                    q = qmap[i]
                    j = q[chosen[0, q] >= 0]
                    target_rows.append((j, chosen[0, j]))
                target = local_targets(target_rows, cv, bank, y, valid, cut)
                if len(target):
                    calib[held, seed] = np.stack(((f['prob']-target)**2, (f['prior_rate']-target)**2)).mean(1)
        for r, (ix, _) in records.items():
            s = str(d['sites'][ix[0]])
            roster[r] = s
            if (chosen[2, ix] >= 0).sum() != (chosen[3, ix] >= 0).sum():
                raise ValueError('Matched recording switch budget changed')
            paths = {k: run.model.selected_paths(v[ix], bank[ix], cv[ix]) for k, v in zip(ARMS, chosen)}
            val = run.parent.readout.score_recording({BASE: cv[ix], OLS: ols[ix], **paths}, pairs,
                y[ix], valid[ix], d['queries'][ix], r, easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(val, seed=seed, source=s))
        beat(seed=seed, state='held_scoring_complete')
    result = prior.prev.prior.grouped.grouped_readout(rows, roster, SEEDS)
    common = set(roster.values())
    data = [np.mean([v for (name, _), v in calib.items() if name == s], axis=0)
            for s in sorted(common) if any(name == s for name, _ in calib)]
    return dict(grouped=result, brier_sources=len(data), brier_source_seed_cells=len(calib),
        brier_estimated_and_fit_prevalence=np.mean(data, axis=0).tolist() if data else None,
        threshold_indices={arm: dict(Counter(str(v['threshold_indices'][i]) for v in cells))
                           for i, arm in enumerate(ARMS[1:3])},
        donor_all_views_pass={arm: sum(d['supported'] > 0 and d['supported'] == d['views']
             for v in cells for d in [v['donor'][i]]) for i, arm in enumerate(ARMS[1:3])},
        outer_folds=len(cells), external_access=False, independent_confirmation=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = p.parse_args().phase
    cut = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit post-search protocol before fitting')
    start = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            print(row, flush=True)

        d, refs = run.population(), run.register()
        z = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = cal.prior.h.previous.prior.blocks(d, z['frames'])
        labels = (z['target_eval']-d['origin'][:, None], z['valid'])
        cache = PRIVATE/'freeze.json'
        if phase == 'fit':
            _, specs, _ = roles.h.cost.banks()
            cells = [c for seed in SEEDS for c in fit_seed(d, refs, labels, records, cut, seed, specs[seed], beat)]
            if len(cells) != 42:
                raise ValueError('Require every fold')
            run.once(cache, run.stable(dict(cells=cells)))
            out = cache
        else:
            run.checked(run.read(PUBLIC/'fit_execution.json')['artifact'])
            if phase == 'replay':
                run.checked(run.read(PUBLIC/'score_execution.json')['artifact'])
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(d, refs, labels, records, cut, run.read(cache)['cells'], beat)))
        receipt = PUBLIC/f'{phase}_execution.json'
        if not receipt.exists():
            run.once(receipt, dict(artifact=run.artifact(out), seconds=time.monotonic()-start, pid=os.getpid()))
        elif run.read(receipt)['artifact'] != run.artifact(out):
            raise ValueError('Replay differs from original score')


if __name__ == '__main__':
    main()
