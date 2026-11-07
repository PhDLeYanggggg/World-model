"""Distill feasible whole-recording decisions into causal path-and-recipient ranks."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_exact_count_joint_capacity as cap
from src.evaluation.m3w_recording_switch_ceiling import matched_count, moments, policy

prev, run, np = cap.prev, cap.run, cap.np
BASE, OLS, SEEDS = cap.BASE, cap.OLS, cap.prev.SEEDS
PUBLIC = cap.PUBLIC.parent/'native_structured_imitation_v1'
PRIVATE = cap.PRIVATE.parent/'native_structured_imitation_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_structured_imitation.py')
MATCHED = (*prev.MATCHED, 'student_nomination', 'student_full')
ARMS = (*MATCHED, 'student_screen')
RIDGE = .01


def register():
    cut = cap.register()
    prev.receipt(cap.PUBLIC, 'replay')
    parents = (cap.REG, cap.PUBLIC/'replay_execution.json', prev.REG, prev.PUBLIC/'freeze_execution.json',
               Path(cap.__file__), Path(matched_count.__code__.co_filename), Path(run.model.__file__), *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in parents], cut=cut, seeds=SEEDS,
        arms=ARMS, ridge=RIDGE, fixed_counts=True, primary='student_full_vs_gain',
        teacher='recording_local_cv_utility_exact_risk', risk=.02, external_access=False)))
    return cut


def dominant_actions(cv, bank, y, valid, cut):
    best = np.full(len(cv), -np.inf)
    action = np.full(len(cv), -1, dtype=np.int16)
    for k in range(bank.shape[1]):
        m = moments(cv, bank[:, k], y, valid, cut)
        take = (m['radius'] > 0.) & (m['utility'] > best)
        action[take], best[take] = k, m['utility'][take]
    return action


def verified_teacher(cv, bank, y, valid, queries, rec, cut):
    a = dominant_actions(cv, bank, y, valid, cut['easy'])
    path = run.model.selected_paths(a, bank, cv)
    m = moments(cv, path, y, valid, cut['easy'])
    _, inv, count = np.unique(queries, return_inverse=True, return_counts=True)
    w = 1./count[inv]
    for seconds in (3., 12.):
        take, status = matched_count(m, queries, w, seconds)
        if status['infeasible']:
            return np.full_like(a, -1), 'infeasible'
        if take is None:
            continue
        pred = np.where(take[:, None, None], path, cv)
        r = policy(pred, cv, cv, y, valid, np.full(len(cv), rec),
                   easy_cut=cut['easy'], hard_cut=cut['hard'])['recordings'][rec]
        if r['supported']:
            np.testing.assert_array_equal(np.bincount(inv[take], minlength=len(count)), np.minimum(2, count-1))
            return np.where(take, a, -1), 'verified'
    return np.full_like(a, -1), 'unresolved'


def features(h, bank, cv, crowd):
    x, _ = run.model.descriptors(h, bank, cv, crowd)
    identity = np.broadcast_to(np.eye(bank.shape[1]), (*bank.shape[:2], bank.shape[1]))
    return np.concatenate((x, identity), axis=-1)


def paired_moments(x, legal, teacher):
    """Return query-averaged normal equations for observed teacher preferences."""
    picked = np.flatnonzero(teacher >= 0)
    if not len(picked):
        return None
    negative = x[legal & (teacher < 0)[:, None]]
    result = []
    for i in picked:
        k = teacher[i]
        if not legal[i, k]:
            raise ValueError('Teacher selected an identity path')
        other = x[i, legal[i] & (np.arange(x.shape[1]) != k)]
        families = [x[i, k]-z for z in (negative, other) if len(z)]
        if families:
            result.append((np.mean([z.T @ z/len(z) for z in families], axis=0),
                           np.mean([z.mean(0) for z in families], axis=0)))
    if not result:
        return None
    return tuple(np.mean([a[k] for a in result], axis=0) for k in (0, 1))


def fit_seed(d, refs, labels, records, cut, seed, beat):
    final = PRIVATE/f'fit_{seed}.json'
    reg = run.sha(REG)
    if not final.exists():
        cv, bank = run.candidates(d, refs, seed)
        y, valid = labels
        teacher = np.full(len(cv), -1, dtype=np.int16)
        states = {}
        for rec, (ix, _) in records.items():
            mark = PRIVATE/f'teacher_{seed}_{rec}.json'
            if not mark.exists():
                a, status = verified_teacher(cv[ix], bank[ix], y[ix], valid[ix], d['queries'][ix], rec, cut)
                run.once(mark, run.stable(dict(registration=reg, status=status, action=a.tolist())))
            doc = run.read(mark)
            if doc['registration'] != reg or len(doc['action']) != len(ix):
                raise ValueError('Stale teacher checkpoint')
            teacher[ix] = doc['action']
            states[doc['status']] = states.get(doc['status'], 0)+1
            beat(seed=seed, completed=sum(states.values()), state='recording_teacher_checked')
        sources = sorted(set(d['sites']))
        cells = {s: [] for s in sources}
        for at in d['groups']:
            if not (teacher[at] >= 0).any():
                continue
            x = features(d['history'][at], bank[at], cv[at], d['crowd'][at])
            legal = (bank[at] != cv[at, None]).any(axis=(2, 3))
            m = paired_moments(x, legal, teacher[at])
            if m is not None:
                cells[d['sites'][at[0]]].append(m)
        counts = {s: len(cells[s]) for s in sources}
        stats = {s: tuple(np.mean([v[k] for v in cells[s]], axis=0) for k in (0, 1))
                 for s in sources if cells[s]}
        fits = {}
        for held in sources:
            train = [stats[s] for s in sources if s != held and s in stats]
            gram, rhs = (np.mean([x[k] for x in train], axis=0) for k in (0, 1))
            fits[held] = np.linalg.solve(gram+RIDGE*np.eye(len(rhs)), rhs).tolist()
        run.once(final, run.stable(dict(registration=reg, seed=seed, fit_sources=sources,
            coefficients=fits, usable_queries=sum(counts.values()), teacher_status=states)))
    item = run.read(final)
    if item['registration'] != reg or item['seed'] != seed:
        raise ValueError('Incompatible student checkpoint')
    beat(seed=seed, state='excluded_source_students_frozen')
    return run.artifact(final)


def choose(scores, legal, k):
    ranked = np.where(legal, scores, -np.inf)
    a = ranked.argmax(1)
    best = ranked[np.arange(len(a)), a]
    rows = np.argsort(-best, kind='stable')[:k]
    if len(rows) != k or not np.isfinite(best[rows]).all():
        raise ValueError('Cannot match the original genuine count')
    result = np.full(len(a), -1, dtype=np.int16)
    result[rows] = a[rows]
    return result


def freeze(d, refs, fits, beat):
    original = prev.receipt(prev.PUBLIC, 'freeze')
    transport = prev.receipt(prev.old.prior.PUBLIC, 'freeze')
    cells = []
    for old, donor, f in zip(original['cells'], transport['cells'], fits['cells'], strict=True):
        head = run.read(run.checked(f))
        seed = head['seed']
        if seed != old['seed'] or seed != donor['seed'] or head['registration'] != run.sha(REG):
            raise ValueError('Mixed seed or fit provenance')
        p = PRIVATE/f'choices_{seed}.npz'
        mark = p.with_suffix('.json')
        if not mark.exists():
            cv, bank = run.candidates(d, refs, seed)
            with np.load(run.checked(old['artifact']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'])
                original_choices, nominee = z['choice'][:3], z['nominee']
            with np.load(run.checked(donor['cache']), allow_pickle=False) as z:
                costs = dict(zip(z['sources'], z['excluded_source_coefficients']))
            choices = np.full((len(ARMS), len(cv)), -1, dtype=np.int16)
            choices[:3] = original_choices
            for at in d['groups']:
                s = str(d['sites'][at[0]])
                h, b, c, crowd = d['history'][at], bank[at], cv[at], d['crowd'][at]
                x = features(h, b, c, crowd)
                score = x @ np.asarray(head['coefficients'][s])
                f, scale = run.model.descriptors(h, b, c, crowd)
                u, a, e, legal = run.model.decision_costs(f, costs[s], b, c, scale)
                k = int((choices[0, at] >= 0).sum())
                choices[3, at] = choose(score, legal & (np.arange(b.shape[1]) == nominee[at, None]), k)
                full = choose(score, legal, k)
                choices[4, at] = full
                cost = np.einsum('nad,adc->nac', f, costs[s])
                ref = np.minimum(cost[..., 4].clip(0.), cost[..., 3].clip(0.))*scale[:, None]
                choices[5, at] = prev.old.prior.model.gate(full, u, a, e, ref)
            tmp = p.with_suffix('.pending.npz')
            np.savez(tmp, ids=d['global_ids'], choices=choices)
            tmp.replace(p)
            run.once(mark, dict(registration=run.sha(REG), seed=seed, artifact=run.artifact(p)))
        item = run.read(mark)
        if item['registration'] != run.sha(REG) or item['seed'] != seed:
            raise ValueError('Incompatible choices')
        run.checked(item['artifact'])
        cells.append(item)
        beat(seed=seed, state='past_only_choices_frozen')
    path = PRIVATE/'freeze.json'
    run.once(path, run.stable(dict(registration=run.sha(REG), cells=cells, score_labels_read=False)))
    return path


def score(d, refs, lab, frozen, cut, beat):
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    contrasts = {f'{a}_vs_{b}': (b, a, BASE) for a in ARMS for b in (BASE, OLS)}
    for a in ('student_full', 'student_nomination'):
        for b in prev.MATCHED:
            contrasts[f'{a}_vs_{b}'] = (b, a, BASE)
    contrasts['student_full_vs_student_nomination'] = ('student_nomination', 'student_full', BASE)
    rows, relative = [], {}
    groups = sorted(set(roster.values()))
    for f in frozen['cells']:
        seed = f['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(f['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            choice = z['choices']
        cells = {s: [] for s in groups}
        for rec, (ix, queries) in records.items():
            paths = {BASE: cv[ix], OLS: bank[ix, run.NAMES.index(OLS)],
                     **{a: run.model.selected_paths(c[ix], bank[ix], cv[ix]) for a, c in zip(ARMS, choice)}}
            changed = np.stack([(paths[a] != cv[ix]).any(axis=(1, 2)) for a in MATCHED])
            for at in queries:
                n = changed[:, at].sum(1)
                if not np.all(n == n[0]):
                    raise ValueError('Genuine counts do not match')
                p = np.stack([paths[a][at] for a in ('gain', 'student_full')])
                _, value, _, _ = prev.pair.query(d['history'][ix[at]], p, y[ix[at]], valid[ix[at]], [(0, 1)])
                cells[roster[rec]].append(value[0])
            row = run.parent.readout.score_recording(paths, contrasts, y[ix], valid[ix], d['queries'][ix],
                                                      rec, easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, seed=seed, source=roster[rec]))
        for s, v in cells.items():
            relative[s, seed] = prev.pair.mean_finite(v)
        beat(seed=seed, state='frozen_predictions_scored')
    agg = prev.old.prior.grouped_readout(rows, roster, SEEDS)
    old = run.read(prev.PUBLIC/'analysis.json')['grouped']['risk']
    if any(agg['risk'][a] != old[a] for a in prev.MATCHED):
        raise ValueError('Matched operating controls did not replay')
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    values = np.asarray([[relative[s, seed] for seed in SEEDS] for s in groups])
    rel = {key: prev.pair.bootstrap(values[:, :, i], draw) for i, key in enumerate(prev.pair.PAIR_FIELDS)}
    return dict(grouped=agg, relative_pair=rel, external_access=False, independent_confirmation=False, deployment=False,
                stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'fit', 'freeze', 'score', 'replay'))
    phase = p.parse_args().phase
    cut = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the training design first')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    t = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-t, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', event)
            if v.get('completed', 34) % 10 == 0 or v.get('completed', 34) == 34:
                print(event, flush=True)

        d, refs = run.population(), run.register()
        if phase == 'fit':
            lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
            records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
            labels = (lab['target_eval']-d['origin'][:, None], lab['valid'])
            cells = [fit_seed(d, refs, labels, records, cut, seed, beat) for seed in SEEDS]
            out = PRIVATE/'fit.json'
            run.once(out, run.stable(dict(registration=run.sha(REG), cells=cells)))
        elif phase == 'freeze':
            fits = prev.receipt(PUBLIC, 'fit')
            if fits['registration'] != run.sha(REG):
                raise ValueError('Student fit does not match')
            out = freeze(d, refs, fits, beat)
        else:
            frozen = prev.receipt(PUBLIC, 'freeze')
            if frozen['registration'] != run.sha(REG) or frozen['score_labels_read']:
                raise ValueError('Require pre-score decision freeze')
            if phase == 'replay':
                prev.receipt(PUBLIC, 'score')
            lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(d, refs, lab, frozen, cut, beat)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-t, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Replay differs from registered result')


if __name__ == '__main__':
    main()
