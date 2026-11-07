"""Isolate label-availability modeling at the same neural replacement counts."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_direct_neural_increment as prev

run, np, pair = prev.run, prev.np, prev.pair
BASE, OLS, MENU, LEARNED, SEEDS = prev.BASE, prev.OLS, prev.MENU, prev.LEARNED, prev.SEEDS
PUBLIC = prev.PUBLIC.parent/'native_visibility_increment_v1'
PRIVATE = prev.PRIVATE.parent/'native_visibility_increment_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_visibility_increment.py')
ARMS = ('anchor', 'cv_difference', 'direct_matched', 'minimum_disagreement', 'constant_unknown', 'fitted_unknown')


def receipt(folder, name):
    return run.read(run.checked(run.read(folder/f'{name}_execution.json')['artifact']))


def register():
    cut = prev.register()
    for name in ('fit', 'freeze', 'replay'):
        run.checked(run.read(prev.PUBLIC/f'{name}_execution.json')['artifact'])
    refs = (prev.REG, Path(prev.__file__), prev.PUBLIC/'fit_execution.json',
            prev.PUBLIC/'freeze_execution.json', prev.PUBLIC/'replay_execution.json', *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in refs], cut=cut, arms=ARMS, seeds=SEEDS,
        primary='fitted_unknown_vs_constant_unknown', count='frozen_cv_difference_query_neural_count',
        fit_source_roles=[1, 13, 12], ridge=.002, bootstrap_draws=3000, risk=.02, external_access=False)))
    return cut


def factorized(gain, radius, probability):
    p = np.clip(probability, 0., 1.)[:, None]
    return (1.-p)*np.clip(gain, -radius, radius)-p*radius


def rank(anchor, legal, scores, k):
    """Replace exactly k distinct anchor paths with genuine neural paths."""
    value = np.where(legal & (anchor >= 0)[:, None], scores, -np.inf)
    best = value.argmax(1)
    top = value[np.arange(len(best)), best]
    at = np.argsort(-top, kind='stable')[:k]
    if k < 0 or len(at) != k or not np.isfinite(top[at]).all():
        raise ValueError('Matched neural replacement count is not attainable')
    out = anchor.copy()
    out[at] = LEARNED[best[at]]
    return out


def covariates(h, cv, anchor, crowd):
    x, _ = run.model.descriptors(h, anchor[:, None], cv, crowd)
    return x


def train_seed(d, refs, y, valid, cut, cell, beat):
    seed = cell['seed']
    cv, bank = run.candidates(d, refs, seed)
    sources = sorted(prev.coeffs(cell))
    groups = {s: [] for s in sources}
    for at in d['groups']:
        groups[d['sites'][at[0]]].append(at)
    stats = {}
    for s in sources:
        at = np.flatnonzero(d['sites'] == s)
        b = bank[at][:, MENU]
        x, scale = run.model.descriptors(d['history'][at], b, cv[at], d['crowd'][at])
        stats[s] = run.model.fitting_moments(x, b, cv[at], y[at], valid[at], scale,
                                            1./d['crowd'][at], cut['easy'])
    oi = run.NAMES.index(OLS)
    fits = []
    for i, held in enumerate(sources):
        mark = PRIVATE/f'fit_{seed}_{i:02}.json'
        if not mark.exists():
            conditional, missing, rates = [], [], []
            for donor in sources:
                if donor == held:
                    continue
                coef = run.model.coefficients([stats[s] for s in prev.fitting_sources(sources, held, donor)])
                rows, actions = [], []
                for at in groups[donor]:
                    a = prev.prior.prior.model.decide(d['history'][at], bank[at][:, MENU], cv[at],
                                                     bank[at, oi], d['crowd'][at], coef)['joint']
                    take = a >= 0
                    if take.any():
                        rows.append(at[take])
                        actions.append(MENU[a[take]])
                at, a = np.concatenate(rows), np.concatenate(actions)
                if not (d['sites'][at] == donor).all():
                    raise ValueError('Misaligned donor rows')
                h, crowd = d['history'][at], d['crowd'][at]
                anchor, cand = bank[at, a], bank[at][:, LEARNED]
                known = valid[at].any(1)
                x, scale = run.model.descriptors(h, cand, anchor, crowd)
                target = prev.lower_target(anchor, cand, y[at], valid[at])/scale[:, None]
                conditional.append(prev.moments(x[known], target[known], crowd[known]))
                missing.append(prev.moments(covariates(h, cv[at], anchor, crowd), (~known)[:, None], crowd))
                rates.append(float(np.average(~known, weights=1./crowd)))
            c = run.model.coefficients(conditional)[..., 0]
            p = run.model.coefficients(missing)[0, :, 0]
            run.once(mark, run.stable(dict(seed=seed, held=str(held), registration=run.sha(REG),
                donors=[str(s) for s in sources if s != held], conditional=c.tolist(),
                probability=p.tolist(), prevalence=float(np.mean(rates)))))
        fit = run.read(mark)
        if (fit['seed'], fit['held'], fit['registration'], fit['donors']) != (
                seed, str(held), run.sha(REG), [str(s) for s in sources if s != held]):
            raise ValueError('Incompatible nested fit')
        fits.append(run.artifact(mark))
        beat(seed=seed, completed=i+1, state='factorized_head_frozen')
    return dict(seed=seed, fits=fits)


def frozen_fits(packet, registration):
    values = [run.read(run.checked(r)) for r in packet['fits']]
    if any(r['registration'] != registration or r['seed'] != packet['seed'] for r in values):
        raise ValueError('Misaligned fit provenance')
    fits = {r['held']: r for r in values}
    if len(fits) != 14:
        raise ValueError('Incomplete held-source roster')
    return fits


def freeze_seed(d, refs, cell, trained, direct):
    seed = trained['seed']
    cv, bank = run.candidates(d, refs, seed)
    models = frozen_fits(trained, run.sha(REG))
    old = frozen_fits(direct, run.sha(prev.REG))
    with np.load(run.checked(cell['artifact']), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], d['global_ids'])
        a, control = z['choices'][:2]
    if not np.array_equal(a >= 0, control >= 0):
        raise ValueError('Original matched recipients changed')
    choices = np.stack((a, control, a, a, a, a))
    probs = np.zeros((2, len(cv)))
    for at in d['groups']:
        s = str(d['sites'][at[0]])
        fit = models[s]
        anchor = run.model.selected_paths(a[at], bank[at], cv[at])
        cand = bank[at][:, LEARNED]
        h, crowd = d['history'][at], d['crowd'][at]
        x, scale = run.model.descriptors(h, cand, anchor, crowd)
        rad = np.linalg.norm(cand-anchor[:, None], axis=-1).max(2)
        legal = (rad > 0.) & (np.linalg.norm(cand-cv[at, None], axis=-1).max(2) > 0.)
        p = np.clip(covariates(h, cv[at], anchor, crowd)[:, 0] @ np.asarray(fit['probability']), 0., 1.)
        q = np.full(len(at), fit['prevalence'])
        probs[:, at] = (q, p)
        conditional = np.einsum('nkd,kd->nk', x, np.asarray(fit['conditional']))*scale[:, None]
        delta = np.einsum('nkd,kd->nk', x, np.asarray(old[s]['coefficients']))*scale[:, None]
        k = int(np.isin(control[at], LEARNED).sum())
        for i, value in enumerate((np.clip(delta, -rad, rad), -rad,
                                   factorized(conditional, rad, q), factorized(conditional, rad, p)), 2):
            choices[i, at] = rank(a[at], legal, value, k)
    return choices, probs


def score(d, refs, lab, frozen, cut, beat):
    rows, brier, pairs = [], {}, {}
    records = prev.prior.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
    groups = sorted(set(roster.values()))
    contrasts = {f'{a}_vs_{b}': (b, a, BASE) for a in ARMS for b in (BASE, OLS)}
    contrasts.update({f'{a}_vs_{b}': (b, a, BASE) for i, a in enumerate(ARMS) for b in ARMS[:i]})
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    for cell in frozen['cells']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            choice, p = z['choices'], z['probability']
        pair_seed = {s: [] for s in groups}
        for s in groups:
            at = (d['sites'] == s) & (choice[0] >= 0)
            brier[s, seed] = np.average((p[:, at]-(~valid[at].any(1))[None])**2, axis=1,
                                       weights=1./d['crowd'][at])
        for r, (ix, queries) in records.items():
            s = roster[r]
            paths = {BASE: cv[ix], OLS: bank[ix, run.NAMES.index(OLS)],
                     **{a: run.model.selected_paths(c[ix], bank[ix], cv[ix]) for a, c in zip(ARMS, choice)}}
            switches = np.stack([(paths[a] != cv[ix]).any(axis=(1, 2)) for a in ARMS])
            np.testing.assert_array_equal(switches, np.broadcast_to(choice[0, ix] >= 0, switches.shape))
            for at in queries:
                counts = np.isin(choice[1:, ix[at]], LEARNED).sum(1)
                if np.any(counts != counts[0]):
                    raise ValueError('Actual neural counts differ')
                ppaths = np.stack([paths[a][at] for a in ARMS[-2:]])
                _, a, _, _ = pair.query(d['history'][ix[at]], ppaths, y[ix[at]], valid[ix[at]], [(0, 1)])
                pair_seed[s].append(a[0])
            row = run.parent.readout.score_recording(paths, contrasts, y[ix], valid[ix], d['queries'][ix], r,
                                                      easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, seed=seed, source=s))
        for s, a in pair_seed.items():
            pairs[s, seed] = pair.mean_finite(a)
        beat(seed=seed, state='matched_predictions_scored')
    agg = prev.prior.prior.grouped_readout(rows, roster, SEEDS)
    before = run.read(prev.PUBLIC/'analysis.json')['grouped']['risk']
    if any(agg['risk'][a] != before[a] for a in ARMS[:2]):
        raise ValueError('Frozen anchor or CV-difference risk did not replay')
    bs = np.asarray([[brier[s, seed] for seed in SEEDS] for s in groups])
    pp = np.asarray([[pairs[s, seed] for seed in SEEDS] for s in groups])
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    return dict(grouped=agg, learned_replacements=dict(zip(ARMS, np.sum(
        [c['learned'] for c in frozen['cells']], 0).tolist())),
        unknown_brier=dict(constant=pair.bootstrap(bs[:, :, 0], draw), fitted=pair.bootstrap(bs[:, :, 1], draw),
                           fitted_minus_constant=pair.bootstrap(bs[:, :, 1]-bs[:, :, 0], draw)),
        relative_pair={k: pair.bootstrap(pp[:, :, i], draw) for i, k in enumerate(pair.PAIR_FIELDS)},
        external_access=False, independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'fit', 'freeze', 'score', 'replay'))
    phase = p.parse_args().phase
    cut = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the design before execution')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    t = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**fields):
            event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-t, **fields)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', event)
            if fields.get('completed', 14) in (1, 7, 14):
                print(event, flush=True)

        d, refs = run.population(), run.register()
        if phase == 'fit':
            parent = receipt(prev.prior.prior.PUBLIC, 'freeze')
            lab = run.old.prior.arrays(('target_eval', 'valid'))
            y = lab['target_eval']-d['origin'][:, None]
            cells = [train_seed(d, refs, y, lab['valid'], cut, c, beat) for c in parent['cells']]
            out = PRIVATE/'fit.json'
            run.once(out, run.stable(dict(registration=run.sha(REG), cells=cells)))
        elif phase == 'freeze':
            trained, direct, choices = receipt(PUBLIC, 'fit'), receipt(prev.PUBLIC, 'fit'), receipt(prev.PUBLIC, 'freeze')
            if trained['registration'] != run.sha(REG):
                raise ValueError('Incompatible fit')
            cells = []
            for c, f, b in zip(choices['cells'], trained['cells'], direct['cells'], strict=True):
                seed = f['seed']
                if c['seed'] != seed or b['seed'] != seed:
                    raise ValueError('Incompatible seed roster')
                path = PRIVATE/f'choices_{seed}.npz'
                mark = path.with_suffix('.json')
                if not mark.exists():
                    z, prob = freeze_seed(d, refs, c, f, b)
                    tmp = path.with_suffix('.pending.npz')
                    np.savez(tmp, ids=d['global_ids'], choices=z, probability=prob)
                    tmp.replace(path)
                    run.once(mark, dict(seed=seed, registration=run.sha(REG), artifact=run.artifact(path),
                                       learned=np.isin(z, LEARNED).sum(1).tolist()))
                item = run.read(mark)
                if item['seed'] != seed or item['registration'] != run.sha(REG):
                    raise ValueError('Incompatible choice checkpoint')
                run.checked(item['artifact'])
                cells.append(item)
                beat(seed=seed, state='past_only_choices_frozen')
            out = PRIVATE/'freeze.json'
            run.once(out, run.stable(dict(cells=cells, registration=run.sha(REG), score_labels_read=False)))
        else:
            frozen = receipt(PUBLIC, 'freeze')
            if frozen['registration'] != run.sha(REG) or frozen['score_labels_read']:
                raise ValueError('Require complete pre-score decision freeze')
            if phase == 'replay':
                receipt(PUBLIC, 'score')
            lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(d, refs, lab, frozen, cut, beat)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-t, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Replay differs from original artifact')


if __name__ == '__main__':
    main()
