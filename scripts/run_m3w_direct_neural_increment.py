"""Fit a direct neural-minus-motion critic without changing recipients or risk reference."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_causal_menu_control as prior

run, np, pair = prior.run, prior.np, prior.pair
BASE, OLS, MENU, LEARNED, SEEDS = prior.BASE, prior.OLS, prior.MENU, prior.LEARNED, prior.SEEDS
PUBLIC = prior.PUBLIC.parent/'native_direct_increment_v1'
PRIVATE = prior.PRIVATE.parent/'native_direct_increment_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_direct_neural_increment.py')
MATCHED = ('anchor', 'cv_difference', 'direct_difference')
ARMS = (*MATCHED, *(f'{a}_screen' for a in MATCHED))


def register():
    cut = prior.register()
    for name in ('freeze', 'score', 'replay'):
        run.checked(run.read(prior.PUBLIC/f'{name}_execution.json')['artifact'])
    refs = (prior.REG, prior.PUBLIC/'freeze_execution.json', prior.PUBLIC/'replay_execution.json',
            Path(prior.__file__), Path(run.model.__file__), *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in refs], cut=cut, seeds=SEEDS,
        arms=ARMS, outer_donor_anchor_fit=[1, 13, 12], ridge=.002, switch_margin=0.,
        primary='direct_difference_vs_cv_difference', risk=.02, external_access=False)))
    return cut


def fitting_sources(sources, held, donor):
    if held == donor or held not in sources or donor not in sources:
        raise ValueError('A scored source cannot be an anchor donor')
    return [s for s in sources if s not in (held, donor)]


def lower_target(anchor, candidates, y, valid):
    ref, _ = run.model.errors(anchor, y, valid)
    radius = np.linalg.norm(candidates-anchor[:, None], axis=-1).max(2)
    known = valid.any(1)
    return np.column_stack([
        np.where(known, ref-run.model.errors(candidates[:, i], y, valid)[0], -radius[:, i])
        for i in range(candidates.shape[1])])


def moments(x, y, crowd):
    w = (1./crowd)
    w /= w.sum()
    return (np.einsum('nkd,n,nkf->kdf', x, w, x),
            np.einsum('nkd,n,nk->kd', x, w, y)[..., None])


def coeffs(parent_cell):
    with np.load(run.checked(parent_cell['cache']), allow_pickle=False) as f:
        return dict(zip(f['sources'], f['excluded_source_coefficients']))


def fit_seed(d, refs, y, valid, cut, parent_cell, beat):
    seed = parent_cell['seed']
    cv, bank = run.candidates(d, refs, seed)
    original = coeffs(parent_cell)
    sites = d['sites']
    sources = sorted(original)
    by_source = {s: [] for s in sources}
    for at in d['groups']:
        by_source[sites[at[0]]].append(at)
    stats = {}
    for s in sources:
        at = np.flatnonzero(sites == s)
        x, scale = run.model.descriptors(d['history'][at], bank[at][:, MENU], cv[at], d['crowd'][at])
        stats[s] = run.model.fitting_moments(x, bank[at][:, MENU], cv[at], y[at], valid[at], scale,
                                            1./d['crowd'][at], cut['easy'])
    causal_ols = run.NAMES.index(OLS)
    receipts = []
    for held in sources:
        mark = PRIVATE/f'fit_{seed}_{sources.index(held):02}.json'
        if not mark.exists():
            verify = run.model.coefficients([stats[s] for s in sources if s != held])
            np.testing.assert_allclose(verify, original[held][MENU], rtol=1e-8, atol=1e-9)
            fitting, seen = [], 0
            for donor in sources:
                if donor == held:
                    continue
                c = run.model.coefficients([stats[s] for s in fitting_sources(sources, held, donor)])
                indices, anchors = [], []
                for at in by_source[donor]:
                    a = prior.prior.model.decide(d['history'][at], bank[at][:, MENU], cv[at],
                                                bank[at, causal_ols], d['crowd'][at], c)['joint']
                    take = a >= 0
                    if take.any():
                        indices.append(at[take])
                        anchors.append(MENU[a[take]])
                if not indices:
                    raise ValueError('No causal anchor support on a fit source')
                at, act = np.concatenate(indices), np.concatenate(anchors)
                if np.any(sites[at] != donor):
                    raise ValueError('Increment fit recipient outside its donor')
                anchor = bank[at, act]
                candidates = bank[at][:, LEARNED]
                x, scale = run.model.descriptors(d['history'][at], candidates, anchor, d['crowd'][at])
                target = lower_target(anchor, candidates, y[at], valid[at])/scale[:, None]
                fitting.append(moments(x, target, d['crowd'][at]))
                seen += len(at)
            c = run.model.coefficients(fitting)[..., 0]
            run.once(mark, run.stable(dict(registration=run.sha(REG), seed=seed, held=str(held),
                donor_sources=[str(s) for s in sources if s != held], fitted_rows=seen, coefficients=c.tolist())))
        item = run.read(mark)
        if (item['registration'], item['seed'], item['held'], item['donor_sources']) != (
                run.sha(REG), seed, str(held), [str(s) for s in sources if s != held]):
            raise ValueError('Incompatible source fit checkpoint')
        receipts.append(run.artifact(mark))
        beat(seed=seed, completed=len(receipts), state='excluded_source_head_frozen')
    return dict(seed=seed, fits=receipts)


def replace_actions(anchor, legal, incremental):
    out = anchor.copy()
    rows = np.flatnonzero(anchor >= 0)
    score = np.where(legal[rows], incremental[rows], -np.inf)
    best = score.argmax(1)
    good = score[np.arange(len(rows)), best] > 0.
    out[rows[good]] = LEARNED[best[good]]
    return out


def choose(history, bank, cv, crowd, anchor, original, direct):
    x, scale = run.model.descriptors(history, bank, cv, crowd)
    gain, harm_all, harm_easy, legal = run.model.decision_costs(x, original, bank, cv, scale)
    expected = np.einsum('nad,adc->nac', x, original)
    ref = np.minimum(expected[..., 4].clip(0.), expected[..., 3].clip(0.))*scale[:, None]
    p = run.model.selected_paths(anchor, bank, cv)
    candidates = bank[:, LEARNED]
    desc, norm = run.model.descriptors(history, candidates, p, crowd)
    radius = np.linalg.norm(candidates-p[:, None], axis=-1).max(2)
    delta = np.einsum('nkd,kd->nk', desc, direct)*norm[:, None]
    delta = np.clip(delta, -radius, radius)
    candidate_legal = legal[:, LEARNED] & (radius > 0.)
    cv_delta = gain[:, LEARNED]-gain[np.arange(len(anchor)), anchor.clip(0), None]
    cv_delta = np.clip(cv_delta, -radius, radius)
    arms = (anchor, replace_actions(anchor, candidate_legal, cv_delta),
            replace_actions(anchor, candidate_legal, delta))
    screened = [prior.prior.model.gate(a, gain, harm_all, harm_easy, ref) for a in arms]
    return np.stack((*arms, *screened))


def freeze_seed(d, refs, parent_cell, motion, fitted):
    seed = fitted['seed']
    cv, bank = run.candidates(d, refs, seed)
    original = coeffs(parent_cell)
    direct = {}
    for spec in fitted['fits']:
        fit = run.read(run.checked(spec))
        if fit['seed'] != seed or fit['registration'] != run.sha(REG) or fit['held'] in direct:
            raise ValueError('Incorrect source fit')
        direct[fit['held']] = np.asarray(fit['coefficients'])
    if set(original) != set(direct):
        raise ValueError('Incomplete excluded-source fit roster')
    with np.load(run.checked(motion['artifact']), allow_pickle=False) as f:
        np.testing.assert_array_equal(f['ids'], d['global_ids'])
        anchor = f['choices'][prior.ARMS.index('causal_joint')]
    choices = np.full((len(ARMS), len(cv)), -1, dtype=np.int16)
    for at in d['groups']:
        s = str(d['sites'][at[0]])
        choices[:, at] = choose(d['history'][at], bank[at], cv[at], d['crowd'][at], anchor[at],
                                original[s], direct[s])
    if not np.all((choices[:3] >= 0) == (anchor >= 0)):
        raise ValueError('Changed matched recipients')
    return choices


def pooled_accuracy(rows):
    out = {}
    for arm in (BASE, OLS, *ARMS):
        data = [r['metrics'][arm] for r in rows]
        out[arm] = {}
        for name, count, error in (
                ('available_ADE', 'known_rows', 'known_ADE_pixel'),
                ('available_endpoint_FDE', 'endpoint_rows', 'available_endpoint_FDE_diagnostic_pixel')):
            n = sum(d[count] for d in data)
            out[arm][name] = dict(rows=n, mean_pixel=sum(d[count]*d[error] for d in data if d[count])/n if n else None)
    return out


def score(d, refs, lab, frozen, cut, beat):
    records = prior.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
    groups = sorted(set(roster.values()))
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    contrasts = {f'{a}_vs_{b}': (b, a, BASE) for a in ARMS for b in (BASE, OLS)}
    for a in ('cv_difference', 'direct_difference'):
        for b in ('anchor', 'cv_difference'):
            if a != b:
                contrasts[f'{a}_vs_{b}'] = (b, a, BASE)
                contrasts[f'{a}_screen_vs_{b}_screen'] = (b+'_screen', a+'_screen', BASE)
    rows, relative = [], {}
    for cell in frozen['cells']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        ols = bank[:, run.NAMES.index(OLS)]
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            choices = z['choices']
        site_pair = {g: [] for g in groups}
        for rec, (ix, queries) in records.items():
            src = roster[rec]
            paths = {BASE: cv[ix], OLS: ols[ix], **{a: run.model.selected_paths(c[ix], bank[ix], cv[ix])
                     for a, c in zip(ARMS, choices)}}
            for a in MATCHED:
                np.testing.assert_array_equal((paths[a] != cv[ix]).any(axis=(1, 2)), choices[0, ix] >= 0)
            for at in queries:
                p = np.stack([paths[a][at] for a in MATCHED])
                _, values, _, _ = pair.query(d['history'][ix[at]], p, y[ix[at]], valid[ix[at]], [(1, 2), (0, 2)])
                site_pair[src].append(values)
            result = run.parent.readout.score_recording(paths, contrasts, y[ix], valid[ix], d['queries'][ix],
                                                         rec, easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(result, seed=seed, source=src))
        for g, values in site_pair.items():
            relative[g, seed] = pair.mean_finite(values)
        beat(seed=seed, state='frozen_decisions_scored')
    agg = prior.prior.grouped_readout(rows, roster, SEEDS)
    baseline = run.read(prior.PUBLIC/'analysis.json')['grouped']['risk']
    if any(agg['risk'][a] != baseline[b] for a, b in (('anchor', 'causal_joint'), ('anchor_screen', 'causal_screen'))):
        raise ValueError('The frozen motion control did not replay')
    x = np.asarray([[relative[g, s] for s in SEEDS] for g in groups])
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    return dict(grouped=agg, pooled_descriptive_accuracy=pooled_accuracy(rows), relative_pair={
        a: {k: pair.bootstrap(x[:, :, j, i], draw) for i, k in enumerate(pair.PAIR_FIELDS)}
        for j, a in enumerate(('direct_vs_cv_difference', 'direct_vs_anchor'))},
        learned_replacements=dict(zip(ARMS, np.sum([r['learned'] for r in frozen['cells']], 0).tolist())),
        independent_confirmation=False, external_access=False, deployment=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'fit', 'freeze', 'score', 'replay'))
    phase = p.parse_args().phase
    cut = register()
    if phase == 'register':
        return
    for file in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{file.relative_to(ROOT)}'], cwd=ROOT) != file.read_bytes():
            raise ValueError('Commit the complete design before execution')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', event)
            if v.get('completed', 14) in (1, 7, 14):
                print(event, flush=True)

        def receipt(folder, key):
            return run.read(run.checked(run.read(folder/f'{key}_execution.json')['artifact']))

        d, refs = run.population(), run.register()
        transport = receipt(prior.prior.PUBLIC, 'freeze')
        if [c['seed'] for c in transport['cells']] != list(SEEDS):
            raise ValueError('Unexpected forecaster seed roster')
        if phase == 'fit':
            lab = run.old.prior.arrays(('target_eval', 'valid'))
            y = lab['target_eval']-d['origin'][:, None]
            cells = [fit_seed(d, refs, y, lab['valid'], cut, c, beat) for c in transport['cells']]
            out = PRIVATE/'fit.json'
            run.once(out, run.stable(dict(cells=cells, registration=run.sha(REG))))
        elif phase == 'freeze':
            fit, old = receipt(PUBLIC, 'fit'), receipt(prior.PUBLIC, 'freeze')
            if fit['registration'] != run.sha(REG) or any(c['mismatched_queries'] for c in old['cells']):
                raise ValueError('Require the exact matched anchor and direct fit')
            cells = []
            for parent, motion, trained in zip(transport['cells'], old['cells'], fit['cells'], strict=True):
                seed = trained['seed']
                if seed != parent['seed'] or seed != motion['seed']:
                    raise ValueError('Mixed seed bindings')
                path = PRIVATE/f'choice_{seed}.npz'
                mark = path.with_suffix('.json')
                if not mark.exists():
                    choices = freeze_seed(d, refs, parent, motion, trained)
                    tmp = path.with_suffix('.pending.npz')
                    np.savez(tmp, ids=d['global_ids'], choices=choices)
                    tmp.replace(path)
                    run.once(mark, dict(seed=seed, registration=run.sha(REG), artifact=run.artifact(path),
                                       learned=np.isin(choices, LEARNED).sum(1).tolist()))
                item = run.read(mark)
                if item['registration'] != run.sha(REG) or item['seed'] != seed:
                    raise ValueError('Incompatible decision checkpoint')
                run.checked(item['artifact'])
                cells.append(item)
                beat(seed=seed, state='past_only_choices_frozen')
            out = PRIVATE/'freeze.json'
            run.once(out, run.stable(dict(registration=run.sha(REG), cells=cells, score_labels_read=False)))
        else:
            frozen = receipt(PUBLIC, 'freeze')
            if frozen['registration'] != run.sha(REG) or frozen['score_labels_read']:
                raise ValueError('Require frozen decision outputs before reading score labels')
            if phase == 'replay':
                receipt(PUBLIC, 'score')
            lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(d, refs, lab, frozen, cut, beat)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Experiment replay differs')


if __name__ == '__main__':
    main()
