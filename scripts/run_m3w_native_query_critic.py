"""Nested source-excluded validation of a current-query switch-set critic."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import freeze_m3w_dronecrowd_transport_candidate as prior
from src.world_model import m3w_query_risk_critic as critic

run, np, pair, BASE, OLS = prior.run, prior.np, prior.pair, prior.BASE, prior.OLS
PUBLIC = prior.PUBLIC.parent/'native_query_risk_critic_v1'
PRIVATE = run.PRIVATE/'native_query_risk_critic_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(critic.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_query_risk_critic.py')
ARMS = (BASE, OLS, 'joint', 'gated_joint', 'recalibrated_sum', 'query_critic')
PAIRS = {f'{a}_vs_cv': (BASE, a, BASE) for a in ARMS[1:]}
PAIRS.update({f'{a}_vs_ols4': (OLS, a, BASE) for a in ARMS[2:]})
PAIRS.update(critic_vs_old=('gated_joint', 'query_critic', BASE),
             critic_vs_sum=('recalibrated_sum', 'query_critic', BASE))


def register():
    prior.register()
    analysis = prior.PUBLIC/'analysis.json'
    score = prior.PUBLIC/'score_execution.json'
    replay = prior.PUBLIC/'replay_execution.json'
    for path in (score, replay):
        if run.read(path)['artifact'] != run.artifact(analysis):
            raise ValueError('Require an exact prior rehearsal replay')
    freeze = prior.PRIVATE/'freeze.json'
    binding = run.read(prior.PUBLIC/'freeze_execution.json')['artifact']
    if binding != run.artifact(freeze):
        raise ValueError('Require the immutable prior coefficients')
    parents = (prior.REG, prior.PUBLIC/'freeze_execution.json', score, replay, analysis,
               Path(run.model.__file__), Path(prior.model.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], arms=ARMS, contrasts=PAIRS,
        ridge=critic.RIDGE, budget=critic.BUDGET, support_threshold=.5,
        scope='exposed_development_only', external_access=False, bootstrap_draws=3000,
        training_query_proposal_excludes_outer_and_training_source=True)))
    return run.read(freeze), run.read(prior.REG)['cut']


def proposal(q, d, bank, floor, x, scale, coef):
    history, paths, cv = d['history'][q], bank[q], floor[q]
    qx, qs = x[q], scale[q]
    u, a, e, legal = run.model.decision_costs(qx, coef, paths, cv, qs)
    ols = paths[:, run.NAMES.index(OLS)]
    choice = run.model.choose_query(history, cv, paths, ols, qs, u, legal)[3]
    if (choice < 0).all():
        return None
    p = np.einsum('nad,adc->nac', qx, coef)
    support = np.minimum(p[..., 4].clip(0.), p[..., 3].clip(0.))*qs[:, None]
    old = prior.model.gate(choice, u, a, e, support)
    features = critic.proposal_features(history, cv, paths, choice, (u, a, e), qs, support)
    return choice, (old >= 0).any(), features


def fit_seed(d, bank, floor, y, valid, seed, cut, old_cell, dest, beat):
    sites = sorted(np.unique(d['sites']))
    x, scale = run.model.descriptors(d['history'], bank, floor, d['crowd'])
    moments, queries = {}, {}
    actions = np.stack([critic.action_targets(floor, bank[:, k], y, valid, cut)
                        for k in range(bank.shape[1])], axis=1)
    for s in sites:
        ix = np.flatnonzero(d['sites'] == s)
        moments[s] = run.model.fitting_moments(x[ix], bank[ix], floor[ix], y[ix], valid[ix],
            scale[ix], 1./d['crowd'][ix], cut)
        queries[s] = [q for q in d['groups'] if d['sites'][q[0]] == s]
    with np.load(run.checked(old_cell['cache']), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['sources'], sites)
        old = dict(zip(sites, z['excluded_source_coefficients']))
    decisions = np.full((len(ARMS)-2, len(y)), -1, dtype=np.int16)
    anchor, estimate, accepted, coef_sum, coef_query = [], [], [], [], []
    for held in sites:
        small, large, targets, sources = [], [], [], []
        for donor in sites:
            if donor == held:
                continue
            coef = run.model.coefficients([moments[s] for s in sites if s not in (held, donor)])
            for q in queries[donor]:
                p = proposal(q, d, bank, floor, x, scale, coef)
                if p is None:
                    continue
                a, _, (xs, xl, norm) = p
                mask = a >= 0
                targets.append(critic.targets(actions[q[mask], a[mask]], norm))
                small.append(xs)
                large.append(xl)
                sources.append(donor)
        fits = [critic.fit(np.asarray(z), np.asarray(targets), sources) for z in (small, large)]
        coef_sum.append(fits[0])
        coef_query.append(fits[1])
        for q in queries[held]:
            p = proposal(q, d, bank, floor, x, scale, old[held])
            if p is None:
                continue
            a, old_ok, (xs, xl, _) = p
            predictions = np.vstack([critic.predict(z[None], c) for z, c in zip((xs, xl), fits)])
            flags = (True, old_ok, *critic.accept(predictions))
            for j, keep in enumerate(flags):
                if keep:
                    decisions[j, q] = a
            anchor.append(q[0])
            estimate.append(predictions)
            accepted.append(flags)
        beat(state='outer_source_decisions', seed=seed, sources=len(coef_sum))
    prior.h.prune.previous.history.feedback.write_array(dest, ids=d['global_ids'], decisions=decisions,
        anchor=np.asarray(anchor), estimate=np.asarray(estimate), accepted=np.asarray(accepted),
        coefficient_sum=np.asarray(coef_sum), coefficient_query=np.asarray(coef_query),
        fit_order=np.asarray(sites))


def evaluate(d, refs, y, valid, records, cut, freeze, beat):
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
    sites, seeds = sorted(set(roster.values())), list(run.kin.prior.SEEDS)
    rows, quality, totals = [], {}, np.zeros((4, 3), dtype=int)
    for cell in freeze['cells']:
        seed = cell['seed']
        floor, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            decision, anchor, predictions, accepted = (z[k] for k in ('decisions', 'anchor', 'estimate', 'accepted'))
        paths = {BASE: floor, OLS: bank[:, run.NAMES.index(OLS)]}
        paths.update({a: run.model.selected_paths(c, bank, floor) for a, c in zip(ARMS[2:], decision)})
        for r, (ix, _) in records.items():
            row = run.parent.readout.score_recording({k: v[ix] for k, v in paths.items()}, PAIRS,
                y[ix], valid[ix], d['queries'][ix], r, easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, seed=seed, source=roster[r]))
        at = {q[0]: q for q in d['groups']}
        target = []
        for k in anchor:
            q = at[k]
            sel = q[decision[0, q] >= 0]
            steps = np.linalg.norm(np.diff(d['history'][sel], axis=1), axis=-1)
            norm = (1.+6.5*np.median(steps, axis=1)).mean()
            target.append(critic.targets(critic.action_targets(floor[sel], paths['joint'][sel],
                y[sel], valid[sel], cut['easy']), norm))
        target = np.asarray(target)
        actual = critic.accept(target)
        totals[:, 0] += accepted.sum(0)
        totals[:, 1] += accepted[actual].sum(0)
        totals[:, 2] += accepted[target[:, -1] == 0].sum(0)
        err = np.abs(predictions-target[:, None])
        err[..., -1] **= 2
        for s in sites:
            quality[s, seed] = err[d['sites'][anchor] == s].mean(0)
        beat(state='readout_seed', seed=seed)
    grouped = prior.grouped_readout(rows, roster, seeds)
    old = run.read(prior.PUBLIC/'analysis.json')['grouped']
    for arm in (BASE, OLS, 'joint', 'gated_joint'):
        if grouped['risk'][arm] != old['risk'][arm]:
            raise ValueError('Original proposal or gate failed to replay')
    for key, value in old['effects'].items():
        if key in grouped['effects'] and grouped['effects'][key] != value:
            raise ValueError('Original effects failed to replay')
    mat = np.asarray([[quality[s, seed] for seed in seeds] for s in sites])
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    prediction_quality = {key: dict(sum=float(mat[:, :, 0, j].mean()),
        query=float(mat[:, :, 1, j].mean()), reduction=pair.bootstrap(mat[:, :, 0, j]-mat[:, :, 1, j], draw))
        for j, key in enumerate(critic.HEADS)}
    return dict(grouped=grouped, matched_proposal_quality=prediction_quality,
        proposals=dict(zip(ARMS[2:], [dict(accepted=int(n), actual_feasible=int(f), missing_easy_support=int(m))
                                   for n, f, m in totals])),
        original_gate_exact=True, original_rules_unchanged=True, stage5c_executed=False, smc_enabled=False,
        external_inputs_opened=False, external_outcomes_opened=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = p.parse_args().phase
    old, cut = register()
    if phase == 'register':
        return
    for path in (*FILES, REG):
        committed = subprocess.check_output(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT)
        if committed != path.read_bytes():
            raise ValueError('Commit protocol and code before fitting')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**values):
            row = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **values)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            print(row, flush=True)

        d, refs = run.population(), run.register()
        labels = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        y, valid = labels['target_eval']-d['origin'][:, None], labels['valid']
        freeze = PUBLIC/'freeze.json'
        if phase == 'fit':
            cells = []
            for seed, cell in zip(run.kin.prior.SEEDS, old['cells'], strict=True):
                assert seed == cell['seed']
                path = PRIVATE/f'seed_{seed}.npz'
                done = path.with_suffix('.json')
                if not done.exists():
                    floor, bank = run.candidates(d, refs, seed)
                    fit_seed(d, bank, floor, y, valid, seed, cut['easy'], cell, path, beat)
                    run.once(done, dict(seed=seed, registration=run.sha(REG), artifact=run.artifact(path)))
                row = run.read(done)
                if row['seed'] != seed or row['registration'] != run.sha(REG):
                    raise ValueError('Resume identity differs')
                run.checked(row['artifact'])
                cells.append(row)
            run.once(freeze, dict(cells=cells, phase='decisions_frozen', registration=run.sha(REG)))
            out = freeze
        else:
            doc = run.read(freeze)
            if doc['registration'] != run.sha(REG) or run.artifact(freeze) != run.read(PUBLIC/'fit_execution.json')['artifact']:
                raise ValueError('Require sealed predictions before readout')
            out = PUBLIC/'analysis.json'
            if phase == 'replay' and run.artifact(out) != run.read(PUBLIC/'score_execution.json')['artifact']:
                raise ValueError('No verified score to replay')
            records = prior.h.previous.prior.blocks(d, labels['frames'])
            run.once(out, run.stable(evaluate(d, refs, y, valid, records, cut, doc, beat)))
        dest = PUBLIC/f'{phase}_execution.json'
        rec = dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out))
        if dest.exists():
            if run.read(dest)['artifact'] != rec['artifact']:
                raise ValueError('Previous execution bound a different artifact')
        else:
            run.once(dest, rec)


if __name__ == '__main__':
    main()
