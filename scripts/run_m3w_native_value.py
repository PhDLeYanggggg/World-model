"""Regress benefit and harm magnitudes on the frozen native activation options."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_native_activation as prior
from src.world_model import m3w_native_value as model
from sklearn.ensemble import HistGradientBoostingRegressor

run, np, pair, BASE, OLS = prior.run, prior.np, prior.pair, prior.BASE, prior.OLS
slate, switch = prior.prev, prior.model
PUBLIC = prior.PUBLIC.parent/'native_value_activation_v1'
PRIVATE = run.PRIVATE/'native_value_activation_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_native_value.py')
OLD = ('support_refit', 'matched_gain', 'matched_probability', 'causal_gain', 'causal_probability')
NEW = ('matched_value', 'matched_value_risk', 'causal_value', 'causal_value_risk')
ARMS = (BASE, OLS, *OLD, *NEW)
PAIRS = {f'{a}_vs_cv': (BASE, a, BASE) for a in ARMS[1:]}
PAIRS.update({f'{a}_vs_ols4': (OLS, a, BASE) for a in (*OLD, *NEW)})
CONTRASTS = dict(value_risk_vs_gain=('matched_gain', 'matched_value_risk', BASE),
                 value_vs_gain=('matched_gain', 'matched_value', BASE),
                 value_risk_vs_value=('matched_value', 'matched_value_risk', BASE),
                 causal_value_risk_vs_gain=('causal_gain', 'causal_value_risk', BASE))
PAIRS.update(CONTRASTS)
CONFIG = dict(slate.HGB, max_iter=45)


def register():
    _, cut = prior.register()
    freeze, analysis = (prior.PUBLIC/k for k in ('freeze.json', 'analysis.json'))
    for kind, p in (('fit', freeze), ('score', analysis), ('replay', analysis)):
        if run.read(prior.PUBLIC/f'{kind}_execution.json')['artifact'] != run.artifact(p):
            raise ValueError('Missing verified activation artifacts')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in (prior.REG, freeze, analysis, Path(slate.__file__),
                                           Path(switch.__file__))],
        roles=[10, 3, 1], model_settings=CONFIG, targets=model.NAMES, eligible_weight=4.,
        primary='value_risk_vs_gain_matched_by_recording', original_budget=.02,
        cut=cut, external_access=False)))
    return run.read(freeze), cut


def fit(d, cv, bank, y, valid, cut, old, output, checkpoint, beat):
    h, site, crowd = (d[k] for k in ('history', 'sites', 'crowd'))
    x, scale = run.model.descriptors(h, bank, cv, crowd)
    z = slate.prev.model.features(h, bank, cv, crowd, d['groups'])
    action = np.stack([model.action_targets(cv, bank[:, a], y, valid, cut) for a in range(bank.shape[1])], 1)
    queries, moments = {}, {}
    for s in np.unique(site):
        ix = np.flatnonzero(site == s)
        queries[s] = [q for q in d['groups'] if site[q[0]] == s]
        moments[s] = run.model.fitting_moments(x[ix], bank[ix], cv[ix], y[ix], valid[ix], scale[ix],
                                               1./crowd[ix], cut)
    with np.load(run.checked(old['decisions']), allow_pickle=False) as f:
        np.testing.assert_array_equal(f['ids'], d['global_ids'])
        control = f['choices']
    parent = {r[0]: r[1:] for r in slate.joblib.load(run.checked(old['checkpoint']))}
    choices = np.full((len(NEW), len(h)), -1, dtype=np.int16)
    heads, facts, anchors, estimates, selected_sets = [], [], [], [], []
    for fold in old['folds']:
        held, fitting, cal = fold['held'], fold['fitting'], fold['calibration']
        assert (fitting, cal) == switch.split(sorted(queries), held)
        coef, support, classifier = parent[held]
        np.testing.assert_array_equal(coef, run.model.coefficients([moments[s] for s in fitting]))
        prob_easy = slate.prev.model.probability(z, support)
        tx, ty, tw = [], [], []
        for donor in fitting:
            donor_coef = run.model.coefficients([moments[s] for s in fitting if s != donor])
            for q in queries[donor]:
                options, features, allowed = slate.slate_for_query(q, d, bank, cv, x, scale, z, donor_coef)
                tx.append(features)
                ty.append(model.option_targets(action, q, options, scale[q]))
                w = np.where(allowed, 4., 1.)
                tw.append(w/(w.sum()*len(queries[donor])))
        xx, yy, w = np.concatenate(tx), np.concatenate(ty), np.concatenate(tw)
        fits = [HistGradientBoostingRegressor(random_state=old['seed'], **CONFIG).fit(xx, t, sample_weight=w/w.mean())
                for t in yy.T]
        heads.append((held, fits))

        def predicted(qs):
            raw, eligible, proposal, count, active = [], [], [], [], []
            for q in qs:
                options, features, allowed = slate.slate_for_query(q, d, bank, cv, x, scale, z, coef)
                p = classifier.predict_proba(features)[:, 1]
                best = int(np.argmax(np.where(allowed, p, -np.inf)))
                common = (options == options[0]).all(0) & (options[0] >= 0)
                v = ((options >= 0) & ~common[None]) @ prob_easy[q]
                active.append(bool(allowed[v.argmax()]))
                raw.append([est.predict(features[best:best+1])[0] for est in fits])
                eligible.append(allowed.any())
                proposal.append(options[best])
                count.append((options[best] >= 0).sum())
            raw, eligible = np.asarray(raw), np.asarray(eligible)
            score = np.where(eligible[:, None], model.score(raw), -np.inf)
            return score, raw, eligible, proposal, np.asarray(count), np.asarray(active)

        donor_q = [q for s in cal for q in queries[s]]
        cal_score, _, _, _, _, cal_old = predicted(donor_q)
        assert int(cal_old.sum()) == fold['donor_support_queries']
        thresholds = [switch.threshold(v, int(cal_old.sum())) for v in cal_score.T]
        held_q = queries[held]
        scores, raw, eligible, proposed, count, active = predicted(held_q)
        current = np.array([(control[0, q] >= 0).any() for q in held_q])
        np.testing.assert_array_equal(active, current)
        rec = np.asarray([d['recordings'][q[0]] for q in held_q])
        masks = [*(switch.matched_mask(v, current, count, rec) for v in scores.T),
                 *(switch.causal_mask(v, t) for v, t in zip(scores.T, thresholds))]
        for j, q in enumerate(held_q):
            for p in control[1:]:
                if (p[q] >= 0).any():
                    np.testing.assert_array_equal(p[q], proposed[j])
            if eligible[j]:
                anchors.append(q[0])
                selected_sets.append(proposed[j])
                estimates.append(raw[j])
            for k, mask in enumerate(masks):
                if mask[j]:
                    choices[k, q] = proposed[j]
        facts.append(dict(held=held, thresholds=thresholds, target=int(cal_old.sum()),
            donor_activated=[int(switch.causal_mask(v, t).sum()) for v, t in zip(cal_score.T, thresholds)],
            held_activated=[int(m.sum()) for m in masks]))
        beat(state='continuous_heads_done', seed=old['seed'], sources=len(facts))
    tmp = checkpoint.with_suffix('.pending')
    slate.joblib.dump(heads, tmp, compress=3)
    tmp.replace(checkpoint)
    # Choices are ragged; retain only their padded two agent slots for later target readout.
    rows = np.full((len(anchors), 2, 2), -1, dtype=int)
    for i, p in enumerate(selected_sets):
        at = np.flatnonzero(p >= 0)
        rows[i, :len(at)] = np.column_stack((at, p[at]))
    slate.prior.h.prune.previous.history.feedback.write_array(output, ids=d['global_ids'], choices=choices,
        anchors=np.asarray(anchors), selected=rows, prediction=np.asarray(estimates))
    return facts


def score(d, refs, y, valid, cut, packet, prior_packet, records, beat):
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
    seeds, sites = list(run.kin.prior.SEEDS), sorted(set(roster.values()))
    rows, rel, target_error, n_scored = [], {}, [], 0
    counts = np.zeros((len(ARMS)-2, 2), int)
    for cell, old in zip(packet['cells'], prior_packet['cells'], strict=True):
        seed = cell['seed']
        assert seed == old['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(old['decisions']), allow_pickle=False) as f:
            np.testing.assert_array_equal(f['ids'], d['global_ids'])
            reference = f['choices']
        with np.load(run.checked(cell['decisions']), allow_pickle=False) as f:
            np.testing.assert_array_equal(f['ids'], d['global_ids'])
            choices = np.concatenate((reference, f['choices']))
            anchors, sl, pred = (f[k] for k in ('anchors', 'selected', 'prediction'))
        groups = {q[0]: q for q in d['groups']}
        actual = []
        for start, actions in zip(anchors, sl):
            ix, a = actions[actions[:, 0] >= 0].T
            q = groups[start][ix]
            norm = 1.+6.5*np.median(np.linalg.norm(np.diff(d['history'][q], axis=1), axis=-1), axis=1)
            actual.append(model.action_targets(cv[q], bank[q, a], y[q], valid[q], cut['easy']).sum(0)/norm.mean())
        target_error.append(np.abs(pred-actual))
        n_scored += len(anchors)
        paths = {BASE: cv, OLS: bank[:, run.NAMES.index(OLS)]}
        paths.update({a: run.model.selected_paths(c, bank, cv) for a, c in zip(ARMS[2:], choices)})
        labels = [slate.crit.action_targets(cv, paths[a], y, valid, cut['easy']) for a in ARMS[2:]]
        pair_arms = tuple(dict.fromkeys(k for a in CONTRASTS.values() for k in a[:2]))
        lookup = [(pair_arms.index(a), pair_arms.index(b)) for a, b, _ in CONTRASTS.values()]
        for r, (ix, blocks) in records.items():
            view = {k: v[ix] for k, v in paths.items()}
            row = run.parent.readout.score_recording(view, PAIRS, y[ix], valid[ix], d['queries'][ix],
                r, easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, seed=seed, source=roster[r]))
            if len({row['metrics'][a]['risk']['selected'] for a in ('matched_gain', *NEW[:2])}) != 1:
                raise ValueError('Switch counts differ within recording')
            for g in blocks:
                q = ix[g]
                for j, outcome in enumerate(labels):
                    take = q[choices[j, q] >= 0]
                    counts[j] += (len(take) > 0, slate.crit.accept(slate.crit.targets(outcome[take], 1.)[None])[0])
                _, v, _, _ = pair.query(d['history'][q], np.stack([view[k][g] for k in pair_arms]),
                                         y[q], valid[q], lookup)
                rel.setdefault((roster[r], seed), []).append(v)
        beat(state='readout_seed', seed=seed)
    summary = slate.prior.grouped_readout(rows, roster, seeds)
    original = run.read(prior.PUBLIC/'analysis.json')['grouped']
    for arm in (BASE, OLS, *OLD):
        if summary['risk'][arm] != original['risk'][arm]:
            raise ValueError('Activation control changed')
    geom = np.asarray([[pair.mean_finite(rel[s, seed]) for seed in seeds] for s in sites])
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    return dict(grouped=summary, counts=dict(zip(ARMS[2:], [dict(active=int(a), feasible=int(f)) for a, f in counts])),
        prediction_MAE=dict(zip(model.NAMES, np.concatenate(target_error).mean(0).tolist())),
        predicted_eligible_queries=n_scored, pair_geometry={name: {
            k: pair.bootstrap(geom[:, :, i, j], draw) for j, k in enumerate(pair.PAIR_FIELDS)}
            for i, name in enumerate(CONTRASTS)}, original_controls_replayed=True,
        original_rules_unchanged=True, offline_matched_not_deployable=True,
        independent_confirmation=False, external_outcomes_opened=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = p.parse_args().phase
    previous, cut = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit registration before fitting')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            d = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-started, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', d)
            print(d, flush=True)

        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
        freeze = PUBLIC/'freeze.json'
        if phase == 'fit':
            cells = []
            for old in previous['cells']:
                seed = old['seed']
                path = PRIVATE/f'seed_{seed}.npz'
                mark, weights = path.with_suffix('.json'), path.with_suffix('.joblib')
                if not mark.exists():
                    cv, bank = run.candidates(d, refs, seed)
                    folds = fit(d, cv, bank, y, valid, cut['easy'], old, path, weights, beat)
                    run.once(mark, dict(seed=seed, registration=run.sha(REG), decisions=run.artifact(path),
                                       checkpoint=run.artifact(weights), folds=folds))
                cell = run.read(mark)
                if cell['seed'] != seed or cell['registration'] != run.sha(REG):
                    raise ValueError('Changed resume identity')
                run.checked(cell['decisions'])
                run.checked(cell['checkpoint'])
                cells.append(cell)
            run.once(freeze, dict(cells=cells, registration=run.sha(REG), decisions_frozen=True))
            out = freeze
        else:
            packet = run.read(freeze)
            if packet['registration'] != run.sha(REG) or run.read(PUBLIC/'fit_execution.json')['artifact'] != run.artifact(freeze):
                raise ValueError('Require sealed fit')
            records = slate.prior.h.previous.prior.blocks(d, lab['frames'])
            out = PUBLIC/'analysis.json'
            if phase == 'replay' and run.read(PUBLIC/'score_execution.json')['artifact'] != run.artifact(out):
                raise ValueError('Require verified prior score')
            run.once(out, run.stable(score(d, refs, y, valid, cut, packet, previous, records, beat)))
        dest = PUBLIC/f'{phase}_execution.json'
        receipt = dict(pid=os.getpid(), seconds=time.monotonic()-started, artifact=run.artifact(out))
        if not dest.exists():
            run.once(dest, receipt)
        elif run.read(dest)['artifact'] != receipt['artifact']:
            raise ValueError('Existing receipt refers to a different artifact')


if __name__ == '__main__':
    main()
