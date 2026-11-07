"""Isolate query activation from the action ranker on source-separated folds."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_native_feasible_slate as prev
from src.world_model import m3w_native_activation as model

run, np, pair, BASE, OLS = prev.run, prev.np, prev.pair, prev.BASE, prev.OLS
PUBLIC = prev.PUBLIC.parent/'native_activation_v1'
PRIVATE = run.PRIVATE/'native_activation_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_native_activation.py')
ARMS = (BASE, OLS, 'previous_support', 'support_refit', 'matched_gain', 'matched_probability',
        'causal_gain', 'causal_probability')
PAIRS = {f'{a}_vs_cv': (BASE, a, BASE) for a in ARMS[1:]}
PAIRS.update({f'{a}_vs_ols4': (OLS, a, BASE) for a in ARMS[2:]})
PAIRS.update(matched_prob_vs_gain=('matched_gain', 'matched_probability', BASE),
             matched_prob_vs_support=('support_refit', 'matched_probability', BASE),
             causal_prob_vs_gain=('causal_gain', 'causal_probability', BASE),
             causal_prob_vs_support=('support_refit', 'causal_probability', BASE))
POLICIES = ARMS[3:]
PAIR_CONTRASTS = ('matched_prob_vs_gain', 'matched_prob_vs_support', 'causal_prob_vs_gain')


def register():
    _, support, cut = prev.register()
    score = prev.PUBLIC/'analysis.json'
    if (run.read(prev.PUBLIC/'replay_execution.json')['artifact'] != run.artifact(score)
            or not run.read(prev.PUBLIC/'checkpoint_audit.json')['prediction_replay_exact']):
        raise ValueError('Require prior result and verified checkpoints')
    parents = (prev.REG, score, prev.PUBLIC/'checkpoint_audit.json', Path(prev.model.__file__),
               Path(prev.prev.model.__file__), Path(prev.crit.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], policies=POLICIES, contrasts=PAIRS,
        tree_settings=prev.HGB, sklearn=prev.sklearn_version, cut=cut, calibration_sources=3,
        primary='offline_count_matched_probability_minus_gain',
        external_access=False, original_budget=.02)))
    return support, cut


def fit(d, cv, bank, y, valid, cut, seed, cache, weight_path, beat):
    h, site, crowd = (d[k] for k in ('history', 'sites', 'crowd'))
    names = np.unique(site)
    x, scale = run.model.descriptors(h, bank, cv, crowd)
    z = prev.prev.model.features(h, bank, cv, crowd, d['groups'])
    e, _ = prev.crit.errors(cv, y, valid)
    easy = valid.any(1) & (e > 0.) & (e <= cut)
    actions = np.stack([prev.crit.action_targets(cv, bank[:, a], y, valid, cut) for a in range(bank.shape[1])], 1)
    moments, queries = {}, {}
    for name in names:
        ix = np.flatnonzero(site == name)
        moments[name] = run.model.fitting_moments(x[ix], bank[ix], cv[ix], y[ix], valid[ix], scale[ix],
                                                  1./crowd[ix], cut)
        queries[name] = [q for q in d['groups'] if site[q[0]] == name]
    choices = np.full((len(POLICIES), len(h)), -1, dtype=np.int16)
    fitted, cells = [], []
    for held in names:
        fitting, calibration = model.split(names, held)
        coef = run.model.coefficients([moments[s] for s in fitting])
        ix = np.isin(site, fitting)
        support = prev.prev.model.train(z[ix], easy[ix], site[ix], crowd[ix], held)
        prob_easy = prev.prev.model.probability(z, support)
        tx, ty, tw = [], [], []
        for donor in fitting:
            donor_coef = run.model.coefficients([moments[s] for s in fitting if s != donor])
            for q in queries[donor]:
                slate, v, _ = prev.slate_for_query(q, d, bank, cv, x, scale, z, donor_coef)
                tx.append(v)
                ty.append(prev.option_targets(actions, q, slate))
                tw.append(np.full(len(slate), 1./(len(slate)*len(queries[donor]))))
        w = np.concatenate(tw)
        clf = prev.HistGradientBoostingClassifier(random_state=seed, **prev.HGB)
        clf.fit(np.concatenate(tx), np.concatenate(ty), sample_weight=w/w.mean())

        def decisions(qs):
            scores, old, old_choice, new_choice, counts = [], [], [], [], []
            for q in qs:
                slate, v, allowed = prev.slate_for_query(q, d, bank, cv, x, scale, z, coef)
                # Every option differs from the common anchor only at its replaceable row.
                common = (slate == slate[0]).all(0) & (slate[0] >= 0)
                variable = (slate >= 0) & ~common[None]
                values = (variable * prob_easy[q]).sum(1)
                control = int(values.argmax())
                p = clf.predict_proba(v)[:, 1]
                best = int(np.argmax(np.where(allowed, p, -np.inf)))
                scores.append((v[best, 0], p[best]) if allowed.any() else (-np.inf, -np.inf))
                old.append(allowed[control])
                old_choice.append(slate[control])
                new_choice.append(slate[best])
                counts.append(int((slate[best] >= 0).sum()))
            return np.asarray(scores), np.asarray(old), old_choice, new_choice, np.asarray(counts)

        cq = [q for s in calibration for q in queries[s]]
        cal_score, cal_old, _, _, _ = decisions(cq)
        cuts = [model.threshold(cal_score[:, j], int(cal_old.sum())) for j in range(2)]
        qs = queries[held]
        scores, old, old_path, new_path, count = decisions(qs)
        rec = np.asarray([d['recordings'][q[0]] for q in qs])
        masks = [old, *(model.matched_mask(scores[:, j], old, count, rec) for j in range(2)),
                 *(model.causal_mask(scores[:, j], cuts[j]) for j in range(2))]
        for j, q in enumerate(qs):
            for k, mask in enumerate(masks):
                if mask[j]:
                    choices[k, q] = old_path[j] if k == 0 else new_path[j]
        cells.append(dict(held=str(held), fitting=list(map(str, fitting)), calibration=list(map(str, calibration)),
            donor_queries=len(cq), donor_support_queries=int(cal_old.sum()), score_thresholds=cuts,
            donor_activated=[int(model.causal_mask(cal_score[:, j], cuts[j]).sum()) for j in range(2)],
            held_queries=len(qs), held_activated=[int(mask.sum()) for mask in masks]))
        fitted.append((str(held), coef, support, clf))
        beat(state='held_source_done', seed=seed, done=len(cells))
    tmp = weight_path.with_suffix('.pending')
    prev.joblib.dump(fitted, tmp, compress=3)
    tmp.replace(weight_path)
    prev.prior.h.prune.previous.history.feedback.write_array(cache, ids=d['global_ids'], choices=choices)
    return cells


def score(d, refs, target, valid, cut, packet, support, records, beat):
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
    sites, seeds = sorted(set(roster.values())), list(run.kin.prior.SEEDS)
    rows, relative, local = [], {}, {}
    for cell, old in zip(packet['cells'], support['cells'], strict=True):
        seed = cell['seed']
        assert seed == old['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['decisions']), allow_pickle=False) as f:
            np.testing.assert_array_equal(f['ids'], d['global_ids'])
            selections = f['choices']
        with np.load(run.checked(old['artifact']), allow_pickle=False) as f:
            np.testing.assert_array_equal(f['ids'], d['global_ids'])
            legacy = f['choices'][5]
        paths = {BASE: cv, OLS: bank[:, run.NAMES.index(OLS)],
                 'previous_support': run.model.selected_paths(legacy, bank, cv)}
        paths.update({a: run.model.selected_paths(k, bank, cv) for a, k in zip(POLICIES, selections)})
        labels = [prev.crit.action_targets(cv, paths[a], target, valid, cut['easy']) for a in POLICIES]
        for r, (ix, groups) in records.items():
            view = {a: p[ix] for a, p in paths.items()}
            row = run.parent.readout.score_recording(view, PAIRS, target[ix], valid[ix], d['queries'][ix], r,
                                                       easy_cut=cut['easy'], hard_cut=cut['hard'])
            s = roster[r]
            rows.append(dict(row, seed=seed, source=s))
            for g in groups:
                q = ix[g]
                verdict = []
                for k, a in enumerate(labels):
                    taken = q[selections[k, q] >= 0]
                    v = prev.crit.targets(a[taken], 1.)
                    verdict.append((len(taken) > 0, bool(prev.crit.accept(v[None])[0]),
                                    len(taken) > 0 and v[-1] == 0.))
                local.setdefault((s, seed), []).append(verdict)
                _, x, _, _ = pair.query(d['history'][q], np.stack([view[a][g] for a in POLICIES]),
                                         target[q], valid[q], [(1, 2), (0, 2), (3, 4)])
                relative.setdefault((s, seed), []).append(x)
    grouped = prev.prior.grouped_readout(rows, roster, seeds)
    if grouped['risk']['previous_support'] != run.read(prev.prev.PUBLIC/'analysis.json')['grouped']['risk']['gated_support']:
        raise ValueError('Previous support control did not replay')
    if len({grouped['risk'][a]['switches'] for a in POLICIES[:3]}) != 1:
        raise ValueError('Offline control switch count differs')
    for row in rows:
        if len({row['metrics'][a]['risk']['selected'] for a in POLICIES[:3]}) != 1:
            raise ValueError('Offline count not matched within recording')
    values = np.asarray([[np.asarray(local[s, k]).mean(0) for k in seeds] for s in sites])
    totals = np.concatenate([np.asarray(v) for v in local.values()]).sum(0)
    geom = np.asarray([[pair.mean_finite(relative[s, k]) for k in seeds] for s in sites])
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    return dict(grouped=grouped, counts=dict(zip(POLICIES, [dict(zip(
        ('active_queries', 'locally_feasible_queries', 'active_without_easy_reference'), v.tolist())) for v in totals])),
        local_feasible_pp={name: pair.bootstrap(100*(values[:, :, j, 1]-values[:, :, i, 1]), draw)
                           for name, (i, j) in zip(PAIR_CONTRASTS, ((1, 2), (0, 2), (3, 4)))},
        pair_geometry={name: {k: pair.bootstrap(geom[:, :, i, j], draw) for j, k in enumerate(pair.PAIR_FIELDS)}
                       for i, name in enumerate(PAIR_CONTRASTS)},
        causal_thresholds_donor_only=True, offline_matched_not_deployable=True, original_rules_unchanged=True,
        external_outcomes_opened=False, independent_confirmation=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = p.parse_args().phase
    support, cut = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit protocol before fitting')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), seconds=time.monotonic()-start, phase=phase, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            print(row, flush=True)

        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
        freeze = PUBLIC/'freeze.json'
        if phase == 'fit':
            cells = []
            for seed in run.kin.prior.SEEDS:
                path = PRIVATE/f'seed_{seed}.npz'
                mark, weights = path.with_suffix('.json'), path.with_suffix('.joblib')
                if not mark.exists():
                    cv, bank = run.candidates(d, refs, seed)
                    folds = fit(d, cv, bank, y, valid, cut['easy'], seed, path, weights, beat)
                    run.once(mark, dict(seed=seed, registration=run.sha(REG), decisions=run.artifact(path),
                        checkpoint=run.artifact(weights), folds=folds))
                row = run.read(mark)
                if row['seed'] != seed or row['registration'] != run.sha(REG):
                    raise ValueError('Resume identity differs')
                run.checked(row['decisions'])
                run.checked(row['checkpoint'])
                cells.append(row)
            run.once(freeze, dict(cells=cells, phase='decisions_frozen', registration=run.sha(REG)))
            out = freeze
        else:
            packet = run.read(freeze)
            if packet['registration'] != run.sha(REG) or run.artifact(freeze) != run.read(PUBLIC/'fit_execution.json')['artifact']:
                raise ValueError('Require sealed decisions before scoring')
            out = PUBLIC/'analysis.json'
            if phase == 'replay' and run.artifact(out) != run.read(PUBLIC/'score_execution.json')['artifact']:
                raise ValueError('Require verified score to replay')
            records = prev.prior.h.previous.prior.blocks(d, lab['frames'])
            run.once(out, run.stable(score(d, refs, y, valid, cut, packet, support, records, beat)))
        receipt = PUBLIC/f'{phase}_execution.json'
        doc = dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out))
        if not receipt.exists():
            run.once(receipt, doc)
        elif run.read(receipt)['artifact'] != doc['artifact']:
            raise ValueError('Execution refers to a different artifact')


if __name__ == '__main__':
    main()
