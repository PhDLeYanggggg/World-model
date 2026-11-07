"""Train a direct joint feasibility ranker on source-excluded candidate slates."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_native_support_retrieval as prev
from src.world_model import m3w_feasible_slate as model
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

run, np, pair, BASE, OLS = prev.run, prev.np, prev.pair, prev.BASE, prev.OLS
crit, prior = prev.prev.critic, prev.prior
PUBLIC = prev.PUBLIC.parent/'native_feasible_slate_v1'
PRIVATE = run.PRIVATE/'native_feasible_slate_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_feasible_slate.py')
HGB = dict(max_iter=60, max_leaf_nodes=15, min_samples_leaf=32, l2_regularization=2., early_stopping=False)
ARMS = (BASE, OLS, 'gated_support', 'matched_feasible', 'support_retrieved', 'feasible_top1')
PAIRS = {f'{a}_vs_cv': (BASE, a, BASE) for a in ARMS[1:]}
PAIRS.update({f'{a}_vs_ols4': (OLS, a, BASE) for a in ARMS[2:]})
PAIRS.update(matched_vs_old=('gated_support', 'matched_feasible', BASE),
             top1_vs_old=('support_retrieved', 'feasible_top1', BASE))


def register():
    transport, cut = prev.register()
    score, freeze = (prev.PUBLIC/k for k in ('analysis.json', 'freeze.json'))
    for name, artifact in (('replay', score), ('fit', freeze)):
        if run.read(prev.PUBLIC/f'{name}_execution.json')['artifact'] != run.artifact(artifact):
            raise ValueError('Require replayed retrieval and its frozen actions')
    parents = (prev.REG, score, freeze, prev.PUBLIC/'gate_retention.json',
               Path(crit.__file__), Path(run.model.__file__), Path(prior.model.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], tree_settings=HGB, sklearn=sklearn_version,
        original_budget=.02, scope='exposed_development_only', external_access=False,
        primary='same_slate_top1_feasibility', matched_policy='frozen_support_query_mask_and_gate')))
    return transport, run.read(freeze), cut


def slate_for_query(q, data, bank, floor, x, scale, row_features, coef):
    h, cv, paths, s = data['history'][q], floor[q], bank[q], scale[q]
    u, a, e, legal = run.model.decision_costs(x[q], coef, paths, cv, s)
    orig = run.model.choose_query(h, cv, paths, paths[:, run.NAMES.index(OLS)], s, u, legal)[3]
    slate, _ = model.options(orig, u, s, legal)
    if not len(slate):
        raise ValueError('Fixed query roster no longer proposes a switch')
    p = np.einsum('nad,adc->nac', x[q], coef)
    ref = np.minimum(p[..., 4].clip(0.), p[..., 3].clip(0.))*s[:, None]
    z = model.descriptors(slate, h, row_features[q], paths, cv, s, u, a, e, ref)
    allowed = np.array([(prior.model.gate(choice, u, a, e, ref) >= 0).any() for choice in slate])
    return slate, z, allowed


def option_targets(actions, q, slate):
    mask = slate >= 0
    pos = q[np.where(mask)[1]].reshape(len(slate), -1)
    y = actions[pos, slate[mask].reshape(pos.shape)].sum(1)
    y[:, 3] = y[:, 3] > 0
    return crit.accept(y)


def fit(d, cv, bank, target, valid, cut, seed, tc, sc, dest, weights, beat):
    h, sites, crowd = (d[k] for k in ('history', 'sites', 'crowd'))
    names = np.unique(sites)
    x, scale = run.model.descriptors(h, bank, cv, crowd)
    z = prev.model.features(h, bank, cv, crowd, d['groups'])
    labels = np.stack([crit.action_targets(cv, bank[:, k], target, valid, cut) for k in range(bank.shape[1])], 1)
    moments, queries = {}, {}
    for name in names:
        ix = np.flatnonzero(sites == name)
        moments[name] = run.model.fitting_moments(x[ix], bank[ix], cv[ix], target[ix], valid[ix], scale[ix],
            1./crowd[ix], cut)
        queries[name] = [q for q in d['groups'] if sites[q[0]] == name]
    with np.load(run.checked(tc['cache']), allow_pickle=False) as p:
        np.testing.assert_array_equal(p['sources'], names)
        original = dict(zip(names, p['excluded_source_coefficients']))
    with np.load(run.checked(sc['artifact']), allow_pickle=False) as p:
        np.testing.assert_array_equal(p['ids'], d['global_ids'])
        support, gated = p['choices'][[2, 5]]
    selected = np.full((2, len(h)), -1, dtype=np.int16)
    models, q_anchor, p_top, slate_counts = [], [], [], []
    for held in names:
        tx, ty, tw = [], [], []
        for donor in names:
            if donor == held:
                continue
            coef = run.model.coefficients([moments[k] for k in names if k not in (held, donor)])
            q_list = queries[donor]
            for q in q_list:
                slate, v, _ = slate_for_query(q, d, bank, cv, x, scale, z, coef)
                tx.append(v)
                ty.append(option_targets(labels, q, slate))
                tw.append(np.full(len(slate), 1./(len(slate)*len(q_list))))
        w = np.concatenate(tw)
        clf = HistGradientBoostingClassifier(random_state=seed, **HGB)
        clf.fit(np.concatenate(tx), np.concatenate(ty), sample_weight=w/w.mean())
        models.append(clf)
        for q in queries[held]:
            slate, v, allowed = slate_for_query(q, d, bank, cv, x, scale, z, original[held])
            control = (slate == support[q]).all(1)
            if control.sum() != 1 or (allowed[control][0] != (gated[q] >= 0).any()):
                raise ValueError('Prior support proposal/gate not reproduced on the same slate')
            p = clf.predict_proba(v)[:, 1]
            best, deployed = model.choose(p, allowed, bool((gated[q] >= 0).any()))
            selected[0, q] = slate[best]
            if deployed is not None:
                selected[1, q] = slate[deployed]
            q_anchor.append(q[0])
            p_top.append(p[best])
            slate_counts.append(len(slate))
        beat(state='held_source_done', seed=seed, done=len(models))
    tmp = weights.with_suffix('.pending')
    joblib.dump(dict(names=names, estimators=models), tmp, compress=3)
    tmp.replace(weights)
    prior.h.prune.previous.history.feedback.write_array(dest, ids=d['global_ids'], choices=selected,
        anchor=np.asarray(q_anchor), p_top=np.asarray(p_top), slate_counts=np.asarray(slate_counts))


def score(d, refs, target, valid, cut, packet, support_packet, records, beat):
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
    sites, seeds = sorted(set(roster.values())), list(run.kin.prior.SEEDS)
    rows, cells, pair_cells, changes = [], {}, {}, np.zeros(2, int)
    n_options = 0
    for cell, sup in zip(packet['cells'], support_packet['cells'], strict=True):
        seed = cell['seed']
        assert seed == sup['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['decisions']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            learned = z['choices']
            n_options += int(z['slate_counts'].sum())
        with np.load(run.checked(sup['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            controls = z['choices'][[2, 5]]
        choices = (controls[1], learned[1], controls[0], learned[0])
        paths = {BASE: cv, OLS: bank[:, run.NAMES.index(OLS)]}
        paths.update({k: run.model.selected_paths(a, bank, cv) for k, a in zip(ARMS[2:], choices)})
        moments = [crit.action_targets(cv, paths[k], target, valid, cut['easy']) for k in ARMS[2:]]
        for r, (ix, groups) in records.items():
            counts, pairs = [], []
            view = {k: v[ix] for k, v in paths.items()}
            for g in groups:
                q = ix[g]
                active = [(c[q] >= 0).sum() for c in choices]
                if active[0] != active[1] or active[2] != active[3]:
                    raise ValueError('Counts must match at each query, not only in aggregate')
                changes += [int((a[q] != b[q]).any()) for a, b in (choices[:2], choices[2:])]
                verdict = []
                for choice, action in zip(choices, moments):
                    taken = q[choice[q] >= 0]
                    verdict.append(bool(crit.accept(crit.targets(action[taken], 1.)[None])[0]))
                counts.append(verdict)
                _, rel, _, _ = pair.query(d['history'][q], np.stack([view[k][g] for k in ARMS[2:]]),
                    target[q], valid[q], [(0, 1), (2, 3)])
                pairs.append(rel)
            row = run.parent.readout.score_recording(view, PAIRS, target[ix], valid[ix], d['queries'][ix],
                r, easy_cut=cut['easy'], hard_cut=cut['hard'])
            site = roster[r]
            rows.append(dict(row, seed=seed, source=site))
            cells.setdefault((site, seed), []).append(np.asarray(counts))
            pair_cells.setdefault((site, seed), []).extend(pairs)
        beat(state='readout_seed', seed=seed)
    grouped = prior.grouped_readout(rows, roster, seeds)
    old = run.read(prev.PUBLIC/'analysis.json')['grouped']
    for k in (BASE, OLS, 'gated_support', 'support_retrieved'):
        if grouped['risk'][k] != old['risk'][k]:
            raise ValueError('Support control failed exact replay')
    view_counts = np.stack([a.any(0) for v in cells.values() for a in v]).sum(0)
    counts = np.stack([a.sum(0) for v in cells.values() for a in v]).sum(0)
    data = np.asarray([[np.concatenate(cells[s, k]).mean(0) for k in seeds] for s in sites])
    geom = np.asarray([[pair.mean_finite(pair_cells[s, k]) for k in seeds] for s in sites])
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    return dict(grouped=grouped, feasible_queries=dict(zip(ARMS[2:], counts.tolist())),
        views_with_feasible=dict(zip(ARMS[2:], view_counts.tolist())), slate_options=n_options,
        changed_queries=dict(zip(('matched', 'all_top1'), changes.tolist())),
        feasibility_pp={key: pair.bootstrap(100.*(data[:, :, j]-data[:, :, j-1]), draw)
                        for key, j in (('matched', 1), ('all_top1', 3))},
        pair_geometry={key: {v: pair.bootstrap(geom[:, :, i, j], draw) for j, v in enumerate(pair.PAIR_FIELDS)}
                       for key, i in (('matched', 0), ('all_top1', 1))},
        original_rules_unchanged=True, external_outcomes_opened=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = p.parse_args().phase
    transport, support, cut = register()
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
            row = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            print(row, flush=True)

        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
        freeze = PUBLIC/'freeze.json'
        if phase == 'fit':
            cells = []
            for seed, tc, sc in zip(run.kin.prior.SEEDS, transport['cells'], support['cells'], strict=True):
                assert seed == tc['seed'] == sc['seed']
                cache = PRIVATE/f'seed_{seed}.npz'
                mark, weight = cache.with_suffix('.json'), cache.with_suffix('.joblib')
                if not mark.exists():
                    cv, bank = run.candidates(d, refs, seed)
                    fit(d, cv, bank, y, valid, cut['easy'], seed, tc, sc, cache, weight, beat)
                    run.once(mark, dict(seed=seed, registration=run.sha(REG),
                        decisions=run.artifact(cache), checkpoint=run.artifact(weight)))
                row = run.read(mark)
                if row['seed'] != seed or row['registration'] != run.sha(REG):
                    raise ValueError('Resume identity differs')
                for k in ('decisions', 'checkpoint'):
                    run.checked(row[k])
                cells.append(row)
            run.once(freeze, dict(cells=cells, phase='decisions_frozen', registration=run.sha(REG)))
            out = freeze
        else:
            packet = run.read(freeze)
            if packet['registration'] != run.sha(REG) or run.artifact(freeze) != run.read(PUBLIC/'fit_execution.json')['artifact']:
                raise ValueError('Require sealed decisions before scoring')
            out = PUBLIC/'analysis.json'
            if phase == 'replay' and run.artifact(out) != run.read(PUBLIC/'score_execution.json')['artifact']:
                raise ValueError('Require verified score before replay')
            records = prior.h.previous.prior.blocks(d, lab['frames'])
            run.once(out, run.stable(score(d, refs, y, valid, cut, packet, support, records, beat)))
        receipt = PUBLIC/f'{phase}_execution.json'
        doc = dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out))
        if not receipt.exists():
            run.once(receipt, doc)
        elif run.read(receipt)['artifact'] != doc['artifact']:
            raise ValueError('Execution refers to a different artifact')


if __name__ == '__main__':
    main()
