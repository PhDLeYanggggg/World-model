"""Measure support-aware retrieval with matched proposal counts and a fixed gate."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_native_query_critic as prev
from src.world_model import m3w_native_support_retrieval as model

run, np, pair, BASE, OLS = prev.run, prev.np, prev.pair, prev.BASE, prev.OLS
prior = prev.prior
PUBLIC = prior.PUBLIC.parent/'native_support_retrieval_v1'
PRIVATE = run.PRIVATE/'native_support_retrieval_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_native_support_retrieval.py')
MATCHED = ('joint', 'gain_retrieved', 'support_retrieved')
GATED = ('gated_joint', 'gated_gain', 'gated_support')
ARMS = (BASE, OLS, *MATCHED, *GATED)
PAIRS = {f'{a}_vs_cv': (BASE, a, BASE) for a in ARMS[1:]}
PAIRS.update({f'{a}_vs_ols4': (OLS, a, BASE) for a in (*MATCHED, *GATED)})
PAIRS.update(support_vs_gain=(MATCHED[1], MATCHED[2], BASE),
             gated_support_vs_gain=(GATED[1], GATED[2], BASE),
             gated_support_vs_original=(GATED[0], GATED[2], BASE))
FIELDS = ('query_easy_support', 'query_feasible', 'view_easy_support', 'view_witness',
          'gated_view_pass')


def register():
    old, cut = prev.register()
    parents = [prev.REG, prev.PUBLIC/'analysis.json', prev.PUBLIC/'replay_execution.json',
               prev.PUBLIC/'proposal_capacity.json', Path(run.model.__file__),
               Path(prior.model.__file__), Path(prev.critic.__file__)]
    result = run.artifact(prev.PUBLIC/'analysis.json')
    if (run.read(prev.PUBLIC/'replay_execution.json')['artifact'] != result
            or run.read(prev.PUBLIC/'proposal_capacity.json')['analysis'] != result):
        raise ValueError('Require the sealed previous experiment and its capacity audit')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], arms=ARMS, contrasts=PAIRS, cut=cut,
        shortlist=model.SHORTLIST, ridge=model.RIDGE, original_gate_unchanged=True,
        original_budget=.02, held_source_exclusion=True, bootstrap_draws=3000,
        scope='exposed_development_only', external_access=False)))
    return old, cut


def decide(d, bank, cv, target, valid, cut, cell):
    h, crowd, sources = (d[k] for k in ('history', 'crowd', 'sites'))
    ref, _ = prev.critic.errors(cv, target, valid)
    easy = valid.any(1) & (ref > 0.) & (ref <= cut)
    z = model.features(h, bank, cv, crowd, d['groups'])
    probability = np.empty(len(z))
    folds = []
    with np.load(run.checked(cell['cache']), allow_pickle=False) as f:
        names = np.unique(sources)
        np.testing.assert_array_equal(f['sources'], names)
        coef = dict(zip(names, f['excluded_source_coefficients']))
    for name in names:
        fitted = model.train(z, easy, sources, crowd, name)
        ix = sources == name
        probability[ix] = model.probability(z[ix], fitted)
        folds.append(np.concatenate(fitted))
    choice = np.full((6, len(h)), -1, dtype=np.int16)
    for q in d['groups']:
        s = sources[q[0]]
        x, scale = run.model.descriptors(h[q], bank[q], cv[q], crowd[q])
        u, a, e, legal = run.model.decision_costs(x, coef[s], bank[q], cv[q], scale)
        ols = bank[q, run.NAMES.index(OLS)]
        original = run.model.choose_query(h[q], cv[q], bank[q], ols, scale, u, legal)[3]
        matched = (original, *model.rerank(original, u, scale, legal, probability[q]))
        if len({int((v >= 0).sum()) for v in matched}) != 1:
            raise ValueError('Agent slots per query changed')
        p = np.einsum('nad,adc->nac', x, coef[s])
        support = np.minimum(p[..., 4].clip(0.), p[..., 3].clip(0.))*scale[:, None]
        gates = [prior.model.gate(v, u, a, e, support) for v in matched]
        choice[:, q] = np.stack((*matched, *gates))
    return choice, probability, np.stack(folds)


def score(d, refs, y, valid, records, cut, packet, beat):
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
    sites, seeds = sorted(set(roster.values())), list(run.kin.prior.SEEDS)
    rows, retrieval, brier, geometry, changes = [], {}, {}, {}, np.zeros(2, dtype=int)
    for cell in packet['cells']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as f:
            np.testing.assert_array_equal(f['ids'], d['global_ids'])
            choices, p = f['choices'], f['probability']
        ref, _ = prev.critic.errors(cv, y, valid)
        easy = valid.any(1) & (ref > 0.) & (ref <= cut['easy'])
        paths = {BASE: cv, OLS: bank[:, run.NAMES.index(OLS)]}
        paths.update({a: run.model.selected_paths(v, bank, cv) for a, v in zip(ARMS[2:], choices)})
        targets = {a: prev.critic.action_targets(cv, paths[a], y, valid, cut['easy']) for a in MATCHED}
        for s in sites:
            ix = d['sites'] == s
            brier[s, seed] = np.average((p[ix]-easy[ix])**2, weights=1./d['crowd'][ix])
        for r, (ix, groups) in records.items():
            site = roster[r]
            out = np.zeros((3, len(FIELDS)), dtype=int)
            rec_paths = {k: v[ix] for k, v in paths.items()}
            row = run.parent.readout.score_recording(rec_paths, PAIRS, y[ix], valid[ix],
                d['queries'][ix], r, easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, seed=seed, source=site))
            for g in groups:
                q = ix[g]
                changes += [int((choices[i, q] != choices[0, q]).any()) for i in (1, 2)]
                for i, a in enumerate(MATCHED):
                    use = q[choices[i, q] >= 0]
                    if len(use):
                        t = prev.critic.targets(targets[a][use], 1.)
                        out[i, 0] += int(t[-1])
                        out[i, 1] += int(prev.critic.accept(t[None])[0])
                _, v, _, _ = pair.query(d['history'][q],
                    np.stack([rec_paths[a][g] for a in (*MATCHED, *GATED)]),
                    y[q], valid[q], [(1, 2), (4, 5)])
                geometry.setdefault((site, seed), []).append(v)
            out[:, 2:4] = out[:, :2] > 0
            out[:, 4] = [row['metrics'][a]['risk']['supported'] for a in GATED]
            retrieval.setdefault((site, seed), []).append((out, len(groups)))
        beat(state='readout_seed', seed=seed)
    grouped = prior.grouped_readout(rows, roster, seeds)
    orig = run.read(prev.PUBLIC/'analysis.json')['grouped']
    for a in (BASE, OLS, MATCHED[0], GATED[0]):
        if orig['risk'][a] != grouped['risk'][a]:
            raise ValueError('Fixed control does not replay')
    matched_counts = {grouped['risk'][a]['switches'] for a in MATCHED}
    if len(matched_counts) != 1:
        raise ValueError('Matched forecast counts changed')
    totals = sum((x for group in retrieval.values() for x, _ in group))
    capacity = run.read(prev.PUBLIC/'proposal_capacity.json')
    if (grouped['views']-totals[0, 2] != capacity['views_without_easy_in_any_original_proposal']
            or totals[0, 3] != capacity['views_with_feasible_single_query_witness']):
        raise ValueError('Original proposal capacity failed to replay')
    normalized, raw_geometry = [], []
    for s in sites:
        local, local_pair = [], []
        for seed in seeds:
            blocks = retrieval[s, seed]
            n = np.array([sum(nq for _, nq in blocks)]*2+[len(blocks)]*3)
            local.append(sum(x for x, _ in blocks)/n)
            local_pair.append(pair.mean_finite(geometry[s, seed]))
        normalized.append(local)
        raw_geometry.append(local_pair)
    normalized, geom = np.asarray(normalized), np.asarray(raw_geometry)
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    effects = {k: pair.bootstrap(100.*(normalized[:, :, 2, i]-normalized[:, :, 1, i]), draw)
               for i, k in enumerate(FIELDS)}
    rel = {name: {k: pair.bootstrap(geom[:, :, i, j], draw) for j, k in enumerate(pair.PAIR_FIELDS)}
           for i, name in enumerate(('matched_support_vs_gain', 'gated_support_vs_gain'))}
    return dict(grouped=grouped, counts=dict(zip(MATCHED, [dict(zip(FIELDS, v.tolist())) for v in totals])),
        support_vs_gain_pp=effects, relative_pair_geometry=rel, proposals=sum(n for x in retrieval.values() for _, n in x),
        changed_vs_original=dict(zip(MATCHED[1:], changes.tolist())),
        support_brier_source_weighted=float(np.mean(list(brier.values()))),
        original_replayed_exact=True, original_rules_unchanged=True, external_inputs_opened=False,
        external_outcomes_opened=False, independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False, submission_ready=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = p.parse_args().phase
    old, cut = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit code and registration before fitting')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
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
                mark = path.with_suffix('.json')
                if not mark.exists():
                    cv, bank = run.candidates(d, refs, seed)
                    v, p, folds = decide(d, bank, cv, y, valid, cut['easy'], cell)
                    prior.h.prune.previous.history.feedback.write_array(path, ids=d['global_ids'],
                        choices=v, probability=p, coefficients=folds)
                    run.once(mark, dict(seed=seed, registration=run.sha(REG), artifact=run.artifact(path)))
                row = run.read(mark)
                if row['seed'] != seed or row['registration'] != run.sha(REG):
                    raise ValueError('Resume identity differs')
                run.checked(row['artifact'])
                cells.append(row)
                beat(state='seed_decisions_frozen', seed=seed)
            run.once(freeze, dict(cells=cells, phase='decisions_frozen', registration=run.sha(REG)))
            out = freeze
        else:
            doc = run.read(freeze)
            if doc['registration'] != run.sha(REG) or run.artifact(freeze) != run.read(PUBLIC/'fit_execution.json')['artifact']:
                raise ValueError('Require frozen fit before readout')
            out = PUBLIC/'analysis.json'
            if phase == 'replay' and run.artifact(out) != run.read(PUBLIC/'score_execution.json')['artifact']:
                raise ValueError('No verified score to replay')
            records = prior.h.previous.prior.blocks(d, labels['frames'])
            run.once(out, run.stable(score(d, refs, y, valid, records, cut, doc, beat)))
        dest = PUBLIC/f'{phase}_execution.json'
        rec = dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out))
        if not dest.exists():
            run.once(dest, rec)
        elif run.read(dest)['artifact'] != rec['artifact']:
            raise ValueError('Previous execution bound a different artifact')


if __name__ == '__main__':
    main()
