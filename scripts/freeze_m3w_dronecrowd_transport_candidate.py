"""Freeze the native transport candidate and score only excluded development sources."""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_dronecrowd_exact_bank as lineage
from src.evaluation.m3w_grouped_recording_readout import grouped_readout
from src.evaluation import m3w_joint_path_audit as pair
from src.world_model import m3w_native_transport_policy as model

run, np, h = lineage.run, lineage.np, lineage.h
PUBLIC = lineage.PUBLIC.parent/'dronecrowd_transport_candidate_v1'
PRIVATE = run.PRIVATE/'native_transport_candidate'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_native_transport_policy.py')
BASE = run.parent.CV
OLS = run.parent.OLS
PAIRS = {f'{a}_vs_cv': (BASE, a, BASE) for a in (OLS, *model.ARMS)}
PAIRS.update({f'{a}_vs_ols4': (OLS, a, BASE) for a in model.ARMS})
PAIRS.update({f'joint_vs_{a}': (a, 'joint', BASE) for a in model.MATCHED[:-1]})
PAIRS['gated_joint_vs_joint'] = ('joint', 'gated_joint', BASE)
JOINT_PAIRS = ('joint_vs_independent', 'joint_vs_scene', 'joint_vs_pair_additive',
               'gated_joint_vs_cv', 'gated_joint_vs_ols4')


def register():
    a, r = (lineage.PUBLIC/k for k in ('analysis.json', 'replay_execution.json'))
    q = run.read(a)
    if (run.read(r)['analysis_sha256'] != run.sha(a) or q['external_outcomes_opened']
            or q['predictive_admission'] or q['external_controller_frozen']):
        raise ValueError('Require the pre-outcome lineage receipt')
    for k, digest in q['bindings'].items():
        if run.sha(ROOT/k) != digest:
            raise ValueError('Changed lineage input')
    candidate = q['exact_comparator_fit_cut_candidate']
    cut = {k: candidate[k] for k in ('easy', 'hard')}
    parents = (a, r, run.REG, run.parent.REG, Path(run.model.__file__), Path(pair.__file__),
               run.old.prior.PARENT/'population.json')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], cut=cut, predictors=q['neural_checkpoints'],
        cut_derivation=candidate['criterion'], risk_budget=.02, accuracy_reference=OLS,
        risk_reference=BASE, arms=model.ARMS, fixed_candidate_bank=list(run.NAMES),
        scope='candidate_and_development_rehearsal_only', external_role_registry_unchanged=True,
        external_inputs_allowed=False, external_outcomes_allowed=False, predictive_admission=False)))
    return cut


def freeze(d, refs, y, valid, cut, beat):
    sources = sorted(str(s) for s in np.unique(d['sites']))
    if len(sources) != 14:
        raise ValueError('Require the complete development controller roster')
    path = PRIVATE/'freeze.json'
    old = run.read(path) if path.exists() else None
    cells = []
    for seed in run.kin.prior.SEEDS:
        dest = PRIVATE/f'seed_{seed}.npz'
        mark = dest.with_suffix('.json')
        if not mark.exists():
            floor, bank = run.candidates(d, refs, seed)
            stats = {}
            for s in sources:
                ix = np.flatnonzero(d['sites'] == s)
                x, scale = run.model.descriptors(d['history'][ix], bank[ix], floor[ix], d['crowd'][ix])
                stats[s] = run.model.fitting_moments(x, bank[ix], floor[ix], y[ix], valid[ix], scale,
                    1./d['crowd'][ix], cut['easy'])
            loso = np.stack([run.model.coefficients([stats[s] for s in sources if s != held])
                             for held in sources])
            pooled = run.model.coefficients(list(stats.values()))
            h.prune.previous.history.feedback.write_array(dest, sources=np.asarray(sources),
                transport_coefficients=pooled, excluded_source_coefficients=loso)
            run.once(mark, run.stable(dict(seed=seed, registration=run.sha(REG), cache=run.artifact(dest),
                transport_fit_sources=sources, development_leave_one_source_out=True)))
        row = run.read(mark)
        if row['seed'] != seed or row['registration'] != run.sha(REG) or row['transport_fit_sources'] != sources:
            raise ValueError('Resumed controller differs from its frozen identity')
        run.checked(row['cache'])
        cells.append(row)
        beat(state='coefficients_frozen', seed=seed)
    doc = run.stable(dict(cells=cells, registration=run.sha(REG), controller_candidate_frozen=True,
        external_admission=False, external_confirmatory_execution=False))
    if old is not None and old != doc:
        raise ValueError('A different controller is already frozen')
    run.once(path, doc)
    return path


def score(d, refs, y, valid, records, cut, packet, beat):
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
    sources, seeds = sorted(set(roster.values())), list(run.kin.prior.SEEDS)
    rows, pair_cells = [], {}
    comparisons = [(PAIRS[k][0], PAIRS[k][1]) for k in JOINT_PAIRS]
    names = (BASE, OLS, *model.ARMS)
    lookup = [tuple(names.index(k) for k in p) for p in comparisons]
    for cell in packet['cells']:
        seed = cell['seed']
        floor, bank = run.candidates(d, refs, seed)
        ols = bank[:, run.NAMES.index(OLS)]
        with np.load(run.checked(cell['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['sources'], sources)
            coefficients = dict(zip(sources, z['excluded_source_coefficients']))
        site_pairs = {s: [] for s in sources}
        for rec, (ix, groups) in records.items():
            site = roster[rec]
            paths = {BASE: floor[ix], OLS: ols[ix]}
            paths.update({a: np.empty_like(paths[BASE]) for a in model.ARMS})
            moments = []
            for g in groups:
                q = ix[g]
                history = d['history'][q]
                decision = model.decide(history, bank[q], floor[q], ols[q], d['crowd'][q], coefficients[site])
                for name, choice in decision.items():
                    paths[name][g] = run.model.selected_paths(choice, bank[q], floor[q])
                _, v, _, _ = pair.query(history, np.stack([paths[n][g] for n in names]), y[q], valid[q], lookup)
                moments.append(v)
            row = run.parent.readout.score_recording(paths, PAIRS, y[ix], valid[ix], d['queries'][ix], rec,
                                                      easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, seed=seed, source=site))
            site_pairs[site].extend(moments)
        for site, values in site_pairs.items():
            pair_cells[site, seed] = pair.mean_finite(values)
        beat(state='excluded_source_rehearsal_scored', seed=seed, views=len(rows))
    grouped = grouped_readout(rows, roster, seeds)
    total = [grouped['risk'][a]['switches'] for a in model.MATCHED]
    if len(set(total)) != 1:
        raise ValueError('Matched intervention counts changed')
    a = np.asarray([[pair_cells[s, k] for k in seeds] for s in sources])
    draw = np.random.default_rng(73107).integers(len(sources), size=(3000, len(sources)))
    pair_stats = {name: {key: pair.bootstrap(a[:, :, i, j], draw) for j, key in enumerate(pair.PAIR_FIELDS)}
                  for i, name in enumerate(JOINT_PAIRS)}
    return dict(grouped=grouped, pair_contrasts=pair_stats, common_cut=cut, fixed_coefficients=True,
        no_policy_selected_on_rehearsal_outcomes=True, external_inputs_opened=False, external_outcomes_opened=False,
        external_prediction_admission=False, independent_confirmation=False,
        original_rules_unchanged=True, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'freeze', 'score', 'replay'))
    phase = p.parse_args().phase
    cut = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit candidate definition before fitting')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            print(msg, flush=True)

        d, refs = run.population(), run.register()
        labels = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = h.previous.prior.blocks(d, labels['frames'])
        y, valid = labels['target_eval']-d['origin'][:, None], labels['valid']
        if phase == 'freeze':
            out = freeze(d, refs, y, valid, cut, beat)
        else:
            frozen = PRIVATE/'freeze.json'
            if run.read(PUBLIC/'freeze_execution.json')['artifact']['sha256'] != run.sha(frozen):
                raise ValueError('Require complete frozen controller')
            if phase == 'replay' and not (PUBLIC/'score_execution.json').exists():
                raise ValueError('Score before replay')
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(d, refs, y, valid, records, cut, run.read(frozen), beat)))
        receipt = dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out),
                       python_architecture=platform.machine())
        dest = PUBLIC/f'{phase}_execution.json'
        if dest.exists():
            if run.read(dest)['artifact'] != receipt['artifact']:
                raise ValueError('Previous execution bound a different artifact')
        else:
            run.once(dest, receipt)


if __name__ == '__main__':
    main()
