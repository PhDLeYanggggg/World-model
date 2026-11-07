"""Read out frozen joint and sparse policies without new fitting or model selection."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_sparse_quarter_exchange as sparse
from src.evaluation import m3w_joint_path_audit as model

run, np, h = sparse.run, sparse.np, sparse.h
PUBLIC = sparse.PUBLIC.parent/'moving_joint_path_audit_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_joint_path_audit.py')
ARMS = ('cv', 'ols4', 'independent', 'scene', 'pair_additive', 'joint',
        'quarter', 'dense_quarter', 'sparse_risk', 'sparse_support')
PAIRS = (('independent', 'joint'), ('scene', 'joint'), ('pair_additive', 'joint'),
         ('cv', 'joint'), ('ols4', 'joint'), ('cv', 'quarter'),
         ('quarter', 'dense_quarter'), ('quarter', 'sparse_risk'),
         ('quarter', 'sparse_support'), ('sparse_risk', 'sparse_support'))


def register():
    names = ('analysis.json', 'score_execution.json', 'replay_execution.json')
    s = [sparse.PUBLIC/n for n in names]
    if run.read(s[2])['artifact']['sha256'] != run.sha(s[0]):
        raise ValueError('Require exact sparse result')
    j = [run.PUBLIC/n for n in (*names, 'freeze.json')]
    if not run.read(j[3])['frozen_before_readout']:
        raise ValueError('Require frozen original decisions')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in (*s, *j, Path(run.model.__file__))],
        arms=ARMS, comparisons=PAIRS, primary='joint_vs_pair_additive_grid_ade_lower',
        draws=3000, bootstrap_seed=73107, no_new_policy=True, external_outcomes_opened=False)))


def decision_arrays(m, sparse_cell, dense_cell, joint_cell):
    d = m.d
    with np.load(run.checked(joint_cell['cache']), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], d['global_ids'])
        joint = {name: z['choices'][run.model.ARMS.index(name)]
                 for name in ('independent', 'scene', 'pair_additive', 'joint')}
    control = np.full((3, len(m.direct)), -2, dtype=np.int16)
    dense = {f['identity']['held']: f for f in dense_cell['folds']}
    for f in sparse_cell['folds']:
        site = f['identity']['held']
        ix = m.ix[site]
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][ix])
            control[1:, ix] = z['action'][2:4]
        with np.load(run.checked(dense[site]['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][ix])
            control[0, ix] = z['action'][1]
    if (control < -1).any():
        raise ValueError('Frozen policies must cover every row')
    return joint, dict(zip(ARMS[7:], control), quarter=m.direct)


def run_audit():
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = h.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    j = {r['seed']: r for r in run.read(run.PUBLIC/'freeze.json')['seeds']}
    cache = [sparse.PRIVATE/'freeze.json', sparse.parent.PRIVATE/'freeze.json']
    for p, experiment in zip(cache, (sparse, sparse.parent)):
        if run.read(experiment.PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(p):
            raise ValueError('Require exact frozen decisions for new readout')
    a, b = ({r['seed']: r for r in run.read(p)['cells']} for p in cache)
    cells, controls, ranks = h.cost.banks()
    edges = [tuple(ARMS.index(k) for k in p) for p in PAIRS]
    source = sorted(set(str(x) for x in d['sites']))
    seeds = list(run.kin.prior.SEEDS)
    query_scores = {s: [] for s in source}
    metrics = {}
    support = np.zeros(4, np.int64)
    changed = np.zeros(len(PAIRS), np.int64)
    switches = {name: 0 for name in ARMS}
    changed_action = queries = 0
    for cell in cells:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        joint, other = decision_arrays(m, a[m.seed], b[m.seed], j[m.seed])
        blocks = {s: [] for s in source}
        for rec, (ix, groups) in records.items():
            floor, bank = m.floor[ix], m.bank[ix]
            full = {k: run.model.selected_paths(v[ix], bank, floor) for k, v in joint.items()}
            quarter = sparse.parent.paths(m, ix, {k: v[ix] for k, v in other.items()})
            paths = np.stack([dict(cv=floor, ols4=bank[:, run.NAMES.index(run.parent.OLS)],
                                  **full, **quarter)[k] for k in ARMS])
            for name, v in (*joint.items(), *other.items()):
                switches[name] += int((v[ix] >= 0).sum())
            switches['ols4'] += int((paths[1] != floor).any((1, 2)).sum())
            arms, pairs = [], []
            for g in groups:
                q = ix[g]
                if len({int((z[q] >= 0).sum()) for z in joint.values()}) != 1 or (
                        len({int((z[q] >= 0).sum()) for z in other.values()}) != 1):
                    raise ValueError('Require identical query counts inside each comparison family')
                changed_action += int(np.any(joint['joint'][q] != joint['pair_additive'][q]))
                ar, pair, hit, counts = model.query(d['history'][q], paths[:, g], y[q], valid[q], edges)
                arms.append(ar); pairs.append(pair)
                support += counts
                changed += hit
                queries += 1
            site = str(d['sites'][ix[0]])
            blocks[site].append((np.asarray(arms), np.asarray(pairs)))
            obs = np.asarray(arms)[:, :, 0]
            if np.isfinite(obs).any():
                metrics[rec, m.seed] = np.nanmean(obs, axis=0)
        for site in source:
            x = np.concatenate([z[0] for z in blocks[site]])
            p = np.concatenate([z[1] for z in blocks[site]])
            query_scores[site].append((m.seed, model.mean_finite(x), model.mean_finite(p),
                                       np.nanquantile(x[:, :, 0], .95, axis=0)))
        print(dict(state='seed_scored', seed=m.seed, queries=queries), flush=True)
        del m
    if len(metrics) != 102 or len(source) != 14 or len(seeds) != 3:
        raise ValueError('Keep all original recording views and sources')
    arm, pair, tail = (np.asarray([[v[k] for v in sorted(query_scores[s], key=lambda z: z[0])]
                                  for s in source]) for k in (1, 2, 3))
    draw = np.random.default_rng(73107).integers(len(source), size=(3000, len(source)))
    comparisons = {f'{b}_vs_{a}': {k: model.bootstrap(pair[:, :, i, j], draw)
                   for j, k in enumerate(model.PAIR_FIELDS)}
                   for i, (a, b) in enumerate(PAIRS)}
    old = run.read(run.PUBLIC/'analysis.json')
    for left, field in (('independent', 'pair_error_reduction_joint_vs_independent'),
                        ('pair_additive', 'pair_error_reduction_joint_vs_additive')):
        val = comparisons[f'joint_vs_{left}']['observed_ade_reduction']
        np.testing.assert_allclose([val['point'], *val['interval95']],
            [old[field]['point'], *old[field]['interval95']], rtol=1e-10, atol=1e-10)
    if changed_action != old['joint_changes_vs_additive'] or queries != old['compared_queries']:
        raise ValueError('Original matched decision comparison changed')
    for key in ('independent', 'scene', 'pair_additive', 'joint'):
        if switches[key] != old['matched_switches_per_arm']:
            raise ValueError('Original intervention totals changed')
    score = {k: {field: model.bootstrap(arm[:, :, i, j], draw) for j, field in enumerate(model.ARM_FIELDS)}
             for i, k in enumerate(ARMS)}
    for i, key in enumerate(ARMS):
        score[key]['source_averaged_p95_query_pair_ade'] = model.bootstrap(tail[:, :, i], draw)
        score[key]['worst_recording_view_pair_ade'] = max(float(z[i]) for z in metrics.values())
    return dict(comparisons=comparisons, arms=score, switches=switches, views=len(metrics),
        source_groups=len(source), queries=queries, original_pair_contrasts_reproduced=True,
        support=dict(zip(('requested_pair_steps', 'observed_pair_steps', 'requested_terminal_pairs',
                          'observed_terminal_pairs'), map(int, support))),
        changed_pair_geometry_queries={f'{b}_vs_{a}': int(changed[i]) for i, (a,b) in enumerate(PAIRS)},
        original_joint_choice_changes=changed_action, original_rules_unchanged=True,
        scientific_roles_assigned=False, external_labels_opened=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = p.parse_args().phase
    register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit audit protocol before scoring')
    start = time.monotonic()
    if phase == 'replay' and not (PUBLIC/'score_execution.json').exists():
        raise ValueError('Run score before replay')
    analysis = PUBLIC/'analysis.json'
    run.once(analysis, run.stable(run_audit()))
    info = dict(phase=phase, seconds=time.monotonic()-start, analysis_sha256=run.sha(analysis))
    receipt = PUBLIC/f'{phase}_execution.json'
    if receipt.exists():
        if run.read(receipt)['analysis_sha256'] != info['analysis_sha256']:
            raise ValueError('Receipt mismatch')
    else:
        run.once(receipt, info)
    print(info, flush=True)


if __name__ == '__main__':
    main()
