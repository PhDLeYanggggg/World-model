"""Fit and score causal sparse exchanges against matched-count controls."""
import argparse
from collections import Counter
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_quarter_swap_capacity as capacity
from src.world_model import m3w_sparse_quarter_exchange as model

parent, h, run, np = capacity.parent, capacity.h, capacity.run, capacity.np
torch = parent.model.torch
PUBLIC = parent.PUBLIC.parent/'moving_sparse_quarter_exchange_v1'
PRIVATE = run.PRIVATE/'sparse_quarter_exchange'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_sparse_quarter_exchange.py')
ARMS = ('risk_probe', 'support_probe', 'risk_calibrated', 'support_calibrated')


def register():
    a, r = (capacity.PUBLIC/name for name in ('analysis.json', 'replay_execution.json'))
    if run.read(r)['analysis_sha256'] != run.sha(a) or run.read(a)['external_outcomes_opened']:
        raise ValueError('Require replayed, exposed capacity result')
    paths = (a, r, capacity.REG, Path(capacity.model.__file__), parent.REG, Path(parent.__file__),
             Path(parent.model.__file__), parent.PUBLIC/'analysis.json',
             parent.PUBLIC/'replay_execution.json')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in paths], cap=model.CAP, thresholds=model.THRESHOLDS,
        support_weight=model.SUPPORT_WEIGHT, pairs_per_query=model.FITTING_PAIRS_PER_QUERY,
        steps=parent.model.STEPS, batch=parent.model.BATCH, budget=.02,
        primary='support_calibrated_vs_quarter', external_outcomes_opened=False)))


def design(m, ix, groups, co):
    dc, legal, i, k, x = parent.design(m, ix, groups, co)
    moves, query = model.roster(groups, dc, legal)
    action = np.zeros((*legal.shape, x.shape[1]), np.float32)
    action[i, k] = x
    return dc, moves, query, action


def samples(m, ix, groups, co, y, valid, cut, rng):
    dc, moves, query, action = design(m, ix, groups, co)
    if not len(moves):
        return None
    truth = capacity.model.prepare(m.floor[ix], m.bank[ix], dc, groups, y[ix], valid[ix], cut)
    np.testing.assert_array_equal(moves, truth['moves'])
    np.testing.assert_array_equal(query, truth['query'])
    blocks = np.split(np.arange(len(query)), np.flatnonzero(np.diff(query))+1)
    sample = [rng.choice(i, model.FITTING_PAIRS_PER_QUERY, replace=False)
              if len(i) > model.FITTING_PAIRS_PER_QUERY else i for i in blocks]
    take = np.concatenate(sample)
    weight = np.repeat([1./len(i) for i in sample], [len(i) for i in sample])
    x, scale = model.pair_features(action, dc, moves[take], m.scale[ix])
    query_size = np.asarray([len(g) for g in groups])[query[take]]
    t = np.column_stack((truth['gains'][take]*query_size, truth['changes'][take]))
    t[:, :4] /= scale[:, None]
    return x, t, weight/weight.sum()


def options(net, normal, m, ix, groups, co):
    dc, moves, q, action = design(m, ix, groups, co)
    predictions = []
    for i in range(0, len(moves), 4096):
        x, _ = model.pair_features(action, dc, moves[i:i+4096], m.scale[ix])
        predictions.append(model.predict(net, normal(x)))
    pred = np.concatenate(predictions) if predictions else np.empty((0, 5))
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite pair prediction')
    return dc, moves, [model.first_options(pred, q, support=s) for s in (False, True)]


def donor_selection(records, names):
    eligible = []
    for j, name in enumerate(names):
        new = lost = easy_bad = 0
        gain = 0.
        for record in records:
            before, after = (record['metrics'][k]['risk'] for k in ('quarter', name))
            new += int(after['supported'] and not before['supported'])
            lost += int(before['supported'] and not after['supported'])
            easy_bad += int((after['known_easy_net_degradation'] or 0.) > .02
                            and (before['known_easy_net_degradation'] or 0.) <= .02)
            pair = record['comparisons'][name]
            gain += pair['known_delta']-pair['unknown_radius']
        if new and not lost and not easy_bad and gain >= 0.:
            eligible.append((new, gain, -j))
    return model.THRESHOLDS[-max(eligible)[-1]] if eligible else None


def fit(m, controls, records, y, valid, beat):
    d, cuts = m.d, run.cuts_by_source()
    stats = {s: run.model.fitting_moments(m.x[ix], m.bank[ix], m.floor[ix], y[ix], valid[ix],
        m.scale[ix], 1./d['crowd'][ix], cuts[s]['easy']) for s, ix in m.ix.items()}
    rec_at = {int(ix[0]): rec for rec, (ix, _) in records.items()}
    names = [[f'{a}_{j}' for j in range(len(model.THRESHOLDS))] for a in ('risk', 'support')]
    folds = []
    for item in controls:
        r = item['identity']
        site, learn, cal = r['held'], r['fit'], r['cal_a']+r['cal_b']
        if (len(learn), len(cal), len(set(learn+cal+[site]))) != (7, 6, 14):
            raise ValueError('Require disjoint source roles')
        ident = dict(seed=m.seed, held=site, fit=learn, cal=cal, registration=run.sha(REG))
        mark = PRIVATE/f'{m.seed}_{site}.json'
        if not mark.exists():
            seed = int.from_bytes(hashlib.sha256(f'sparse-quarter:{m.seed}:{site}'.encode()).digest()[:4], 'little')
            rng = np.random.default_rng(seed)
            blocks = []
            for s in learn:
                co = run.model.coefficients([stats[t] for t in learn if t != s])
                local = [samples(m, ix, g, co, y, valid, cuts[s]['easy'], rng) for ix, g in m.groups[s]]
                local = [block for block in local if block is not None]
                blocks.extend((x, t, w/len(local)) for x, t, w in local)
            x, t, w = [np.concatenate(v) for v in zip(*blocks)]
            center = np.average(x, weights=w, axis=0)
            spread = np.sqrt(np.average((x-center)**2, weights=w, axis=0)).clip(.01)

            def normal(a):
                return np.clip((a-center)/spread, -8., 8.).astype(np.float32)

            ck = mark.with_suffix('.pt')
            net = model.train(normal(x), t, w, seed, ck,
                lambda **v: beat(state='training', seed=m.seed, source=site, **v))
            co = m.co[site]
            donors = []
            for s in cal:
                for ix, g in m.groups[s]:
                    dc, moves, opts = options(net, normal, m, ix, g, co)
                    picks = {name: model.apply(dc, moves, *v, z) for keys, v in zip(names, opts)
                             for name, z in zip(keys, model.THRESHOLDS)}
                    donors.append(parent.readout(m, ix, rec_at[int(ix[0])], dict(quarter=dc, **picks),
                        {k: ('quarter', k) for k in picks}, y, valid, cuts))
            thresholds = [donor_selection(donors, name) for name in names]
            at = m.ix[site]
            action = np.full((len(ARMS), len(at)), -2, dtype=np.int16)
            for ix, g in m.groups[site]:
                dc, moves, opts = options(net, normal, m, ix, g, co)
                np.testing.assert_array_equal(dc, m.direct[ix])
                action[:, np.searchsorted(at, ix)] = np.stack(
                    [model.apply(dc, moves, *v, 0.) for v in opts]
                    +[model.apply(dc, moves, *v, z) for v, z in zip(opts, thresholds)])
            cache = mark.with_suffix('.npz')
            h.prune.previous.history.feedback.write_array(cache, ids=d['global_ids'][at], action=action)
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache),
                checkpoint=run.artifact(ck), thresholds=thresholds, fitting_pairs=len(x))))
        f = run.read(mark)
        if f['identity'] != ident:
            raise ValueError('Incompatible sparse exchange fold')
        for artifact in (f['cache'], f['checkpoint']):
            run.checked(artifact)
        folds.append(f)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def score(m, records, frozen, dense, y, valid):
    d, cv, cuts = m.d, run.parent.CV, run.cuts_by_source()
    dense_folds = {r['identity']['held']: r for r in dense['folds']}
    action = np.full((len(ARMS)+1, len(m.direct)), -2, dtype=np.int16)
    for f in frozen['folds']:
        site = f['identity']['held']
        at = m.ix[site]
        for record, dest, arm in ((f, slice(0, len(ARMS)), slice(None)),
                                  (dense_folds[site], len(ARMS), 1)):
            with np.load(run.checked(record['cache']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
                action[dest, at] = z['action'][arm]
    if (action < -1).any():
        raise ValueError('Incomplete action cache')
    rows = []
    for rec, (ix, groups) in records.items():
        dc = m.direct[ix]
        choices = dict(zip((*ARMS, 'gain_harm'), action[:, ix]), quarter=dc)
        edits = {}
        for name, v in choices.items():
            inserted = (dc < 0) & (v >= 0)
            if any((dc[q] >= 0).sum() != (v[q] >= 0).sum() for q in groups):
                raise ValueError('Query count changed')
            if name in ARMS and (inserted.sum() > model.CAP or any(inserted[q].sum() > 1 for q in groups)
                    or ((v != dc) & (dc >= 0) & (v >= 0)).any()):
                raise ValueError('Sparse action changed its registered budget')
            edits[name] = int(inserted.sum())
        pairs = {f'{a}_vs_cv': (cv, a) for a in choices}
        pairs.update({f'{a}_vs_quarter': ('quarter', a) for a in choices if a != 'quarter'})
        pairs.update(support_probe_vs_risk_probe=('risk_probe', 'support_probe'),
                     support_calibrated_vs_risk_calibrated=('risk_calibrated', 'support_calibrated'))
        scored = parent.readout(m, ix, rec, choices, pairs, y, valid, cuts)
        if any(scored['metrics'][a]['risk']['selected'] != (dc >= 0).sum() for a in choices):
            raise ValueError('Scorer disagrees with frozen intervention count')
        rows.append(dict(scored, source=str(d['sites'][ix[0]]), seed=m.seed, edits=edits))
    return rows


def report(rows, roster, frozen):
    result = parent.previous.parent.grouped_readout(rows, roster, list(run.kin.prior.SEEDS))
    old = run.read(parent.PUBLIC/'analysis.json')['grouped']
    for name in ('quarter', 'gain_harm'):
        if (result['risk'][name] != old['risk'][name]
                or result['effects'][f'{name}_vs_cv'] != old['effects'][f'{name}_vs_cv']):
            raise ValueError('Original and dense controls must replay exactly')
    folds = [v for cell in frozen['cells'] for v in cell['folds']]
    run.once(PUBLIC/'analysis.json', run.stable(dict(grouped=result, models=len(folds),
        fitting_pairs=sum(f['fitting_pairs'] for f in folds),
        replaced_agents={a: sum(r['edits'][a] for r in rows) for a in rows[0]['edits']},
        thresholds={name: dict(Counter(str(f['thresholds'][j]) for f in folds))
                    for j, name in enumerate(('risk', 'support'))},
        original_rules_unchanged=True, independent_confirmation=False, external_labels_opened=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = parser.parse_args().phase
    register()
    if phase == 'register':
        return
    for p in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))],
                                   cwd=ROOT) != p.read_bytes():
            raise ValueError('Commit exchange protocol before execution')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resume from saved exchange checkpoints')
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            if v.get('state') != 'training' or v.get('step') == parent.model.STEPS:
                print(msg, flush=True)

        beat(state='start')
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = h.previous.prior.blocks(d, lab['frames'])
        roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
        y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
        cells, controls, ranks = h.cost.banks()
        out = PRIVATE/'freeze.json'
        if phase != 'fit':
            if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(out):
                raise ValueError('Require frozen sparse choices')
            dense = run.read(parent.PRIVATE/'freeze.json')
            if run.read(parent.PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(parent.PRIVATE/'freeze.json'):
                raise ValueError('Require unchanged frozen dense choices')
            packet = run.read(out)
            frozen = {c['seed']: c for c in packet['cells']}
            dense_by_seed = {c['seed']: c for c in dense['cells']}
        if phase == 'replay' and not (PUBLIC/'score_execution.json').exists():
            raise ValueError('Run score before replay')
        fits, rows = [], []
        for cell in cells:
            m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
            if phase == 'fit':
                fits.append(fit(m, controls[m.seed], records, y, valid, beat))
            else:
                rows.extend(score(m, records, frozen[m.seed], dense_by_seed[m.seed], y, valid))
            beat(state='seed_complete', seed=m.seed)
            del m
        if phase == 'fit':
            run.once(out, run.stable(dict(cells=fits, held_excluded=True)))
        else:
            report(rows, roster, packet)
            out = PUBLIC/'analysis.json'
        receipt = dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out),
            python_architecture=platform.machine(), torch_threads=torch.get_num_threads())
        p = PUBLIC/f'{phase}_execution.json'
        if p.is_file():
            if run.read(p)['artifact'] != receipt['artifact']:
                raise ValueError('Previous execution bound a different artifact')
        else:
            run.once(p, receipt)


if __name__ == '__main__':
    main()
