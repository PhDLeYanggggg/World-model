"""Compare learned candidate costs while fixing per-query intervention counts."""
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

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_calibrated_amplitude as previous
from src.world_model import m3w_quarter_allocation as model

h, run, np = previous.h, previous.run, previous.np
PUBLIC = previous.PUBLIC.parent/'moving_quarter_allocation_v1'
PRIVATE = run.PRIVATE/'quarter_allocation'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_quarter_allocation.py')
ARMS = ('quarter', 'gain_only', 'gain_harm', 'donor_calibrated')


def register():
    a, receipt = (previous.PUBLIC/k for k in ('analysis.json', 'replay_execution.json'))
    if run.read(receipt)['artifact']['sha256'] != run.sha(a) or run.read(a)['external_labels_opened']:
        raise ValueError('Require verified amplitude replay with external labels sealed')
    paths = (a, receipt, previous.REG, Path(previous.__file__), Path(h.prune.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in paths], arms=ARMS, prices=model.PRICES,
        steps=model.STEPS, batch=model.BATCH, alpha=.25, budget=.02,
        primary='gain_harm_vs_gain_only', external_outcomes_opened=False)))


def design(m, ix, groups, co):
    dc = h.prune.frozen_direct(m, ix, groups, co)[0]
    scale = m.scale[ix]
    u, all_r, easy_r, legal = run.model.decision_costs(m.x[ix], co, m.bank[ix], m.floor[ix], scale)
    active = np.zeros(len(dc), bool)
    for q in groups:
        if (dc[q] >= 0).any():
            active[q] = True
    i, k = np.nonzero(legal & active[:, None])
    x = np.column_stack((m.x[ix[i], k], np.eye(legal.shape[1])[k], dc[i] == k, dc[i] >= 0,
                         *[np.arcsinh(v[i, k]/scale[i]) for v in (u, all_r, easy_r)]))
    return dc, legal, i, k, x


def paths(m, ix, choices):
    c, b = m.floor[ix], m.bank[ix]
    return {name: previous.parent.model.shrink(c, run.model.selected_paths(z, b, c), .25)
            for name, z in choices.items()}


def readout(m, ix, rec, choices, contrasts, y, valid, cuts):
    p = run.parent
    site = str(m.d['sites'][ix[0]])
    return p.readout.score_recording({p.CV: m.floor[ix], **paths(m, ix, choices)},
        {k: (*v, p.CV) for k, v in contrasts.items()}, y[ix], valid[ix],
        m.d['queries'][ix], rec, easy_cut=cuts[site]['easy'], hard_cut=cuts[site]['hard'])


def choose_price(records):
    eligible = []
    for j, price in enumerate(model.PRICES):
        arm = str(price)
        gain = 0.
        new = lost = violations = 0
        for record in records:
            old, new_r = (record['metrics'][k]['risk'] for k in ('quarter', arm))
            new += int(new_r['supported'] and not old['supported'])
            lost += int(old['supported'] and not new_r['supported'])
            violations += int(new_r['known_easy_net_degradation'] is not None
                and new_r['known_easy_net_degradation'] > .02
                and (old['known_easy_net_degradation'] or 0.) <= .02)
            pair = record['comparisons'][arm]
            gain += pair['known_delta']-pair['unknown_radius']
        if new and not lost and not violations and gain >= 0.:
            eligible.append((new, gain, -j, price))
    return max(eligible)[-1] if eligible else None


def fit(m, controls, records, y, valid, beat):
    d, cuts = m.d, run.cuts_by_source()
    original = {s: run.model.fitting_moments(m.x[ix], m.bank[ix], m.floor[ix], y[ix], valid[ix],
                    m.scale[ix], 1./d['crowd'][ix], cuts[s]['easy']) for s, ix in m.ix.items()}
    rec_by_start = {int(ix[0]): rec for rec, (ix, _) in records.items()}
    folds = []
    for item in controls:
        r = item['identity']
        site, learn, cal = r['held'], r['fit'], r['cal_a']+r['cal_b']
        if (len(learn), len(cal), len(set(learn+cal+[site]))) != (7, 6, 14):
            raise ValueError('Require disjoint source roles')
        ident = dict(seed=m.seed, held=site, fit=learn, cal=cal, registration=run.sha(REG))
        mark = PRIVATE/f'{m.seed}_{site}.json'
        if not mark.exists():
            blocks = []
            for s in learn:
                co = run.model.coefficients([original[t] for t in learn if t != s])
                local = []
                for ix, g in m.groups[s]:
                    _, legal, i, k, x = design(m, ix, g, co)
                    if len(i):
                        t = model.targets(m.floor[ix], m.bank[ix], y[ix], valid[ix], i, k, cuts[s]['easy'])
                        w = 1./(d['crowd'][ix[i]]*legal[i].sum(1))
                        local.append((x, t/m.scale[ix[i], None], w/w.sum()))
                blocks.extend((x, t, w/len(local)) for x, t, w in local)
            x, t, w = [np.concatenate(v) for v in zip(*blocks)]
            center = np.average(x, weights=w, axis=0)
            spread = np.sqrt(np.average((x-center)**2, weights=w, axis=0)).clip(.01)

            def normal(a):
                return np.clip((a-center)/spread, -8., 8.).astype(np.float32)

            seed = int.from_bytes(hashlib.sha256(f'quarter:{m.seed}:{site}'.encode()).digest()[:4], 'little')
            ck = mark.with_suffix('.pt')
            net = model.train(normal(x), t, w, seed, ck,
                lambda **v: beat(state='training', seed=m.seed, source=site, **v))
            co = m.co[site]

            def estimate(ix, g):
                dc, legal, i, k, x = design(m, ix, g, co)
                cost = np.zeros((*legal.shape, 5))
                cost[i, k] = model.predict(net, normal(x))*m.scale[ix[i], None]
                return dc, legal, cost

            donors = []
            for s in cal:
                for ix, g in m.groups[s]:
                    dc, legal, cost = estimate(ix, g)
                    choices = {'quarter': dc, **{str(p): model.allocate(g, dc, cost, legal, p)
                                                 for p in model.PRICES}}
                    donors.append(readout(m, ix, rec_by_start[int(ix[0])], choices,
                        {str(p): ('quarter', str(p)) for p in model.PRICES}, y, valid, cuts))
            price = choose_price(donors)
            at = m.ix[site]
            action = np.full((len(ARMS)-1, len(at)), -1, dtype=np.int16)
            for ix, g in m.groups[site]:
                dc, legal, cost = estimate(ix, g)
                np.testing.assert_array_equal(dc, m.direct[ix])
                action[:, np.searchsorted(at, ix)] = np.stack((
                    model.allocate(g, dc, cost, legal, 0.), model.allocate(g, dc, cost, legal, 1.),
                    dc if price is None else model.allocate(g, dc, cost, legal, price)))
            cache = mark.with_suffix('.npz')
            h.prune.previous.history.feedback.write_array(cache, ids=d['global_ids'][at],
                action=action, center=center, spread=spread)
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache),
                checkpoint=run.artifact(ck), price=price, fitting_candidates=len(x))))
        f = run.read(mark)
        if f['identity'] != ident:
            raise ValueError('Incompatible allocation fold')
        for artifact in (f['cache'], f['checkpoint']):
            run.checked(artifact)
        folds.append(f)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def score(m, records, frozen, y, valid):
    d, cv, cuts = m.d, run.parent.CV, run.cuts_by_source()
    action = np.full((len(ARMS)-1, len(m.floor)), -2, np.int16)
    for f in frozen['folds']:
        at = m.ix[f['identity']['held']]
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            action[:, at] = z['action']
    if (action < -1).any():
        raise ValueError('Incomplete allocation cache')
    rows = []
    for rec, (ix, groups) in records.items():
        site = str(d['sites'][ix[0]])
        dc, bank, floor = m.direct[ix], m.bank[ix], m.floor[ix]
        legal = np.linalg.norm(bank-floor[:, None], axis=-1).max(2) > 0.
        active = np.concatenate([q for q in groups if (dc[q] >= 0).any()]) if (dc >= 0).any() else []
        allowed = np.zeros(len(dc), bool)
        allowed[active] = True
        i, k = np.nonzero(legal & allowed[:, None])
        cost = np.zeros((*legal.shape, 5))
        cost[i, k] = model.targets(floor, bank, y[ix], valid[ix], i, k, cuts[site]['easy'])
        oracle = model.allocate(groups, dc, cost, legal, 1.)
        choices = dict(zip(ARMS, [dc, *action[:, ix]]), hindsight_additive=oracle)
        changed = {}
        for name, a in choices.items():
            pick = np.flatnonzero(a >= 0)
            if not legal[pick, a[pick]].all() or any((a[q] >= 0).sum() != (dc[q] >= 0).sum()
                                                   for q in groups):
                raise ValueError('Illegal action or changed query count')
            changed[name] = [int(((dc < 0) & (a >= 0)).sum()),
                             int(((dc >= 0) & (a >= 0) & (a != dc)).sum())]
        pairs = {f'{a}_vs_cv': (cv, a) for a in choices}
        pairs.update({f'{a}_vs_quarter': ('quarter', a) for a in choices if a != 'quarter'})
        pairs['gain_harm_vs_gain_only'] = ('gain_only', 'gain_harm')
        out = readout(m, ix, rec, choices, pairs, y, valid, cuts)
        if any(out['metrics'][a]['risk']['selected'] != (dc >= 0).sum() for a in choices):
            raise ValueError('Scorer disagrees with intervention count')
        rows.append(dict(out, source=site, seed=m.seed, changed=changed))
    return rows


def report(rows, roster, frozen):
    result = previous.parent.grouped_readout(rows, roster, list(run.kin.prior.SEEDS))
    old = run.read(previous.PUBLIC/'analysis.json')['grouped']
    if (result['risk']['quarter'] != old['risk']['quarter']
            or result['effects']['quarter_vs_cv'] != old['effects']['quarter_vs_cv']):
        raise ValueError('Require exact fixed-quarter replay')
    folds = [v for c in frozen['cells'] for v in c['folds']]
    run.once(PUBLIC/'analysis.json', run.stable(dict(grouped=result, models=len(folds),
        changes={a: np.sum([r['changed'][a] for r in rows], axis=0).tolist()
                 for a in rows[0]['changed']},
        fitting_candidate_occurrences=sum(f['fitting_candidates'] for f in folds),
        calibrated_prices=dict(Counter(str(f['price']) for f in folds)),
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
            raise ValueError('Commit allocation protocol before execution')
    model.torch.set_num_threads(4)
    model.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resume from the last saved update')
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            if v.get('state') != 'training' or v.get('step') == model.STEPS:
                print(msg, flush=True)

        beat(state='start')
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = h.previous.prior.blocks(d, lab['frames'])
        roster = {rec: str(d['sites'][ix[0]]) for rec, (ix, _) in records.items()}
        y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
        cells, controls, ranks = h.cost.banks()
        out = PRIVATE/'freeze.json'
        frozen = {}
        if phase != 'fit':
            if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(out):
                raise ValueError('Require frozen allocation')
            packet = run.read(out)
            frozen = {c['seed']: c for c in packet['cells']}
        if phase == 'replay' and not (PUBLIC/'score_execution.json').exists():
            raise ValueError('Score before replay')
        fits, rows = [], []
        for cell in cells:
            m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
            if phase == 'fit':
                fits.append(fit(m, controls[m.seed], records, y, valid, beat))
            else:
                rows.extend(score(m, records, frozen[m.seed], y, valid))
            beat(state='seed_complete', seed=m.seed)
            del m
        if phase == 'fit':
            run.once(out, run.stable(dict(cells=fits, held_excluded=True)))
        else:
            report(rows, roster, packet)
            out = PUBLIC/'analysis.json'
        receipt = dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out),
            python_architecture=platform.machine(), torch_threads=model.torch.get_num_threads())
        p = PUBLIC/f'{phase}_execution.json'
        if p.is_file():
            if run.read(p)['artifact'] != receipt['artifact']:
                raise ValueError('Previous receipt bound a different artifact')
        else:
            run.once(p, receipt)


if __name__ == '__main__':
    main()
