"""Fit and freeze count-preserving amplitude controllers on source-separated development."""
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
from scripts import run_m3w_frozen_amplitude as parent
from src.world_model import m3w_calibrated_amplitude as model

h, run, np = parent.h, parent.run, parent.np
PUBLIC = parent.PUBLIC.parent/'moving_calibrated_amplitude_v1'
PRIVATE = run.PRIVATE/'calibrated_amplitude'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_calibrated_amplitude.py')
ARMS = ('direct', 'quarter', 'uniform_calibrated', 'adaptive_calibrated', 'matched_query_budget')


def register():
    a, replay = (parent.PUBLIC/n for n in ('analysis.json', 'replay_execution.json'))
    if run.read(replay)['analysis_sha256'] != run.sha(a) or run.read(a)['external_labels_opened']:
        raise ValueError('Require exact preceding development replay')
    paths = (a, replay, parent.REG, Path(parent.__file__), Path(h.prune.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in paths], alphas=model.ALPHA.tolist(),
        prices=model.PRICES, steps=model.STEPS, batch=model.BATCH,
        original_budget=.02, primary='adaptive_calibrated_vs_quarter',
        matched_probe='matched_query_budget_vs_quarter', independent_outcomes_opened=False)))


def calibrate(records, keys, choices):
    best = None
    for j, key in enumerate(keys):
        new = lost = bad = 0
        gain = 0.
        for real, pred, base in records:
            passed, easy_bad, lower = model.observe(real, choices(pred, key))
            new += int(passed and not base[0])
            lost += int(base[0] and not passed)
            bad += int(easy_bad and not base[1])
            gain += lower
        if new and not lost and not bad and gain >= 0.:
            value = (new, gain, -j)
            best = value if best is None else max(best, value)
    return None if best is None else keys[-best[-1]]


def uniform(pred, k):
    return np.full(len(pred), model.QUARTER if k is None else k, dtype=int)


def adaptive(pred, price):
    return uniform(pred, None) if price is None else model.choose(pred, price)


def fit(m, controls, y, valid, beat):
    d, cuts = m.d, run.cuts_by_source()
    original = {s: run.model.fitting_moments(m.x[ix], m.bank[ix], m.floor[ix], y[ix], valid[ix],
                    m.scale[ix], 1./d['crowd'][ix], cuts[s]['easy']) for s, ix in m.ix.items()}

    def realized(ix, i, family, site):
        return model.realized(m.floor[ix], m.bank[ix[i], family], y[ix], valid[ix],
                              i, 1./d['crowd'][ix], cuts[site]['easy'])

    folds = []
    for item in controls:
        r = item['identity']
        site, learn, cal = r['held'], r['fit'], r['cal_a']+r['cal_b']
        ident = dict(seed=m.seed, held=site, fit=learn, cal=cal, registration=run.sha(REG))
        if (len(learn), len(cal), len(set(learn+cal+[site]))) != (7, 6, 14):
            raise ValueError('Require disjoint source roles')
        mark = PRIVATE/f'{m.seed}_{site}.json'
        if not mark.exists():
            blocks = []
            for s in learn:
                co = run.model.coefficients([original[t] for t in learn if t != s])
                local = []
                for ix, g in m.groups[s]:
                    _, i, k, x, _, _ = h.prune.frozen_direct(m, ix, g, co)
                    if len(i):
                        t = realized(ix, i, k, s)
                        w = t['weights']
                        local.append((x, t['target']/m.scale[ix[i], None, None], w/w.sum()))
                blocks.extend((x, y, w/len(local)) for x, y, w in local)
            x, target, weight = [np.concatenate(v) for v in zip(*blocks)]
            center = np.average(x, weights=weight, axis=0)
            spread = np.sqrt(np.average((x-center)**2, weights=weight, axis=0)).clip(.01)

            def normal(x):
                return np.clip((x-center)/spread, -8., 8.).astype(np.float32)

            seed = int.from_bytes(hashlib.sha256(f'amplitude:{m.seed}:{site}'.encode()).digest()[:4], 'little')
            ck = mark.with_suffix('.pt')
            net = model.train(normal(x), target, weight, seed, ck,
                lambda **v: beat(state='training', seed=m.seed, source=site, **v))
            co = m.co[site]
            donors = []
            for s in cal:
                for ix, g in m.groups[s]:
                    _, i, k, x, _, _ = h.prune.frozen_direct(m, ix, g, co)
                    real = realized(ix, i, k, s)
                    pred = model.predict(net, normal(x))
                    donors.append((real, pred, model.observe(real, uniform(pred, None))))
            uniform_key = calibrate(donors, list(range(len(model.ALPHA))), uniform)
            price = calibrate(donors, model.PRICES, adaptive)
            at = m.ix[site]
            action = np.full((3, len(at)), model.QUARTER, dtype=np.int16)
            for ix, g in m.groups[site]:
                dc, i, _, x, q, _ = h.prune.frozen_direct(m, ix, g, co)
                np.testing.assert_array_equal(dc, m.direct[ix])
                pred = model.predict(net, normal(x))
                j = np.searchsorted(at, ix[i])
                action[:, j] = np.stack((uniform(pred, uniform_key),
                                         adaptive(pred, price), model.matched(pred, q)))
            cache = mark.with_suffix('.npz')
            h.prune.previous.history.feedback.write_array(cache, ids=d['global_ids'][at],
                action=action, center=center, spread=spread)
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache),
                checkpoint=run.artifact(ck), uniform_alpha_index=uniform_key, price=price, fitting_switches=len(x))))
        f = run.read(mark)
        if f['identity'] != ident:
            raise ValueError('Incompatible amplitude fold')
        for artifact in (f['cache'], f['checkpoint']):
            run.checked(artifact)
        folds.append(f)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def score(m, records, frozen, y, valid):
    d, p, cuts = m.d, run.parent, run.cuts_by_source()
    action = np.full((3, len(m.floor)), -1, np.int16)
    for f in frozen['folds']:
        at = m.ix[f['identity']['held']]
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            action[:, at] = z['action']
    rows = []
    for rec, (ix, groups) in records.items():
        s = str(d['sites'][ix[0]])
        e, hard = cuts[s]['easy'], cuts[s]['hard']
        dc, c, b = m.direct[ix], m.floor[ix], m.bank[ix]
        original = run.model.selected_paths(dc, b, c)
        a = action[:, ix]
        if (a < 0).any() or (a >= len(model.ALPHA)).any():
            raise ValueError('Incomplete frozen amplitude cache')
        for q in groups:
            j = q[dc[q] >= 0]
            if model.ALPHA[a[2, j]].sum() != .25*len(j):
                raise ValueError('Matched per-query amplitude budget changed')
        paths = {p.CV: c, 'direct': original, 'quarter': parent.model.shrink(c, original, .25)}
        paths.update({k: parent.model.shrink(c, original, model.ALPHA[v]) for k, v in zip(ARMS[2:], a)})
        pairs = {f'{k}_vs_cv': (p.CV, k, p.CV) for k in ARMS}
        pairs.update({f'{k}_vs_quarter': ('quarter', k, p.CV) for k in ARMS[2:]})
        out = p.readout.score_recording(paths, pairs, y[ix], valid[ix], d['queries'][ix], rec,
            easy_cut=e, hard_cut=hard)
        if any(out['metrics'][k]['risk']['selected'] != int((dc >= 0).sum()) for k in ARMS):
            raise ValueError('Frozen switch roster changed')
        rows.append(dict(out, source=s, seed=m.seed,
            amplitude_hist=[np.bincount(v[dc >= 0], minlength=len(model.ALPHA)).tolist() for v in a]))
    return rows


def report(rows, roster, frozen):
    result = parent.grouped_readout(rows, roster, list(run.kin.prior.SEEDS))
    old = run.read(parent.PUBLIC/'analysis.json')['grouped']
    for k in ('direct', 'quarter'):
        if result['risk'][k] != old['risk'][k] or result['effects'][k+'_vs_cv'] != old['effects'][k+'_vs_cv']:
            raise ValueError('Original and fixed-quarter controls must replay exactly')
    folds = [v for c in frozen['cells'] for v in c['folds']]
    run.once(PUBLIC/'analysis.json', run.stable(dict(grouped=result, models=len(folds),
        fitting_switch_occurrences=sum(f['fitting_switches'] for f in folds),
        amplitude_hist=dict(zip(ARMS[2:], np.sum([r['amplitude_hist'] for r in rows], axis=0).tolist())),
        uniform_alpha_choices=dict(Counter(str(f['uniform_alpha_index']) for f in folds)),
        adaptive_prices=dict(Counter(str(f['price']) for f in folds)),
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
            raise ValueError('Commit amplitude protocol before execution')
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
        roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
        y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
        cells, controls, ranks = h.cost.banks()
        out = PUBLIC/'freeze.json'
        frozen = {}
        if phase != 'fit':
            if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(out):
                raise ValueError('Require frozen amplitudes')
            packet = run.read(out)
            frozen = {c['seed']: c for c in packet['cells']}
        records_out, rows = [], []
        for cell in cells:
            m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
            if phase == 'fit':
                records_out.append(fit(m, controls[m.seed], y, valid, beat))
            else:
                rows.extend(score(m, records, frozen[m.seed], y, valid))
            beat(state='seed_complete', seed=m.seed)
            del m
        if phase == 'fit':
            run.once(out, run.stable(dict(cells=records_out, held_excluded=True)))
        else:
            report(rows, roster, packet)
            out = PUBLIC/'analysis.json'
        receipt = dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out),
            python_architecture=platform.machine(), torch_threads=model.torch.get_num_threads())
        p = PUBLIC/f'{phase}_execution.json'
        if p.is_file():
            if run.read(p)['artifact'] != receipt['artifact']:
                raise ValueError('Previous execution bound a different artifact')
        else:
            run.once(p, receipt)


if __name__ == '__main__':
    main()
