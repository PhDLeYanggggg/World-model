"""Train source-separated context adapters and score matched original recipients."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_gated_motion_residual as prior
from src.evaluation import m3w_joint_path_audit as pair
from src.world_model.m3w_neighbor_motion_adapter import ARMS, NeighborMotionAdapter

run, prev, np, torch = prior.run, prior.prev, prior.np, prior.torch
PUBLIC = prior.PUBLIC.parent/'native_neighbor_motion_adapter_v1'
PRIVATE = prior.PRIVATE.parent/'native_neighbor_motion_adapter_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), Path(NeighborMotionAdapter.forward.__code__.co_filename),
         PUBLIC/'protocol.md', ROOT/'tests/test_m3w_neighbor_motion_adapter.py')
BASE, OLS, GAIN = prior.BASE, prior.OLS, prior.CONTROL
PATH_NAMES = (GAIN, *ARMS)
CONTRASTS = ((ARMS[0], ARMS[1]), (ARMS[1], ARMS[2]), *((GAIN, a) for a in ARMS))
PAIRS = {f'{b}_vs_{a}': (a, b, BASE) for a, b in CONTRASTS}


def register():
    cut, jobs = prior.register()
    trained = prev.receipt(prior.PUBLIC, 'train')
    if trained['overlap'] or not trained['initialization_and_draws_matched']:
        raise ValueError('Require disjoint original forecaster checkpoints')
    backbones = {f['identity']['seed']: f['checkpoint'] for f in trained['fits'] if f['identity']['arm'] == 'ungated'}
    if tuple(sorted(backbones)) != prev.SEEDS:
        raise ValueError('Expected all three frozen backbones')
    binds = (*FILES, prior.REG, prior.PUBLIC/'train_execution.json', prior.PUBLIC/'replay_execution.json',
             Path(pair.__file__), Path(prior.native.__file__))
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in binds], cut=cut, backbones=backbones,
        arms=ARMS, seeds=prev.SEEDS, settings=prior.SETTINGS, exact_recipients=True, original_risk_unchanged=True,
        external_access=False, model_selection=False, data_loader_workers=0, torch_threads=2)))
    return cut, jobs, backbones


def model(arm, backbone):
    base = prior.GatedMotionResidual(False)
    base.load_state_dict(backbone)
    return NeighborMotionAdapter(base, arm)


def train(jobs, backbones, resume, beat):
    path = PRIVATE/'train.json'
    if path.exists():
        record = run.read(path)
        if record['registration'] != run.sha(REG):
            raise ValueError('Different fit registration')
        for f in record['fits']:
            run.checked(f['checkpoint'])
        return path
    d = prior.fit_source.fitting_only(jobs[0]['identity'], beat)
    if len(d['ids']) != 69746 or set(d['sites']) & set(run.old.prior.arrays(('sites',))['sites']):
        raise ValueError('Training sources overlap the exposed development roster')
    g = d['geometry'][:, :16].reshape(-1, 8, 2)
    rms = np.maximum(np.sqrt(np.square(np.diff(g, axis=1)).sum(2).mean(1)), .05)
    fold = dict(train_ids=np.arange(len(g)), held_ids=np.empty(0, dtype=int), groups=d['groups'], factors=1./rms)
    exact = prior.fit_source.source.exact
    fits = []
    for seed in prev.SEEDS:
        old = torch.load(run.checked(backbones[seed]), map_location='cpu', weights_only=False)
        if old['identity']['fit_global_ids'] != jobs[0]['identity']['train_ids_sha256']:
            raise ValueError('Unexpected backbone training roster')
        init, draw = None, None
        for arm in ARMS:
            torch.manual_seed(seed+103)
            net = model(arm, old['model'])
            if init is None:
                init = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                exact(init, net.state_dict())
            ident = dict(seed=seed, arm=arm, registration=run.sha(REG), backbone=backbones[seed]['sha256'])
            directory = PRIVATE/'fits'/f'{seed}_{arm}'
            cp = directory/'checkpoint.pt'
            out = prior.native.fit_trial(net, d, fold, seed=seed, settings=prior.SETTINGS,
                                         identity=ident, directory=directory, resume=resume and cp.exists(),
                                         heartbeat=lambda **v: beat(seed=seed, arm=arm, **v))
            if not out['complete'] or out['held_rows_sampled'] or out['total_draws'] != 900*128:
                raise ValueError('Incomplete registered fit')
            final = torch.load(cp, map_location='cpu', weights_only=False)
            exact(old['model'], net.backbone.state_dict())
            rng = tuple(final[k] for k in ('draws', 'sampler_rng', 'torch_rng'))
            if draw is None:
                draw = rng
            else:
                exact(draw, rng)
            fits.append(dict(identity=ident, result=out, checkpoint=run.artifact(cp)))
    run.once(path, run.stable(dict(fits=fits, registration=run.sha(REG), fit_rows=len(g),
                                    fit_groups=len(d['groups']), backbone_and_draws_exact=True)))
    return path


def infer(beat):
    fits = prev.receipt(PUBLIC, 'train')
    if not fits['backbone_and_draws_exact'] or len(fits['fits']) != 9:
        raise ValueError('Complete all context ablations before forecast freeze')
    d = run.population()
    arrays = run.old.prior.arrays(('geometry', 'global_ids'))
    np.testing.assert_array_equal(arrays['global_ids'], d['global_ids'])
    h = (d['history']-d['origin'][:, None]).astype(np.float32)
    np.testing.assert_array_equal(arrays['geometry'][:, :16].reshape(h.shape), h)
    ids = np.arange(len(h))
    receipts = []
    for f in fits['fits']:
        arm, seed = (f['identity'][k] for k in ('arm', 'seed'))
        path = PRIVATE/f'prediction_{seed}_{arm}.npz'
        mark = path.with_suffix('.json')
        if not mark.exists():
            state = torch.load(run.checked(f['checkpoint']), map_location='cpu', weights_only=False)
            if state['identity'] != f['identity'] or state['step'] != prior.SETTINGS['steps']:
                raise ValueError('Expected final checkpoint')
            net = model(arm, prior.GatedMotionResidual(False).state_dict())
            net.load_state_dict(state['model'])
            p = prior.native.predict(net, arrays, ids, 512)
            if not np.isfinite(p).all() or p.shape != (len(ids), 12, 2):
                raise ValueError('Invalid past-only forecasts')
            tmp = path.with_suffix('.pending.npz')
            np.savez(tmp, ids=d['global_ids'], prediction=p)
            tmp.replace(path)
            run.once(mark, dict(forecast=run.artifact(path), checkpoint=f['checkpoint'], arm=arm, seed=seed))
        r = run.read(mark)
        if (r['seed'], r['arm'], r['checkpoint']) != (seed, arm, f['checkpoint']):
            raise ValueError('Changed forecast checkpoint')
        run.checked(r['forecast'])
        receipts.append(r)
        beat(state='forecast_frozen', seed=seed, arm=arm, rows=len(ids))
    out = PRIVATE/'freeze.json'
    run.once(out, run.stable(dict(forecasts=receipts, uses_past_only=True, registration=run.sha(REG))))
    return out


def score(cut, beat):
    f = prev.receipt(PUBLIC, 'infer')
    if not f['uses_past_only'] or f['registration'] != run.sha(REG):
        raise ValueError('Freeze all nine past-only forecasts before scoring')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
    groups = sorted(set(roster.values()))
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    rows, sub, support = [], 0, np.zeros(4, dtype=int)
    cells = {}
    indices = [(PATH_NAMES.index(a), PATH_NAMES.index(b)) for a, b in CONTRASTS]
    for c in prev.receipt(prev.PUBLIC, 'freeze')['cells']:
        seed = c['seed']
        cv, bank = run.candidates(d, refs, seed)
        pred = {}
        for t in f['forecasts']:
            if t['seed'] == seed:
                with np.load(run.checked(t['forecast']), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], d['global_ids'])
                    pred[t['arm']] = z['prediction'].astype(float)
        if set(pred) != set(ARMS):
            raise ValueError('Missing forecast arm')
        with np.load(run.checked(c['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            selected = z['choice'][0]
        per_source = {g: [] for g in groups}
        for rec, (ix, queries) in records.items():
            floor, bank_local = cv[ix], bank[ix]
            origin = run.model.selected_paths(selected[ix], bank_local, floor)
            switched = (origin != floor).any(axis=(1, 2))
            use = switched & np.logical_and.reduce([np.linalg.norm(pred[a][ix]-floor, axis=2).max(1) > 1e-6 for a in ARMS])
            matched = {a: np.where(use[:, None, None], pred[a][ix], origin) for a in ARMS}
            for p in matched.values():
                np.testing.assert_array_equal((p != floor).any(axis=(1, 2)), switched)
            sub += int(use.sum())
            paths = {BASE: floor, OLS: bank_local[:, run.NAMES.index(OLS)], GAIN: origin, **matched}
            row = run.parent.readout.score_recording(paths, PAIRS, y[ix], valid[ix], d['queries'][ix],
                                                     rec, easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, source=roster[rec], seed=seed))
            for q in queries:
                _, p, _, counts = pair.query(d['history'][ix[q]], np.stack([paths[a][q] for a in PATH_NAMES]),
                                              y[ix[q]], valid[ix[q]], indices)
                support += counts
                per_source[roster[rec]].append(p)
        for g in groups:
            cells[g, seed] = pair.mean_finite(per_source[g])
        beat(state='matched_context_scored', seed=seed, recording_views=len(rows))
    grouped = prev.old.prior.grouped_readout(rows, roster, prev.SEEDS)
    if grouped['risk'][GAIN] != prev.receipt(prev.PUBLIC, 'replay')['grouped']['risk'][GAIN]:
        raise ValueError('Matched original policy did not replay')
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    x = np.asarray([[cells[g, s] for s in prev.SEEDS] for g in groups])
    pair_results = {name: {k: pair.bootstrap(x[:, :, a, j], draw) for j, k in enumerate(pair.PAIR_FIELDS)}
                    for a, name in enumerate(PAIRS)}
    return dict(grouped=grouped, relative_geometry=pair_results, shared_substitution_rows=sub,
                pair_support=support.tolist(), independent_confirmation=False, external_access=False,
                deployment=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'train', 'infer', 'score', 'replay'))
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cut, jobs, backbones = register()
    if args.phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the design before training')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)

        def beat(**v):
            event = dict(pid=os.getpid(), phase=args.phase, seconds=time.monotonic()-start, **v)
            prior.fit_source.source.atomic(PRIVATE/'heartbeat.json', event)
            print(event, flush=True)

        if args.phase == 'train':
            out = train(jobs, backbones, args.resume, beat)
        elif args.phase == 'infer':
            out = infer(beat)
        else:
            if args.phase == 'replay':
                prev.receipt(PUBLIC, 'score')
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(cut, beat)))
        mark = PUBLIC/f'{args.phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Context audit replay differs')


if __name__ == '__main__':
    main()
