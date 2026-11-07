"""Fit and evaluate two matched blends over frozen individual and pair forecasts."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_relative_path_training as relative
from src.world_model.m3w_completion_motion_blend import CompletionMotionBlend, objectives

prior, context, pair, run, prev, np, torch = (
    relative.prior, relative.context, relative.pair, relative.run, relative.prev, relative.np, relative.torch)
PUBLIC = relative.PUBLIC.parent/'native_completion_motion_blend_v1'
PRIVATE = relative.PRIVATE.parent/'native_completion_motion_blend_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), Path(objectives.__code__.co_filename),
         PUBLIC/'protocol.md', ROOT/'tests/test_m3w_completion_motion_blend.py')
ARMS = ('observed_fit', 'completion_fit')
REF, HALF = 'reference', 'fixed_half'
BASE, OLS, GAIN = relative.BASE, relative.OLS, relative.GAIN
NAMES = (GAIN, REF, HALF, *ARMS)
CONTRASTS = ((REF, ARMS[1]), (ARMS[0], ARMS[1]), (HALF, ARMS[1]), (REF, ARMS[0]), (REF, HALF), (GAIN, ARMS[1]))
PAIRS = {f'{b}_vs_{a}': (a, b, BASE) for a, b in CONTRASTS}
SET = relative.SETTINGS


def register():
    cut, jobs, _ = relative.register()
    p = prev.receipt(relative.PUBLIC, 'train')
    f = prev.receipt(relative.PUBLIC, 'infer')
    prev.receipt(relative.PUBLIC, 'replay')
    if not p['fit_sources_exclude_development'] or p['fit_rows'] != 69746 or not f['uses_past_only']:
        raise ValueError('Require source-excluded parent weights and frozen past forecasts')
    binds = (*FILES, relative.REG, relative.PUBLIC/'train_execution.json',
             relative.PUBLIC/'infer_execution.json', relative.PUBLIC/'replay_execution.json',
             Path(relative.training_graph.__code__.co_filename), Path(pair.__file__))
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in binds], cut=cut, arms=ARMS,
        settings=SET, seed_list=prev.SEEDS, exact_recipients=True, no_new_base_fits=True, external_access=False,
        original_risk_unchanged=True, torch_threads=2, workers=0)))
    return cut, jobs, p, f


def parent_fit_predictions(d, parent, seed):
    paths = {}
    ids = np.arange(len(d['ids']))
    for f in parent['fits']:
        ident = f['identity']
        if ident['seed'] != seed:
            continue
        state = torch.load(run.checked(f['checkpoint']), map_location='cpu', weights_only=False)
        if state['identity'] != ident or state['step'] != SET['steps'] or ident['registration'] != run.sha(relative.REG):
            raise ValueError('Wrong final frozen parent fit')
        net = context.model(context.ARMS[-1], prior.GatedMotionResidual(False).state_dict())
        net.load_state_dict(state['model'])
        paths[ident['arm']] = prior.native.predict(net, d, ids, 512)
    if set(paths) != set(relative.ARMS):
        raise ValueError('Missing both frozen paths')
    return [torch.from_numpy(paths[a]) for a in relative.ARMS]


def fit(d, edges, edge_groups, scales, h, ref, proposal, ident, resume, beat):
    seed, arm = ident['seed'], ident['arm']
    directory = PRIVATE/'fits'/f'{seed}_{arm}'
    directory.mkdir(parents=True, exist_ok=True)
    cp = directory/'checkpoint.pt'
    torch.manual_seed(seed+557)
    net = CompletionMotionBlend()
    opt = torch.optim.AdamW(net.parameters(), lr=SET['learning_rate'], weight_decay=SET['weight_decay'])
    rng = torch.Generator().manual_seed(seed+7919)
    draws, edge_draws = np.zeros(len(h), dtype=np.int64), np.zeros(len(edges), dtype=np.int64)
    step, trace = 0, []
    if cp.exists():
        if not resume:
            raise ValueError('Use --resume to continue the registered fit')
        saved = torch.load(cp, map_location='cpu', weights_only=False)
        if saved['identity'] != ident:
            raise ValueError('Blend resume lineage differs')
        net.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        rng.set_state(saved['sampler_rng'])
        torch.set_rng_state(saved['torch_rng'])
        step, trace, draws, edge_draws = (saved[k] for k in ('step', 'trace', 'draws', 'edge_draws'))
    while step < SET['steps']:
        single = prior.native.draw_batch(d['groups'], SET['single_rows'], rng)
        drawn = prior.native.draw_batch(edge_groups, SET['edge_rows'], rng)
        a, b = edges[drawn].T
        ids = np.r_[single, a, b]
        np.add.at(draws, ids, 1)
        np.add.at(edge_draws, drawn, 1)
        n, k = len(single), len(a)
        left, right = torch.arange(n, n+k), torch.arange(n+k, n+2*k)
        p, q = ref[ids], proposal[ids]
        alpha = net(h[ids], p, q)
        pred = p+alpha[..., None]*(q-p)
        main, geom, debit, geom_debit = objectives(pred, p, torch.from_numpy(d['target'][ids]),
            torch.from_numpy(d['valid'][ids]), scales[ids], left, right)
        weight = SET['pair_weight']
        loss = main+weight*geom
        if arm == ARMS[1]:
            loss = loss+debit+weight*geom_debit
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite completion-aware blend objective')
        low = SET['minimum_lr_ratio']
        for group in opt.param_groups:
            group['lr'] = SET['learning_rate']*(low+(1.-low)*(1.+math.cos(math.pi*step/(SET['steps']-1)))/2.)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), SET['gradient_clip'], error_if_nonfinite=True)
        opt.step()
        step += 1
        if step == 1 or step % SET['checkpoint_every'] == 0:
            rec = dict(step=step, objective=float(loss.detach()), individual_debit=float(debit.detach()),
                       pair_debit=float(geom_debit.detach()), alpha=float(alpha.detach().mean()))
            trace.append(rec)
            beat(state='training', seed=seed, arm=arm, **rec)
        if step % SET['checkpoint_every'] == 0 or step == SET['steps']:
            tmp = cp.with_suffix('.tmp')
            torch.save(dict(identity=ident, step=step, trace=trace, draws=draws, edge_draws=edge_draws,
                model=net.state_dict(), optimizer=opt.state_dict(), sampler_rng=rng.get_state(),
                torch_rng=torch.get_rng_state()), tmp)
            tmp.replace(cp)
    if draws.sum() != 900*128 or edge_draws.sum() != 900*32:
        raise ValueError('Incomplete matched batch budget')
    return cp, trace, (draws, edge_draws, rng.get_state(), torch.get_rng_state())


def train(jobs, parent, resume, beat):
    out = PRIVATE/'train.json'
    if out.exists():
        for f in run.read(out)['fits']:
            if f['identity']['registration'] != run.sha(REG):
                raise ValueError('Training registration differs')
            run.checked(f['checkpoint'])
        return out
    ident = jobs[0]['identity']
    d = prior.fit_source.fitting_only(ident, beat)
    if len(d['ids']) != 69746 or set(d['sites']) & set(run.old.prior.arrays(('sites',))['sites']):
        raise ValueError('Training sources must exclude the exposed development sources')
    edges, edge_groups, graph = relative.training_graph(d, ident, beat)
    if graph != parent['graph']:
        raise ValueError('Frozen pair-training graph changed')
    h = torch.from_numpy(d['geometry'][:, :16].reshape(-1, 8, 2).copy())
    scales = h.diff(dim=1).square().sum(2).mean(1).sqrt().clamp_min(.05)
    results = []
    for seed in prev.SEEDS:
        ref, proposal = parent_fit_predictions(d, parent, seed)
        draws = None
        for arm in ARMS:
            ident = dict(registration=run.sha(REG), graph=graph['edge_sha256'], seed=seed, arm=arm)
            cp, trace, r = fit(d, edges, edge_groups, scales, h, ref, proposal, ident, resume, beat)
            if draws is None:
                draws = r
            else:
                prior.fit_source.source.exact(draws, r)
            results.append(dict(identity=ident, checkpoint=run.artifact(cp), trace=trace))
    run.once(out, run.stable(dict(fits=results, matched_draws=True, fit_rows=len(h), graph=graph)))
    return out


def anchors(d, frozen, seed):
    paths = {}
    for f in frozen['forecasts']:
        if f['identity']['seed'] == seed:
            with np.load(run.checked(f['forecast']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'])
                paths[f['identity']['arm']] = z['prediction']
    return [paths[a] for a in relative.ARMS]


def infer(frozen, beat):
    fits = prev.receipt(PUBLIC, 'train')
    if not fits['matched_draws'] or len(fits['fits']) != 6:
        raise ValueError('Expected both matched heads for three seeds')
    d = run.population()
    h = torch.from_numpy((d['history']-d['origin'][:, None]).astype(np.float32))
    out, rows = PRIVATE/'freeze.json', []
    for seed in prev.SEEDS:
        a, b = anchors(d, frozen, seed)
        x, y = torch.from_numpy(a), torch.from_numpy(b)
        for f in fits['fits']:
            ident = f['identity']
            if ident['seed'] != seed:
                continue
            arm = ident['arm']
            state = torch.load(run.checked(f['checkpoint']), map_location='cpu', weights_only=False)
            if state['identity'] != ident or state['step'] != SET['steps']:
                raise ValueError('Expected frozen final blend')
            net = CompletionMotionBlend()
            net.load_state_dict(state['model'])
            with torch.no_grad():
                alpha = net(h, x, y).numpy()
            if not np.isfinite(alpha).all() or alpha.min() < 0. or alpha.max() > 1.:
                raise ValueError('Invalid bounded interpolation')
            path = PRIVATE/f'alpha_{seed}_{arm}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], d['global_ids'])
                    np.testing.assert_array_equal(z['alpha'], alpha)
            else:
                tmp = path.with_suffix('.pending.npz')
                np.savez(tmp, ids=d['global_ids'], alpha=alpha)
                tmp.replace(path)
            rows.append(dict(seed=seed, arm=arm, artifact=run.artifact(path)))
            beat(state='blend_frozen', seed=seed, arm=arm, rows=len(h))
    run.once(out, run.stable(dict(cells=rows, registration=run.sha(REG), past_only=True)))
    return out


def score(cut, parents, beat):
    f = prev.receipt(PUBLIC, 'infer')
    if f['registration'] != run.sha(REG) or not f['past_only']:
        raise ValueError('Require frozen blend outputs before label access')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {rec: str(d['sites'][ix[0]]) for rec, (ix, _) in records.items()}
    groups = sorted(set(roster.values()))
    target, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    indices = [(NAMES.index(a), NAMES.index(b)) for a, b in CONTRASTS]
    rows, rel, used = [], {}, 0
    selected_weights = {a: [] for a in ARMS}
    for c in prev.receipt(prev.PUBLIC, 'freeze')['cells']:
        seed = c['seed']
        cv, bank = run.candidates(d, refs, seed)
        a, b = (x.astype(float) for x in anchors(d, parents, seed))
        delta = b-a
        weights = {}
        pred = {REF: a, HALF: a+.5*delta}
        for x in f['cells']:
            if x['seed'] == seed:
                with np.load(run.checked(x['artifact']), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], d['global_ids'])
                    weights[x['arm']] = z['alpha'][:, 0]
                pred[x['arm']] = a+weights[x['arm']][:, None, None]*delta
        if set(weights) != set(ARMS):
            raise ValueError('Missing frozen blend head')
        with np.load(run.checked(c['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            choice = z['choice'][0]
        cells = {s: [] for s in groups}
        for rec, (ix, queries) in records.items():
            base, bnk = cv[ix], bank[ix]
            gain = run.model.selected_paths(choice[ix], bnk, base)
            switched = (gain != base).any(axis=(1, 2))
            use = switched & np.logical_and.reduce([np.linalg.norm(p[ix]-base, axis=2).max(1) > 1e-6 for p in pred.values()])
            used += int(use.sum())
            for k in ARMS:
                selected_weights[k].extend(weights[k][ix][use].tolist())
            matched = {k: np.where(use[:, None, None], p[ix], gain) for k, p in pred.items()}
            for p in matched.values():
                np.testing.assert_array_equal((p != base).any(axis=(1, 2)), switched)
            paths = {BASE: base, OLS: bnk[:, run.NAMES.index(OLS)], GAIN: gain, **matched}
            row = run.parent.readout.score_recording(paths, PAIRS, target[ix], valid[ix], d['queries'][ix], rec,
                                                     easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, seed=seed, source=roster[rec]))
            for q in queries:
                _, value, _, _ = pair.query(d['history'][ix[q]], np.stack([paths[k][q] for k in NAMES]),
                                            target[ix[q]], valid[ix[q]], indices)
                cells[roster[rec]].append(value)
        for g in groups:
            rel[g, seed] = pair.mean_finite(cells[g])
        beat(state='matched_readout', seed=seed, views=len(rows))
    grouped = prev.old.prior.grouped_readout(rows, roster, prev.SEEDS)
    if grouped['risk'][GAIN] != prev.receipt(prev.PUBLIC, 'replay')['grouped']['risk'][GAIN]:
        raise ValueError('Original matched control changed')
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    cube = np.asarray([[rel[g, s] for s in prev.SEEDS] for g in groups])
    geometry = {k: {m: pair.bootstrap(cube[:, :, i, j], draw) for j, m in enumerate(pair.PAIR_FIELDS)}
                for i, k in enumerate(PAIRS)}
    return dict(grouped=grouped, relative_geometry=geometry, common_substitutions=used,
        alpha_quantiles={k: np.quantile(v, [.1, .5, .9]).tolist() for k, v in selected_weights.items()},
        external_access=False, independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'train', 'infer', 'score', 'replay'))
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cut, jobs, parent, frozen = register()
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
            out = train(jobs, parent, args.resume, beat)
        elif args.phase == 'infer':
            out = infer(frozen, beat)
        else:
            if args.phase == 'replay':
                prev.receipt(PUBLIC, 'score')
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(cut, frozen, beat)))
        mark = PUBLIC/f'{args.phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Replay differs')


if __name__ == '__main__':
    main()
