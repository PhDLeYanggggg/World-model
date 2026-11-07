"""Isolate relative-path supervision at matched inputs, batches and recipients."""
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
from scripts import run_m3w_neighbor_motion_adapter as context
from src.world_model.m3w_relative_path_supervision import relative_path_loss

prior, pair, run, prev, np, torch = context.prior, context.pair, context.run, context.prev, context.np, context.torch
PUBLIC = context.PUBLIC.parent/'native_relative_path_training_v1'
PRIVATE = context.PRIVATE.parent/'native_relative_path_training_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), Path(relative_path_loss.__code__.co_filename),
         PUBLIC/'protocol.md', ROOT/'tests/test_m3w_relative_path_supervision.py')
ARMS = ('individual', 'with_relative')
SETTINGS = dict(prior.SETTINGS, single_rows=64, edge_rows=32, pair_weight=.25)
BASE, OLS, GAIN = context.BASE, context.OLS, context.GAIN
NAMES = (GAIN, *ARMS)
PAIRS = {f'{b}_vs_{a}': (a, b, BASE) for a, b in ((ARMS[0], ARMS[1]), (GAIN, ARMS[0]), (GAIN, ARMS[1]))}


def register():
    cut, jobs, backbones = context.register()
    prev.receipt(context.PUBLIC, 'replay')
    bindings = (*FILES, context.REG, context.PUBLIC/'replay_execution.json', Path(context.__file__),
                Path(pair.local_edges.__code__.co_filename), Path(prior.native.__file__))
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in bindings], cut=cut, settings=SETTINGS,
        backbones=backbones, arms=ARMS, seeds=prev.SEEDS, exact_recipients=True,
        external_access=False, original_risk_unchanged=True, torch_threads=2, workers=0)))
    return cut, jobs, backbones


def training_graph(d, identity, beat):
    src = prior.fit_source
    prep = src.prep
    cfg = run.read(src.source.CONFIG)
    admission = prep.load_admission(ROOT, ROOT/cfg['admission'])
    lookup = {r['clip']: r for r in run.read(src.bank.old.HOME/'movingdronecrowd_intake_v1/analysis.json')['recordings']}
    edges, global_at, local_at, queries = [], 0, 0, 0
    for node, meta in sorted(admission['recordings'].items()):
        clip = lookup[meta['clip']]
        n = clip['window_support']['obs8_pred12_native_stride1']['history_complete_agent_instances']
        start, global_at = global_at, global_at+n
        if meta['exclusion_group'] not in identity['normalizers'] or n == 0:
            continue
        path = ROOT/meta['source_annotation_path']
        prep.require_fit_record(admission, node, 'fit', path)
        rec = prep.mot_record(path.read_bytes(), meta['clip'], range(1, clip['image_frames']+1))
        end = local_at+n
        np.testing.assert_array_equal(d['ids'][local_at:end], np.arange(start, global_at))
        for q in range(prep.SPEC.history_span, rec.valid.shape[1]-12):
            inputs = prep.past_inputs(rec, q, prep.SPEC)
            h = inputs['positions'][inputs['history_mask'].all(1)]
            if not len(h):
                continue
            stop = local_at+len(h)
            np.testing.assert_array_equal(d['geometry'][local_at:stop, :16],
                (h-h[:, -1:]).reshape(-1, 16).astype(np.float32))
            links = pair.local_edges(h)
            if links.shape[1]:
                edges.append(links.T+local_at)
            local_at = stop
            queries += 1
        if local_at != end:
            raise ValueError('Training query boundaries no longer align')
        beat(state='verified_training_graph_recording', rows=local_at)
    if local_at != len(d['ids']):
        raise ValueError('Training query graph did not cover the fit roster')
    edges = np.concatenate(edges)
    groups = [np.flatnonzero(d['sites'][edges[:, 0]] == s) for s in sorted(set(d['sites']))]
    if not all(len(g) for g in groups) or not np.array_equal(d['sites'][edges[:, 0]], d['sites'][edges[:, 1]]):
        raise ValueError('Require same-query edges and every fitting source')
    both = d['valid'][edges[:, 0]] & d['valid'][edges[:, 1]]
    return edges, groups, dict(queries=queries, edges=len(edges), coobserved_edges=int(both.any(1).sum()),
                               coobserved_steps=int(both.sum()), edge_sha256=prior.fit_source.source.array_hash(edges))


def fit(net, d, edges, edge_groups, scales, ident, directory, resume, beat):
    directory.mkdir(parents=True, exist_ok=True)
    cp = directory/'checkpoint.pt'
    rng = torch.Generator().manual_seed(ident['seed']+7919)
    opt = torch.optim.AdamW(net.parameters(), lr=SETTINGS['learning_rate'], weight_decay=SETTINGS['weight_decay'])
    draw, edge_draw = np.zeros(len(d['ids']), dtype=np.int64), np.zeros(len(edges), dtype=np.int64)
    trace, step = [], 0
    if cp.exists():
        if not resume:
            raise ValueError('Use --resume with the existing checkpoint')
        old = torch.load(cp, map_location='cpu', weights_only=False)
        if old['identity'] != ident:
            raise ValueError('Resume registration or training graph differs')
        net.load_state_dict(old['model'])
        opt.load_state_dict(old['optimizer'])
        rng.set_state(old['sampler_rng'])
        torch.set_rng_state(old['torch_rng'])
        trace, step, draw, edge_draw = (old[k] for k in ('trace', 'step', 'draws', 'edge_draws'))
    net.train()
    weight = SETTINGS['pair_weight'] if ident['arm'] == ARMS[1] else 0.
    while step < SETTINGS['steps']:
        single = prior.native.draw_batch(d['groups'], SETTINGS['single_rows'], rng)
        sampled_edges = prior.native.draw_batch(edge_groups, SETTINGS['edge_rows'], rng)
        endpoints = edges[sampled_edges]
        ids = np.r_[single, endpoints[:, 0], endpoints[:, 1]]
        np.add.at(draw, ids, 1)
        np.add.at(edge_draw, sampled_edges, 1)
        p = net(prior.native.pack_geometry(d['geometry'][ids]))
        y = torch.from_numpy(d['target'][ids])
        valid = torch.from_numpy(d['valid'][ids])
        scale = torch.from_numpy(scales[ids])
        main, _ = prior.native.masked_objective(p, y, valid, 1./scale)
        i = slice(len(single), len(single)+len(endpoints))
        j = slice(len(single)+len(endpoints), len(ids))
        relational = relative_path_loss(p[i], p[j], y[i], y[j], valid[i], valid[j],
                                         (scale[i]*scale[j]).sqrt())
        loss = main+weight*relational
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite individual or pair loss')
        progress = step/(SETTINGS['steps']-1)
        floor = SETTINGS['minimum_lr_ratio']
        for group in opt.param_groups:
            group['lr'] = SETTINGS['learning_rate']*(floor+(1.-floor)*(1.+math.cos(math.pi*progress))/2.)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), SETTINGS['gradient_clip'], error_if_nonfinite=True)
        opt.step()
        step += 1
        if step == 1 or step % SETTINGS['checkpoint_every'] == 0:
            row = dict(step=step, individual=float(main.detach()), pair=float(relational.detach()),
                       supervised_pairs=int((valid[i] & valid[j]).any(1).sum()))
            trace.append(row)
            beat(state='training', seed=ident['seed'], arm=ident['arm'], **row)
        if step % SETTINGS['checkpoint_every'] == 0 or step == SETTINGS['steps']:
            temp = cp.with_suffix('.tmp')
            torch.save(dict(identity=ident, step=step, model=net.state_dict(), optimizer=opt.state_dict(),
                            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(),
                            trace=trace, draws=draw, edge_draws=edge_draw), temp)
            temp.replace(cp)
    if draw.sum() != 900*128 or edge_draw.sum() != 900*32:
        raise ValueError('Incomplete original training budget')
    return cp, (draw, edge_draw, rng.get_state(), torch.get_rng_state()), trace


def train(jobs, backbones, resume, beat):
    out = PRIVATE/'train.json'
    if out.exists():
        for f in run.read(out)['fits']:
            run.checked(f['checkpoint'])
            if f['identity']['registration'] != run.sha(REG):
                raise ValueError('Stale training checkpoint')
        return out
    identity = jobs[0]['identity']
    d = prior.fit_source.fitting_only(identity, beat)
    if len(d['ids']) != 69746 or set(d['sites']) & set(run.old.prior.arrays(('sites',))['sites']):
        raise ValueError('Require the original disjoint fitting sources')
    edges, groups, graph = training_graph(d, identity, beat)
    h = d['geometry'][:, :16].reshape(-1, 8, 2)
    scales = np.maximum(np.sqrt(np.square(np.diff(h, axis=1)).sum(2).mean(1)), .05)
    fits = []
    exact = prior.fit_source.source.exact
    for seed in prev.SEEDS:
        base = torch.load(run.checked(backbones[seed]), map_location='cpu', weights_only=False)
        if base['identity']['fit_global_ids'] != identity['train_ids_sha256']:
            raise ValueError('Wrong frozen backbone fit roster')
        initial, control = None, None
        for arm in ARMS:
            torch.manual_seed(seed+419)
            net = context.model(context.ARMS[-1], base['model'])
            if initial is None:
                initial = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                exact(initial, net.state_dict())
            ident = dict(seed=seed, arm=arm, registration=run.sha(REG),
                         graph=graph['edge_sha256'], backbone=backbones[seed]['sha256'])
            cp, drawn, trace = fit(net, d, edges, groups, scales, ident,
                                   PRIVATE/'fits'/f'{seed}_{arm}', resume, beat)
            exact(base['model'], net.backbone.state_dict())
            if control is None:
                control = drawn
            else:
                exact(control, drawn)
            fits.append(dict(identity=ident, checkpoint=run.artifact(cp), trace=trace))
    run.once(out, run.stable(dict(fits=fits, fit_rows=len(d['ids']), graph=graph,
        frozen_backbone_and_draws_exact=True, fit_sources_exclude_development=True)))
    return out


def infer(beat):
    f = prev.receipt(PUBLIC, 'train')
    if not f['frozen_backbone_and_draws_exact'] or len(f['fits']) != 6:
        raise ValueError('Finish all matched fits before inference')
    d = run.population()
    arrays = run.old.prior.arrays(('geometry', 'global_ids'))
    np.testing.assert_array_equal(arrays['global_ids'], d['global_ids'])
    hist = (d['history']-d['origin'][:, None]).astype(np.float32)
    np.testing.assert_array_equal(arrays['geometry'][:, :16].reshape(hist.shape), hist)
    receipts = []
    for cell in f['fits']:
        ident = cell['identity']
        seed, arm = ident['seed'], ident['arm']
        path = PRIVATE/f'prediction_{seed}_{arm}.npz'
        mark = path.with_suffix('.json')
        if not mark.exists():
            state = torch.load(run.checked(cell['checkpoint']), map_location='cpu', weights_only=False)
            if state['identity'] != ident or state['step'] != SETTINGS['steps'] or ident['registration'] != run.sha(REG):
                raise ValueError('Expected final registered fit')
            net = context.model(context.ARMS[-1], prior.GatedMotionResidual(False).state_dict())
            net.load_state_dict(state['model'])
            pred = prior.native.predict(net, arrays, np.arange(len(hist)), 512)
            if not np.isfinite(pred).all() or pred.shape != (len(hist), 12, 2):
                raise ValueError('Bad past-only predictions')
            tmp = path.with_suffix('.pending.npz')
            np.savez(tmp, ids=d['global_ids'], prediction=pred)
            tmp.replace(path)
            run.once(mark, dict(identity=ident, forecast=run.artifact(path), checkpoint=cell['checkpoint']))
        r = run.read(mark)
        if r['identity'] != ident or r['checkpoint'] != cell['checkpoint']:
            raise ValueError('Stale forecast')
        run.checked(r['forecast'])
        receipts.append(r)
        beat(state='forecast_frozen', seed=seed, arm=arm, rows=len(hist))
    out = PRIVATE/'freeze.json'
    run.once(out, run.stable(dict(forecasts=receipts, uses_past_only=True, registration=run.sha(REG))))
    return out


def score(beat):
    frozen = prev.receipt(PUBLIC, 'infer')
    if frozen['registration'] != run.sha(REG) or not frozen['uses_past_only']:
        raise ValueError('Require immutable past-only predictions before scoring')
    cut = run.read(REG)['cut']
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
    groups = sorted(set(roster.values()))
    target, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    rows, cells, sub = [], {}, 0
    support = np.zeros(4, dtype=int)
    comp = [(NAMES.index(a), NAMES.index(b)) for a, b, _ in PAIRS.values()]
    for cell in prev.receipt(prev.PUBLIC, 'freeze')['cells']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        pred = {}
        for r in frozen['forecasts']:
            if r['identity']['seed'] == seed:
                with np.load(run.checked(r['forecast']), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], d['global_ids'])
                    pred[r['identity']['arm']] = z['prediction'].astype(float)
        if set(pred) != set(ARMS):
            raise ValueError('Both forecast heads are required')
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            selected = z['choice'][0]
        effects = {g: [] for g in groups}
        for r, (ix, queries) in records.items():
            base, bank_r = cv[ix], bank[ix]
            gain = run.model.selected_paths(selected[ix], bank_r, base)
            switched = (gain != base).any(axis=(1, 2))
            use = switched & np.logical_and.reduce([np.linalg.norm(pred[a][ix]-base, axis=2).max(1) > 1e-6 for a in ARMS])
            sub += int(use.sum())
            paths = {BASE: base, OLS: bank_r[:, run.NAMES.index(OLS)], GAIN: gain,
                     **{a: np.where(use[:, None, None], pred[a][ix], gain) for a in ARMS}}
            for a in ARMS:
                np.testing.assert_array_equal((paths[a] != base).any(axis=(1, 2)), switched)
            row = run.parent.readout.score_recording(paths, PAIRS, target[ix], valid[ix], d['queries'][ix],
                                                     r, easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, seed=seed, source=roster[r]))
            for q in queries:
                _, e, _, s = pair.query(d['history'][ix[q]], np.stack([paths[a][q] for a in NAMES]),
                                        target[ix[q]], valid[ix[q]], comp)
                effects[roster[r]].append(e)
                support += s
        for g in groups:
            cells[g, seed] = pair.mean_finite(effects[g])
        beat(state='scored_fixed_recipients', seed=seed, views=len(rows))
    grouped = prev.old.prior.grouped_readout(rows, roster, prev.SEEDS)
    if grouped['risk'][GAIN] != prev.receipt(prev.PUBLIC, 'replay')['grouped']['risk'][GAIN]:
        raise ValueError('Original gain comparator changed')
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    x = np.asarray([[cells[g, s] for s in prev.SEEDS] for g in groups])
    metrics = {k: {field: pair.bootstrap(x[:, :, i, j], draw) for j, field in enumerate(pair.PAIR_FIELDS)}
               for i, k in enumerate(PAIRS)}
    return dict(grouped=grouped, relative_geometry=metrics, shared_substitutions=sub,
                pair_support=support.tolist(), training_graph=prev.receipt(PUBLIC, 'train')['graph'],
                external_access=False, independent_confirmation=False, deployment=False,
                stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'train', 'infer', 'score', 'replay'))
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    _, jobs, backbones = register()
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
            run.once(out, run.stable(score(beat)))
        mark = PUBLIC/f'{args.phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Replay differs')


if __name__ == '__main__':
    main()
