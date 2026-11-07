"""Isolate the effect of a two-reference loss on a matched forecast head."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_gated_motion_residual as base
from src.world_model import m3w_dual_reference_forecast as cost

run, prev, np, torch = base.run, base.prev, base.np, base.torch
PUBLIC = base.PUBLIC.parent/'native_dual_reference_forecast_v1'
PRIVATE = base.PRIVATE.parent/'native_dual_reference_forecast_v1'
REG = PUBLIC/'registration.json'
ARMS = ('cv_only', 'dual_reference')
BASE, OLS, CONTROL = base.BASE, base.OLS, base.CONTROL
FILES = (Path(__file__).resolve(), Path(cost.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_dual_reference_forecast.py')
PAIRS = {f'{a}_vs_{b}': (b, a, BASE) for a, b in (
    ('dual_reference', 'cv_only'), ('dual_reference', OLS), ('cv_only', OLS),
    ('dual_reference', BASE), ('cv_only', BASE),
    ('matched_dual_reference', 'matched_cv_only'),
    *[(f'matched_{a}', b) for a in ARMS for b in (CONTROL, BASE, OLS)])}


def register():
    cut, jobs = base.register()
    bindings = (*FILES, base.REG, *base.FILES, base.fit_source.PUBLIC/'fit_population.json')
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in bindings], arms=ARMS,
        seeds=prev.SEEDS, cut=cut, settings=base.SETTINGS, bootstrap=3000,
        primary='matched_dual_reference_vs_matched_cv_only',
        fixed_gain_switches=9675, model_selection=False, external_access=False, workers=0,
        original_risk_unchanged=True, stage5c_executed=False, smc_enabled=False)))
    return cut, jobs


def fit(net, data, tensors, seed, arm, identity, cp, resume, beat):
    cfg = base.SETTINGS
    h, y, valid, cv, ols, scale = tensors
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = torch.Generator().manual_seed(seed+7919)
    step, draws, trace = 0, np.zeros(len(h), np.int64), []
    if cp.exists():
        if not resume:
            raise FileExistsError('Existing checkpoint requires --resume')
        saved = torch.load(cp, map_location='cpu', weights_only=False)
        if saved['identity'] != identity or not 0 < saved['step'] <= cfg['steps']:
            raise ValueError('Invalid resume identity or step')
        net.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        rng.set_state(saved['sampler_rng']); torch.set_rng_state(saved['torch_rng'])
        step, draws, trace = saved['step'], saved['draws'], saved['trace']
    net.train()
    while step < cfg['steps']:
        ids = base.native.draw_batch(data['groups'], cfg['batch_size'], rng)
        progress = step/max(cfg['steps']-1, 1)
        lr = cfg['learning_rate']*(cfg['minimum_lr_ratio']+
             (1-cfg['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*progress)))
        for group in opt.param_groups:
            group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        loss, components, penalty = cost.objective(net({'history': h[ids]}), y[ids], valid[ids],
            cv[ids], ols[ids], scale[ids], data['easy_cut'], dual=arm == 'dual_reference')
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite forecast loss; resume from last atomic checkpoint')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
        opt.step()
        step += 1
        np.add.at(draws, ids, 1)
        report = step == 1 or step % cfg['heartbeat_every'] == 0 or step == cfg['steps']
        if report:
            trace.append(dict(step=step, loss=float(loss.detach()), cv=float(components[0].detach()),
                ols4=float(components[1].detach()), cv_penalty=float(penalty.detach()),
                gradient=float(grad), learning_rate=lr, supported=int(valid[ids].any(1).sum())))
        if step % cfg['checkpoint_every'] == 0 or step == cfg['steps']:
            tmp = cp.with_suffix('.pending')
            torch.save(dict(identity=identity, step=step, draws=draws, trace=trace, model=net.state_dict(),
                optimizer=opt.state_dict(), sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state()), tmp)
            tmp.replace(cp)
        if report:
            beat(state='training', seed=seed, arm=arm, **trace[-1])
    if draws.sum() != cfg['steps']*cfg['batch_size']:
        raise ValueError('Fitting draw count differs from registration')
    return dict(identity=identity, checkpoint=run.artifact(cp), step=step, draws=int(draws.sum()), trace=trace,
                parameters=sum(p.numel() for p in net.parameters())), (draws, rng.get_state(), torch.get_rng_state())


def train(jobs, beat, resume):
    data = base.fit_source.fitting_only(jobs[0]['identity'], beat)
    sites = set(data['sites'])
    if len(data['ids']) != 69746 or len(sites) != 9 or sites & set(run.old.prior.arrays(('sites',))['sites']):
        raise ValueError('Expected unchanged fitting roster with no development overlap')
    h = torch.from_numpy(data['geometry'][:, :16].reshape(-1, 8, 2).copy())
    cv, ols = cost.causal_references(h)
    np.testing.assert_allclose(cv.numpy(), data['geometry'][:, 332:356].reshape(-1, 12, 2), atol=1e-4)
    scale = h.diff(dim=1).square().sum(2).mean(1).sqrt().clamp_min(.05)
    tensors = h, torch.from_numpy(data['target']), torch.from_numpy(data['valid']), cv, ols, scale
    fits = []
    for seed in prev.SEEDS:
        initial, states = None, []
        for arm in ARMS:
            torch.manual_seed(seed)
            net = base.GatedMotionResidual(False)
            if initial is None:
                initial = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                base.fit_source.source.exact(initial, net.state_dict())
            identity = dict(seed=seed, arm=arm, registration=run.sha(REG),
                            fit_ids=jobs[0]['identity']['train_ids_sha256'])
            f, state = fit(net, data, tensors, seed, arm, identity, PRIVATE/f'{seed}_{arm}.pt', resume, beat)
            fits.append(f); states.append(state)
        base.fit_source.source.exact(*states)
    out = PRIVATE/'train.json'
    run.once(out, run.stable(dict(fits=fits, fit_rows=len(h), fit_sites=len(sites),
                                  matched_initialization_and_draws=True, fit_easy_cut=data['easy_cut'])))
    return out


def infer(d, trained, beat):
    h = (d['history']-d['origin'][:, None]).astype(np.float32)
    cv, _ = cost.causal_references(torch.from_numpy(h))
    refs = run.register()
    records = []
    if len(trained['fits']) != len(ARMS)*len(prev.SEEDS) or not trained['matched_initialization_and_draws']:
        raise ValueError('Matched train receipts required')
    for f in trained['fits']:
        seed, arm = f['identity']['seed'], f['identity']['arm']
        st = torch.load(run.checked(f['checkpoint']), map_location='cpu', weights_only=False)
        if st['identity'] != f['identity'] or st['step'] != base.SETTINGS['steps'] or st['identity']['registration'] != run.sha(REG):
            raise ValueError('Expected frozen final checkpoint')
        if arm == ARMS[0]:
            floor, bank = run.candidates(d, refs, seed)
            _, ols = cost.causal_references(torch.from_numpy(h))
            np.testing.assert_allclose(cv.numpy(), floor, atol=.002)
            np.testing.assert_allclose(ols.numpy(), bank[:, run.NAMES.index(OLS)], atol=.002)
        net = base.GatedMotionResidual(False)
        net.load_state_dict(st['model']); net.eval()
        with torch.inference_mode():
            pred = np.concatenate([net({'history': torch.from_numpy(h[i:i+1024])}).numpy()
                                   for i in range(0, len(h), 1024)])
        if pred.shape != (len(h), 12, 2) or not np.isfinite(pred).all():
            raise ValueError('Invalid forecast freeze')
        path = PRIVATE/f'prediction_{seed}_{arm}.npz'
        tmp = path.with_suffix('.pending.npz')
        np.savez(tmp, ids=d['global_ids'], prediction=pred)
        tmp.replace(path)
        records.append(dict(seed=seed, arm=arm, checkpoint=f['checkpoint'], forecast=run.artifact(path)))
        beat(state='forecast_frozen', seed=seed, arm=arm, rows=len(h))
    out = PRIVATE/'freeze.json'
    run.once(out, run.stable(dict(registration=run.sha(REG), uses_past_only=True, forecasts=records)))
    return out


def score(d, frozen, lab, cut, beat):
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {rec: str(d['sites'][ix[0]]) for rec, (ix, _) in records.items()}
    original = prev.receipt(prev.PUBLIC, 'replay')['grouped']
    refs = run.register()
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    rows, replaced = [], 0
    for f in prev.receipt(prev.PUBLIC, 'freeze')['cells']:
        seed = f['seed']
        cv, bank = run.candidates(d, refs, seed)
        preds = {}
        for r in frozen['forecasts']:
            if r['seed'] == seed:
                with np.load(run.checked(r['forecast']), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], d['global_ids'])
                    preds[r['arm']] = z['prediction'].astype(float)
        if set(preds) != set(ARMS):
            raise ValueError('Expected both frozen arms for each seed')
        with np.load(run.checked(f['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            selected = z['choice'][0]
        for rec, (ix, _) in records.items():
            floor, b = cv[ix], bank[ix]
            gain = run.model.selected_paths(selected[ix], b, floor)
            use = (gain != floor).any(axis=(1, 2))
            local = {a: preds[a][ix] for a in ARMS}
            both = use & np.logical_and.reduce([np.linalg.norm(p-floor, axis=2).max(1) > 1e-6
                                                 for p in local.values()])
            matched = {'matched_'+a: np.where(both[:, None, None], p, gain) for a, p in local.items()}
            for p in matched.values():
                np.testing.assert_array_equal((p != floor).any(axis=(1, 2)), use)
            replaced += int(both.sum())
            paths = {BASE: floor, OLS: b[:, run.NAMES.index(OLS)], CONTROL: gain, **local, **matched}
            row = run.parent.readout.score_recording(paths, PAIRS, y[ix], valid[ix], d['queries'][ix], rec,
                                                     easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, seed=seed, source=roster[rec]))
        beat(state='fixed_recipient_scored', seed=seed, views=len(rows))
    grouped = prev.old.prior.grouped_readout(rows, roster, prev.SEEDS)
    if grouped['risk'][CONTROL] != original['risk'][CONTROL] or grouped['risk'][CONTROL]['switches'] != 9675:
        raise ValueError('Fixed-recipient control changed')
    return dict(grouped=grouped, replaced_in_both=replaced, matched_recipients=True, source_disjoint=True,
                external_access=False, independent_confirmation=False, deployment=False,
                stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'train', 'infer', 'score', 'replay'))
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cut, jobs = register()
    if args.phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the protocol and code before running')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2); torch.set_num_interop_threads(1)

        def beat(**v):
            event = dict(pid=os.getpid(), phase=args.phase, seconds=time.monotonic()-start, **v)
            base.fit_source.source.atomic(PRIVATE/'heartbeat.json', event)
            print(event, flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resource guard stopped work; atomic checkpoints preserved')

        mark = PUBLIC/f'{args.phase}_execution.json'
        if mark.exists():
            prev.receipt(PUBLIC, args.phase)
            return
        if args.phase == 'train':
            out = train(jobs, beat, args.resume)
        elif args.phase == 'infer':
            out = infer(run.population(), prev.receipt(PUBLIC, 'train'), beat)
        else:
            frozen = prev.receipt(PUBLIC, 'infer')
            if frozen['registration'] != run.sha(REG) or not frozen['uses_past_only']:
                raise ValueError('Freeze all forecasts before reading development targets')
            if args.phase == 'replay':
                prev.receipt(PUBLIC, 'score')
            lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(run.population(), frozen, lab, cut, beat)))
        run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))


if __name__ == '__main__':
    main()
