"""Matched warm fits test pointwise positive harm against CV and OLS4."""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_dual_reference_forecast as parent
from src.world_model import m3w_pointwise_reference_forecast as model

base, run, prev, np, torch = parent.base, parent.run, parent.prev, parent.np, parent.torch
BASE, OLS, GAIN = parent.BASE, parent.OLS, parent.CONTROL
PUBLIC = parent.PUBLIC.parent/'native_pointwise_reference_forecast_v1'
PRIVATE = parent.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_pointwise_reference_forecast.py')
ARMS = ('cv_continued','pointwise')
SETTINGS = dict(steps=300, batch=128, lr=.0001, checkpoint=100, torch_threads=2, workers=0)
PAIRS = {f'{a}_vs_{b}': (b,a,BASE) for a,b in (
    ('matched_pointwise','matched_cv_continued'), ('pointwise','cv_continued'),
    *[(f'matched_{x}', y) for x in (*ARMS, 'warm') for y in (GAIN, BASE, OLS)],
    *[(f'matched_{x}', 'matched_warm') for x in ARMS],
    *[(x,y) for x in ARMS for y in (BASE,OLS)])}


def register():
    cut, jobs = parent.register()
    trained = prev.receipt(parent.PUBLIC, 'train')
    old = prev.receipt(parent.PUBLIC, 'replay')
    warm = [f for f in trained['fits'] if f['identity']['arm'] == 'cv_only']
    if tuple(f['identity']['seed'] for f in warm) != tuple(prev.SEEDS) or old['replaced_in_both'] != 9675:
        raise ValueError('Verified parent checkpoints and recipient budget required')
    for f in warm:
        run.checked(f['checkpoint'])
    sources = (*FILES, parent.REG, *(parent.PUBLIC/f'{p}_execution.json' for p in ('train','infer','replay')))
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in sources], warm=warm,
        settings=SETTINGS, seeds=prev.SEEDS, arms=ARMS, cut=cut, matched_recipients=9675,
        primary='matched_pointwise_vs_matched_cv_continued', bootstrap=3000,
        official_rule_unchanged=True, external_access=False)))
    return cut, jobs, warm


def train(jobs, warm, beat, resume):
    data = base.fit_source.fitting_only(jobs[0]['identity'], beat)
    sites = set(data['sites'])
    if len(data['ids']) != 69746 or len(sites) != 9 or sites & set(run.old.prior.arrays(('sites',))['sites']):
        raise ValueError('Source-separated fit roster changed')
    h = torch.from_numpy(data['geometry'][:, :16].reshape(-1, 8, 2).copy())
    y, valid = torch.from_numpy(data['target']), torch.from_numpy(data['valid'])
    cv, ols = parent.cost.causal_references(h)
    scale = h.diff(dim=1).square().sum(2).mean(1).sqrt().clamp_min(.05)
    fits = []
    for w in warm:
        seed = w['identity']['seed']
        origin = torch.load(run.checked(w['checkpoint']), map_location='cpu', weights_only=False)
        if origin['step'] != 900 or origin['identity'] != w['identity']:
            raise ValueError('Expected the matched final CV-only warm checkpoint')
        paired = []
        for arm in ARMS:
            net = base.GatedMotionResidual(False)
            opt = torch.optim.AdamW(net.parameters())
            rng = torch.Generator()
            cp = PRIVATE/f'{seed}_{arm}.pt'
            ident = dict(registration=run.sha(REG), seed=seed, arm=arm, warm=w['checkpoint'])
            state = origin
            step, trace, draws = 0, [], np.zeros(len(h), dtype=np.int64)
            if cp.exists():
                if not resume:
                    raise FileExistsError('Existing checkpoints require --resume')
                state = torch.load(cp, map_location='cpu', weights_only=False)
                if state['identity'] != ident or not 0 < state['step'] <= SETTINGS['steps']:
                    raise ValueError('Stale continuation checkpoint')
                step, trace, draws = (state[k] for k in ('step','trace','draws'))
            net.load_state_dict(state['model']); opt.load_state_dict(deepcopy(state['optimizer']))
            rng.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
            for group in opt.param_groups:
                group['lr'] = SETTINGS['lr']
            while step < SETTINGS['steps']:
                ids = base.native.draw_batch(data['groups'], SETTINGS['batch'], rng)
                opt.zero_grad(set_to_none=True)
                loss, refs, pen, extra, active = model.loss(net({'history': h[ids]}), y[ids], valid[ids], cv[ids],
                    ols[ids], scale[ids], data['easy_cut'], treatment=arm == 'pointwise')
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss; keep last atomic checkpoint')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
                opt.step()
                step += 1; np.add.at(draws, ids, 1)
                if step % SETTINGS['checkpoint'] == 0:
                    trace.append(dict(step=step, loss=float(loss.detach()), cv_cost=float(refs[0].detach()),
                        ols4_cost=float(refs[1].detach()), original_penalty=float(pen.detach()),
                        pointwise_excess=float(extra.detach()), differential_gradient_rows=int(active),
                        grad=float(grad)))
                    tmp = cp.with_suffix('.pending')
                    torch.save(dict(identity=ident, model=net.state_dict(), optimizer=opt.state_dict(),
                        sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), trace=trace,
                        step=step, draws=draws), tmp)
                    tmp.replace(cp)
                    beat(state='fit_checkpoint', arm=arm, seed=seed, **trace[-1])
            if draws.sum() != SETTINGS['steps']*SETTINGS['batch']:
                raise ValueError('Continuation fitting budget differs')
            paired.append((draws, rng.get_state(), torch.get_rng_state()))
            fits.append(dict(identity=ident, checkpoint=run.artifact(cp), trace=trace, step=step,
                             draws=int(draws.sum()), parameters=sum(p.numel() for p in net.parameters())))
        base.fit_source.source.exact(*paired)
    out = PRIVATE/'train.json'
    run.once(out, run.stable(dict(fits=fits, fit_rows=len(h), fitting_sites=len(sites), paired_draws=True)))
    return out


def infer(d, trained, beat):
    h = (d['history']-d['origin'][:, None]).astype(np.float32)
    if not trained['paired_draws'] or len(trained['fits']) != 6:
        raise ValueError('Complete paired training is required')
    forecasts = []
    for f in trained['fits']:
        ident = f['identity']
        state = torch.load(run.checked(f['checkpoint']), map_location='cpu', weights_only=False)
        if state['identity'] != ident or state['step'] != SETTINGS['steps'] or ident['registration'] != run.sha(REG):
            raise ValueError('Expected frozen final continuation checkpoint')
        seed, arm = ident['seed'], ident['arm']
        net = base.GatedMotionResidual(False)
        net.load_state_dict(state['model']); net.eval()
        with torch.inference_mode():
            p = np.concatenate([net({'history': torch.from_numpy(h[i:i+1024])}).numpy() for i in range(0,len(h),1024)])
        if p.shape != (len(h),12,2) or not np.isfinite(p).all():
            raise ValueError('Invalid frozen path')
        path = PRIVATE/f'{seed}_{arm}.npz'
        tmp = path.with_suffix('.pending.npz')
        np.savez(tmp, ids=d['global_ids'], prediction=p)
        tmp.replace(path)
        forecasts.append(dict(seed=seed, arm=arm, artifact=run.artifact(path)))
        beat(state='past_only_forecast_frozen', arm=arm, seed=seed, rows=len(h))
    out = PRIVATE/'freeze.json'
    run.once(out, run.stable(dict(forecasts=forecasts, registration=run.sha(REG), uses_past_only=True)))
    return out


def score(cut, freeze, beat):
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval','valid','frames'))
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {r: str(d['sites'][ix[0]]) for r,(ix,_) in records.items()}
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    warm = [dict(seed=f['seed'], arm='warm', artifact=f['forecast'])
            for f in prev.receipt(parent.PUBLIC,'infer')['forecasts'] if f['arm'] == 'cv_only']
    rows, replacements = [], 0
    for f in prev.receipt(prev.PUBLIC,'freeze')['cells']:
        seed = f['seed']
        cv, bank = run.candidates(d, refs, seed)
        predictions = {}
        for p in (*freeze['forecasts'], *warm):
            if p['seed'] == seed:
                with np.load(run.checked(p['artifact']), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], d['global_ids'])
                    predictions[p['arm']] = z['prediction'].astype(float)
        if set(predictions) != {*ARMS,'warm'}:
            raise ValueError('Expected both arms and common initial forecast')
        with np.load(run.checked(f['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            selected = z['choice'][0]
        for rec,(ix,_) in records.items():
            floor, b = cv[ix], bank[ix]
            gain = run.model.selected_paths(selected[ix], b, floor)
            uses = np.any(gain != floor, axis=(1,2))
            pred = {k: p[ix] for k,p in predictions.items()}
            common = uses & np.logical_and.reduce([np.linalg.norm(p-floor, axis=2).max(1) > 1e-6 for p in pred.values()])
            matched = {'matched_'+k: np.where(common[:,None,None], p, gain) for k,p in pred.items()}
            for p in matched.values():
                np.testing.assert_array_equal(np.any(p != floor, axis=(1,2)), uses)
            replacements += int(common.sum())
            paths = {BASE:floor, OLS:b[:,run.NAMES.index(OLS)], GAIN:gain, **pred, **matched}
            row = run.parent.readout.score_recording(paths, PAIRS, y[ix], valid[ix], d['queries'][ix], rec,
                                                     easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, seed=seed, source=roster[rec]))
        beat(state='matched_forecast_scored', seed=seed, views=len(rows))
    grouped = prev.old.prior.grouped_readout(rows, roster, prev.SEEDS)
    old = prev.receipt(parent.PUBLIC,'replay')['grouped']['risk']
    if grouped['risk'][GAIN] != old[GAIN] or grouped['risk'][GAIN]['switches'] != 9675:
        raise ValueError('Fixed gain-policy control changed')
    return dict(grouped=grouped, replacements=replacements, matched_recipients=True, official_rule_changed=False,
        deployment=False, independent_confirmation=False, external_access=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register','train','infer','score','replay'))
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    cut, jobs, warm = register()
    if a.phase == 'register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show',f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit protocol and code before running')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(SETTINGS['torch_threads']); torch.set_num_interop_threads(1)

        def beat(**v):
            info = dict(pid=os.getpid(), phase=a.phase, seconds=time.monotonic()-start, **v)
            base.fit_source.source.atomic(PRIVATE/'heartbeat.json',info)
            print(info, flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resource guard stopped with atomic checkpoints preserved')

        mark = PUBLIC/f'{a.phase}_execution.json'
        if mark.exists():
            prev.receipt(PUBLIC, a.phase)
            return
        if a.phase == 'train':
            out = train(jobs,warm,beat,a.resume)
        elif a.phase == 'infer':
            out = infer(run.population(),prev.receipt(PUBLIC,'train'),beat)
        else:
            frozen = prev.receipt(PUBLIC,'infer')
            if not frozen['uses_past_only'] or frozen['registration'] != run.sha(REG):
                raise ValueError('Freeze past-only forecasts before scoring')
            if a.phase == 'replay':
                prev.receipt(PUBLIC,'score')
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(cut,frozen,beat)))
        run.once(mark,dict(pid=os.getpid(),seconds=time.monotonic()-start,artifact=run.artifact(out)))


if __name__ == '__main__':
    main()
