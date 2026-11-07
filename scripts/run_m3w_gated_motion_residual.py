"""Fit OLS4-relative motion heads, freeze forecasts and compare fixed recipients."""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch, not Rosetta')

from scripts import audit_m3w_dual_reference_capacity as dual
from scripts import run_m3w_easy_aligned_forecast as fit_source
from src.world_model import m3w_native_forecast as native
from src.world_model.m3w_gated_motion_residual import GatedMotionResidual

run, prev, np, torch = dual.run, dual.prev, dual.np, fit_source.torch
PUBLIC = dual.PUBLIC.parent/'native_gated_motion_residual_v1'
PRIVATE = dual.PRIVATE.parent/'native_gated_motion_residual_v1'
REG = PUBLIC/'registration.json'
ARMS = ('ungated', 'gated')
BASE, OLS, CONTROL = dual.parent.BASE, dual.parent.OLS, 'gain'
SETTINGS = dict(steps=900, batch_size=128, learning_rate=.001, minimum_lr_ratio=.1, weight_decay=.0001,
                gradient_clip=5., checkpoint_every=150, heartbeat_every=150)
FILES = (Path(__file__).resolve(), ROOT/'src/world_model/m3w_gated_motion_residual.py',
         PUBLIC/'protocol.md', ROOT/'tests/test_m3w_gated_motion_residual.py')
PAIRS = {f'{a}_vs_{b}': (b, a, BASE) for a, b in (
    ('gated', 'ungated'), ('gated', OLS), ('ungated', OLS), ('gated', BASE), ('ungated', BASE),
    ('matched_gated', 'matched_ungated'), ('matched_gated', CONTROL), ('matched_ungated', CONTROL))}


def register():
    cut, _ = dual.register()
    jobs = fit_source.registration()
    manifest = fit_source.PUBLIC/'fit_population.json'
    old = run.read(manifest)
    if old['rows'] != 69746 or len(old['groups']) != 9 or old['A_target_opened']:
        raise ValueError('Expected the original source-separated fitting roster')
    bindings = (*FILES, manifest, fit_source.REG, Path(fit_source.__file__), Path(native.__file__), dual.REG,
                prev.PUBLIC/'freeze_execution.json', prev.PUBLIC/'replay_execution.json')
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in bindings], cut=cut, settings=SETTINGS,
        arms=ARMS, seeds=prev.SEEDS, model_selection=False, external_access=False,
        torch_threads=2, data_loader_workers=0, original_risk_unchanged=True, pairs=PAIRS)))
    return cut, jobs


def train(jobs, beat, resume):
    path = PRIVATE/'train.json'
    if path.exists():
        trained = run.read(path)
        for f in trained['fits']:
            run.checked(f['checkpoint'])
            if f['identity']['registration'] != run.sha(REG):
                raise ValueError('Training receipt registration differs')
        return path
    data = fit_source.fitting_only(jobs[0]['identity'], beat)
    fitting = set(data['sites'])
    dev = set(run.old.prior.arrays(('sites',))['sites'])
    if len(data['ids']) != 69746 or fitting & dev:
        raise ValueError('Fitting is not source separated from development')
    g = data['geometry'][:, :16].reshape(-1, 8, 2)
    rms = np.maximum(np.sqrt(np.square(np.diff(g, axis=1)).sum(2).mean(1)), .05)
    fold = dict(train_ids=np.arange(len(g)), held_ids=np.empty(0, dtype=int),
                groups=data['groups'], factors=1./rms)
    fits = []
    for seed in prev.SEEDS:
        initial = None
        states = []
        for arm in ARMS:
            torch.manual_seed(seed)
            net = GatedMotionResidual(arm == 'gated')
            if initial is None:
                initial = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                fit_source.source.exact(initial, net.state_dict())
            directory = PRIVATE/'fits'/f'{seed}_{arm}'
            cp = directory/'checkpoint.pt'
            identity = dict(seed=seed, arm=arm, registration=run.sha(REG),
                            fit_global_ids=jobs[0]['identity']['train_ids_sha256'])
            result = native.fit_trial(net, data, fold, seed=seed, settings=SETTINGS, identity=identity,
                                      directory=directory, resume=resume and cp.exists(),
                                      heartbeat=lambda **v: beat(seed=seed, arm=arm, **v))
            if not result['complete'] or result['held_rows_sampled'] or result['total_draws'] != 900*128:
                raise ValueError('Expected complete matched training budget without held data')
            state = torch.load(cp, map_location='cpu', weights_only=False)
            states.append(tuple(state[k] for k in ('draws', 'sampler_rng', 'torch_rng')))
            fits.append(dict(identity=identity, checkpoint=run.artifact(cp), result=result))
        fit_source.source.exact(*states)
    run.once(path, run.stable(dict(fits=fits, initialization_and_draws_matched=True,
                                    fit_rows=len(g), fit_groups=len(fitting), overlap=0)))
    return path


def infer(d, trained, beat):
    history = (d['history']-d['origin'][:, None]).astype(np.float32)
    if trained['fit_rows'] != 69746 or trained['overlap'] or len(trained['fits']) != 6:
        raise ValueError('Expected fully trained source-excluded arm pair on every seed')
    refs = []
    for f in trained['fits']:
        ident = f['identity']
        seed, arm = ident['seed'], ident['arm']
        path = PRIVATE/f'prediction_{seed}_{arm}.npz'
        mark = path.with_suffix('.json')
        if not mark.exists():
            st = torch.load(run.checked(f['checkpoint']), map_location='cpu', weights_only=False)
            if st['identity'] != ident or st['step'] != SETTINGS['steps'] or ident['registration'] != run.sha(REG):
                raise ValueError('Expected frozen final source-separated checkpoint')
            net = GatedMotionResidual(arm == 'gated')
            net.load_state_dict(st['model'])
            net.eval()
            with torch.no_grad():
                pred = np.concatenate([net({'history': torch.from_numpy(history[i:i+1024])}).numpy()
                                       for i in range(0, len(history), 1024)])
            if pred.shape != (len(history), 12, 2) or not np.isfinite(pred).all():
                raise ValueError('Invalid forecast freeze')
            tmp = path.with_suffix('.pending.npz')
            np.savez(tmp, ids=d['global_ids'], prediction=pred)
            tmp.replace(path)
            run.once(mark, dict(checkpoint=f['checkpoint'], forecast=run.artifact(path), seed=seed, arm=arm))
        r = run.read(mark)
        if (r['seed'], r['arm'], r['checkpoint']) != (seed, arm, f['checkpoint']):
            raise ValueError('Stale frozen forecast')
        run.checked(r['forecast'])
        refs.append(r)
        beat(state='forecast_frozen', seed=seed, arm=arm, rows=len(history))
    path = PRIVATE/'freeze.json'
    run.once(path, run.stable(dict(forecasts=refs, registration=run.sha(REG), uses_past_only=True)))
    return path


def score(d, refs, lab, cut, frozen, choices, beat):
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {rec: str(d['sites'][ix[0]]) for rec, (ix, _) in records.items()}
    original = prev.receipt(prev.PUBLIC, 'replay')['grouped']
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    rows, n_sub = [], 0
    for f in choices['cells']:
        seed = f['seed']
        cv, bank = run.candidates(d, refs, seed)
        all_pred = {}
        for r in frozen['forecasts']:
            if r['seed'] == seed:
                with np.load(run.checked(r['forecast']), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], d['global_ids'])
                    all_pred[r['arm']] = z['prediction'].astype(float)
        if set(all_pred) != set(ARMS):
            raise ValueError('Expected both frozen arms for each seed')
        with np.load(run.checked(f['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            selected = z['choice'][0]
        for rec, (ix, _) in records.items():
            floor, b = cv[ix], bank[ix]
            gain = run.model.selected_paths(selected[ix], b, floor)
            use = (gain != floor).any(axis=(1, 2))
            local = {a: all_pred[a][ix] for a in ARMS}
            both = use & np.logical_and.reduce([np.linalg.norm(p-floor, axis=2).max(1) > 1e-6
                                                 for p in local.values()])
            matched = {'matched_'+a: np.where(both[:, None, None], p, gain) for a, p in local.items()}
            for p in matched.values():
                np.testing.assert_array_equal((p != floor).any(axis=(1, 2)), use)
            n_sub += int(both.sum())
            paths = {BASE: floor, OLS: b[:, run.NAMES.index(OLS)], CONTROL: gain, **local, **matched}
            row = run.parent.readout.score_recording(paths, PAIRS, y[ix], valid[ix], d['queries'][ix], rec,
                                                     easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(row, seed=seed, source=roster[rec]))
        beat(state='fixed_policy_scored', seed=seed, views=len(rows))
    grouped = prev.old.prior.grouped_readout(rows, roster, prev.SEEDS)
    if grouped['risk'][CONTROL] != original['risk'][CONTROL] or grouped['risk'][CONTROL]['switches'] != 9675:
        raise ValueError('Fixed-recipient original gain control differs')
    return dict(grouped=grouped, original_recipient_masks_equal=True, both_head_substitutions=n_sub,
                fit_sources_exclude_development=True, external_access=False, independent_confirmation=False,
                deployment=False, stage5c_executed=False, smc_enabled=False)


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
            raise ValueError('Commit the experiment before real training')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)

        def beat(**v):
            event = dict(pid=os.getpid(), phase=args.phase, seconds=time.monotonic()-start, **v)
            fit_source.source.atomic(PRIVATE/'heartbeat.json', event)
            print(event, flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Preserved existing atomic checkpoints for resume')

        if args.phase == 'train':
            out = train(jobs, beat, args.resume)
        elif args.phase == 'infer':
            out = infer(run.population(), prev.receipt(PUBLIC, 'train'), beat)
        else:
            frozen = prev.receipt(PUBLIC, 'infer')
            if frozen['registration'] != run.sha(REG) or not frozen['uses_past_only']:
                raise ValueError('Freeze all forecasts before scoring')
            if args.phase == 'replay':
                prev.receipt(PUBLIC, 'score')
            lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
            choices = prev.receipt(prev.PUBLIC, 'freeze')
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(run.population(), run.register(), lab, cut, frozen, choices, beat)))
        mark = PUBLIC/f'{args.phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Outcome replay differs')


if __name__ == '__main__':
    main()
