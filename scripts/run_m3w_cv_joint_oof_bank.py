"""Build PCT/EqMotion donor predictions without fitting on A or donor rows."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 .venv-pytorch')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_joint_feasibility as study
from src.world_model.m3w_european_source_forecast import SourceForecaster
from src.world_model.m3w_native_forecast import fit_trial, predict

prior, prefix = study.prior, study.parent.parent.prefix
np, torch = study.np, prior.torch
read, sha, once, stable, artifact = study.read, study.sha, study.once, study.stable, study.artifact
checked = prior.checked
OUT = study.OUT.parent / 'moving_cv_joint_oof_v1'
PRIVATE = study.PRIVATE.parent / OUT.name
SEEDS, FAMILIES = study.SEEDS, ('pct', 'eqmotion')
STEPS = 600


def folds(sites):
    names, sizes = np.unique(sites, return_counts=True)
    if len(names) != 9:
        raise ValueError('Require the nine nonempty original fitting sources')
    groups = [[] for _ in range(3)]
    totals = [0, 0, 0]
    for i in sorted(range(len(names)), key=lambda j: (-sizes[j], str(names[j]))):
        g = min((j for j in range(3) if len(groups[j]) < 3), key=lambda j: (totals[j], j))
        groups[g].append(str(names[i]))
        totals[g] += int(sizes[i])
    return groups


def setup(beat):
    old_id, jobs = study.parent.parent.register()
    fit = prior.prior.fitting_only(jobs[0]['identity'], beat)
    h = fit['geometry'][:, :16].reshape(-1, 8, 2)
    from src.evaluation.m3w_native_metrics import native_errors
    err, _ = native_errors(study.motion.bank(h)[:, 1], fit['target'], fit['valid'], np.ones(len(h)))
    groups = folds(fit['sites'])
    archived = read(prior.bank.old.PARENTS[-2])['prediction_receipts']
    a_sites = set(study.parent.parent.a_inputs()['sites'])
    overlap = {str(i): len(set(next(r for r in archived if r['identity']['fold'] == i)['identity']['fit_groups']) & a_sites)
               for i in range(3)}
    if len(a_sites) != 14 or set(fit['sites']) & a_sites or overlap != {'0': 6, '1': 8, '2': 0}:
        raise ValueError('Fitting/source lineage changed')
    config_path = prior.prior.source.CONFIG
    cfg = read(config_path)
    settings = dict(cfg['training'], steps=STEPS, batch_size=16, checkpoint_every=100, heartbeat_every=100)
    reg = OUT / 'registration.json'
    once(reg, stable(dict(parent_joint=artifact(study.OUT / 'readout.json'), parent_fit=old_id,
        files=[artifact(p) for p in (Path(__file__), config_path, Path(fit_trial.__code__.co_filename),
            Path(SourceForecaster.forward.__code__.co_filename), OUT / 'protocol.md')],
        donor_groups=groups, families=FAMILIES, seeds=SEEDS, settings=settings, baseline_index=4,
        a_training_overlap_with_archived_models=overlap, fit_ids=jobs[0]['identity']['train_ids_sha256'],
        a_labels_accessed=False, confirmation_access=False, original_2pct=True)))
    return sha(reg), fit, groups, settings, cfg['families'], err


def design(sites, held_sites, ade):
    held = np.isin(sites, held_sites)
    train, test = np.flatnonzero(~held), np.flatnonzero(held)
    factors = np.zeros(len(sites))
    groups = []
    for s in sorted(set(sites[train])):
        ix = np.flatnonzero(sites == s)
        measured = np.isfinite(ade[ix])
        if not measured.any() or ade[ix][measured].mean() <= 0.:
            raise ValueError('Donor normalizer has no support')
        factors[ix] = len(ix) / measured.sum() / ade[ix][measured].mean()
        groups.append(ix)
    return dict(train_ids=train, held_ids=test, factors=factors, groups=groups)


def run(reg, data, groups, settings, arches, err, beat, replay):
    receipts, updates = [], 0
    for fold, held in enumerate(groups):
        d = design(data['sites'], held, err)
        ids = d['held_ids']
        for seed in SEEDS:
            states = []
            for family in FAMILIES:
                name = f'{fold}_{seed}_{family}'
                root = PRIVATE / name
                ck = root / 'checkpoint.pt'
                info = dict(registration=reg, fold=fold, seed=seed, family=family,
                    donor_held=held, train_sites=sorted(set(data['sites'][d['train_ids']])),
                    train_ids=prior.prior.source.array_hash(data['ids'][d['train_ids']]),
                    held_ids=prior.prior.source.array_hash(data['ids'][ids]))
                torch.manual_seed(seed)
                net = SourceForecaster(arches[family], 4)
                if replay and (not ck.exists() or torch.load(ck, map_location='cpu', weights_only=False)['step'] != STEPS):
                    raise ValueError('Replay requires completed checkpoint')
                fit = fit_trial(net, data, d, seed=seed, settings=settings, identity=info,
                    directory=root, resume=ck.exists(), heartbeat=lambda **v: beat(cell=name, **v))
                state = torch.load(ck, map_location='cpu', weights_only=False)
                if not fit['complete'] or fit['held_rows_sampled'] or fit['total_draws'] != STEPS * settings['batch_size']:
                    raise ValueError('Incomplete or exposed donor producer')
                updates += fit['new_updates']
                states.append([state[k] for k in ('draws', 'sampler_rng', 'torch_rng')])
                pred = predict(net, {'geometry': data['geometry']}, ids, 128)
                if pred.shape != (len(ids), 12, 2) or not np.isfinite(pred).all():
                    raise ValueError('Expected a full finite OOF prediction array')
                path = root / 'donor.npz'
                if path.exists():
                    with np.load(path, allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'], data['ids'][ids])
                        np.testing.assert_array_equal(z['prediction'], pred)
                else:
                    tmp = path.with_suffix('.pending.npz')
                    np.savez(tmp, ids=data['ids'][ids], prediction=pred)
                    tmp.replace(path)
                receipts.append(dict(identity=info, checkpoint=artifact(ck), forecast=artifact(path),
                                     held_rows=len(ids), steps=STEPS))
                beat(state='oof_frozen', cell=name, completed=len(receipts), updates=updates)
            # Architects have different initialization, but must use identical data draws.
            prior.prior.source.exact(states[0][:2], states[1][:2])
    for seed in SEEDS:
        for family in FAMILIES:
            r = [v for v in receipts if v['identity']['seed'] == seed and v['identity']['family'] == family]
            if len(r) != 3 or sum(v['held_rows'] for v in r) != len(data['ids']):
                raise ValueError('Incomplete OOF roster')
    once(OUT / 'freeze.json', stable(dict(registration=reg, cells=receipts, fits=len(receipts),
        fit_rows=len(data['ids']), updates=len(receipts)*STEPS, a_labels_accessed=False,
        confirmation_access=False, held_source_draws=0)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            if shutil.disk_usage(PRIVATE).free < 10 * 2**30 or time.monotonic() - start > 12 * 3600:
                raise RuntimeError('Resource limit reached; resume the last atomic checkpoint')
            state = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            prior.prior.source.atomic(PRIVATE / 'heartbeat.json', state)
            print(state, flush=True)

        result = run(*setup(beat), beat, args.replay)
        beat(state='complete', updates=result, replay=args.replay,
             max_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == '__main__':
    main()
