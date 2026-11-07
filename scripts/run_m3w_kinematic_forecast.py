"""Fit and score matched direct kinematic forecasts on the exposed A roster."""
import argparse
import fcntl
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_easy_aligned_forecast as prior
from src.world_model import m3w_kinematic_forecast as model
from src.world_model import m3w_native_forecast as native

bank, np, torch, read, artifact, checked, sha, once, stable = (
    prior.bank, prior.np, prior.torch, prior.read, prior.artifact, prior.checked, prior.sha, prior.once, prior.stable)
parent = bank.parent
PUBLIC = bank.old.HOME/'moving_direct_kinematic_forecast_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_direct_kinematic_forecast_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_kinematic_forecast.py')
ARMS = ('self', 'peer')
SETTINGS = dict(steps=2400, batch_size=128, learning_rate=.001, minimum_lr_ratio=.1,
                weight_decay=.0001, gradient_clip=5., checkpoint_every=100, heartbeat_every=100)
PAIRS = {'peer_minus_self': ('self', 'peer', parent.CV), **{
    f'{arm}_vs_{key}': (name, arm, name if key != 'eqmotion' else parent.OLS)
    for arm in ARMS for key, name in (('cv', parent.CV), ('ols4', parent.OLS), ('eqmotion', 'eqmotion'))}}


def register():
    paths = [prior.REG, prior.PUBLIC/'fit_population.json', parent.REG, bank.REG, bank.PUBLIC/'analysis.json',
             bank.PUBLIC/'replay_execution.json', Path(native.__file__), Path(prior.__file__),
             Path(parent.__file__), Path(parent.readout.__file__), Path(bank.api.__file__), *bank.old.PARENTS]
    jobs = read(prior.REG)['jobs']
    if tuple(j['seed'] for j in jobs) != prior.SEEDS or len({tuple(j['identity']['fit_groups']) for j in jobs}) != 1:
        raise ValueError('Three registered matched fitting populations required')
    if read(paths[5])['artifact']['sha256'] != sha(paths[4]):
        raise ValueError('Previous candidate ceiling not verified')
    once(REG, stable(dict(files=[artifact(p) for p in FILES], parents=[artifact(p) for p in paths],
        arms=ARMS, settings=SETTINGS, seeds=prior.SEEDS, source_bootstrap_draws=3000,
        comparisons=PAIRS, raw_risk_reference=parent.CV, budget=.02, freeze_before_A_labels=True,
        development_only=True, external_outcomes_opened=False)))
    return jobs


def train(jobs, beat, prefix):
    frozen = PUBLIC/'training_freeze.json'
    if frozen.exists():
        if prefix:
            raise ValueError('The full fit is already frozen')
        fits = read(frozen)['fits']
        for f in fits:
            checked(f['checkpoint'])
            if f['identity']['registration_sha256'] != sha(REG):
                raise ValueError('Cannot reuse a different registration')
        return dict(fits=len(fits), new_updates=0)
    data = prior.fitting_only(jobs[0]['identity'], beat)
    ids = np.arange(len(data['ids'])); fits = []
    for job in jobs:
        seed = job['seed']; warm = torch.load(checked(job['checkpoint']), map_location='cpu', weights_only=False)
        i = job['identity']
        if warm['identity'] != i or warm['step'] != 4000 or prior.source.array_hash(warm['factors']) != i['factors_sha256']:
            raise ValueError('Changed training weights or provenance')
        np.testing.assert_array_equal(warm['train_ids'], data['ids'])
        fold = dict(train_ids=ids, held_ids=ids[:0], groups=data['groups'], factors=warm['factors'][data['ids']])
        states = []
        for arm in ARMS:
            torch.manual_seed(seed)
            net = model.KinematicForecaster(arm == 'peer')
            key = f'{seed}_{arm}'; directory = PRIVATE/'fits'/key
            identity = dict(seed=seed, arm=arm, registration_sha256=sha(REG),
                            global_fit_rows_sha256=i['train_ids_sha256'])
            checkpoint = directory/'checkpoint.pt'
            r = native.fit_trial(net, data, fold, seed=seed, settings=SETTINGS, identity=identity,
                directory=directory, resume=checkpoint.exists(), stop_at=20 if prefix else None,
                heartbeat=lambda **v: beat(key=key, **v))
            if prefix:
                return dict(prefix_step=r['step'], checkpoint=artifact(checkpoint), seed=seed, arm=arm)
            if not r['complete'] or r['held_rows_sampled'] or r['total_draws'] != SETTINGS['steps'] * SETTINGS['batch_size']:
                raise ValueError('Incomplete or non-matched training')
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            states.append((state['draws'], state['sampler_rng'], state['torch_rng']))
            fits.append(dict(key=key, seed=seed, arm=arm, identity=identity, checkpoint=artifact(checkpoint), **r))
        prior.source.exact(*states)
    once(frozen, dict(fits=fits, fit_rows=len(ids), matched_sampling_and_rng=True,
         total_updates=len(fits)*SETTINGS['steps'], before_A_inference_and_labels=True))
    return dict(fits=len(fits), new_updates=sum(r['new_updates'] for r in fits))


def infer(jobs, beat):
    a = bank.old.prior.arrays(('global_ids', 'geometry', 'sites')); ids = a['global_ids']
    if len(ids) != 125540 or set(a['sites']) & set(jobs[0]['identity']['fit_groups']):
        raise ValueError('Keep the full source-excluded A roster')
    refs = []
    for f in read(PUBLIC/'training_freeze.json')['fits']:
        state = torch.load(checked(f['checkpoint']), map_location='cpu', weights_only=False)
        if state['identity'] != f['identity'] or state['step'] != SETTINGS['steps']:
            raise ValueError('Require a frozen completed fit')
        net = model.KinematicForecaster(f['arm'] == 'peer'); net.load_state_dict(state['model'])
        def forward(g):
            return native.predict(net, {'geometry': g}, np.arange(len(g)), 256)
        parts = []; key = f['key']
        for k, offset in enumerate(range(0, len(ids), 4096)):
            ix = slice(offset, offset+4096)
            p, _ = prior.spool_batch(PRIVATE/'chunks'/key, k, f['checkpoint'], ids[ix], a['geometry'][ix], forward)
            parts.append(p)
            if k % 8 == 0:
                beat(state='inference', key=key, rows=min(offset+4096, len(ids)))
        pred = np.concatenate(parts); path = PRIVATE/f'{key}.npz'
        if not path.exists():
            tmp = path.with_suffix('.pending.npz'); np.savez(tmp, ids=ids, prediction=pred); tmp.replace(path)
        else:
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], ids); np.testing.assert_array_equal(z['prediction'], pred)
        refs.append(dict(seed=f['seed'], arm=f['arm'], checkpoint=f['checkpoint'], forecast=artifact(path)))
    if {(r['seed'], r['arm']) for r in refs} != {(s, a) for s in prior.SEEDS for a in ARMS} or len(refs) != 6:
        raise ValueError('Both arms must cover all three seeds')
    once(PUBLIC/'forecast_freeze.json', dict(forecasts=refs, all_before_A_labels=True, rows=len(refs)*len(ids)))
    return dict(forecasts=len(refs), rows=len(refs)*len(ids))


def score(beat):
    frozen = read(PUBLIC/'forecast_freeze.json'); refs = frozen['forecasts']; old = bank.old
    if not frozen['all_before_A_labels'] or len(refs) != 6:
        raise ValueError('Freeze complete forecasts before readout')
    a = old.prior.arrays(('global_ids', 'history', 'origin', 'queries', 'sites', 'recordings'))
    y = old.prior.arrays(('target_eval', 'valid'))
    ids, q, sites, rec = (a[k] for k in ('global_ids', 'queries', 'sites', 'recordings'))
    cuts = read(old.PARENTS[4])['cuts']
    cut = {s: cuts[str(x['job']['producer_fold'])] for x in read(old.PARENTS[3])['forecasts'] for s in x['job']['donor_sites']}
    comparison = parent.readout
    bank_ref = {(r['seed'], r['recording']): r['arms']['full'] for r in read(bank.PUBLIC/'analysis.json')['recording_views']}
    records = []
    for seed in prior.SEEDS:
        preds = parent.read_banks(read(parent.REG)['forecasts'], seed, ids, sites)
        for r in refs:
            if r['seed'] == seed:
                with np.load(checked(r['forecast']), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], ids); preds[r['arm']] = z['prediction'].astype(float)
        for recording in np.unique(rec):
            ix = np.flatnonzero(rec == recording); scene = str(sites[ix[0]]); key = str(recording)
            if not np.all(sites[ix] == scene):
                raise ValueError('Recording crosses sources')
            h = a['history'][ix].astype(float); o = a['origin'][ix, None]; c = cut[scene]
            cv = np.arange(1, 13)[None, :, None] * (h[:, -1] - h[:, -2])[:, None]
            causal = {k: parent.baseline_numpy(h, j) - o for j, k in enumerate(parent.BASELINES) if k != parent.CV}
            paths = {parent.CV: cv, **causal, **{k: v[ix] for k, v in preds.items()}}
            truth, valid = y['target_eval'][ix]-o, y['valid'][ix]
            scored = comparison.score_recording({k: paths[k] for k in (*ARMS, parent.CV, parent.OLS, 'eqmotion')},
                PAIRS, truth, valid, q[ix], key, easy_cut=c['easy'], hard_cut=c['hard'])
            _, inverse, counts = np.unique(q[ix], return_inverse=True, return_counts=True); w = 1./counts[inverse]
            previous = [paths[k] for k in ('eqmotion', 'pct', *causal)]
            solve = lambda p: bank.api.solve(cv, np.stack(p), truth, valid, w, key, c['easy'], c['hard'])
            if solve(previous) != bank_ref[seed, key]:
                raise ValueError('Original full-bank result failed exact replay')
            oracle = {k: solve(previous + [paths[n] for n in group]) for k, group in
                      (('self', ('self',)), ('peer', ('peer',)), ('both', ARMS))}
            records.append(dict(seed=seed, source=scene, bank_ceiling=oracle, **scored))
        beat(state='scored_seed', seed=seed, views=len(records))
    if len(records) != 102 or len(np.unique(sites)) != 14:
        raise ValueError('All fixed views are required')
    names = sorted(set(sites)); draw = np.random.default_rng(73107).integers(len(names), size=(3000, len(names)))
    merged = {(seed, s): {k: comparison.merge_comparisons([r['comparisons'][k] for r in records
        if r['source'] == s and r['seed'] == seed]) for k in PAIRS} for s in names for seed in prior.SEEDS}
    def interval(k, field):
        v = [[merged[seed, s][k][field] for seed in prior.SEEDS] for s in names]
        if any(x is None for a in v for x in a):
            return None
        x = np.asarray(v).mean(1)
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())
    contrasts = {k: {f: interval(k, f) for f in ('known_pp', 'completion_lower_pp')} for k in PAIRS}
    support = {k: dict(feasible=sum(r['bank_ceiling'][k]['feasible'] for r in records),
                      positive=sum((r['bank_ceiling'][k]['completion_gain_mass'] or 0) > 0 for r in records))
               for k in (*ARMS, 'both')}
    risk = {k: sum(r['metrics'][k]['risk']['supported'] for r in records) for k in (*ARMS, parent.OLS, 'eqmotion')}
    once(PUBLIC/'analysis.json', stable(dict(contrasts=contrasts, bank_support=support, raw_risk_passes=risk,
        records=records, original_bank_replay_exact=True, oracle_row_choices_exported=False, original_rules_unchanged=True,
        external_outcomes_opened=False, independent_confirmation=False, deployment_changed=False, submission_ready=False)))
    return dict(analysis=artifact(PUBLIC/'analysis.json'), contrasts=contrasts, bank_support=support, raw_risk_passes=risk)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'prefix', 'train', 'infer', 'score', 'replay'))
    phase = p.parse_args().phase; jobs = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit registered implementation before fitting')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Keep existing atomic checkpoints for resume')
            event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            prior.source.atomic(PRIVATE/'heartbeat.json', event); print(event, flush=True)
        beat(state='started')
        if phase in ('prefix', 'train'):
            out = train(jobs, beat, phase == 'prefix')
        elif phase == 'infer':
            out = infer(jobs, beat)
        else:
            out = score(beat)
        once(PUBLIC/f'{phase}_execution.json', stable(dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **out)))


if __name__ == '__main__':
    main()
