"""Freeze and compare source-deletion cost scores at identical query budgets."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import resource
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 .venv-pytorch')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_motion_choice_cost as parent
from src.world_model import m3w_motion_cost_stability as policy

np, torch, read, artifact, checked, sha, once, stable = (
    parent.np, parent.torch, parent.read, parent.artifact, parent.checked, parent.sha, parent.once, parent.stable)
prior, model, trainer, prefix = parent.prior, parent.model, parent.trainer, parent.prefix
OUT = parent.OUT.parent / 'moving_motion_cost_stability_v1'
PRIVATE = parent.PRIVATE.parent / OUT.name
CV, OLS, SEEDS, STEPS = parent.CV, parent.OLS, parent.SEEDS, parent.STEPS
ARMS = ('full_fit', 'omit_mean', 'omit_min')
PAIRS = dict(min_vs_mean=('omit_mean', 'omit_min'), min_vs_full=('full_fit', 'omit_min'),
             mean_vs_full=('full_fit', 'omit_mean'), **{f'{a}_vs_ols4': (OLS, a) for a in ARMS})


def register():
    reg, jobs = parent.register()
    freeze, score = parent.OUT / 'freeze.json', parent.OUT / 'readout.json'
    a = read(score)
    if a['registration'] != reg or a['choice_freeze'] != sha(freeze) or a['independent_confirmation']:
        raise ValueError('Require the verified full-fit pooled model')
    file = OUT / 'registration.json'
    once(file, stable(dict(parent=[artifact(p) for p in (freeze, score, parent.OUT / 'registration.json')],
        files=[artifact(p) for p in (Path(__file__), Path(policy.__file__), Path(parent.__file__),
                                    Path(trainer.__file__), OUT / 'protocol.md')],
        arms=ARMS, seeds=SEEDS, steps=STEPS, omitted_groups_per_seed=9, fixed_budget_per_query=1,
        fit_id_hash=jobs[0]['identity']['train_ids_sha256'], original_2pct=True,
        independent_confirmation=False, freeze_before_score=True)))
    return sha(file), jobs


@torch.inference_mode()
def all_scores(net, x):
    return np.concatenate([model.scores(net(b), 'pooled_cost').numpy() for b in x.split(2048)])


def a_inputs():
    return prior.bank.old.prior.arrays(('global_ids', 'history', 'queries', 'recordings', 'sites'))


def save_actions(path, data):
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(data):
                raise ValueError('Saved action schema changed')
            for k, v in data.items():
                np.testing.assert_array_equal(z[k], v)
    else:
        tmp = path.with_suffix('.pending.npz')
        np.savez(tmp, **data)
        tmp.replace(path)
    return artifact(path)


def fit_freeze(reg, jobs, replay, beat):
    d = prefix.fitting(jobs, beat)
    samples, fit_sites = parent.fit_inputs(d)
    sources = sorted(set(fit_sites))
    a = a_inputs()
    if len(sources) != 9 or len(a['global_ids']) != 125540 or set(a['sites']) & set(sources):
        raise ValueError('Require fixed disjoint fitting and development rosters')
    groups = [np.flatnonzero(fit_sites == s) for s in sources]
    ols, _, bank, legal, radius = model.candidates(a['history'])
    x = torch.from_numpy(model.inputs(a['history'], ols, bank, radius))
    chosen_query = np.unique(np.stack((a['recordings'], a['queries'].astype(str)), axis=1), axis=0,
                             return_inverse=True)[1].reshape(-1)
    roster = {r['seed']: r for r in read(parent.OUT / 'freeze.json')['cells'] if r['arm'] == 'pooled_cost'}
    rows, updates = [], 0
    for seed in SEEDS:
        checkpoint = torch.load(checked(roster[seed]['fit']['artifact']), weights_only=True, map_location='cpu')
        if checkpoint['identity'] != roster[seed]['fit']['identity'] or checkpoint['step'] != STEPS:
            raise ValueError('Full fit checkpoint identity differs')
        net = model.Head(x.shape[1]).eval()
        net.load_state_dict(checkpoint['net'])
        base = all_scores(net, x)
        chosen = model.choose(base, legal)
        with np.load(checked(roster[seed]['choice']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], a['global_ids'])
            np.testing.assert_array_equal(z['choice'], chosen)
        ii = np.arange(len(chosen))
        base = np.where(chosen >= 0, base[ii, chosen], 0.)
        pairs = policy.contests(chosen_query, radius, chosen >= 0, base)
        estimates, fits = [], []
        for j, omitted in enumerate(sources):
            remain = [g for i, g in enumerate(groups) if i != j]
            ident = dict(registration=reg, seed=seed, omitted_index=j, omitted=str(omitted),
                         fit_ids=jobs[0]['identity']['train_ids_sha256'], retained_rows=sum(map(len, remain)))
            sub, fit, steps = trainer.fit(ident, f'omit_{j}_{seed}', seed, lambda: model.Head(x.shape[1]), STEPS,
                prefix.sample_fn(samples, remain), lambda net, *v: model.loss(net, 'pooled_cost', *v), replay, beat)
            fits.append(fit)
            val = all_scores(sub, x)[ii, chosen]
            estimates.append(np.where(chosen >= 0, val, 0.))
            updates += steps
        e = np.stack(estimates)
        winners = policy.decide(base, e, pairs)
        data = dict(ids=a['global_ids'], chosen=chosen, pairs=pairs, **winners)
        rows.append(dict(seed=seed, fits=fits, n_pairs=len(pairs),
            min_mean_disagreement=int((winners['omit_min'] != winners['omit_mean']).sum()),
            radius_sum=float(radius[pairs].min(1).sum()),
            pivotal_sources=np.bincount(e.argmin(0)[winners['omit_min']], minlength=len(sources)).tolist(),
            choices=save_actions(PRIVATE / f'choices_{seed}.npz', data)))
        beat(phase='actions_frozen', seed=seed, updates=updates)
    file = OUT / 'freeze.json'
    once(file, stable(dict(registration=reg, before_A_labels=True, rows=rows,
        development_rows=len(a['global_ids']), fits=len(rows)*len(sources), total_updates=len(rows)*len(sources)*STEPS)))
    return file, updates


def score(reg, file):
    frozen = read(file)
    if frozen['registration'] != reg or not frozen['before_A_labels'] or len(frozen['rows']) != 3:
        raise ValueError('Missing action freeze')
    a = a_inputs()
    ols, cv, bank, legal, radius = model.candidates(a['history'])
    paths = {}
    for cell in frozen['rows']:
        with np.load(checked(cell['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], a['global_ids'])
            pairs, chosen = z['pairs'], z['chosen']
            flat = pairs.ravel()
            if len(np.unique(flat)) != len(flat) or not legal[flat, chosen[flat]].all():
                raise ValueError('Require disjoint legal contests')
            picked = bank[np.arange(len(chosen)), chosen]
            paths[cell['seed']] = {arm: policy.edited_path(ols, picked, radius, pairs, z[arm]) for arm in ARMS}
    old = prior.bank.old
    labels = old.prior.arrays(('target_eval', 'valid', 'origin'))
    target, valid = labels['target_eval'] - labels['origin'][:, None], labels['valid']
    safe = np.where(valid[..., None], target, 0.)
    anchor_error = np.linalg.norm(ols - safe, axis=-1)
    cuts = read(old.PARENTS[4])['cuts']
    cut = {s: cuts[str(v['job']['producer_fold'])] for v in read(old.PARENTS[3])['forecasts']
           for s in v['job']['donor_sites']}
    rows, roster = [], {}
    for seed in SEEDS:
        for recording in np.unique(a['recordings']):
            ix = np.flatnonzero(a['recordings'] == recording)
            source = str(a['sites'][ix[0]])
            if not np.all(a['sites'][ix] == source):
                raise ValueError('Recording crosses sources')
            roster[str(recording)] = source
            local = {CV: cv[ix], OLS: ols[ix], **{arm: p[ix] for arm, p in paths[seed].items()}}
            q = a['queries'][ix]
            _, inv, n = np.unique(q, return_inverse=True, return_counts=True)
            w = 1. / n[inv]
            r = prior.parent.readout.score_recording(local, {}, target[ix], valid[ix], q, str(recording),
                easy_cut=cut[source]['easy'], hard_cut=cut[source]['hard'])
            r['comparisons'] = {name: prefix.model.full_masses(local[left], local[right], local[CV],
                target[ix], valid[ix], w) for name, (left, right) in PAIRS.items()}
            r['mechanism'] = {arm: dict(observed_harm=float(w @ np.where(valid[ix], np.maximum(
                    np.linalg.norm(local[arm] - safe[ix], axis=-1) - anchor_error[ix], 0.), 0.).sum(1)),
                observed_edits=int(((local[arm] != local[OLS]).any((1, 2)) & valid[ix].any(1)).sum()),
                missing_distance=r['comparisons'][f'{arm}_vs_ols4']['unknown_radius']) for arm in ARMS}
            rows.append(dict(r, source=source, seed=seed))
    summary = prefix.grouped.grouped_readout(rows, roster, SEEDS)
    if (summary['recordings'], summary['source_groups'], summary['views']) != (34, 14, 102):
        raise ValueError('A roster changed')
    mechanism = {arm: {k: float(np.sum([r['mechanism'][arm][k] for r in rows])) for k in
                        ('observed_harm', 'observed_edits', 'missing_distance')} for arm in ARMS}
    once(OUT / 'readout.json', stable(dict(registration=reg, freeze=sha(file), summary=summary, mechanism=mechanism,
        fits=frozen['fits'], total_updates=frozen['total_updates'], contests=[r['n_pairs'] for r in frozen['rows']],
        changed_vs_mean=[r['min_mean_disagreement'] for r in frozen['rows']],
        original_2pct=True, independent_confirmation=False, deployment_changed=False, submission_ready=False)))
    once(PRIVATE / 'recording_scores.json', stable(dict(readout=sha(OUT / 'readout.json'), rows=rows)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    trainer.PRIVATE = PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            state = dict(pid=os.getpid(), seconds=round(time.monotonic() - start, 3), **v)
            prior.prior.source.atomic(PRIVATE / 'heartbeat.json', state)
            if v.get('step', STEPS) in (100, STEPS):
                print(state, flush=True)

        reg, jobs = register()
        file, updates = fit_freeze(reg, jobs, args.replay, beat)
        score(reg, file)
        beat(phase='complete', updates=updates, replay=args.replay,
             max_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == '__main__':
    main()
