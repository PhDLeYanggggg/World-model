"""Train and replay the matched moving-source rank/cost experiment."""

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
from scripts import run_m3w_prefix_expert_mixture as prefix
from src.world_model import m3w_motion_choice_cost as model

np, torch, prior, trainer = prefix.np, prefix.torch, prefix.prior, prefix.trainer
read, artifact, checked, sha, once, stable = prefix.read, prefix.artifact, prefix.checked, prefix.sha, prefix.once, prefix.stable
OUT = prefix.OUT.parent / 'moving_motion_choice_cost_v1'
PRIVATE = prefix.PRIVATE.parent / OUT.name
SEEDS, STEPS = prefix.SEEDS, prefix.STEPS
CV, OLS = prefix.CV, prefix.OLS
PAIRS = {f'{a}_vs_ols4': (OLS, a) for a in model.ARMS}
PAIRS.update(pooled_vs_rank=('rank', 'pooled_cost'), lead_vs_rank=('rank', 'lead_cost'),
             lead_vs_pooled=('pooled_cost', 'lead_cost'))


def register():
    parent, jobs = prefix.register()
    earlier = prefix.OUT / 'lead_alignment.json'
    f = OUT / 'registration.json'
    once(f, stable(dict(parent=parent, prior_lead_analysis=artifact(earlier),
        files=[artifact(p) for p in (Path(__file__), Path(model.__file__), Path(prefix.__file__),
               Path(prefix.model.__file__), Path(trainer.__file__), Path(prefix.grouped.__file__),
               Path(prior.parent.readout.__file__), Path(prior.parent.readout.policy.__code__.co_filename),
               OUT / 'protocol.md')],
        input_grid=[8, 12], arms=model.ARMS, seeds=SEEDS, steps=STEPS, batch=prefix.BATCH,
        experts=[prefix.model.EXPERTS[i] for i in model.EXPERTS], radius_fraction=model.RADIUS_FRACTION,
        primary='lead cost minus pooled cost, paired full-grid lower versus common CV denominator',
        recipients='force exact matched edits when at least two causal options reach fixed radius',
        independent_confirmation=False, original_2pct=True, no_readout_selection=True)))
    return sha(f), jobs


def fit_inputs(d):
    ols, _, bank, legal, radius = model.candidates(d['history'])
    keep = (legal.sum(1) >= 2) & d['valid'].any(1)
    x = model.inputs(d['history'][keep], ols[keep], bank[keep], radius[keep])
    y = model.targets(ols[keep], bank[keep], radius[keep], d['target'][keep], d['valid'][keep])
    if not np.isfinite(y).all() or (np.abs(y[legal[keep]]) > 1.00001).any():
        raise ValueError('Paired targets must obey the displacement bound')
    return (tuple(torch.from_numpy(z) for z in (x, legal[keep], y, d['valid'][keep])), d['sites'][keep])


@torch.inference_mode()
def infer(net, x, legal, arm):
    result = np.concatenate([model.scores(net(t), arm).numpy() for t in x.split(2048)])
    return model.choose(result, legal)


def freeze_choice(name, ids, choice):
    f = PRIVATE / f'{name}.npz'
    if f.exists():
        with np.load(f, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            np.testing.assert_array_equal(z['choice'], choice)
    else:
        tmp = f.with_suffix('.pending.npz')
        np.savez(tmp, ids=ids, choice=choice)
        tmp.replace(f)
    return artifact(f)


def train_and_freeze(reg, jobs, replay, beat):
    d = prefix.fitting(jobs, beat)
    samples, fit_sites = fit_inputs(d)
    groups = [np.flatnonzero(fit_sites == s) for s in sorted(set(d['sites']))]
    a = prior.bank.old.prior.arrays(('global_ids', 'history', 'sites'))
    if len(a['global_ids']) != 125540 or set(a['sites']) & set(d['sites']) or not all(len(g) for g in groups):
        raise ValueError('Expected full A roster with distinct source groups and fitting support')
    ols, _, bank, legal, radius = model.candidates(a['history'])
    x = torch.from_numpy(model.inputs(a['history'], ols, bank, radius))
    sample = prefix.sample_fn(samples, groups)
    cells, updates = [], 0
    for seed in SEEDS:
        rng = []
        for arm in model.ARMS:
            name = f'{arm}_{seed}'
            identity = dict(registration=reg, seed=seed, arm=arm, fitting=jobs[0]['identity']['train_ids_sha256'])
            net, fit, steps = trainer.fit(identity, name, seed, lambda: model.Head(x.shape[1]),
                STEPS, sample, lambda net, *data: model.loss(net, arm, *data), replay, beat)
            state = torch.load(checked(fit['artifact']), weights_only=True, map_location='cpu')
            rng.append(state['rng'])
            updates += steps
            choice = infer(net, x, legal, arm)
            cells.append(dict(seed=seed, arm=arm, fit=fit, choice=freeze_choice(name, a['global_ids'], choice)))
            beat(phase='choice_frozen', name=name, cells=len(cells), updates=updates)
        if any(r != rng[0] for r in rng[1:]):
            raise ValueError('Matched RNG states diverged')
    f = OUT / 'freeze.json'
    once(f, stable(dict(registration=reg, cells=cells, before_A_labels=True, matched_rng=True,
        fit_rows=len(d['ids']), fit_eligible=len(fit_sites), A_rows=len(x), edits_per_seed=int((legal.sum(1) >= 2).sum()),
        parameters=sum(p.numel() for p in net.parameters()), updates=len(cells)*STEPS)))
    return f, updates


def readout(reg, f):
    freeze, old = read(f), prior.bank.old
    if freeze['registration'] != reg or not freeze['before_A_labels'] or len(freeze['cells']) != 9:
        raise ValueError('Require all matched choices frozen before scoring')
    a = old.prior.arrays(('global_ids', 'history', 'origin', 'queries', 'sites', 'recordings'))
    ols, cv, bank, legal, radius = model.candidates(a['history'])
    picks = {}
    for cell in freeze['cells']:
        with np.load(checked(cell['choice']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], a['global_ids'])
            picks[cell['seed'], cell['arm']] = z['choice']
    paths = {key: model.apply(ols, bank, legal, radius, choice) for key, choice in picks.items()}
    target = old.prior.arrays(('target_eval', 'valid'))
    y, valid = target['target_eval'] - a['origin'][:, None], target['valid']
    cuts = read(old.PARENTS[4])['cuts']
    cut = {s: cuts[str(x['job']['producer_fold'])] for x in read(old.PARENTS[3])['forecasts']
           for s in x['job']['donor_sites']}
    used = legal.sum(1) >= 2
    available = used & valid.any(1)
    safe = np.where(valid[..., None], y, 0.)
    anchor_err = np.linalg.norm(ols - safe, axis=-1)
    gain = np.where(valid[:, None], anchor_err[:, None] - np.linalg.norm(bank - safe[:, None], axis=-1), 0.)
    gain_sum = gain.sum(-1)
    oracle = np.where(legal[available], gain_sum[available], -np.inf).max(1)
    diagnostic = {}
    for arm in model.ARMS:
        v, u, wins, lead_harm = [], [], [], []
        for seed in SEEDS:
            selected = gain[available, picks[seed, arm][available]]
            s = selected.sum(1)
            v.append(float(s.sum()))
            u.append(float((oracle - s).sum()))
            wins.append(float((oracle - s < 1e-7).mean()))
            lead_harm.append(np.maximum(-selected, 0.).sum(0))
        diagnostic[arm] = dict(action_histograms=[np.bincount(picks[s, arm][used], minlength=len(model.EXPERTS)).tolist()
            for s in SEEDS], known_regret_px_per_labeled_step=float(np.mean(u)/valid[available].sum()),
            known_win_rate=float(np.mean(wins)), net_gain_sum=float(np.mean(v)),
            lead_harm_mass=np.mean(lead_harm, axis=0).tolist())
    rows, roster = [], {}
    for seed in SEEDS:
        for recording in np.unique(a['recordings']):
            ix = np.flatnonzero(a['recordings'] == recording)
            key, source = str(recording), str(a['sites'][ix[0]])
            if not np.all(a['sites'][ix] == source):
                raise ValueError('Recording crosses source groups')
            roster[key] = source
            local = {CV: cv[ix], OLS: ols[ix], **{arm: paths[seed, arm][ix] for arm in model.ARMS}}
            q = a['queries'][ix]
            _, inv, n = np.unique(q, return_inverse=True, return_counts=True)
            w = 1. / n[inv]
            r = prior.parent.readout.score_recording(local, {}, y[ix], valid[ix], q, key,
                                                     easy_cut=cut[source]['easy'], hard_cut=cut[source]['hard'])
            for name, (left, right) in PAIRS.items():
                for end, sl in (('', slice(None)), ('_lead12', slice(-1, None))):
                    r['comparisons'][name + end] = prefix.model.full_masses(
                        local[left][:, sl], local[right][:, sl], local[CV][:, sl], y[ix, sl], valid[ix, sl], w)
            rows.append(dict(r, source=source, seed=seed))
    summary = prefix.grouped.grouped_readout(rows, roster, SEEDS)
    if summary['views'] != 102 or summary['source_groups'] != 14:
        raise ValueError('Require all registered A views')
    payload = stable(dict(registration=reg, choice_freeze=sha(f), summary=summary,
        diagnostics=diagnostic, fit_eligible=freeze['fit_eligible'], edits_per_seed=int(used.sum()),
        known_edited=int(available.sum()), lead12_edited=int((used & valid[:, -1]).sum()),
        oracle_diagnostic_gain_vs_ols4=float(100*oracle.sum()/np.where(valid[available], anchor_err[available], 0.).sum()),
        parameters=freeze['parameters'], updates=freeze['updates'], original_2pct=True,
        independent_confirmation=False, deployment_changed=False, submission_ready=False))
    once(OUT / 'readout.json', payload)
    once(PRIVATE / 'recording_scores.json', stable(dict(readout=sha(OUT / 'readout.json'), rows=rows)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true', help='Require completed checkpoints, no updates')
    args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    trainer.PRIVATE = PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**values):
            message = dict(pid=os.getpid(), seconds=round(time.monotonic() - start, 3), **values)
            prior.prior.source.atomic(PRIVATE / 'heartbeat.json', message)
            if values.get('step', STEPS) in (100, STEPS):
                print(message, flush=True)

        reg, jobs = register()
        f, updates = train_and_freeze(reg, jobs, args.replay, beat)
        readout(reg, f)
        beat(phase='complete', updates=updates, replay=args.replay,
             max_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == '__main__':
    main()
