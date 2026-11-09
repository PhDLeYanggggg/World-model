"""Find and independently replay CV-anchored joint feasibility witnesses."""

import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_anchor_budget as parent
from src.evaluation import m3w_cv_joint_feasibility as model
from src.evaluation.m3w_selection_forecast_readout import policy

np, read, sha, once, stable, artifact = parent.np, parent.read, parent.sha, parent.once, parent.stable, parent.artifact
motion = parent.parent.model.motion
prior = parent.parent.prior
OUT = parent.OUT.parent / 'moving_cv_joint_feasibility_v1'
PRIVATE = parent.parent.PRIVATE.parent / OUT.name
CAPS, SEEDS, SECONDS = (8, 32), parent.parent.SEEDS, 1.5
STATES = ('verified', 'nonpositive_bound', 'risk_infeasible', 'undefined_support', 'unresolved')


def register():
    refs = prior.parent.register()
    source = parent.OUT / 'readout.json'
    old = read(source)
    if old['independent_confirmation'] or old['registration'] != sha(parent.OUT / 'registration.json'):
        raise ValueError('Require the checked prior CV capacity result')
    f = OUT / 'registration.json'
    once(f, stable(dict(parent=[artifact(p) for p in (source, prior.parent.REG)],
        files=[artifact(p) for p in (Path(__file__), Path(model.__file__), Path(parent.audit.__file__),
            Path(motion.__file__), Path(policy.__code__.co_filename), OUT / 'protocol.md')],
        analytic_names=motion.EXPERTS[1:], additional_names=['eqmotion', 'pct'], caps=CAPS, seeds=SEEDS,
        seconds_per_lp_and_mip=SECONDS, independent_confirmation=False, original_2pct=True)))
    return sha(f), refs


def verify(actions, cv, ols, bank, y, mask, q, easy, hard, cap):
    edits = np.asarray(actions, dtype=int).reshape(-1, 2)
    row, arm = edits.T
    if ((row < 0) | (row >= len(cv))).any() or ((arm < 0) | (arm >= bank.shape[1])).any():
        raise ValueError('Action outside registered menu')
    if len(np.unique(row)) != len(row) or np.unique(q[row], return_counts=True)[1].max(initial=0) > cap:
        raise ValueError('Budget violation')
    path = cv.copy()
    path[row] = bank[row, arm]
    actual = policy(path, cv, cv, y, mask, np.repeat('r', len(q)), easy_cut=easy, hard_cut=hard)['recordings']['r']
    _, inv, counts = np.unique(q, return_inverse=True, return_counts=True)
    weight = 1. / counts[inv]
    terms = [motion.full_masses(p, path, cv, y, mask, weight) for p in (cv, ols)]
    norm = terms[0]['reference']
    gain = [100*(v['known_delta']-v['unknown_radius'])/norm for v in terms] if norm > 0. else None
    ok = bool(gain is not None and min(gain) > model.POSITIVE_PP and actual['supported'] and actual['selected'] == len(row))
    return stable(dict(verified=ok, gain_pp=gain, original_risk=actual, edits=len(row),
                       mean_peak=float(np.linalg.norm(path[row]-cv[row], axis=-1).max(1).mean()) if len(row) else None,
                       reference=norm))


def cell(path, identity, data, args, cap, inherited, replay):
    if not path.exists():
        if replay:
            raise ValueError('Missing solver checkpoint')
        take, solver = model.solve(data, cap, seconds=SECONDS)
        actions = np.column_stack((data['row'][take], data['arm'][take])).tolist() if take is not None else None
        receipt = verify(actions, *args, cap) if actions is not None else None
        status = 'verified' if receipt is not None and receipt['verified'] else solver['status']
        reused = False
        if status != 'verified' and inherited is not None:
            if status != 'unresolved':
                raise ValueError('Solver negative contradicts a verified nested witness')
            receipt = verify(inherited, *args, cap)
            if not receipt['verified']:
                raise ValueError('Inherited action does not verify')
            actions, status, reused = inherited, 'verified', True
        once(path, stable(dict(identity=identity, status=status, solver=solver, actions=actions,
                              witness=receipt, reused_nested_witness=reused, before_prune=data['actions_before_prune'],
                              after_prune=len(data['row']))))
    r = read(path)
    if r['identity'] != identity or (r['actions'] is not None and verify(r['actions'], *args, cap) != r['witness']):
        raise ValueError('Solver witness or identity failed exact replay')
    return r


def summarize(cells):
    by = {(r['recording'], r['cap'], r['seed']): r for r in cells}
    groups = sorted({r['source'] for r in cells})
    result = {}
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    for cap in CAPS:
        a = [r for r in cells if r['cap'] == cap and r['seed'] is None]
        n = [r for r in cells if r['cap'] == cap and r['seed'] is not None]
        if (len(a), len(n)) != (34, 102):
            raise ValueError('A roster mismatch')
        paired = [(by[r['recording'], cap, None], r) for r in n]
        proof = [(l, r) for l, r in paired if r['status'] == 'verified' and l['status'] not in ('verified', 'unresolved')]
        if any(l['status'] == 'verified' and r['status'] != 'verified' for l, r in paired):
            raise ValueError('Nested verified witness was lost')
        rate = np.array([np.mean([100. * ((r['status'] == 'verified') - (l['status'] == 'verified'))
                                  for l, r in paired if r['source'] == g]) for g in groups])
        result[str(cap)] = dict(
            analytic={k: sum(r['status'] == k for r in a) for k in STATES},
            expanded={k: sum(r['status'] == k for r in n) for k in STATES},
            confirmed_new_views=len(proof), confirmed_new_recordings=len({r['recording'] for _, r in proof}),
            confirmed_new_sources=len({r['source'] for _, r in proof}),
            verified_rate_delta_pp=float(rate.mean()), interval95_pp=np.quantile(rate[draw].mean(1), [.025,.975]).tolist(),
            analytic_mean_verified_edits=mean_edits(a), expanded_mean_verified_edits=mean_edits(n))
    return dict(caps=result, source_groups=len(groups), recording_seed_views=102, bootstrap_draws=len(draw))


def mean_edits(rows):
    use = [r['witness']['edits'] for r in rows if r['status'] == 'verified']
    return float(np.mean(use)) if use else None


def execute(reg, refs, replay, beat):
    a = parent.parent.a_inputs()
    neural = {s: prior.parent.read_banks(refs, s, a['global_ids'], a['sites']) for s in SEEDS}
    lab = prior.bank.old.prior.arrays(('target_eval', 'origin', 'valid'))
    y = lab['target_eval'] - lab['origin'][:, None]
    mask = lab['valid']
    analytic = motion.bank(a['history'])
    old = prior.bank.old
    cuts = read(old.PARENTS[4])['cuts']
    cut = {s: cuts[str(v['job']['producer_fold'])] for v in read(old.PARENTS[3])['forecasts']
           for s in v['job']['donor_sites']}
    cells = []
    for rec in np.unique(a['recordings']):
        ix = a['recordings'] == rec
        source, = np.unique(a['sites'][ix])
        c = cut[source]
        cv, ols = analytic[ix, 0], analytic[ix, 1]
        controls = {}
        for seed in (None, *SEEDS):
            bank = analytic[ix, 1:]
            if seed is not None:
                bank = np.concatenate((bank, np.stack([neural[seed][k][ix] for k in ('eqmotion', 'pct')], axis=1)), axis=1)
            args = (cv, ols, bank, y[ix], mask[ix], a['queries'][ix], c['easy'], c['hard'])
            d = model.coefficients(*args[:-1])
            smaller = None
            for cap in CAPS:
                identity = dict(registration=reg, recording=str(rec), source=str(source), seed=seed, cap=cap)
                inherit = controls.get(cap) or smaller
                r = cell(PRIVATE / f'{rec}_{seed}_{cap}.json', identity, d, args, cap, inherit, replay)
                if r['status'] == 'verified':
                    smaller = r['actions']
                    if seed is None:
                        controls[cap] = smaller
                cells.append({**identity, **{k: r[k] for k in ('status', 'solver', 'witness', 'reused_nested_witness')}})
        beat(recording=str(rec), cells=len(cells), total=272, replay=replay)
    if len(cells) != 272 or len(a['history']) != 125540:
        raise ValueError('Expected full exposed A roster')
    score = summarize(cells)
    once(OUT / 'readout.json', stable(dict(registration=reg, summary=score, labels_hindsight_only=True,
        original_2pct=True, independent_confirmation=False, deployment_changed=False, submission_ready=False)))
    once(PRIVATE / 'recording_receipt.json', stable(dict(registration=reg, cells=cells)))
    print(score, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            state = dict(pid=os.getpid(), seconds=round(time.monotonic() - start, 3), **v)
            prior.prior.source.atomic(PRIVATE / 'heartbeat.json', state)
            print(state, flush=True)

        execute(*register(), args.replay, beat)


if __name__ == '__main__':
    main()
