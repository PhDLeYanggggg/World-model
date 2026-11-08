"""Run the fixed, location-held hindcast residual experiment on OpenDD."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_opendd_location as prior
from scripts import run_m3w_tumdot_ing_frozen_transfer as frozen

PUBLIC = prior.PUBLIC.parent / 'opendd_hindcast_residual_v1'
PRIVATE = prior.PRIVATE.parent / PUBLIC.name
SITES = frozen.SITES
SEEDS = prior.SEEDS
STEPS, BATCH = 600, 128
PAIRS = ('cv', 'hindcast', 'incumbent')


def register():
    code = [Path(__file__).resolve(), Path(frozen.__file__), Path(prior.__file__),
            PUBLIC / 'protocol.md', frozen.VELOCITY / 'registration.json',
            frozen.CALIBRATION / 'registration.json', prior.PUBLIC / 'registration.json']
    for name in ('world_model/m3w_hindcast_residual', 'world_model/m3w_ols_replacement',
                 'world_model/m3w_opendd_velocity_candidate', 'evaluation/m3w_opendd_location',
                 'evaluation/m3w_selection_forecast_readout', 'evaluation/m3w_native_metrics'):
        code.append(ROOT / f'src/{name}.py')
    cohorts = {}
    parent = prior.sha(prior.PUBLIC / 'registration.json')
    for site in SITES:
        path = prior.PRIVATE / f'{site}_cohort.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['registration'] != parent or meta['sha256'] != prior.sha(path):
            raise ValueError(f'Cohort identity changed: {site}')
        cohorts[site] = meta['sha256']
    p = PUBLIC / 'registration.json'
    prior.write_json(p, dict(code={str(a.relative_to(ROOT)): prior.sha(a) for a in code},
        sites=cohorts, seeds=SEEDS, steps=STEPS, batch=BATCH, checkpoint_every=100,
        source='opendd_exposed_development_only', official_risk_budget=.02,
        tumdot_access=False, dronecrowd_confirmation_access=False), immutable=True)
    return prior.sha(p)


def load_site(site):
    import torch
    from src.world_model import m3w_ols_replacement as motion
    from src.world_model import m3w_opendd_velocity_candidate as model

    with np.load(prior.PRIVATE / f'{site}_cohort.npz', allow_pickle=False) as f:
        h, y = f['history'], f['future']
        recordings = np.repeat(f['recording'], np.diff(f['offsets']))
    cv, _, old, legal = frozen.motion_bank(h, motion)
    anchor = cv+.05*(old-cv)
    valid = np.isfinite(y).all(2)
    x, rot, scale = model.design(h, np.full((len(h), 2, 2, 8, 2), np.nan))
    if np.any(x[:, 17:]):
        raise ValueError('Expected self-only design')

    def local(a):
        return torch.from_numpy(((a@rot)/scale[:, None, None]).astype(np.float32))

    cv_err = np.linalg.norm(cv-y, axis=-1)
    cv_ade = np.nansum(cv_err, axis=1)/valid.sum(1).clip(1)
    complete_cv = cv_ade[valid.all(1) & (cv_ade>0)]
    return dict(x=torch.from_numpy(x), rot=rot, scale=scale, recordings=recordings,
                cv=cv, old=old, anchor=anchor, y=y, valid=valid, legal=legal, cv_ade=cv_ade,
                complete_cv=complete_cv, pool=np.flatnonzero(legal),
                local=(local(cv), local(anchor), local(y), torch.from_numpy(valid)))


def cuts_for(donor):
    return np.quantile(np.concatenate([d['complete_cv'] for d in donor]), [.25, .75])


def heartbeat(**state):
    prior.write_json(PRIVATE / 'heartbeat.json', dict(pid=os.getpid(), **state))


def fit(sources, held, seed, arm, reg):
    import torch
    from src.world_model import m3w_hindcast_residual as method
    from src.world_model import m3w_opendd_velocity_candidate as model

    train = [s for s in SITES if s != held]
    donor = [sources[s] for s in train]
    easy_cut = cuts_for(donor)[0]
    easy = [torch.from_numpy((d['cv_ade']>0) & (d['cv_ade']<=easy_cut)) for d in donor]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+1776)
    net = model.network(113)
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    path = PRIVATE / f'{held}_{seed}_{arm}.pt'
    ident = dict(registration=reg, held=held, train=train, arm=arm, seed=seed)
    step, trace = 0, []
    if path.exists():
        ck = torch.load(path, weights_only=True, map_location='cpu')
        if ck['identity'] != ident or ck['step'] not in range(100, STEPS+1, 100):
            raise ValueError(f'Checkpoint provenance mismatch: {path.name}')
        net.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        rng.bit_generator.state = ck['rng']
        step, trace = ck['step'], ck['trace']
    updates = 0
    for i in range(step, STEPS):
        draw = rng.integers(len(donor), size=BATCH)
        indices = [(j, rng.choice(d['pool'], size=int((draw==j).sum()))) for j, d in enumerate(donor)]
        x = torch.cat([donor[j]['x'][at] for j, at in indices])
        cv, anchor, future, valid = [torch.cat([donor[j]['local'][k][at] for j, at in indices]) for k in range(4)]
        e = torch.cat([easy[j][at] for j, at in indices])
        loss = method.objective(method.emitted(net(x), anchor), cv, anchor, future, valid, e, arm)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss: {ident}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        updates += 1
        if (i+1)%100 == 0:
            trace.append(float(loss.detach()))
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=ident, model=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, step=i+1, trace=trace), tmp)
            tmp.replace(path)
            heartbeat(phase='train', held=held, seed=seed, arm=arm, step=i+1, loss=trace[-1])
    return net.eval(), dict(identity=ident, trace=trace, checkpoint_sha256=prior.sha(path)), updates


def paired_summary(d, pred, ref, matched):
    from src.evaluation.m3w_opendd_location import paired

    y, valid = d['y'], d['valid']
    cv_err = np.linalg.norm(d['cv']-y, axis=-1)
    observed, lower = paired(ref, pred, y, valid)
    denom = cv_err[valid].sum()
    complete = valid.all(1)
    active = matched & complete
    if denom <= 0 or cv_err[complete].sum() <= 0:
        raise ValueError('OpenDD reference mass is absent')
    return dict(observed_pp=float(100*observed.sum()/denom), lower_pp=float(100*lower.sum()/denom),
                complete_pp=float(100*observed[complete].sum()/cv_err[complete].sum()),
                edited_complete_rows=int(active.sum()))


def readout(d, paths, support, cuts):
    from src.evaluation.m3w_selection_forecast_readout import policy

    cv = d['cv']
    out = {}
    for name, pred in paths.items():
        edited = (pred != cv).any(axis=(1, 2))
        if name != 'cv' and not np.array_equal(edited, support):
            raise ValueError('Edited recipient matching failed')
        r = policy(pred, cv, cv, d['y'], d['valid'], d['recordings'],
                   easy_cut=cuts[0], hard_cut=cuts[1])
        rs = list(r['recordings'].values())
        out[name] = dict(comparisons={ref: paired_summary(d, pred, paths[ref], support) for ref in PAIRS},
            edits=int(edited.sum()), flights=len(rs), passed=sum(v['supported'] for v in rs),
            undefined_full_easy=sum(v['known_easy_net_degradation'] is None for v in rs),
            undefined_selected_easy=sum(v['completion_easy_risk_upper'] is None for v in rs),
            worst_defined_net_easy_pct=max((100*v['known_easy_net_degradation'] for v in rs
                if v['known_easy_net_degradation'] is not None), default=None),
            available_ade=r['known_ADE_pixel'], available_fde=r['available_endpoint_FDE_diagnostic_pixel'],
            hard_observed_gain_percent=r['subsets']['hard']['gain_percent'])
    return out


def summary(rows):
    draws = np.random.default_rng(817).integers(len(SITES), size=(3000, len(SITES)))
    result = {}
    for arm in rows[SITES[0]][0]['arms']:
        values = [[r['arms'][arm] for r in rows[s]] for s in SITES]
        flat = [r for seq in values for r in seq]
        comparisons = {}
        for ref in PAIRS:
            comparisons[ref] = {}
            for k in ('lower_pp', 'complete_pp', 'observed_pp'):
                x = np.array([np.mean([r['comparisons'][ref][k] for r in seq]) for seq in values])
                comparisons[ref][k] = dict(point=float(x.mean()), ci95=np.quantile(x[draws].mean(1),
                    [.025, .975]).tolist(), positive_locations=int((x>0).sum()))
        counts = {k: sum(r[k] for r in flat) for k in
                  ('edits', 'flights', 'passed', 'undefined_full_easy', 'undefined_selected_easy')}
        result[arm] = dict(comparisons=comparisons, **counts,
            worst_defined_net_easy_pct=max(r['worst_defined_net_easy_pct'] for r in flat
                                           if r['worst_defined_net_easy_pct'] is not None))
    return result


def run(reg, replay):
    import torch
    from src.world_model import m3w_hindcast_residual as method
    from src.world_model import m3w_opendd_velocity_candidate as model

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    t0 = monotonic()
    sources = {s: load_site(s) for s in SITES}
    prior_nets = frozen.load_networks(torch, model)
    frozen_receipt = PUBLIC / 'checkpoint_freeze.json'
    expected = None
    if replay:
        expected = json.loads(frozen_receipt.read_text())
        if expected['registration'] != reg or not (PUBLIC / 'analysis.json').is_file():
            raise ValueError('Replay requires a completed matching run')
    nets, metadata, updates = {}, {}, 0
    for site in SITES:
        print(f'{"Verifying" if replay else "Fitting"} {site}', flush=True)
        for seed in SEEDS:
            for arm in method.ARMS:
                k = site, seed, arm
                key = '/'.join(map(str, k))
                if expected is not None:
                    path = PRIVATE / f'{site}_{seed}_{arm}.pt'
                    if prior.sha(path) != expected['fits'][key]['checkpoint_sha256']:
                        raise ValueError('Replay cannot fit or replace a changed checkpoint')
                nets[k], metadata[key], n = fit(sources, *k, reg)
                updates += n
    prior.write_json(frozen_receipt, dict(registration=reg, fits=metadata), immutable=True)
    if replay and updates:
        raise ValueError('Replay unexpectedly fitted a checkpoint')
    rows = {}
    with torch.inference_mode():
        for site, d in sources.items():
            cv = d['cv']
            rows[site] = []
            cuts = cuts_for([sources[s] for s in SITES if s != site])
            for seed in SEEDS:
                direct = model.decode(prior_nets[seed][SITES.index(site)](d['x'])).numpy().astype(float)
                direct = (direct*d['scale'][:, None, None]) @ d['rot'].transpose(0, 2, 1)
                incumbent, old = frozen.blend(cv, direct, d['old'], d['legal'], .05)
                active = (old != cv).any(axis=(1, 2))
                paths = dict(cv=cv, hindcast=old, incumbent=incumbent)
                for arm in method.ARMS:
                    delta = (.25*.05)*model.decode(nets[site, seed, arm](d['x'])).numpy().astype(float)
                    world = (delta*d['scale'][:, None, None]) @ d['rot'].transpose(0, 2, 1)
                    paths[arm] = old+np.where(active[:, None, None], world, 0.)
                rows[site].append(dict(seed=seed, cuts=cuts.tolist(), arms=readout(d, paths, active, cuts)))
    p = PUBLIC / 'analysis.json'
    prior.write_json(p, dict(registration=reg, targets=sum(len(d['y']) for d in sources.values()),
        by_location=rows, summary=summary(rows), completed_fits=len(nets), fixed_alpha=.05,
        formal_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False), immutable=True)
    receipt = PUBLIC / ('replay_execution.json' if replay else 'execution.json')
    if not receipt.exists():
        prior.write_json(receipt, dict(registration=reg, analysis_sha256=prior.sha(p), updates_this_execution=updates,
            fresh_exact_replay=replay, torch_threads=4, workers=0, pid=os.getpid(),
            wall_seconds=round(monotonic()-t0, 3)), immutable=True)
    heartbeat(phase='completed', replay=replay, updates=updates)
    print(prior.sha(p), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'run', 'replay'))
    a = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reg = register()
        if a.phase == 'register':
            print(reg)
        else:
            if platform.system() != 'Darwin' or platform.machine() != 'arm64':
                raise RuntimeError('Use native arm64 .venv-pytorch on this host')
            run(reg, replay=a.phase=='replay')


if __name__ == '__main__':
    main()
