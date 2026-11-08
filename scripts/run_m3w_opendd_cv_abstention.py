"""Nested site-held evaluation of CV-anchored expected gain and abstention."""
import argparse
from collections import Counter
import fcntl
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_opendd_location as prior

PUBLIC = prior.PUBLIC.parent / 'opendd_cv_abstention_v1'
PRIVATE = prior.PRIVATE.parent / PUBLIC.name
ARMS = ('cv_only', 'dual')
THRESHOLDS = (0., .002, .005, .01, .02, .04, .08)
SEEDS = prior.SEEDS


def beat(phase, **extra):
    state = dict(phase=phase, pid=os.getpid(), **extra)
    prior.write_json(PRIVATE / 'heartbeat.json', state)
    print(json.dumps(state), flush=True)


def register():
    paths = (Path(__file__).resolve(), PUBLIC/'protocol.md',
             prior.PUBLIC/'registration.json', prior.PUBLIC/'analysis.json',
             Path(prior.__file__), ROOT/'src/evaluation/m3w_opendd_cv_abstention.py',
             ROOT/'src/evaluation/m3w_opendd_location.py', ROOT/'src/world_model/m3w_ols_replacement.py')
    path = PUBLIC/'registration.json'
    prior.write_json(path, dict(files={str(p.relative_to(ROOT)): prior.sha(p) for p in paths},
        seeds=SEEDS, arms=ARMS, thresholds=THRESHOLDS, steps=prior.STEPS, batch=prior.BATCH,
        role='followup_development_diagnostic', original_risk_budget=.02,
        stage5c_executed=False, smc_enabled=False, dronecrowd_confirmation_access=False), immutable=True)
    return prior.sha(path)


def prepare(site, parent_registration):
    path = prior.PRIVATE / f'{site}_cohort.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta['registration'] != parent_registration or meta['sha256'] != prior.sha(path):
        raise ValueError(f'Input cache identity mismatch for {site}')
    with np.load(path, allow_pickle=False) as f:
        h, y, offsets = f['history'], f['future'], f['offsets']
    valid = np.isfinite(y).all(2)
    x, cv, ols, b, scale = motion.features(h)
    bank = np.concatenate((ols[:, None], b), axis=1)
    dist = np.linalg.norm(bank - cv[:, None], axis=-1)
    # Match the old design for four options; the OLS4 option gets its own bit.
    zero = np.zeros((len(h), 5, 1), dtype=np.float32)
    zero[:, 0] = 1.
    x_ols = x[:, 0].copy()
    x_ols[:, 8:12] = 0.
    x_ols[:, 12] = (dist[:, 0].mean(1) / scale).clip(-16., 16.)
    x_ols[:, 13] = (dist[:, 0, -1] / scale).clip(-16., 16.)
    x_ols[:, 14:] = 0.
    ols_probe = []
    for prefix in (4, 5, 6):
        c, o, _, s = motion.rollout(h[:, :prefix], 8-prefix)
        target = h[:, prefix:] - h[:, prefix-1:prefix]
        ols_probe.append((np.linalg.norm(c-target, axis=-1).mean(1)
                         - np.linalg.norm(o-target, axis=-1).mean(1)) / s)
    x_ols[:, 17] = np.mean(ols_probe, axis=0).clip(-16., 16.)
    x = np.concatenate((np.concatenate((x_ols[:, None], x), axis=1), zero), axis=2)
    hindcast = x[:, :, 17]
    cv_target = motion.targets(cv, bank, y, valid, scale)
    ols_target = motion.targets(ols, bank, y, valid, scale)
    return dict(x=torch.from_numpy(x), cv=cv, ols=ols, bank=bank, offsets=offsets, y=y, valid=valid,
        cv_target=torch.from_numpy(cv_target), ols_target=torch.from_numpy(ols_target),
        dist=dist, legal=(dist > 1e-9).any(2), hindcast=hindcast,
        pool=np.flatnonzero(valid.any(1)), cv_complete_ade=np.linalg.norm(cv-y,axis=-1)[valid.all(1)].mean(1))


def easy_cut(sources, train):
    a = np.concatenate([sources[s]['cv_complete_ade'] for s in train])
    return float(np.quantile(a[a > 0], .25))


def outputs(net, x):
    raw = net(x)
    return torch.stack((raw[..., 0], torch.nn.functional.softplus(raw[..., 1]),
                        raw[..., 2], torch.nn.functional.softplus(raw[..., 3])), dim=-1)


def fit(sources, train, seed, arm, registration):
    d = [sources[s] for s in train]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+1776)
    nn = torch.nn
    net = nn.Sequential(nn.Linear(d[0]['x'].shape[-1],48), nn.SiLU(),
                        nn.Linear(48,32), nn.SiLU(), nn.Linear(32,4))
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    path = PRIVATE / f'{"-".join(train)}_{seed}_{arm}.pt'
    ident = dict(registration=registration, train=list(train), seed=seed, arm=arm)
    step, loss_trace = 0, []
    if path.exists():
        ck = torch.load(path, weights_only=True, map_location='cpu')
        if ck['identity'] != ident or ck['step'] not in (100,200,300):
            raise ValueError(f'Checkpoint identity mismatch for {path.name}')
        step, loss_trace = ck['step'], ck['loss_trace']
        net.load_state_dict(ck['weights'])
        opt.load_state_dict(ck['optimizer'])
        rng.bit_generator.state = ck['rng']
    target2 = 'ols_target' if arm == 'dual' else 'cv_target'
    for i in range(step, prior.STEPS):
        draw = rng.integers(len(d), size=prior.BATCH)
        picks = [(ds, rng.choice(ds['pool'], size=int((draw == j).sum()))) for j, ds in enumerate(d)]
        x = torch.cat([ds['x'][ix] for ds, ix in picks])
        target = torch.cat([torch.cat((ds['cv_target'][ix], ds[target2][ix]), -1) for ds, ix in picks])
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.smooth_l1_loss(outputs(net, x), target, beta=.1)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training objective')
        loss.backward()
        opt.step()
        if (i+1) % 100 == 0:
            loss_trace.append(float(loss.detach()))
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=ident, step=i+1, weights=net.state_dict(),
                optimizer=opt.state_dict(), rng=rng.bit_generator.state, loss_trace=loss_trace), tmp)
            tmp.replace(path)
            beat('checkpoint', train=train, seed=seed, arm=arm, step=i+1, loss=loss_trace[-1])
    return net.eval(), dict(**ident, loss_trace=loss_trace)


def scores(net, d, arm):
    with torch.no_grad():
        x = outputs(net, d['x']).numpy()
    score = x[..., 0] - x[..., 1]
    if arm == 'dual':
        # The first candidate is OLS4 itself, hence its OLS4-relative harm is zero.
        score[:, 1:] -= x[:, 1:, 3]
    return score


def outer_eval(d, score, threshold, cut):
    n = len(d['offsets'])-1
    empty = np.zeros(n, dtype=bool)
    control = {'minimum': -d['dist'].mean(2), 'hindcast': d['hindcast']}
    forced, mask = model.edit(d, score)
    pred, switches = (d['cv'], empty) if threshold is None else model.edit(d, score, threshold=threshold)
    rows = dict(cv=model.metrics(d,d['cv'],empty,cut), ols4=model.metrics(d,d['ols'],empty,cut),
                forced=model.metrics(d,forced,mask,cut), calibrated=model.metrics(d,pred,switches,cut))
    for name, control_score in control.items():
        cf, ok = model.edit(d, control_score)
        cp, same = model.edit(d, control_score, allowed_queries=switches)
        if not np.array_equal(ok, mask) or not np.array_equal(same, switches):
            raise ValueError('Unmatched queries')
        rows[name] = model.metrics(d, cf, ok, cut)
        rows[f'forced_vs_{name}'] = model.metrics(d, forced, mask, cut, ref=cf)
        rows[f'calibrated_vs_{name}'] = model.metrics(d, pred, switches, cut, ref=cp)
    e_cv = np.linalg.norm(d['cv'] - d['y'], axis=-1)
    e_bank = np.linalg.norm(d['bank'] - d['y'][:, None], axis=-1)
    oracle_gain = np.where(d['valid'][:, None], e_cv[:, None]-e_bank, 0.).sum(2)
    oracle, oracle_mask = model.edit(d, oracle_gain, threshold=0.)
    rows['oracle_observed_abstains'] = model.metrics(d, oracle, oracle_mask, cut)
    oracle, oracle_mask = model.edit(d, oracle_gain)
    rows['oracle_observed_forced'] = model.metrics(d, oracle, oracle_mask, cut)
    return rows, (pred, switches), (forced, mask)


def summarize(evidence):
    rng = np.random.default_rng(817)
    index = rng.integers(len(evidence), size=(3000,len(evidence)))
    out = {}
    keys = [(arm, name) for arm in ARMS for name in next(iter(evidence.values()))[0]['arms'][arm]]
    keys += [('paired', 'dual_vs_cv_only_calibrated'), ('paired','dual_vs_cv_only_forced')]
    for arm, name in keys:
        vals = [[seed['arms'][arm][name] if arm != 'paired' else seed[name] for seed in site]
                for site in evidence.values()]
        v = np.asarray([np.mean([r['lower_pp'] for r in rs]) for rs in vals])
        out[f'{arm}/{name}'] = dict(lower_pp=float(v.mean()),
            ci95_pp=np.quantile(v[index].mean(1), [.025,.975]).tolist(),
            positive_sites=int((v>0).sum()),
            worst_site_pp=float(v.min()),
            switches_all_seeds=sum(r['switches'] for rs in vals for r in rs),
            worst_easy_pct=max((r['signed_easy_pct'] for rs in vals for r in rs
                                if r['signed_easy_pct'] is not None), default=None),
            observed_ade=float(np.mean([r['ade'] for rs in vals for r in rs])),
            observed_fde=float(np.mean([r['fde'] for rs in vals for r in rs])))
    return out


def run(registration):
    start = monotonic()
    parent_registration = prior.sha(prior.PUBLIC/'registration.json')
    sites = tuple(json.loads((prior.PUBLIC/'registration.json').read_text())['sites'])
    sources = {site:prepare(site, parent_registration) for site in sites}
    beat('cohort_cached_verified', rows=sum(len(d['y']) for d in sources.values()))
    fits, predictors = [], {}
    # No scoring until every model for the registered matrix is frozen.
    for size in (2,1):
        for excluded in combinations(sites, size):
            train = tuple(s for s in sites if s not in excluded)
            for seed in SEEDS:
                for arm in ARMS:
                    net, meta = fit(sources, train, seed, arm, registration)
                    predictors[excluded, seed, arm] = net
                    fits.append(meta)
    evidence = {}
    thresholds = Counter()
    for outer in sites:
        evidence[outer] = []
        outer_cut = easy_cut(sources, [s for s in sites if s != outer])
        for seed in SEEDS:
            row = dict(seed=seed, calibration={}, arms={})
            calibrated, forced = {}, {}
            for arm in ARMS:
                trials = [dict(threshold=t, sites=[]) for t in THRESHOLDS]
                for inner in (s for s in sites if s != outer):
                    excluded = tuple(sorted((outer,inner)))
                    cut = easy_cut(sources, [s for s in sites if s not in excluded])
                    d = sources[inner]
                    score = scores(predictors[excluded, seed, arm], d, arm)
                    for t in trials:
                        p, edit = model.edit(d, score, threshold=t['threshold'])
                        t['sites'].append(dict(site=inner, **model.metrics(d,p,edit,cut)))
                selected = model.select_threshold(trials)
                thresholds[f'{arm}/'+ ('fallback' if selected is None else str(selected))] += 1
                row['calibration'][arm] = dict(selected=selected, trials=trials)
                d = sources[outer]
                score = scores(predictors[(outer,),seed,arm],d,arm)
                row['arms'][arm], calibrated[arm], forced[arm] = outer_eval(d,score,selected,outer_cut)
            row['dual_vs_cv_only_calibrated'] = model.metrics(d,*calibrated['dual'],outer_cut,ref=calibrated['cv_only'][0])
            row['dual_vs_cv_only_forced'] = model.metrics(d,*forced['dual'],outer_cut,ref=forced['cv_only'][0])
            evidence[outer].append(row)
        beat('held_site_readout', site=outer)
    output = PUBLIC/'analysis.json'
    replay = output.exists()
    prior.write_json(output, dict(registration=registration, fits=fits, sites=evidence, thresholds=dict(thresholds),
        summary=summarize(evidence), runtime=dict(architecture=platform.machine(),torch=str(torch.__version__),
        threads=4,workers=0), data_source='cached_verified', scope='development_diagnostic',
        independent_confirmation=False, deployment_qualified=False), immutable=True)
    beat('complete', elapsed_s=round(monotonic()-start,3), analysis_sha256=prior.sha(output), exact_replay=replay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only',action='store_true')
    args = parser.parse_args()
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Run native arm64 .venv-pytorch/bin/python, not Rosetta')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registration = register()
        if args.register_only:
            print(registration)
        else:
            import numpy as np
            import torch
            from src.evaluation import m3w_opendd_cv_abstention as model
            from src.world_model import m3w_ols_replacement as motion

            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            run(registration)
