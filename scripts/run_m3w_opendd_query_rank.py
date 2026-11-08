"""Train matched pointwise and query-regret rankers on the OpenDD cohort."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_opendd_cv_abstention as parent

PUBLIC = parent.PUBLIC.parent / 'opendd_query_rank_v1'
PRIVATE = parent.PRIVATE.parent / PUBLIC.name
ARMS = ('pointwise', 'rank_cv', 'rank_dual')
STEPS, BATCH = 400, 64
SEEDS = parent.SEEDS
MODES = ('full', 'limited20', 'recipient', 'candidate')


def beat(phase, **extra):
    value = dict(phase=phase, pid=os.getpid(), **extra)
    parent.prior.write_json(PRIVATE/'heartbeat.json', value)
    print(json.dumps(value), flush=True)


def register():
    paths = (Path(__file__).resolve(), PUBLIC/'protocol.md', Path(parent.__file__),
        parent.PUBLIC/'registration.json', parent.PUBLIC/'metrics.json',
        parent.prior.PUBLIC/'registration.json', ROOT/'src/evaluation/m3w_opendd_location.py',
        ROOT/'src/evaluation/m3w_opendd_cv_abstention.py', ROOT/'src/evaluation/m3w_opendd_query_rank.py',
        ROOT/'src/world_model/m3w_ols_replacement.py')
    p = PUBLIC/'registration.json'
    parent.prior.write_json(p, dict(files={str(a.relative_to(ROOT)):parent.prior.sha(a) for a in paths},
        seeds=SEEDS, arms=ARMS, modes=MODES, steps=STEPS, batch=BATCH, checkpoint_every=100,
        data_role='development', official_risk_budget=.02, dronecrowd_confirmation_access=False,
        stage5c_executed=False, smc_enabled=False), immutable=True)
    return parent.prior.sha(p)


def data(site, parent_id):
    d = parent.prepare(site,parent_id)
    lengths = np.diff(d['offsets'])
    max_options = int(lengths.max()) * d['x'].shape[1]
    rows = np.repeat(np.arange(len(lengths)),lengths)
    speed = np.linalg.norm(np.stack((d['cv'][:,0],d['ols'][:,0]),axis=1),axis=-1).max(1)
    scale = np.asarray([max(.05,np.median(speed[a:b])) for a,b in zip(d['offsets'][:-1],d['offsets'][1:],strict=True)])
    y, valid = d['y'], d['valid'][:,None]
    e_bank = np.linalg.norm(d['bank']-y[:,None],axis=-1)
    bounds = []
    for ref in (d['cv'],d['ols']):
        e_ref = np.linalg.norm(ref-y,axis=-1)
        disagreement = np.linalg.norm(d['bank']-ref[:,None],axis=-1)
        bounds.append(np.where(valid,e_ref[:,None]-e_bank,-disagreement).mean(2))
    cv, ols = (v/scale[rows,None] for v in bounds)
    target = np.stack((cv,cv-np.maximum(0.,-ols)),axis=-1).clip(-4.,4.)
    slots = ((np.arange(len(rows))-d['offsets'][rows])[:,None] * d['x'].shape[1]) + np.arange(d['x'].shape[1])
    query_index = rows[:,None]
    padded = np.zeros((len(lengths),max_options,d['x'].shape[-1]),dtype=np.float32)
    padded[query_index,slots] = d['x'].numpy()
    legal = np.zeros(padded.shape[:2],dtype=bool)
    legal[query_index,slots] = d['legal']
    labels = np.zeros((*padded.shape[:2],2),dtype=np.float32)
    labels[query_index,slots] = target
    return {**d, 'query_scale':scale, 'row_scale':scale[rows], 'oracle_cv':bounds[0],
        'batch':tuple(map(torch.from_numpy,(padded,legal,labels))), 'queries':np.flatnonzero(legal.any(1))}


def fit(sources, held, arm, seed, registration):
    train = tuple(s for s in sources if s != held)
    inputs = [sources[s] for s in train]
    max_options = max(t['batch'][0].shape[1] for t in inputs)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+1776)
    nn = torch.nn
    net = nn.Sequential(nn.Linear(inputs[0]['x'].shape[-1],48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,1))
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    path = PRIVATE / f'{held}_{seed}_{arm}.pt'
    ident = dict(registration=registration,held=held,train=list(train),arm=arm,seed=seed)
    step, trace = 0, []
    if path.exists():
        ck = torch.load(path,map_location='cpu',weights_only=True)
        if ck['identity'] != ident or ck['step'] not in (100,200,300,400):
            raise ValueError(f'Checkpoint identity mismatch: {path.name}')
        net.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        rng.bit_generator.state = ck['rng']
        step, trace = ck['step'],ck['trace']
    for i in range(step,STEPS):
        draw = rng.integers(len(train),size=BATCH)
        samples = []
        for j,t in enumerate(inputs):
            at = rng.choice(t['queries'],size=int((draw==j).sum()))
            x,legal,y = (a[at] for a in t['batch'])
            pad = max_options - legal.shape[1]
            if pad:
                x = nn.functional.pad(x,(0,0,0,pad))
                y = nn.functional.pad(y,(0,0,0,pad))
                legal = nn.functional.pad(legal,(0,pad))
            samples.append((x,legal,y))
        x,legal,y = (torch.cat(a) for a in zip(*samples,strict=True))
        score = net(x).squeeze(-1)
        regret = None if arm=='pointwise' else y[..., int(arm=='rank_dual')]
        loss = rank.training_loss(score,legal,y[...,0],regret)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite objective')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i+1)%100 == 0:
            trace.append(float(loss.detach()))
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=ident,step=i+1,trace=trace,model=net.state_dict(),
                optimizer=opt.state_dict(),rng=rng.bit_generator.state),tmp)
            tmp.replace(path)
            beat('checkpoint',held=held,arm=arm,seed=seed,step=i+1,loss=trace[-1])
    return net.eval(), dict(**ident, trace=trace)


def evaluate(d, learned, easy_cut):
    out = {}
    pool = {**learned, 'minimum':-d['dist'].mean(2)/d['row_scale'][:,None],
            'hindcast':d['hindcast'], 'hindsight_oracle':d['oracle_cv']/d['row_scale'][:,None]}
    for mode in MODES:
        legal = rank.restrict(d['legal'],d['hindcast'],d['offsets'], 'full' if mode=='limited20' else mode)
        scored = {**d, 'legal':legal}
        rows, trajectories = {}, {}
        for name,score in pool.items():
            subset = rank.limited_queries(legal,score,d['offsets'],d['query_scale']) if mode=='limited20' else None
            trajectories[name], switches = evaluate_cv.edit(scored,score,allowed_queries=subset)
            rows[name] = evaluate_cv.metrics(scored,trajectories[name],switches,easy_cut)
        if len({r['switches'] for r in rows.values()}) != 1:
            raise ValueError('Fixed edit budgets differ')
        for a,b in (('rank_cv','pointwise'),('rank_dual','pointwise'),('rank_dual','rank_cv'),
                    ('rank_cv','hindcast'),('rank_cv','minimum'),('rank_dual','hindcast'),('rank_dual','minimum')):
            gains = evaluate_cv.metrics(scored,trajectories[a],switches,easy_cut,ref=trajectories[b])
            rows[f'{a}_vs_{b}'] = {k:gains[k] for k in ('lower_pp','observed_pp')}
        if mode=='full':
            for name in ('cv','ols'):
                rows[name] = evaluate_cv.metrics(scored,d[name],np.zeros_like(switches),easy_cut)
        out[mode] = rows
    return out


def summarize(results):
    sample = np.random.default_rng(817).integers(len(results),size=(3000,len(results)))
    out = {}
    for mode in MODES:
        out[mode] = {}
        for key in next(iter(results.values()))[0]['arms'][mode]:
            views = [[row['arms'][mode][key] for row in rows] for rows in results.values()]
            v = np.asarray([np.mean([x['lower_pp'] for x in vv]) for vv in views])
            flat = [r for vv in views for r in vv]
            stat = dict(lower_pp=float(v.mean()),ci95_pp=np.quantile(v[sample].mean(1),[.025,.975]).tolist(),
                        positive_sites=int((v>0).sum()),worst_site_pp=float(v.min()),
                        observed_pp=float(np.mean([r['observed_pp'] for r in flat])))
            if 'switches' in flat[0]:
                stat.update(switches=sum(r['switches'] for r in flat),
                    worst_easy_pct=max(r['signed_easy_pct'] for r in flat if r['signed_easy_pct'] is not None),
                    ade=float(np.mean([r['ade'] for r in flat])),fde=float(np.mean([r['fde'] for r in flat])))
            out[mode][key] = stat
    return out


def run(registration):
    start = monotonic()
    prior_path = parent.prior.PUBLIC/'registration.json'
    prior_id = parent.prior.sha(prior_path)
    sources = {s:data(s,prior_id) for s in json.loads(prior_path.read_text())['sites']}
    beat('cohort_cached_verified', rows=sum(len(s['y']) for s in sources.values()))
    fits, weights = [], {}
    for held in sources:
        for seed in SEEDS:
            for arm in ARMS:
                net, record = fit(sources,held,arm,seed,registration)
                fits.append(record)
                weights[held,seed,arm] = net
    results = {}
    for held,d in sources.items():
        cut = parent.easy_cut(sources,[s for s in sources if s!=held])
        with torch.no_grad():
            results[held] = [dict(seed=seed,arms=evaluate(d,
                {a:weights[held,seed,a](d['x']).squeeze(-1).numpy() for a in ARMS},cut)) for seed in SEEDS]
    p = PUBLIC/'analysis.json'
    replay = p.exists()
    parent.prior.write_json(p, dict(registration=registration,fits=fits,sites=results,
        summary=summarize(results),runtime=dict(architecture=platform.machine(),torch=str(torch.__version__),
        threads=4,workers=0),scope='development',independent_confirmation=False),immutable=True)
    beat('complete',elapsed_s=round(monotonic()-start,3),analysis_sha256=parent.prior.sha(p),exact_replay=replay)


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
            from src.evaluation import m3w_opendd_cv_abstention as evaluate_cv, m3w_opendd_query_rank as rank
            from src.world_model import m3w_ols_replacement as motion

            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            parent.np,parent.torch,parent.motion = np,torch,motion
            run(registration)
