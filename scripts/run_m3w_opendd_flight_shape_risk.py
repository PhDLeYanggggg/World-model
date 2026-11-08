"""Train equal-radius paths with complete-recording risk objectives."""
import argparse
from collections import Counter
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
from scripts import run_m3w_opendd_radius_shape as shape_run
from scripts import audit_m3w_opendd_radius_risk as audit

base, prior = shape_run.base, shape_run.prior
PUBLIC = shape_run.PUBLIC.parent/'opendd_flight_shape_risk_v1'
PRIVATE = shape_run.PRIVATE.parent/PUBLIC.name
SITES, SEEDS, STEPS = shape_run.SITES, shape_run.SEEDS, shape_run.STEPS


def register():
    old, donors = shape_run.register()
    paths = [Path(__file__).resolve(), Path(audit.__file__), PUBLIC/'protocol.md',
             ROOT/'src/world_model/m3w_flight_shape_risk.py']
    paths += [shape_run.PUBLIC/k for k in ('registration.json','analysis.json','risk_attribution.json',
                                         'checkpoint_freeze.json','replay_execution.json')]
    p = PUBLIC/'registration.json'
    prior.write_json(p, dict(previous_registration=old, original_risk_budget=.02, weight=8,
        steps=STEPS, seeds=SEEDS, sites=SITES, external_access=False, dronecrowd_confirmation_access=False,
        source='opendd_exposed_development_only', files={str(f.relative_to(ROOT)):prior.sha(f) for f in paths}),
        immutable=True)
    return prior.sha(p), donors


def flight_batches(d, c, cut, torch):
    rec = d['recordings']
    known = d['valid'].any(1)
    easy = known & (d['cv_ade']>0) & (d['cv_ade']<=cut)
    scale = d['scale'][:,None,None]
    inc_local = ((c['inc']-d['cv'])@d['rot'])/scale
    target_local = ((d['y']-d['cv'])@d['rot'])/scale

    def t(a):
        return torch.from_numpy(np.asarray(a, dtype=np.float64))

    batches = []
    for name in sorted(set(rec)):
        all_rows = rec==name
        indices = np.flatnonzero(all_rows & (c['r']>0))
        if not len(indices):
            continue
        e = easy[indices]
        all_mass = float(d['cv_ade'][indices].sum())
        easy_mass = float(d['cv_ade'][indices][e].sum())
        full_easy_mass = float(d['cv_ade'][all_rows & easy].sum())
        batches.append(dict(x=c['x'][indices], unit=c['unit'][indices], ratio=c['ratio'][indices,None,None],
            y=d['residual_target'][indices], valid=d['local'][3][indices], easy=torch.from_numpy(e),
            scale=t(d['scale'][indices]), cv_ade=t(d['cv_ade'][indices]), radius=t(c['r'][indices]),
            inc_exact=t(inc_local[indices]), target_exact=t(target_local[indices]),
            mass=(all_mass,easy_mass,full_easy_mass)))
    return batches


def fit(held, seed, arm, batches, reg, modules, expected):
    torch, model, shape, point, risk = modules
    train = [s for s in SITES if s != held]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+1776)
    net = model.network(138)
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    ident = dict(registration=reg, held=held, seed=seed, arm=arm, train=train)
    path = PRIVATE/f'{held}_{seed}_{arm}.pt'
    if expected is not None and prior.sha(path) != expected:
        raise ValueError('Replay refuses a changed checkpoint')
    step, trace = 0, []
    if path.exists():
        ck = torch.load(path,map_location='cpu',weights_only=True)
        if ck['identity']!=ident or ck['step'] not in range(100,STEPS+1,100):
            raise ValueError('Checkpoint identity mismatch')
        net.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        rng.bit_generator.state = ck['rng']
        step, trace = ck['step'], ck['trace']
    if expected is not None and step!=STEPS:
        raise ValueError('Replay cannot train')
    for i in range(step,STEPS):
        batch = batches[train[int(rng.integers(len(train)))]]
        b = batch[int(rng.integers(len(batch)))]
        unit = shape.shape(net(b['x']),b['unit'])
        pred = b['ratio']*unit
        incumbent = b['ratio']*b['unit']
        loss = point.objective(pred,torch.zeros_like(pred),incumbent,b['y'],b['valid'],b['easy'],'easy_guard')
        if arm!='flight_point':
            unit = unit.double()
            max_norm = torch.linalg.vector_norm(unit,dim=-1).amax(1)
            exact = unit*(b['radius']/(b['scale']*max_norm))[:,None,None]
            new, old, signed, debit = risk.quantities(exact,b['inc_exact'],b['target_exact'],
                b['valid'],b['cv_ade'],b['scale'],b['radius'])
            loss = loss+8.*risk.penalty(new,old,signed,debit,b['easy'],*b['mass'],arm)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite recording objective: {ident}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i+1)%100==0:
            trace.append(float(loss.detach()))
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=ident,model=net.state_dict(),optimizer=opt.state_dict(),
                step=i+1,trace=trace,rng=rng.bit_generator.state),tmp)
            tmp.replace(path)
            prior.write_json(PRIVATE/'heartbeat.json',dict(**ident,pid=os.getpid(),step=i+1,loss=trace[-1]))
    return net.eval(), dict(identity=ident,trace=trace,sha256=prior.sha(path)), STEPS-step


def previous_model(site, seed, torch, model, receipt):
    path = shape_run.PRIVATE/f'{site}_{seed}_easy_guard.pt'
    if prior.sha(path)!=receipt['fits'][f'{site}/{seed}/easy_guard']['sha256']:
        raise ValueError('Previous pointwise checkpoint changed')
    net = model.network(138)
    ck = torch.load(path,map_location='cpu',weights_only=True)
    net.load_state_dict(ck['model'])
    return net.eval()


def paths_for(d, c, nets, torch, method):
    cv, r = d['cv'], c['r']
    active = r>0
    paths = dict(cv=cv,incumbent=c['inc'],hindcast=cv+method.at_radius(c['old']-cv,r))
    for arm,net in nets.items():
        with torch.inference_mode():
            sh = method.shape(net(c['x'][active]),c['unit'][active])
        world = np.zeros_like(cv)
        world[active] = sh.numpy().astype(float)@d['rot'][active].transpose(0,2,1)
        paths[arm] = cv+method.at_radius(world,r)
    for name,p in paths.items():
        if name!='cv':
            np.testing.assert_allclose(np.linalg.norm(p-cv,axis=-1).max(1),r,rtol=1e-7,atol=1e-14)
    return paths


def intervals(rows, arms):
    draw = np.random.default_rng(817).integers(len(SITES),size=(3000,len(SITES)))
    result = {}
    for arm in arms:
        site = [np.mean([v['vs_flight_point'][arm]['lower_pp'] for v in rows[s]]) for s in SITES]
        vals = np.asarray(site)
        result[arm] = dict(point=float(vals.mean()),ci95=np.quantile(vals[draw].mean(1),[.025,.975]).tolist())
    return result


def run(reg, donors, replay):
    import torch
    from src.world_model import m3w_opendd_velocity_candidate as model
    from src.world_model import m3w_radius_shape as shape
    from src.world_model import m3w_hindcast_residual as point
    from src.world_model import m3w_flight_shape_risk as risk

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    t0 = monotonic()
    p, cp = PUBLIC/'analysis.json', PUBLIC/'checkpoint_freeze.json'
    expected = json.loads(cp.read_text()) if replay else None
    if replay and (expected['registration']!=reg or not p.is_file()):
        raise ValueError('Replay requires a completed matching readout')
    sources = {s:shape_run.site_data(s,torch,model,shape) for s in SITES}
    outer, inner = shape_run.load_producers(torch,model,donors)
    previous = json.loads((shape_run.PUBLIC/'checkpoint_freeze.json').read_text())
    old_results = json.loads((shape_run.PUBLIC/'analysis.json').read_text())['by_location']
    cuts = {s:base.cuts_for([sources[k] for k in SITES if k!=s]) for s in SITES}
    nets, metadata, updates = {}, {}, 0
    modules = torch,model,shape,point,risk
    for held in SITES:
        print(f'{"Verifying" if replay else "Fitting"} {held}',flush=True)
        for seed in SEEDS:
            batches = {s:flight_batches(sources[s],shape_run.candidate(sources[s],
                inner[tuple(sorted((s,held))),seed],torch,model),cuts[held][0],torch)
                for s in SITES if s!=held}
            for arm in risk.ARMS:
                k = f'{held}/{seed}/{arm}'
                net, meta, n = fit(held,seed,arm,batches,reg,modules,
                                    None if expected is None else expected['fits'][k]['sha256'])
                nets[held,seed,arm], metadata[k] = net,meta
                updates += n
    prior.write_json(cp,dict(registration=reg,fits=metadata),immutable=True)
    rows = {}
    for i,s in enumerate(SITES):
        d = sources[s]
        rows[s] = []
        for seed in SEEDS:
            c = shape_run.candidate(d,outer[seed][i],torch,model)
            prev = previous_model(s,seed,torch,model,previous)
            paths = paths_for(d,c,dict(previous_point=prev,**{a:nets[s,seed,a] for a in risk.ARMS}),torch,shape)
            active = c['r']>0
            arms = base.readout(d,paths,active,cuts[s])
            old = next(v['arms'] for v in old_results[s] if v['seed']==seed)
            if arms['previous_point']!=old['easy_guard'] or arms['incumbent']!=old['incumbent']:
                raise ValueError('Previous comparator does not reproduce its frozen readout')
            transitions = {}
            for a in risk.ARMS:
                q, _, _ = audit.compare(d,paths['incumbent'],paths[a],cuts[s])
                transitions[a] = Counter(r['transition'] for r in q.values())
            rows[s].append(dict(seed=seed,arms=arms,transitions=transitions,pairs=shape_run.pair_errors(d,paths),
                vs_flight_point={a:base.paired_summary(d,paths[a],paths['flight_point'],active)
                                 for a in risk.ARMS if a!='flight_point'}))
    totals = {}
    for a in risk.ARMS:
        counter = Counter()
        for views in rows.values():
            for v in views:
                counter.update(v['transitions'][a])
        totals[a] = dict(counter)
    summary = base.summary(rows)
    primary = summary['flight_relative']
    prior.write_json(p,dict(registration=reg,fits=len(nets),summary=summary,by_location=rows,
        original_unchanged=True,exact_radius_matching=True,paired_vs_flight_point=intervals(rows,risk.ARMS[1:]),
        transitions=totals,primary_recovery=(primary['comparisons']['incumbent']['lower_pp']['ci95'][0]>0
                                             and primary['passed']>=summary['incumbent']['passed']),
        formal_confirmation=False,deployment=False,stage5c_executed=False,smc_enabled=False),immutable=True)
    receipt = PUBLIC/('replay_execution.json' if replay else 'execution.json')
    if not receipt.exists():
        prior.write_json(receipt,dict(registration=reg,analysis_sha256=prior.sha(p),updates_this_execution=updates,
            fresh_exact_replay=replay,pid=os.getpid(),torch_threads=4,workers=0,
            wall_seconds=round(monotonic()-t0,3)),immutable=True)
    prior.write_json(PRIVATE/'heartbeat.json',dict(phase='complete',pid=os.getpid(),replay=replay,updates=updates))
    print(prior.sha(p),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','run','replay'))
    phase = parser.parse_args().phase
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reg, donors = register()
        if phase=='register':
            print(reg)
        else:
            if platform.system()!='Darwin' or platform.machine()!='arm64':
                raise RuntimeError('Use native arm64 .venv-pytorch on this host')
            run(reg,donors,phase=='replay')
