"""Fit donor-only action values, then freeze and score WEPDTOF once."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_wepdtof_external_camera as camera
from scripts import run_m3w_opendd_location as donor
from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_opendd_spline_frames as spline
from src.world_model import m3w_reference_bridge_policy as costs
from src.world_model import m3w_wepdtof_matched_policy as method
import torch

base, np = camera.base,camera.np
OUT = camera.OUT.parent/'wepdtof_external_transfer_v1'
PRIVATE = camera.PRIVATE.parent/OUT.name
SPLINE = OUT.parent/'opendd_matched_spline_v1'
SITES = tuple(f'rdb{i}' for i in range(1,8))
SEEDS,STEPS = donor.SEEDS,800
ARMS = ('cv','ols4','spline',*method.NAMES,'pilot_guarded')


def read(p):
    return json.loads(p.read_text())


def artifact(p):
    return dict(path=str(p.relative_to(base.ROOT)),sha256=base.sha(p))


def checked(record):
    p = base.ROOT/record['path']
    if base.sha(p)!=record['sha256']:
        raise ValueError(f'Artifact changed: {p}')
    return p


def beat(**values):
    v = dict(pid=os.getpid(),**values)
    temp = PRIVATE/'heartbeat.pending'
    temp.write_text(json.dumps(v)+'\n')
    temp.replace(PRIVATE/'heartbeat.json')
    print(v,flush=True)


def register():
    p = SPLINE/'freeze.json'
    reg = base.sha(SPLINE/'registration.json')
    f,r = read(p),read(SPLINE/'readout.json')
    if (not f['before_held_scoring'] or f['registration']!=reg or r['freeze']!=base.sha(p)
            or r['confirmation_access'] or not r['original_2pct']):
        raise ValueError('Require registered donor forecasts')
    cells = {}
    for c in f['cells']:
        site = c['held']
        fit = c['fits']['ols_spline']
        seed = fit['identity']['seed']
        if (set(c['fit'])!=set(SITES)-{site} or fit['identity']['held']!=site or fit['identity']['arm']!='ols_spline'
                or fit['identity']['registration']!=reg or (site,seed) in cells):
            raise ValueError('Donor training exposed its own prediction site')
        checked(c['paths'])
        checked(fit['artifact'])
        cells[site,seed] = c
    if set(cells)!={(s,k) for s in SITES for k in SEEDS}:
        raise ValueError('Missing three-seed donor roster')
    cohorts = {}
    for s in SITES:
        x = donor.PRIVATE/f'{s}_cohort.npz'
        m = read(x.with_suffix('.json'))
        if m['sha256']!=base.sha(x) or m['registration']!=base.sha(donor.PUBLIC/'registration.json'):
            raise ValueError('Original donor roster changed')
        cohorts[s] = artifact(x)
    role = base.OUT/'external_roles.json'
    if read(role)['external_labels_opened_for_scoring'] or not read(role)['roles_frozen']:
        raise ValueError('Require frozen external roles')
    reg_path = OUT/'registration.json'
    base.immutable(reg_path,dict(code=[artifact(x) for x in (
        Path(__file__),Path(method.__file__),Path(costs.__file__),Path(spline.__file__),
        Path(sys.modules[policy.__module__].__file__),OUT/'protocol.md')],
        donor_freeze=artifact(p), donor_cohorts=cohorts, approved_roles=artifact(role),
        seeds=SEEDS, steps=STEPS, batch=128, radius_fraction=method.FRACTION, pair_weight=method.PAIR_WEIGHT,
        only_donor_fitting=True, pilot_cut_quantiles=[.25,.75], no_external_selection=True, original_2pct=True))
    return base.sha(reg_path), cells, cohorts


def donor_examples(cells,cohorts,seed):
    items = []
    cut_rows = []
    for s in SITES:
        with np.load(checked(cohorts[s]),allow_pickle=False) as z:
            h,y,offsets = (z[k] for k in ('history','future','offsets'))
        q = np.repeat(np.arange(len(offsets)-1),np.diff(offsets))
        valid = np.isfinite(y).all(2)
        cv,ols = method.analytic(h)
        with np.load(checked(cells[s,seed]['paths']),allow_pickle=False) as z:
            p = z['ols_spline']
        if p.shape!=cv.shape or not np.isfinite(p).all():
            raise ValueError('Donor prediction rows changed')
        path = method.bank(cv,ols,p,q)
        x,scale,legal,_ = costs.features(h,cv,ols,path)
        e = np.linalg.norm(cv[valid.all(1)]-y[valid.all(1)],axis=-1).mean(1)
        cut_rows.append(e[e>0.])
        items.append((cv,ols,path,y,valid,x,scale,legal))
    cut = float(np.quantile(np.concatenate(cut_rows),.25))
    result = []
    for cv,ols,path,y,v,x,scale,legal in items:
        target = costs.targets(cv,ols,path,np.where(v[...,None],y,0.),v,cut,scale)
        keep = legal.any(1)
        result.append(tuple(torch.from_numpy(a[keep]) for a in (x,target,legal)))
    if any(not len(d[0]) for d in result):
        raise ValueError('Donor site has no eligible action examples')
    return result,cut


def train(reg,cells,cohorts):
    records = []
    for seed in SEEDS:
        items,cut = donor_examples(cells,cohorts,seed)
        dest = PRIVATE/f'value_{seed}.pt'
        torch.manual_seed(seed)
        net = costs.BridgeValue()
        opt = torch.optim.AdamW(net.parameters(),lr=.001)
        rng = np.random.default_rng(seed+871)
        identity = dict(registration=reg,seed=seed,donors=list(SITES),easy_cut=cut)
        start, trace = 0,[]
        if dest.exists():
            z = torch.load(dest,weights_only=True,map_location='cpu')
            if z['identity']!=identity or z['step'] not in range(100,STEPS+1,100):
                raise ValueError('Unexpected donor-head checkpoint')
            net.load_state_dict(z['net'])
            opt.load_state_dict(z['opt'])
            rng.bit_generator.state = z['rng']
            start, trace = z['step'],z['trace']
        for i in range(start+1,STEPS+1):
            n = rng.multinomial(128,np.full(len(items),1./len(items)))
            ix = [rng.integers(len(a[0]),size=k) for a,k in zip(items,n,strict=True)]
            x,y,m = (torch.cat([d[j][t] for d,t in zip(items,ix,strict=True)]) for j in range(3))
            loss = costs.loss(net(x),y,m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
            opt.step()
            if i%100==0:
                trace.append(float(loss.detach()))
                tmp = dest.with_suffix('.pending')
                torch.save(dict(identity=identity,net=net.state_dict(),opt=opt.state_dict(),
                                rng=rng.bit_generator.state,step=i,trace=trace),tmp)
                tmp.replace(dest)
                beat(phase='train',seed=seed,step=i)
        records.append(dict(seed=seed,checkpoint=artifact(dest),identity=identity,trace=trace,
            eligible_rows_by_site=dict(zip(SITES,(len(x[0]) for x in items),strict=True))))
    base.immutable(OUT/'head_freeze.json',dict(registration=reg,models=records,
        configured_updates=STEPS*len(SEEDS), donor_only=True, external_labels_read=False))


def models(reg,cells):
    head = read(OUT/'head_freeze.json')
    if head['registration']!=reg or not head['donor_only'] or head['external_labels_read']:
        raise ValueError('Require frozen donor heads')
    nets = {}
    for r in head['models']:
        seed = r['seed']
        ck = torch.load(checked(r['checkpoint']),map_location='cpu',weights_only=True)
        if ck['identity']!=r['identity'] or ck['step']!=STEPS:
            raise ValueError('Donor head not fully trained')
        net = costs.BridgeValue()
        net.load_state_dict(ck['net'])
        producers = []
        for s in SITES:
            f = cells[s,seed]['fits']['ols_spline']
            z = torch.load(checked(f['artifact']),map_location='cpu',weights_only=True)
            if z['identity']!=f['identity'] or z['step']!=800:
                raise ValueError('Unexpected original spline checkpoint')
            m = spline.Spline()
            m.load_state_dict(z['net'])
            producers.append(m.eval())
        nets[seed] = net.eval(),producers
    if set(nets)!=set(SEEDS):
        raise ValueError('Missing frozen seed')
    return nets


@torch.inference_mode()
def predict(h,q,nets):
    cv,ols = method.analytic(h)
    if not len(h):
        return {s:{k:cv.copy() for k in ARMS[:-1]} for s in SEEDS}
    rel = h-h[:,-1:]
    scale = np.sqrt(np.square(np.diff(h,axis=1)).sum(2).mean(1))
    design,_ = spline.designs(rel/np.maximum(scale,1e-8)[:,None,None])
    x = tuple(torch.from_numpy(v) for v in design['ols_spline'])
    out = {}
    for s,(head,models0) in nets.items():
        p = np.mean([m(*x,False).numpy().astype(float) for m in models0],axis=0)*scale[:,None,None]
        path = method.bank(cv,ols,p,q)
        f,a,legal,_ = costs.features(h,cv,ols,path)
        estimate = head(torch.from_numpy(f)).numpy()*a[:,None,None]
        out[s] = dict(cv=cv,ols4=ols,spline=p,**method.apply(cv,path,method.choose(estimate,legal,q,h,cv,path),q))
    return out


def camera_arrays(windows, pilot=False):
    histories,qs,ids = [],[],[]
    for q,w in enumerate(windows):
        for d in w['paths']:
            if pilot:
                i,_,_ = d
                boxes = np.array([next(r[1:5] for r in f['rows'] if r[0]==i) for f in w['frames']])
                h = .5*(boxes[:,:2]+boxes[:,2:])
            else:
                i,h = d['id'],d['history']
            histories.append(h); qs.append(q); ids.append(i)
    return np.asarray(histories,dtype=float).reshape(-1,8,2),np.array(qs,dtype=int),np.array(ids,dtype=int)


def references(clip):
    s = read(base.OUT/'support.json')
    row, = [x for x in s['clips'] if x['clip']==clip]
    p = base.LOCAL/f'{clip}.json'
    if base.sha(p)!=row['annotation_sha256']:
        raise ValueError('External annotation bytes changed')
    d = read(p)
    frames = {base.data.frame_number(i['file_name'],clip):i['file_name'] for i in d['images']}
    return base.data.annotation_table(d,clip,frames,geometry=True)[0]


def outcomes(windows,h,q,ids,people):
    y = np.zeros((len(h),12,2))
    v = np.zeros((len(h),12),dtype=bool)
    matched = np.zeros(len(h),dtype=bool)
    refs = 0
    for k,w in enumerate(windows):
        maps,sets = [],[]
        for f in w['frames']:
            t = f['frame']
            a = {j:z[t] for j,z in people.items() if t in z}
            maps.append(base.associate(np.asarray(f['rows']).reshape(-1,7),a))
            sets.append(set(a))
        refs += len(set.intersection(*sets))
        match = dict(set.intersection(*maps))
        for r in np.flatnonzero(q==k):
            j = match.get(ids[r])
            if j is None:
                continue
            matched[r] = True
            for t in range(12):
                x = people[j].get(w['start']+(8+t)*base.STRIDE)
                if x is not None:
                    y[r,t] = x[:2]-h[r,-1]
                    v[r,t] = True
    return y,v,matched,refs


def metrics(pred,y,v,q,name,cut,paired_controls=True):
    cv = pred['cv']
    if not len(v):
        return dict(online=0,reference_mass=0.,risk_passes=0,arms={},comparisons={})
    _,inv,count = np.unique(q,return_inverse=True,return_counts=True)
    w = 1./count[inv]
    mass = float(w @ np.where(v,np.linalg.norm(cv-y,axis=-1),0.).sum(1))
    def effect(a,b):
        g,l = paired(b,a,y,v)
        return (100.*np.array([w@g.sum(1),w@l.sum(1)])/mass).tolist() if mass>0. else [None,None]
    rows = {}
    for k,p in pred.items():
        d = policy(p,cv,cv,y,v,np.full(len(p),name),easy_cut=cut[0],hard_cut=cut[1])
        risk = d['recordings'][name]
        z = [effect(p,pred[b]) for b in ('cv','ols4')]
        rows[k] = dict(lower_vs_cv=z[0][1],lower_vs_ols4=z[1][1],known_vs_cv=z[0][0],known_vs_ols4=z[1][0],
            ade_px=d['known_ADE_pixel'],fde_px=d['available_endpoint_FDE_diagnostic_pixel'],
            p95_ade_px=d['p95_known_ADE_pixel'],subsets=d['subsets'],risk=risk,edit_rate=d['switch_rate'],
            dual_pass=bool(risk['supported'] and mass>0. and min(z[0][1],z[1][1])>0.))
    direct = {k:effect(pred['joint'],pred[k]) for k in ('independent','uniform','no_pair')} if paired_controls else {}
    consistency = {k:[] for k in pred}
    for j in np.unique(q):
        at = np.flatnonzero(q==j)
        ii,jj = np.triu_indices(len(at),1)
        i,j0 = at[ii],at[jj]
        valid = v[i]&v[j0]
        if not valid.any():
            continue
        for k,p in pred.items():
            err = np.linalg.norm((p[i]-p[j0])-(y[i]-y[j0]),axis=-1)
            consistency[k].append(float(err[valid].mean()))
    return dict(online=len(v),reference_mass=mass,arms=rows,comparisons=direct,
        pair_relative_error_px={k:float(np.mean(z)) if z else None for k,z in consistency.items()})


def pilot(reg,nets):
    d = read(base.OUT/'rapid_freeze.json')
    if d['annotation_geometry_read'] or d['registration']!=base.sha(camera.detector.REG):
        raise ValueError('Require previously frozen pilot')
    windows = []
    for r in d['windows']:
        p = camera.detector.PRIVATE/f"window_{r['start']:06}.json"
        if base.sha(p)!=r['cache_sha256']:
            raise ValueError('Pilot camera freeze changed')
        windows.append(read(p))
    h,q,ids = camera_arrays(windows,pilot=True)
    pred = predict(h,q,nets)
    y,v,m,_ = outcomes(windows,h,q,ids,references(base.PILOT))
    cv = pred[SEEDS[0]]['cv']
    complete = v.all(1)&m
    cv_ade = np.linalg.norm(cv[complete]-y[complete],axis=-1).mean(1)
    cut = np.quantile(cv_ade[cv_ade>0.],[.25,.75]).tolist()
    values = {str(s):metrics(p,y,v,q,base.PILOT,cut) for s,p in pred.items()}
    enable = all(d['arms']['joint']['dual_pass'] for d in values.values())
    base.immutable(OUT/'pilot_freeze.json',dict(registration=reg, head_sha256=base.sha(OUT/'head_freeze.json'),
        pilot_camera_sha256=base.sha(base.OUT/'rapid_freeze.json'), cut_px=cut, matched=int(m.sum()),
        full=int(complete.sum()), online=len(h), values=values, guarded_uses_joint=enable,
        external_geometry_read=False, original_2pct=True))
    print(dict(cut_px=cut,matched=int(m.sum()),online=len(h),guarded_uses_joint=enable,
               joint_edits=[v['arms']['joint']['risk']['selected'] for v in values.values()]),flush=True)


def frozen_camera():
    key,jobs = camera.register()
    p = camera.OUT/'camera_freeze.json'
    f = read(p)
    if (f['registration']!=key or f['annotations_read'] or f['labels_scored']
            or f['source_groups']!=13 or len(f['records'])!=len(jobs)):
        raise ValueError('External camera freeze incomplete')
    windows = defaultdict(list)
    for row,(g,c,w) in zip(f['records'],jobs,strict=True):
        x = camera.checkpoint_path(c,w['start'])
        if base.sha(x)!=row['cache_sha256'] or row['source']!=g:
            raise ValueError('External observation cache changed')
        item = read(x)
        camera.check_row(item,key,g,c,w)
        windows[c].append(item)
    return p,windows


def freeze(reg,nets):
    f,windows = frozen_camera()
    c = OUT/'pilot_freeze.json'
    p = read(c)
    if p['registration']!=reg or p['external_geometry_read']:
        raise ValueError('Need pilot-only decision')
    rows = []
    for clip,w in windows.items():
        h,q,ids = camera_arrays(w)
        pred = predict(h,q,nets)
        a = dict(q=q,ids=ids)
        for seed,paths in pred.items():
            for k,v in paths.items():
                a[f'{seed}_{k}'] = v
            a[f'{seed}_pilot_guarded'] = paths['joint' if p['guarded_uses_joint'] else 'cv']
        path = PRIVATE/f'{clip}_forecasts.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                if set(a)!=set(z.files):
                    raise ValueError('Frozen forecast arms changed')
                for k,v in a.items():
                    np.testing.assert_array_equal(z[k],v)
        else:
            tmp = path.with_suffix('.pending.npz')
            np.savez(tmp,**a)
            tmp.replace(path)
        rows.append(dict(clip=clip,source=w[0]['source'],online=len(h),artifact=artifact(path)))
    base.immutable(OUT/'forecast_freeze.json',dict(registration=reg,pilot=artifact(c),camera=artifact(f),
        rows=rows,source_groups=len({r['source'] for r in rows}),seeds=SEEDS,arms=ARMS,
        no_external_outcomes_read=True, original_2pct=True))


def summary(rows):
    groups = sorted({r['source'] for r in rows})
    if len(groups)!=13:
        raise ValueError('Incomplete external source roster')
    def interval(get):
        means = []
        for s in groups:
            v = [get(z) for r in rows if r['source']==s for z in r['seeds'].values()]
            if any(x is None for x in v):
                continue
            means.append(np.mean(v))
        a = np.array(means)
        if not len(a):
            return dict(point=None,interval95=None,estimable_sources=0,total_sources=len(groups))
        r = np.random.default_rng(82317).choice(a,size=(3000,len(a)),replace=True).mean(1)
        return dict(point=float(a.mean()),interval95=np.quantile(r,[.025,.975]).tolist(),
                    estimable_sources=len(a),total_sources=len(groups))
    flat = [z for r in rows for z in r['seeds'].values()]
    fields = ('lower_vs_cv','lower_vs_ols4','ade_px','fde_px','p95_ade_px','edit_rate')
    arms = {k:{**{f:interval(lambda r:r['arms'].get(k,{}).get(f)) for f in fields},
                'risk_passes':sum(bool(r['arms'].get(k,{}).get('risk',{}).get('supported')) for r in flat),
                'dual_passes':sum(bool(r['arms'].get(k,{}).get('dual_pass')) for r in flat),
                'recording_seed_views':len(flat),
                'undefined_easy':sum(r['arms'].get(k,{}).get('risk',{}).get('completion_easy_risk_upper') is None for r in flat)}
            for k in ARMS}
    contrasts = {k:interval(lambda r:r['comparisons'].get(k,[None,None])[1])
                 for k in ('independent','uniform','no_pair')}
    return dict(arms=arms,contrasts=contrasts,bootstrap_draws=3000,bootstrap_unit='original_source_after_seed_average',
        online=sum(r['online'] for r in rows),matched=sum(r['matched'] for r in rows),full=sum(r['full'] for r in rows),
        missing=sum(r['missing'] for r in rows),reference_histories=sum(r['reference_histories'] for r in rows))


def score(reg):
    p = OUT/'forecast_freeze.json'
    f = read(p)
    if (not f['no_external_outcomes_read'] or f['registration']!=reg or f['source_groups']!=13
            or f['arms']!=list(ARMS) or f['seeds']!=list(SEEDS)):
        raise ValueError('No complete once-only forecast freeze')
    cut = read(checked(f['pilot']))['cut_px']
    cam,windows = frozen_camera()
    if checked(f['camera'])!=cam or set(windows)!={r['clip'] for r in f['rows']}:
        raise ValueError('Camera freeze or roster differs')
    key = base.sha(p)
    rows = []
    for r in f['rows']:
        clip = r['clip']
        w = windows[clip]
        dest = PRIVATE/f'{clip}_score.json'
        if not dest.exists():
            h,q,ids = camera_arrays(w)
            with np.load(checked(r['artifact']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['q'],q);np.testing.assert_array_equal(z['ids'],ids)
                forecast = {s:{k:z[f'{s}_{k}'] for k in ARMS} for s in SEEDS}
            y,v,m,ref = outcomes(w,h,q,ids,references(clip))
            values = {str(s):metrics(a,y,v,q,clip,cut) for s,a in forecast.items()}
            base.immutable(dest,dict(freeze=key,clip=clip,source=r['source'],online=len(h),
                matched=int(m.sum()),full=int(v.all(1).sum()),missing=int((~v.any(1)).sum()),
                reference_histories=ref,seeds=values))
        z = read(dest)
        if z['freeze']!=key or z['clip']!=clip:
            raise ValueError('Do not reopen outcomes against a changed freeze')
        rows.append(z)
        beat(phase='scored',clip=clip,done=len(rows),total=len(f['rows']))
    a = summary(rows)
    base.immutable(OUT/'readout.json',dict(forecast_freeze=key,rows=rows,summary=a, original_2pct=True,
        external_scores_opened=True, external_model_selection=False, dronecrowd_confirmation_access=False,
        deployment_changed=False,submission_ready=False))
    print({k:a[k] for k in ('online','matched','full','missing','contrasts')},flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('train','pilot','freeze','score'))
    a = p.parse_args()
    for d in (OUT,PRIVATE):
        d.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        key,c,cohorts = register()
        if a.phase=='train':
            train(key,c,cohorts)
        elif a.phase=='score':
            score(key)
        else:
            n = models(key,c)
            if a.phase=='pilot':
                pilot(key,n)
            else:
                freeze(key,n)


if __name__=='__main__':
    main()
