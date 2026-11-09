"""Screen a matched risk-aware causal initial forecast on development sources."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_causal_velocity_spline as previous
from scripts import train_m3w_cv_matched_exchange as exchange
from src.evaluation import m3w_single_exchange_envelope as envelope
from src.world_model import m3w_dual_record_forecast as model

old, base, prior, np, torch = previous.old,previous.base,previous.prior,previous.np,previous.torch
OUT = previous.OUT.parent/'moving_dual_record_forecast_v1'
PRIVATE = previous.PRIVATE.parent/OUT.name
MODES = ('ols_only','dual_risk')
FAMILIES = ('ols4','ols8','completion',*MODES)
ALPHAS = (.02,.10,.35,.70,1.)
STEPS, BATCH = 800,256
CONTROLS = ('ols8','completion','eqmotion','pct','half_anchor')
CHOICES = ('strict','diagnostic','ols_only_screen','dual_risk_screen')


def register():
    refs = [p/k for p in (previous.OUT,exchange.OUT) for k in ('freeze.json','readout.json')]
    f,r,e,s = map(old.read,refs)
    if any(v['freeze']!=old.sha(p) or not v['original_2pct'] or v['a_labels_opened'] or v['independent_confirmation']
           for v,p in ((r,refs[0]),(s,refs[2]))) or not all(z['before_held_scoring'] for z in (f,e)):
        raise ValueError('Require exposed development sources only')
    if [c['identity'] for c in f['cells']] != [c['identity'] for c in e['cells']]:
        raise ValueError('Frozen anchors have different source roles')
    _,donor,_ = base.register()
    p = OUT/'registration.json'
    old.once(p,old.stable(dict(parents=[old.artifact(z) for z in (*refs,base.OUT/'registration.json')],
        files=[old.artifact(z) for z in (Path(__file__),Path(model.__file__),OUT/'protocol.md')],
        steps=STEPS,batch=BATCH,seeds=base.SEEDS,threads=4,workers=0,alphas=ALPHAS,original_2pct=True,
        a_labels_opened=False,confirmation_access=False,select_only_on_calibration=True)))
    return old.sha(p),donor,f['cells'],e['cells'],s


def batcher(d, inputs, cv, ols):
    y = d['target'].astype(np.float32)
    v,w = d['valid'],old.weight(d['query']).astype(np.float32)
    banks = [[np.flatnonzero((d['sites']==s)&(d['recording']==r)) for r in np.unique(d['recording'][d['sites']==s])]
             for s in sorted(set(d['sites']))]
    if len(banks)!=3:
        raise ValueError('Require the three registered fitting sources')
    def draw(rng):
        rr = banks[rng.integers(len(banks))]
        ix = rng.choice(rr[rng.integers(len(rr))],size=BATCH,replace=True)
        return tuple(torch.from_numpy(z[ix]) for z in (*inputs,cv,ols,y,v,w))
    return draw


@torch.inference_mode()
def predict(net, inputs, cv):
    c = cv.astype(np.float32)
    changes = [net(*(torch.from_numpy(a[j:j+2048]) for a in (*inputs,c))).numpy()-c[j:j+2048]
               for j in range(0,len(cv),2048)]
    p = cv+np.concatenate(changes)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite causal forecast')
    return p


def grid(cv, family):
    return {f'{k}_{j}':cv+a*(family[k]-cv) for k in FAMILIES for j,a in enumerate(ALPHAS)}


def calibration(d, cv, ols, candidates, cut, identity):
    names = tuple(candidates)
    parent = previous.parent
    rows = parent.rows_for(d,{base.CV:cv,base.OLS:ols,**candidates},cv,cut,parent.vs_references(names),identity)
    groups = [[r for r in rows if r['source']==s] for s in sorted(set(d['sites']))]
    merge = prior.prior.parent.readout.merge_comparisons
    def lower(x): return (x['known_delta']-x['unknown_radius'])
    table = {}
    for name in names:
        k = [f'{name}_vs_{z}' for z in ('cv','ols4')]
        gain = [[merge([r['comparisons'][q] for r in g])['completion_lower_pp'] for q in k] for g in groups]
        risk = [r['metrics'][name]['risk']['supported'] for r in rows]
        dual = [u and all(lower(r['comparisons'][q])>0. for q in k) for u,r in zip(risk,rows,strict=True)]
        table[name] = dict(risk=int(sum(risk)),dual=int(sum(dual)),macro_lower_pp=np.mean(gain,axis=0).tolist(),views=len(rows))
    def key(k):
        z = table[k]
        return z['dual'],min(z['macro_lower_pp'])
    strict = [k for k,z in table.items() if z['risk']==len(rows) and min(z['macro_lower_pp'])>0.]
    chosen = dict(strict=max(strict,key=key) if strict else None,diagnostic=max(names,key=key),
        **{f'{m}_screen':max((k for k in names if k.startswith(m+'_')),key=key) for m in MODES})
    return table,chosen


def cell(data, inputs, linear, ref, anchor, donor, key, replay, beat):
    i,cut = ref['identity'],ref['cut']
    roles = [set(i[k]) for k in ('train','calibration','held')]
    if any(len(s)!=3 for s in roles) or len(set.union(*roles))!=9:
        raise ValueError('Source roles overlap')
    (tr,t),(ca,c),(he,h) = [base.subset(data,i[k]) for k in ('train','calibration','held')]
    seed,fold = i['seed'],i['fold']
    xtr,xc,xh = ([z[v] for z in inputs] for v in (tr,ca,he))
    cv,ols,ols8 = linear.transpose(1,0,2,3)
    draw = batcher(t,xtr,cv[tr].astype(np.float32),ols[tr].astype(np.float32))
    cal = dict(ols4=ols[ca],ols8=ols8[ca])
    held = dict(ols4=ols[he],ols8=ols8[he])
    fits,updates,last_rng = {},0,None
    for arm in MODES:
        def cost(net, history, peers, mask, frame, scale, velocity, cv0, ols0, y, v, weight):
            p = net(history,peers,mask,frame,scale,velocity,cv0)
            return model.objective(p,cv0,ols0,y,v,weight,cut[0],arm=='dual_risk')
        ident = dict(i,registration=key,arm=arm)
        net,fit,n = prior.prefix.trainer.fit(ident,f'{arm}_{fold}_{seed}',seed,model.Forecast,STEPS,draw,cost,replay,beat)
        fits[arm] = fit
        rng = torch.load(old.checked(fit['artifact']),weights_only=True,map_location='cpu')['rng']
        if last_rng is not None and last_rng != rng:
            raise ValueError('Training RNG changed between matched models')
        last_rng = rng
        updates += n
        cal[arm],held[arm] = predict(net,xc,cv[ca]),predict(net,xh,cv[he])
    meta = ref['fits']['completion']
    z = torch.load(old.checked(meta['artifact']),weights_only=True,map_location='cpu')
    if (z['identity']!=meta['identity'] or z['identity']['train']!=i['train'] or z['identity']['held']!=i['held']
            or z['identity']['calibration']!=i['calibration'] or z['step']!=previous.STEPS):
        raise ValueError('Completion spline crossed a fold')
    net = previous.model.Spline()
    net.load_state_dict(z['net'])
    cal['completion'] = previous.predict(net.eval(),xc,'completion')
    with np.load(old.checked(ref['predictions']),allow_pickle=False) as a:
        np.testing.assert_array_equal(a['ids'],h['ids'])
        held['completion'] = a['completion']
    table,sel = calibration(c,cv[ca],ols[ca],grid(cv[ca],cal),cut,i)
    chosen = grid(cv[he],held)
    raw = {k:v[he] for k,v in prior.read_donor(donor,seed,data).items()}
    mc,mo,bank = prior.bank(h['history'],raw)
    np.testing.assert_array_equal(mc,cv[he]); np.testing.assert_array_equal(mo,ols[he])
    with np.load(old.checked(anchor['choices']),allow_pickle=False) as a:
        np.testing.assert_array_equal(a['ids'],h['ids'])
        half = mc+.5*(exchange.old.slate.path_for(mc,base.expanded(mc,bank),a['anchor'])-mc)
    p = dict(ids=h['ids'],**held,**raw,half_anchor=half,
             **{k:chosen[v] if v is not None else cv[he] for k,v in sel.items()})
    path = prior.study.parent.parent.save_actions(PRIVATE/f'paths_{fold}_{seed}.npz',p)
    return dict(identity=i,cut=cut,fits=fits,paths=path,calibration=table,selected=sel),updates


def evaluate(data, linear, cells, ref):
    rows,abs_by,upper = [],{},[]
    arms = (*MODES,*CONTROLS,*CHOICES)
    pairs = previous.parent.vs_references(arms)
    pairs.update(dual_vs_ols_only=('ols_only','dual_risk'),screen_vs_half=('half_anchor','diagnostic'),
                 screen_risk_vs_ols_only=('ols_only_screen','dual_risk_screen'))
    cv,ols,_ = linear.transpose(1,0,2,3)
    for c in cells:
        i,cut = c['identity'],c['cut']
        ix,h = base.subset(data,i['held'])
        with np.load(old.checked(c['paths']),allow_pickle=False) as z:
            np.testing.assert_array_equal(h['ids'],z['ids'])
            paths = {base.CV:cv[ix],base.OLS:ols[ix],**{k:z[k] for k in arms}}
        rows.extend(previous.parent.rows_for(h,paths,cv[ix],cut,pairs,i))
        for s in np.unique(h['sites']):
            at = h['sites']==s
            sub = {k:h[k][at] for k in ('target','valid','query')}
            abs_by[str(s),i['seed']] = {k:previous.absolute_masses(sub,p[at]) for k,p in paths.items()}
        for r in np.unique(h['recording']):
            on = h['recording']==r
            q,y,v = (h[k][on] for k in ('query','target','valid'))
            z,o = cv[ix][on],ols[ix][on]
            w = old.weight(q)
            mass = float(w@np.where(v,np.linalg.norm(z-y,axis=-1),0.).sum(1))
            terms = {}
            for k in ('diagnostic','half_anchor'):
                _,_,u = envelope.bound(z,np.stack((z,o)),paths[k][on],y,v,prior.query_groups(q),w)
                terms[k] = float(100.*u[1]/mass)
            upper.append(dict(recording=str(r),seed=i['seed'],ols4_upper=terms))
    roster = {r['recording']:r['source'] for r in rows}
    summary = prior.prefix.grouped.grouped_readout(rows,roster,base.SEEDS)
    dual = {}
    for k in arms:
        dual[k] = sum(r['metrics'][k]['risk']['supported'] and all(
            r['comparisons'][f'{k}_vs_{b}']['known_delta']-r['comparisons'][f'{k}_vs_{b}']['unknown_radius']>0.
            for b in ('cv','ols4')) for r in rows)
    if (len(rows)!=72 or len(roster)!=24 or summary['risk']['half_anchor']!=ref['summary']['risk']['half_anchor']
            or any(not np.isfinite(r['ols4_upper']['diagnostic']) for r in upper)):
        raise ValueError('Held cohort or half-peak control changed')
    result = {}
    sites = sorted(set(roster.values()))
    for k in (*arms,base.CV,base.OLS):
        a = np.asarray([abs_by[s,j][k] for s in sites for j in base.SEEDS])
        result[k] = dict(available_ade_pixel=float((a[:,0]/a[:,1]).mean()),endpoint_fde_pixel=float((a[:,2]/a[:,3]).mean()))
    bounds = {k:dict(excluded=sum(r['ols4_upper'][k]<=0. for r in upper),
                    median_pp=float(np.median([r['ols4_upper'][k] for r in upper]))) for k in ('diagnostic','half_anchor')}
    return dict(summary=summary,dual_pass=dual,absolute=result,single_query_upper=bounds,
                per_record_upper=upper,strict_selected=sum(z['selected']['strict'] is not None for z in cells))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    prior.prefix.trainer.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE/'train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**z):
            v = dict(pid=os.getpid(),seconds=round(time.monotonic()-start,3),**z)
            prior.donor.prior.prior.source.atomic(PRIVATE/'heartbeat.json',v)
            if z.get('phase')!='fit':
                print(v,flush=True)
        key,donor,ref,anchor,readout = register()
        data = base.fit_population(beat)
        inputs = previous.model.causal_inputs(data['history'],prior.query_groups(data['query']))
        linear = prior.study.motion.bank(data['history'])[:,:3]
        cells,update = [],0
        for c,h in zip(ref,anchor,strict=True):
            z,u = cell(data,inputs,linear,c,h,donor,key,a.replay,beat)
            update += u
            cells.append(z)
            beat(phase='cell_frozen',fold=z['identity']['fold'],seed=z['identity']['seed'],selected=z['selected'],updates=update)
        f = OUT/'freeze.json'
        old.once(f,old.stable(dict(registration=key,cells=cells,before_held_scoring=True,
                                  optimizer_updates=len(cells)*len(MODES)*STEPS)))
        r = evaluate(data,linear,old.read(f)['cells'],readout)
        old.once(OUT/'readout.json',old.stable(dict(freeze=old.sha(f),**r,original_2pct=True,
            a_labels_opened=False,confirmation_access=False,independent_confirmation=False,deployment_changed=False)))
        beat(phase='complete',updates=update,replay=a.replay)


if __name__ == '__main__':
    main()
