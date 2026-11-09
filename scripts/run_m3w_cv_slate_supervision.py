"""Train and compare recording-feasible supervision at exact query counts."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
from itertools import combinations

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_cv_slate_teachers as teacher
from src.world_model import m3w_cv_slate_supervision as slate

parent,base,prior = teacher.parent,teacher.base,teacher.prior
np,torch,read,sha,once,stable,artifact,checked = (
    parent.np,parent.torch,parent.read,parent.sha,parent.once,parent.stable,parent.artifact,parent.checked)
OUT,PRIVATE = teacher.OUT,teacher.PRIVATE
STEPS = 1000
FITS = ('gain_additive','feasible_additive','feasible_joint')
ARMS = ('frozen','ols_agreement',*FITS,'calibrated')
QUANTILES = (0.,.25,.50,.75)


def x_for(h):
    past,_ = prior.study.motion.features(h,'prefix')
    return past,prior.model.scale(h)


def weight(q):
    _,inv,n = np.unique(q,return_inverse=True,return_counts=True)
    return 1./n[inv]


def examples(frozen,d,cut):
    cv,ols,bank = teacher.analytic(d['history'])
    h,y,v,q = (d[k] for k in ('history','target','valid','query'))
    past,s = x_for(h)
    with np.load(checked(frozen['teacher']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],d['ids'])
        a,verified = z['action'],z['verified']
    features,labels,source = [],[],[]
    stats = dict(active_queries=0,paired_examples=0,feasible_targets_differ=0,alternative_slates_feasible=0)
    for rec in np.unique(d['recording']):
        ix = np.flatnonzero(d['recording']==rec)
        if not verified[ix].any():
            continue
        if not verified[ix].all():
            raise ValueError('Teacher must mark the whole recording')
        w = weight(q[ix])
        p = slate.path_for(cv[ix],bank[ix],a[ix])
        total,ref,offset,cap = slate.contributions(cv[ix],ols[ix],p,y[ix],v[ix],w,cut[0])
        if not slate.verdict(total,ref,offset,cap)[1]:
            raise ValueError('Verified teacher does not satisfy recomputed contributions')
        for g in prior.query_groups(q[ix]):
            at = ix[g]
            c,o,b = cv[at],ols[at],bank[at]
            k = int((a[at]>=0).sum())
            if k==0:
                continue
            stats['active_queries'] += 1
            own,*_ = slate.contributions(c,o,p[g],y[at],v[at],w[g],cut[0])
            menu = [a[at],*slate.menu(c,o,b,k)]
            unique = list({z.tobytes():z for z in menu}.values())
            edges = prior.geometry.local_edges(h[at])
            x,score = [],[]
            for choice in unique:
                forecast = slate.path_for(c,b,choice)
                m,*_ = slate.contributions(c,o,forecast,y[at],v[at],w[g],cut[0])
                x.append(slate.descriptor(past[at],h[at],c,forecast,s[at],choice,edges))
                score.append(slate.verdict(total-own+m,ref,offset,cap))
            stats['alternative_slates_feasible'] += sum(ok for _,ok in score[1:])
            for i,j in combinations(range(len(x)),2):
                label = [slate.preference(score[i],score[j],f) for f in (False,True)]
                if not any(label):
                    continue
                features.append([x[i],x[j]])
                labels.append(label)
                source.append(d['sites'][at[0]])
                stats['feasible_targets_differ'] += label[0]!=label[1]
    stats['paired_examples'] = len(labels)
    if not labels:
        raise ValueError('No slate training pairs')
    return np.asarray(features),np.asarray(labels,np.float32),np.asarray(source),past.shape[1]+25,stats


def sampler(x,y,sites,target):
    groups = [np.flatnonzero(sites==s) for s in sorted(set(sites))]

    def sample(rng):
        i = prior.draw(groups,rng)
        flip = rng.integers(2,size=len(i)).astype(bool)
        sample = x[i].copy()
        sample[flip] = sample[flip,::-1]
        label = y[i,target].copy()
        label[flip] *= -1.
        return torch.from_numpy(sample),torch.from_numpy(label)
    return sample


def menus(d,cv,ols,bank,frozen):
    h = d['history']
    past,s = x_for(h)
    out = []
    for g in prior.query_groups(d['query']):
        k = int((frozen[g]>=0).sum())
        if not k:
            continue
        choices = [frozen[g],*slate.menu(cv[g],ols[g],bank[g],k)]
        edges = prior.geometry.local_edges(h[g])
        x = [slate.descriptor(past[g],h[g],cv[g],slate.path_for(cv[g],bank[g],a),s[g],a,edges) for a in choices]
        out.append((g,np.asarray(choices),np.asarray(x)))
    return out


@torch.inference_mode()
def decode(nets,menu,anchor):
    a = {'frozen':anchor,'ols_agreement':np.full_like(anchor,-1)}
    a.update({k:np.full_like(anchor,-1) for k in FITS})
    quality = {k:[] for k in FITS}
    if not menu:
        return a,quality
    x = torch.from_numpy(np.concatenate([m[2] for m in menu]))
    scores = {k:net(x).numpy().reshape(len(menu),-1) for k,net in nets.items()}
    for i,(g,z,_) in enumerate(menu):
        a['ols_agreement'][g] = z[1]
        for k,s in scores.items():
            j = int(s[i].argmax())
            a[k][g] = z[j]
            quality[k].append(float(s[i,j]))
    return a,quality


def filtered(action,menu,scores,cut):
    a = action.copy()
    for (g,_,_),s in zip(menu,scores,strict=True):
        if s<cut:
            a[g] = -1
    return a


def cal_anchor(cal,cv,ols,bank,old,parent_cell,width):
    screen = parent.screen
    nets = {k:screen.load_head(z,width-25 if k=='visibility' else width,o)
            for k,z,o in zip(('tail','visibility','benefit'),parent_cell['fits'],(5,12,12),strict=True)}
    nets.update(cond=screen.load_head(old['fits'][1],width,5),swap=screen.load_head(old['fits'][2],width,2))
    pred,_ = parent.inference(nets,cv,ols,bank,cal['history'])
    return parent.parent.choose(pred['tail_lead'],prior.query_groups(cal['query']),prior.model.scale(cal['history']))


def calibrate(arms,quality,menu,cv,ols,bank,d,cut):
    n = len(np.unique(d['recording']))
    sites = sorted(set(d['sites']))
    scorer = prior.prior.parent.readout
    table = []
    for k in FITS:
        if not quality[k]:
            continue
        for q in QUANTILES:
            threshold = float(np.quantile(quality[k],q))
            a = filtered(arms[k],menu,quality[k],threshold)
            rows = base.metrics(cv,ols,bank,d,{'strict':a},cut)
            gain = [[scorer.merge_comparisons([r['comparisons'][f'strict_vs_{key}'] for r in rows if r['source']==s])
                    ['completion_lower_pp'] for key in ('cv','ols4')] for s in sites]
            table.append(dict(arm=k,quantile=q,threshold=threshold,edits=int((a>=0).sum()),
                passes=sum(r['metrics']['strict']['risk']['supported'] for r in rows),lower=np.mean(gain,axis=0).tolist()))
    candidates = [r for r in table if r['edits'] and r['passes']==n and min(r['lower'])>0.]
    return table,max(candidates,key=lambda x:(min(x['lower']),-x['edits'])) if candidates else None


def cell(t,d,donor,old,prior_cell,reg,training,beat,replay):
    ident = prior_cell['identity']
    fold,seed = ident['fold'],ident['seed']
    if old['identity']!=ident or t['train']!=ident['train'] or t['fold']!=fold:
        raise ValueError('Teacher, parent and risk fold mismatch')
    x,y,sites,width,stats = training
    nets,fits,updates,state = {},[],0,None
    for arm in FITS:
        target = int(arm!='gain_additive')
        net,fit,u = prior.prefix.trainer.fit(dict(ident,arm=arm,registration=reg),f'{arm}_{fold}_{seed}',seed,
            lambda:slate.SlateScore(width,arm=='feasible_joint'),STEPS,sampler(x,y,sites,target),slate.loss,replay,beat)
        rng = torch.load(checked(fit['artifact']),weights_only=True,map_location='cpu')['rng']
        if state is not None and rng!=state:
            raise ValueError('Training draw streams differ')
        state = rng
        nets[arm] = net
        fits.append(fit)
        updates += u
    _,cal = base.subset(d,ident['calibration'])
    cv,ols,b = teacher.analytic(cal['history'])
    anchor = cal_anchor(cal,cv,ols,b,old,prior_cell,width)
    m = menus(cal,cv,ols,b,anchor)
    decoded,quality = decode(nets,m,anchor)
    ladder,choice = calibrate(decoded,quality,m,cv,ols,b,cal,prior_cell['cut'])
    hi,h = base.subset(d,ident['held'])
    cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
    b = base.expanded(cv,raw)
    with np.load(checked(prior_cell['actions']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],h['ids'])
        anchor = z['tail_lead']
    m = menus(h,cv,ols,b,anchor)
    action,quality = decode(nets,m,anchor)
    action['calibrated'] = (np.full_like(anchor,-1) if choice is None else
        filtered(action[choice['arm']],m,quality[choice['arm']],choice['threshold']))
    path = prior.study.parent.parent.save_actions(PRIVATE/f'choices_{fold}_{seed}.npz',dict(ids=h['ids'],**action))
    return dict(identity=ident,cut=prior_cell['cut'],fits=fits,training=stats,ladder=ladder,
        selected=choice,actions=path),updates


def score(cells,d,donor,beat):
    scorer = prior.prior.parent.readout
    pairs = {f'{k}_vs_{tag}':(b,k) for k in ARMS for tag,b in (('cv',base.CV),('ols4',base.OLS))}
    primary = dict(feasible_supervision=('gain_additive','feasible_additive'),
        joint_composition=('feasible_additive','feasible_joint'),joint_vs_frozen=('frozen','feasible_joint'))
    pairs.update(primary)
    rows,geo = [],{}
    for c in cells:
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        b = base.expanded(cv,raw)
        with np.load(checked(c['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            a = {k:z[k] for k in ARMS}
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            cv0,o = cv[ix],ols[ix]
            q,y,v = (h[k][ix] for k in ('query','target','valid'))
            paths = {base.CV:cv0,base.OLS:o,**{k:slate.path_for(cv0,b[ix],p[ix]) for k,p in a.items()}}
            r = scorer.score_recording(paths,{},y,v,q,str(rec),easy_cut=c['cut'][0],hard_cut=c['cut'][1])
            w = weight(q)
            terms = {k:prior.study.motion.full_masses(paths[b],paths[a],cv0,y,v,w) for k,(b,a) in pairs.items()}
            source, = np.unique(h['sites'][ix])
            rows.append(dict(r,source=str(source),seed=seed,comparisons=terms))
            z = geo.setdefault((str(source),seed),np.zeros((len(primary),4)))
            history = h['history'][ix]
            for g in prior.query_groups(q):
                if len({int((a[k][ix][g]>=0).sum()) for k in ARMS if k!='calibrated'})!=1:
                    raise ValueError('Query-matched count changed')
                arr = np.stack([paths[k][g] for pair in primary.values() for k in pair])
                _,u,_,(steps,known,_,_) = prior.geometry.query(history[g],arr,y[g],v[g],((0,1),(2,3),(4,5)))
                if steps: z[:,1] += u[:,1]; z[:,3] += 1
                if known: z[:,0] += u[:,0]; z[:,2] += 1
        beat(phase='held_scored',fold=ident['fold'],seed=seed)
    roster = {r['recording']:r['source'] for r in rows}
    summary = prior.prefix.grouped.grouped_readout(rows,roster,base.SEEDS)
    if (summary['views'],summary['source_groups'],summary['risk']['frozen']['switches'])!=(72,9,54748):
        raise ValueError('Expected the complete parent roster and matched budget')
    sources = sorted(set(roster.values()))
    z = np.array([[geo[s,k] for k in base.SEEDS] for s in sources])
    p = np.divide(z[...,:2],z[...,2:],out=np.full_like(z[...,:2],np.nan),where=z[...,2:]>0.)
    draw = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    geometry = {k:dict(observed_pixel=prior.geometry.bootstrap(p[:,:,i,0],draw),
                      completion_lower_pixel=prior.geometry.bootstrap(p[:,:,i,1],draw)) for i,k in enumerate(primary)}
    return summary,geometry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',action='store_true')
    args = p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    prior.prefix.trainer.PRIVATE = PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    t = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

        def beat(**v):
            info = dict(pid=os.getpid(),seconds=round(time.monotonic()-t,3),**v)
            prior.donor.prior.prior.source.atomic(PRIVATE/'heartbeat.json',info)
            if v.get('phase') and v.get('step',STEPS) in (100,STEPS):
                print(info,flush=True)

        _,donor,_ = base.register()
        tf,pf,sf = OUT/'teacher_freeze.json',parent.OUT/'freeze.json',parent.parent.OUT/'freeze.json'
        teach,p0,s0 = map(read,(tf,pf,sf))
        if (teach['registration']!=sha(OUT/'teacher_registration.json') or not teach['analytic_only']
                or any(v['independent_confirmation'] or not v['original_2pct'] for v in (teach,))
                or not all(z['before_held_scoring'] and len(z['cells'])==9 for z in (p0,s0))):
            raise ValueError('Need complete frozen teacher and causal control')
        reg = OUT/'registration.json'
        once(reg,stable(dict(parents=[artifact(x) for x in (tf,pf,sf)],files=[artifact(x) for x in
            (Path(__file__),Path(slate.__file__),OUT/'protocol.md')],steps=STEPS,seeds=base.SEEDS,
            arms=ARMS,quantiles=QUANTILES,original_2pct=True,independent_confirmation=False)))
        d = base.fit_population(beat)
        cells,updates,examples_by_fold = [],0,{}
        for z,s in zip(p0['cells'],s0['cells'],strict=True):
            ident = z['identity']
            fold = ident['fold']
            if fold not in examples_by_fold:
                examples_by_fold[fold] = examples(teach['cells'][fold],teacher.fit_role(d,ident),z['cut'])
            c,u = cell(teach['cells'][fold],d,donor,s,z,sha(reg),examples_by_fold[fold],beat,args.replay)
            cells.append(c)
            updates += u
            beat(phase='cell_frozen',fold=fold,seed=ident['seed'],updates=updates)
        fp = OUT/'freeze.json'
        once(fp,stable(dict(registration=sha(reg),cells=cells,optimizer_steps=27*STEPS,before_held_scoring=True)))
        summary,geometry = score(read(fp)['cells'],d,donor,beat)
        once(OUT/'readout.json',stable(dict(freeze=sha(fp),summary=summary,geometry=geometry,
            qualified=sum(c['selected'] is not None for c in cells),
            calibration_max_passes=[max((x['passes'] for x in c['ladder']),default=0) for c in cells],
            a_labels_opened=False,independent_confirmation=False,original_2pct=True)))
        beat(phase='complete',updates=updates,replay=args.replay)


if __name__=='__main__':
    main()
