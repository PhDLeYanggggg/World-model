"""Train on verified source-local action plans; freeze held and camera forecasts."""
import argparse
from collections import Counter
import fcntl
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_full_path_selection as previous
from src.evaluation import m3w_record_peak_allocation as opt
from src.world_model import m3w_slate_imitation as slate

study, base, np, torch = previous.study,previous.base,previous.np,previous.torch
OUT = previous.OUT.parent/'source_excluded_slate_v1'
LOCAL = previous.LOCAL.parent/OUT.name
STEPS, COMPETITORS = previous.STEPS, 15
MISSING = -2


def register():
    parent,cohorts,camera,all_donor = previous.register()
    f = previous.OUT/'freeze.json'
    p = study.read(f)
    if p['registration'] != parent or not p['camera_targets_not_read']:
        raise ValueError('Require frozen donor banks')
    for site in previous.SITES:
        for seed in study.SEEDS:
            for arm in previous.camera.ARMS:
                z = p['predictors'][f'{site}_{seed}_{arm}']['identity']
                if z['held']!=site or set(z['fit_sites'])!=set(previous.SITES)-{site}:
                    raise ValueError('Donor forecasts are not source-excluded')
    r = OUT/'registration.json'
    base.immutable(r,dict(parent=study.artifact(f),deploy_forecasters=study.artifact(previous.camera.OUT/'freeze.json'),
        code=[study.artifact(Path(x)) for x in (__file__,slate.__file__,opt.__file__,previous.__file__,OUT/'protocol.md')],
        cohorts=cohorts,seeds=study.SEEDS,steps=STEPS,workers=0,threads=4,original_2pct=True,
        no_external_reopening=True,threshold_tuning=False,independent_confirmation=False))
    return base.sha(r),cohorts,camera,p,all_donor


def loader(cohorts):
    result = {}
    for site,(h,y,v,q) in previous.loader(cohorts).items():
        with np.load(study.checked(cohorts[site]),allow_pickle=False) as z:
            recording = z['recording']
            if any((np.diff(z['start'][recording==r]) <= 19*study.donor.STRIDE).any() for r in np.unique(recording)):
                raise ValueError('Overlapping or unordered source windows')
        cv,ols = study.method.analytic(h)
        bank = np.stack([ols,previous.camera.linear_menu(h,12)[:,2]],axis=1)
        e = np.linalg.norm(cv[v.all(1)]-y[v.all(1)],axis=-1).mean(1)
        result[site] = dict(h=h,y=y,v=v,q=q,rec=recording[q],cv=cv,ols=ols,bank=bank,e=e[e>0.])
    return result


def qualified(cv,ols,path,y,v,q,cut):
    z = study.policy(path,cv,cv,y,v,np.repeat('record',len(v)),easy_cut=cut[0],hard_cut=cut[1])
    w = 1./np.bincount(q)[q]
    return z['recordings']['record']['supported'] and all(
        w@study.paired(p,path,y,v)[1].sum(1) > 1e-9 for p in (cv,ols))


def teachers(key,site,name,d,bank):
    h,y,v,q,rec,cv,ols = (d[k] for k in ('h','y','v','q','rec','cv','ols'))
    cut = np.quantile(d['e'],[.25,.75])
    output = LOCAL/f'{site}_{name}_teacher.json'
    ident = dict(registration=key,site=site,menu=name,cut=cut.tolist())
    if not output.exists():
        rows, stats = {},Counter()
        for r in np.unique(rec):
            at = np.flatnonzero(rec==r)
            a,b,c,t,m = cv[at],ols[at],bank[at],y[at],v[at]
            g = np.unique(q[at],return_inverse=True)[1]
            n = np.minimum(2,np.bincount(g)-1)
            radii = np.zeros(len(n))
            np.maximum.at(radii,g,np.linalg.norm(c-a[:,None],axis=-1).max((1,2)))
            design = opt.design(a,b,c,t,m,g,cut[0],n,n*radii,'query')
            take,log = opt.solve(design,.75)
            selected = np.full(len(at),MISSING)
            state = log['status']
            if take is not None:
                path = a.copy()
                index,arm = design['row'][take],design['arm'][take]
                if len(np.unique(index))!=len(index) or (np.bincount(g[index],minlength=len(n))>n).any():
                    raise ValueError('Teacher exceeded the candidate budget')
                path[index] = c[index,arm]
                if qualified(a,b,path,t,m,g,cut):
                    selected.fill(-1)
                    selected[index] = arm
                    state = 'verified'
            if state in ('risk_infeasible','nonpositive_bound'):
                selected.fill(-1)
            rows[str(r)] = dict(state=state,actions=selected.tolist())
            stats[state] += 1
        base.immutable(output,dict(identity=ident,records=rows,counts=dict(stats)))
        print(dict(phase='teacher',site=site,menu=name,counts=dict(stats)),flush=True)
    z = study.read(output)
    if z['identity']!=ident:
        raise ValueError('Teacher identity changed')
    labels = np.full(len(h),MISSING)
    for r,record in z['records'].items():
        labels[rec==r] = record['actions']
    return labels,dict(counts=z['counts'],artifact=study.artifact(output),positive_edits=int((labels>=0).sum()))


def examples(d,bank,labels):
    h,cv,ols,q = (d[k] for k in ('h','cv','ols','q'))
    r,k,x = slate.flat_features(h,cv,ols,bank)
    rng = np.random.default_rng(124001)
    rows = []
    for i in np.unique(q):
        query = np.flatnonzero(q==i)
        at = np.flatnonzero(q[r]==i)
        if not len(at) or len(query)<2 or (labels[query]==MISSING).any():
            continue
        options = slate.options(r[at],min(len(query)-1,2))
        good = np.flatnonzero((labels[r[at]]==k[at]) & (labels[r[at]]>=0))
        selected = np.r_[good,[-1]*(2-len(good))]
        true, = np.flatnonzero((options==selected).all(1))
        other = np.delete(np.arange(len(options)),true)
        n = min(len(other),COMPETITORS)
        indices = np.r_[true,rng.choice(other,size=n,replace=False)]
        if true and 0 not in indices:
            indices[-1] = 0
        features = slate.pack(*(v[at] for v in x),options[indices])
        pad = COMPETITORS+1-len(indices)
        rows.append(tuple(np.pad(a,[(0,pad)]+[(0,0)]*(a.ndim-1)) for a in features)+(
            np.arange(COMPETITORS+1)<len(indices),))
    return tuple(np.stack(a) for a in zip(*rows,strict=True)) if rows else None


def expanded(d,source,seed,parent):
    with np.load(study.checked(parent['donor_predictions'][f'{source}_{seed}']),allow_pickle=False) as z:
        np.testing.assert_array_equal(d['q'],z['q'])
        np.testing.assert_array_equal(d['cv'],z['cv'])
        return np.stack([z[k] for k in ('ols4','median4','clean','noise')],axis=1)


def predict(d,net,bank):
    h,cv,ols,q = (d[k] for k in ('h','cv','ols','q'))
    p,unsupported = slate.infer(net,h,cv,ols,bank,q)
    previous.policy.matched(cv,q,{k:p[k] for k in ('joint','matched_independent')})
    for j in np.unique(q):
        at = q==j
        for arm in ('matched_independent','matched_uniform'):
            if arm=='matched_uniform' and j in unsupported:
                continue
            a,b = [np.linalg.norm(p[k][at]-cv[at],axis=-1).max(1) for k in ('joint',arm)]
            np.testing.assert_allclose(np.sort(a),np.sort(b),rtol=1e-8,atol=1e-8)
    return dict(cv=cv,ols4=ols,**p,q=q,uniform_unsupported=unsupported)


def freeze(key,cohorts,camera,parent,all_donor):
    sites = loader(cohorts)
    teachers_analytic, training, fits, forecasts = {},{},{},{}
    for s,d in sites.items():
        y,r = teachers(key,s,'analytic',d,d['bank'])
        teachers_analytic[s] = r
        training[s] = examples(d,d['bank'],y)
    camera_teachers = {}
    for seed in study.SEEDS:
        for held,d in sites.items():
            donors = [s for s in sites if s!=held and training[s] is not None]
            draw = previous.batcher([training[s] for s in donors])
            net,record = previous.fit(key,f'slate_analytic_{held}_{seed}',seed,slate.Slate,
                lambda rng,_: tuple(torch.from_numpy(x) for x in draw(rng)),slate.loss,
                dict(omitted_site=held,fit_sites=donors,arm='analytic'))
            f = f'{held}_{seed}'
            fits[f] = record
            paths = predict(d,net,d['bank'])
            forecasts[f] = previous.save_bank(LOCAL/f'{f}.npz',paths)
        entries = []
        for s,d in sites.items():
            b = expanded(d,s,seed,parent)
            labels,receipt = teachers(key,s,f'expanded_{seed}',d,b)
            camera_teachers[f'{s}_{seed}'] = receipt
            values = examples(d,b,labels)
            if values is not None:
                entries.append((s,values))
        draw = previous.batcher([a for _,a in entries])
        net,record = previous.fit(key,f'slate_camera_{seed}',seed,slate.Slate,
            lambda rng,_: tuple(torch.from_numpy(x) for x in draw(rng)),slate.loss,
            dict(fit_sites=[s for s,_ in entries],arm='camera_expanded_oof'))
        with np.load(study.checked(all_donor['forecasts']['wepdtof'][str(seed)]),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['q'],camera['q'])
            np.testing.assert_array_equal(z['ids'],camera['ids'])
            d = dict(h=camera['history'],cv=z['cv'],ols=z['ols4'],q=z['q'])
            bank = np.stack([z[k] for k in ('ols4','median4','clean','noise')],axis=1)
        f = f'camera_{seed}'
        fits[f] = record
        forecasts[f] = previous.save_bank(LOCAL/f'{f}.npz',dict(**predict(d,net,bank),ids=camera['ids']))
    base.immutable(OUT/'freeze.json',dict(registration=key,fits=fits,forecasts=forecasts,
        analytic_teachers=teachers_analytic,expanded_teachers=camera_teachers,steps=len(fits)*STEPS,
        camera_outcomes_unopened=True,held_site_targets_used_only_for_other_folds=True))


def score_record(p,y,v,q,cut,omit_uniform=False):
    if omit_uniform:
        p = {k:x for k,x in p.items() if k!='matched_uniform'}
    r = study.metrics(p,y,v,q,'record',cut,paired_controls=False)
    cv,active = p['cv'],r['arms']['joint']['risk']['selected']
    w = 1./np.bincount(q)[q]
    direct = {}
    for k in ('matched_independent','matched_uniform'):
        if active and k in p and r['reference_mass']>0.:
            a,b = study.paired(p[k],p['joint'],y,v)
            direct[k] = (100.*np.array([w@a.sum(1),w@b.sum(1)])/r['reference_mass']).tolist()
        else:
            direct[k] = None
    return dict(**r,direct=direct)


def summary(rows):
    stats = {}
    for arm in ('cv','ols4','joint','independent','matched_independent','matched_uniform'):
        a = [r['arms'][arm] for r in rows if arm in r['arms']]
        stats[arm] = dict(views=len(a),edits=sum(r['risk']['selected'] for r in a),
            risk=sum(r['risk']['supported'] for r in a),dual=sum(r['dual_pass'] for r in a))
    return stats


def score(key,cohorts,camera):
    fp = OUT/'freeze.json'
    f = study.read(fp)
    if f['registration']!=key or not f['camera_outcomes_unopened']:
        raise ValueError('Require prediction freeze')
    data = loader(cohorts)
    y,v,stable = previous.camera.targets('wepdtof',camera)
    cut = study.read(study.OUT/'pilot_freeze.json')['cut_px']
    cam,loc = {},{}
    for seed in study.SEEDS:
        for s,d in data.items():
            name = f'{s}_{seed}'
            rec = f['fits'][name]['identity']
            if rec['omitted_site']!=s or s in rec['fit_sites']:
                raise ValueError('Scored donor site entered fit')
            limits = np.quantile(np.concatenate([data[z]['e'] for z in rec['fit_sites']]),[.25,.75])
            with np.load(study.checked(f['forecasts'][name]),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['q'],d['q'])
                p = {k:z[k] for k in z.files if k not in ('q','uniform_unsupported')}
                blocked = z['uniform_unsupported']
            loc[name] = []
            for rec in np.unique(d['rec']):
                at = d['rec']==rec
                q = d['q'][at]
                r = score_record({k:x[at] for k,x in p.items()},d['y'][at],d['v'][at],
                                 np.unique(q,return_inverse=True)[1],limits,bool(np.isin(q,blocked).any()))
                loc[name].append(dict(site=s,record=str(rec),seed=seed,**r))
        with np.load(study.checked(f['forecasts'][f'camera_{seed}']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['q'],camera['q'])
            np.testing.assert_array_equal(z['ids'],camera['ids'])
            p = {k:z[k] for k in z.files if k not in ('ids','q','uniform_unsupported')}
            cam[str(seed)] = score_record(p,y,v,np.unique(camera['q'],return_inverse=True)[1],
                                         cut,bool(len(z['uniform_unsupported'])))
    rows = [r for group in loc.values() for r in group]
    groups = {}
    for s in data:
        sub = [r for r in rows if r['site']==s]
        groups[s] = dict(arms=summary(sub),recordings=len({r['record'] for r in sub}))
        for k in ('matched_independent','matched_uniform'):
            v0 = [r['direct'][k][1] for r in sub if r['direct'][k] is not None]
            groups[s][k] = dict(estimable=len(v0),mean_pp=float(np.mean(v0)) if v0 else None)
    draws = {}
    for k in ('matched_independent','matched_uniform'):
        x = np.array([g[k]['mean_pp'] for g in groups.values() if g[k]['mean_pp'] is not None])
        a = np.random.default_rng(3183).choice(x,size=(3000,len(x))).mean(1) if len(x)>1 else None
        draws[k] = dict(sources=len(x),point_pp=float(x.mean()) if len(x) else None,
                        ci95=np.quantile(a,[.025,.975]).tolist() if a is not None else None)
    stat = dict(opendd=summary(rows),camera=summary(list(cam.values())),
                opendd_matched=draws,camera_matches=int(stable.sum()))
    detail = LOCAL/'held_recording_scores.json'
    base.immutable(detail,dict(registration=key,location_rows=loc))
    base.immutable(OUT/'readout.json',dict(registration=key,freeze=study.artifact(fp),summary=stat,
        camera=cam,site_contrasts=groups,recording_score_cache=study.artifact(detail),original_2pct=True,
        external_test_reopened=False,independent_confirmation=False,deployment_changed=False,submission_ready=False))
    print(stat,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('freeze','score'))
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for d in (OUT,LOCAL):
        d.mkdir(parents=True,exist_ok=True)
    with (LOCAL/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        key,cohorts,camera,parent,all_donor = register()
        if args.phase=='freeze':
            freeze(key,cohorts,camera,parent,all_donor)
        else:
            score(key,cohorts,camera)
