"""Donor-easy-constrained scalar fits; not query safety calibration."""
import hashlib
import statistics
import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from src.evaluation import m3w_coupled_support as parent

base=parent.base
MODES=('parent_blend','easy_only','record_mean_easy','shuffled_easy','easy_constrained')
ARMS=('origin',)+MODES
PRIMARY='easy_constrained'


def coefficients(error,delta,recs,supported):
    alpha,_,_=parent.blend_coefficient(error,delta,recs,supported)
    count=(recs[:,:,None]==recs[:,None,:]).sum(-1)
    weight=1./count;weight/=weight.sum(-1,keepdims=True)
    cross=(weight[...,None]*error*delta).sum(1)
    variance=(weight[...,None]*delta**2).sum(1)
    cap=np.clip(np.divide(-2*cross[:,1],variance[:,1],out=np.ones(len(error)),
        where=variance[:,1]>0),0,1)
    constrained=np.minimum(alpha,cap)
    easy=np.clip(np.divide(-cross[:,1],variance[:,1],out=np.zeros(len(error)),
        where=(variance[:,1]>0)&supported),0,1)
    loss=lambda a:(weight[...,None]*(error+a[:,None,None]*delta)**2).sum(1)
    before=loss(np.zeros(len(error)));previous=loss(alpha);after=loss(constrained)
    if np.any(after[:,1]>before[:,1]+1e-10*np.maximum(1,before[:,1])):
        raise AssertionError('Donor easy constraint failed')
    if np.any(after.sum(1)>before.sum(1)+1e-10*np.maximum(1,before.sum(1))):
        raise AssertionError('Constrained optimum loses its feasible zero blend')
    return dict(parent_alpha=alpha,alpha=constrained,easy_cap=cap,easy_only_alpha=easy,
        before=before,parent_after=previous,after=after,easy_only_after=loss(easy),
        cross=cross,variance=variance,
        conflict=supported & (previous[:,1]>before[:,1]+1e-12) &
            (previous.sum(1)<=before.sum(1)+1e-10*np.maximum(1,before.sum(1))))


def retrieve(bank,query,frozen,*,k=32,heartbeat=lambda **_:None):
    old=parent.retrieve(bank,query,frozen,k=k,heartbeat=heartbeat)
    x,env,y,recs,eligible,scale,sources=(np.asarray(bank[key]) for key in
        ('x','envelope','labels','recordings','eligible','scale','sources'))
    query,frozen=np.asarray(query),np.asarray(frozen)
    n=len(x);indices=old['indices'];pos=np.full(n,-1,int);pos[indices]=np.arange(len(indices))
    origin=frozen[indices][:,[1,4]]
    predictions={a:origin.copy() for a in MODES}
    predictions['parent_blend']=old['predictions'][parent.PRIMARY]
    channel={key:np.zeros((len(indices),2)) for key in ('before','parent_after','after','easy_only_after','cross','variance')}
    channel.update({key:np.zeros(len(indices)) for key in ('parent_alpha','alpha','easy_cap','easy_only_alpha','mean_alpha','shuffled_alpha','conflict')})
    causal=eligible&(env>0);selected=causal&query&np.isfinite(y).all(1)
    caps=np.zeros(n);caps[causal]=np.clip((env[causal]-frozen[causal,0])/env[causal],0,1)
    signed=np.zeros((n,2));signed[selected]=(y[selected][:,[1,4]]+.02*(frozen[selected][:,[2,3]]-y[selected][:,[2,3]]))/env[selected,None]
    _,rec_ids=np.unique(recs,return_inverse=True);_,source_ids=np.unique(sources,return_inverse=True)
    with threadpool_limits(limits=4):
        for receipt in old['references']:
            if not receipt['supported']:continue
            rec=receipt['recording'];ix=np.flatnonzero(query&(recs==rec));p=pos[ix]
            norm=np.flatnonzero(causal&(recs!=rec));donors=np.flatnonzero(selected&(recs!=rec))
            mean=x[norm].astype(float).mean(0);std=np.maximum(x[norm].astype(float).std(0),1e-6)
            search=NearestNeighbors(n_neighbors=k,algorithm='brute',metric='euclidean',n_jobs=1).fit(
                np.clip((x[donors]-mean)/std,-8,8).astype(np.float32))
            counts=np.bincount(source_ids[donors],minlength=len(set(sources)))
            rcounts=np.bincount(rec_ids[donors],minlength=len(set(recs)));trace=hashlib.sha256()
            for start in range(0,len(ix),256):
                q=ix[start:start+256];p0=pos[q]
                _,near=search.kneighbors(np.clip((x[q]-mean)/std,-8,8).astype(np.float32))
                chosen=donors[near];trace.update(chosen.tobytes());rs=rec_ids[chosen];ss=source_ids[chosen]
                square=(env[chosen]/scale[chosen])**2
                nested,support,_=parent.nested_fit(signed[chosen],square,counts[ss],rcounts[rs],rs,ss,caps[chosen])
                nested*=env[chosen,None]
                error=((frozen[chosen][:,:,[1,4]]-.02*frozen[chosen][:,:,[2,3]])-
                    (y[chosen][:,:,[1,4]]-.02*y[chosen][:,:,[2,3]]))/scale[chosen,None]
                delta=(nested-frozen[chosen][:,:,[1,4]])/scale[chosen,None]
                out=coefficients(error,delta,rs,support)
                if not np.array_equal(out['parent_alpha'],old['diagnostics']['alpha'][p0]):
                    raise AssertionError('Parent alpha changed')
                for key,val in out.items():channel[key][p0]=val
                step=old['predictions']['coupled_cost'][p0]-origin[p0]
                predictions[PRIMARY][p0]=origin[p0]+out['alpha'][:,None]*step
                predictions['easy_only'][p0]=origin[p0]+out['easy_only_alpha'][:,None]*step
                heartbeat(state='analogue_queries',recording=rec,completed=min(start+256,len(ix)),total=len(ix))
            if trace.hexdigest()!=receipt['neighbor_sha256']:raise AssertionError('Frozen neighbors changed')
            channel['mean_alpha'][p]=channel['alpha'][p].mean()
            salt=int.from_bytes(hashlib.sha256(('20261029|'+rec).encode()).digest()[:8],'little')
            channel['shuffled_alpha'][p]=channel['alpha'][np.random.default_rng(salt).permutation(p)]
            for arm,key in (('record_mean_easy','mean_alpha'),('shuffled_easy','shuffled_alpha')):
                predictions[arm][p]=origin[p]+channel[key][p,None]*(old['predictions']['coupled_cost'][p]-origin[p])
    return dict(indices=indices,predictions=predictions,channel=channel,references=old['references'],
        supported=old['supported'],nested_supported=old['diagnostics']['nested_supported'])


def summarize(rows,expected):
    keys=[(r['context'],r['source'],r['seed']) for r in rows]
    if len(keys)!=72 or len(set(keys))!=72 or set(keys)!=set(expected):raise ValueError('Complete72 views required')
    means={a:{t:statistics.fmean(r['scores'][a][t]['full_cohort_MSE'] for r in rows)
        for t in ('all','easy')} for a in ARMS}
    prior={a:{t:statistics.fmean(r['prior_MSE'][a][t] for r in rows) for t in ('all','easy')}
           for a in ('independent_cost','uniform')}
    wins=0
    for context,seed in sorted({(r['context'],r['seed']) for r in rows}):
        group=[r for r in rows if (r['context'],r['seed'])==(context,seed)]
        vals={a:statistics.fmean(r['scores'][a]['easy']['full_cohort_MSE'] for r in group)
              for a in ('origin','parent_blend',PRIMARY)}
        ind=statistics.fmean(r['prior_MSE']['independent_cost']['easy'] for r in group)
        wins+=vals[PRIMARY]<min(vals['origin'],vals['parent_blend'],ind)-1e-12
    n=means[PRIMARY]
    return dict(views=72,equal_view_MSE=means,prior_MSE=prior,pair_wins=int(wins),
        risk_MSE_is_secondary_not_policy_advancement_gate=True,
        conditional_assignment_screen={a:n['easy']<means[a]['easy']-1e-12 for a in ('record_mean_easy','shuffled_easy')},
        donor_conflict_occurrences=sum(r['diagnostics']['donor_conflicts'] for r in rows),
        donor_easy_constraint_violations=sum(r['diagnostics']['donor_easy_constraint_violations'] for r in rows),
        occurrences={k:sum(r['scores'][PRIMARY][k] for r in rows) for k in ('rows','known','unknown','supported','unsupported')},
        fitting_diagnostic_only=True,whole_pipeline_OOF=False,held_inference=False,independent_roles_read=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False)
