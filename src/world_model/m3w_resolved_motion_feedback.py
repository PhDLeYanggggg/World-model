"""Past-only feedback from endpoint forecasts already resolved by a query."""
import numpy as np

OWN = 6
SCENE = 5


def features(groups, frames, agents, origin, scale, floor, paths):
    """A record stream; outcomes at t are only positions observed at t."""
    n, candidates = paths.shape[:2]
    if (frames.shape != (n,) or agents.shape != (n,) or origin.shape != (n,2) or scale.shape != (n,)
            or floor.shape != (n,12,2) or paths.shape[2:] != (12,2) or (scale<=0).any()):
        raise ValueError('Aligned observation identities and native 8/12 paths required')
    result = np.zeros((n,candidates,OWN+SCENE),dtype=np.float32)
    previous = {}; latest = {}; past = None
    support = 0.; total = np.zeros((candidates,4)); matches = 0; mature_queries = 0
    for g in groups:
        t = int(frames[g[0]])
        if (past is not None and t<=past or not (frames[g]==t).all() or len(np.unique(agents[g]))!=len(g)):
            raise ValueError('Strict ordered queries and unique observed agents required')
        if past is not None:
            decay = .5**((t-past)/24.); total *= decay; support *= decay
        j = previous.get(t-12)
        if j is not None:
            mature_queries += 1
            who, i0, i1 = np.intersect1d(agents[j],agents[g],assume_unique=True,return_indices=True)
            before, now = j[i0],g[i1]; matches += len(who)
            delta = origin[now]-origin[before]
            ref = np.linalg.norm(floor[before,-1]-delta,axis=-1)/scale[before]
            alt = np.linalg.norm(paths[before,:,-1]-delta[:,None],axis=-1)/scale[before,None]
            gain = ref[:,None]-alt; harm = np.maximum(-gain,0.)
            raw = np.stack((np.broadcast_to(ref[:,None],gain.shape),gain,harm,harm>0.),axis=-1)
            total += raw.sum(0); support += len(who)
            for ident,row in zip(who,raw):
                latest[int(ident)] = (t,row)
        for i in g:
            known = latest.get(int(agents[i]))
            if known is not None:
                when,row = known
                result[i,:,0] = 1.; result[i,:,1] = np.log1p(t-when)
                result[i,:,2:6] = np.column_stack((np.log1p(row[:,0]),np.arcsinh(row[:,1]),np.log1p(row[:,2]),row[:,3]))
        if support > 0.:
            avg = total/support
            result[g,:,OWN:] = np.column_stack((np.full(candidates,np.log1p(support)),np.log1p(avg[:,0]),
                np.arcsinh(avg[:,1]),np.log1p(avg[:,2]),avg[:,3]))
        previous[t] = g; past = t
    return result,dict(exact_maturity_matches=matches,mature_queries=mature_queries,
        own_supported_rows=int((result[:,0,0]>0).sum()),scene_supported_rows=int((result[:,0,OWN]>0).sum()))


def augment(x, feedback, scope):
    if scope not in ('static','own','own_scene'):
        raise ValueError('Registered feature ablation required')
    f = feedback.copy()
    if scope != 'own_scene':
        f[...,OWN:] = 0.
    if scope == 'static':
        f[...,:OWN] = 0.
    return np.concatenate((x,f),axis=-1)
