"""Compare whole intervention sets with count and predicted spend held fixed."""
import numpy as np

from src.world_model.m3w_causal_record_ledger import EPS

PAIR_DIM = 4


def mixed(bank, cv, origin, scale, sets):
    """Mixed relative-motion change, not a collision probability."""
    if sets.shape[1]==1:
        return np.zeros((len(sets),PAIR_DIM))
    i,j = sets[:,:,0].T
    a,b = sets[:,:,1].T
    s = np.sqrt(scale[i]*scale[j])[:,None]
    offset = (origin[i]-origin[j])[:,None]
    def stats(x,y):
        d = np.linalg.norm(offset+x-y,axis=-1)/s
        return np.column_stack((d.min(1),d.mean(1),d[:,-1],(1./(1.+d)).mean(1)))
    p,q = bank[i,a],bank[j,b]
    c,d = cv[i],cv[j]
    return np.clip(stats(p,q)-stats(p,d)-stats(c,q)+stats(c,d),-12.,12.)


def additive(x, sets):
    n,k,d = x.shape
    result = np.zeros((len(sets),k,d))
    rows,acts = sets.transpose(2,0,1)
    np.add.at(result,(np.arange(len(sets))[:,None],acts),x[rows,acts]/sets.shape[1])
    return result.reshape(len(sets),k*d)


def demonstration_differences(groups,x,bank,cv,origin,scale,membership):
    k,d = x.shape[1:]
    diff = []
    for g in groups:
        old,new = (np.argwhere(z[g]>0.) for z in membership)
        if np.array_equal(old,new):
            continue
        if len(old)!=len(new) or len(new) not in (1,2):
            raise ValueError('Demonstrations must have the same current count')
        sets = np.stack((old,new))
        features = np.column_stack((additive(x[g],sets),mixed(bank[g],cv[g],origin[g],scale[g],sets)))
        diff.append(features[1]-features[0])
    return np.asarray(diff).reshape(-1,k*d+PAIR_DIM)


def fit(differences,width,ridge=.002):
    nonempty = [x for x in differences if len(x)]
    if not nonempty:
        raise ValueError('No changed donor pairs')
    x = np.concatenate([z/np.sqrt(len(z)) for z in nonempty])
    w = np.concatenate([np.full(len(z),1./np.sqrt(len(z))) for z in nonempty])
    x /= np.sqrt(len(nonempty))
    w /= np.sqrt(len(nonempty))
    g,b = x.T@x,x.T@w
    return tuple(np.linalg.solve(g[:k,:k]+ridge*np.eye(k),b[:k]) for k in (width,len(b)))


def slate(utility, all_cost, easy_cost, legal, witness, limit=16):
    """A shared causal slate; every option spends no more than the witness."""
    rows = np.flatnonzero(witness>=0)
    take = np.column_stack((rows,witness[rows]))
    n = len(take)
    if n not in (1,2) or not legal[tuple(take.T)].all():
        raise ValueError('A legal original one- or two-action witness required')
    r,k = np.where(legal)
    cand = np.unique(np.concatenate([v[r,k].argsort(kind='stable')[:limit] for v in
                                     (-utility,all_cost,easy_cost)]))
    pool = np.unique(np.concatenate((np.column_stack((r[cand],k[cand])),take)),axis=0)
    if n==1:
        z = pool[:,None]
    else:
        i,j = np.triu_indices(len(pool),1)
        use = pool[i,0]!=pool[j,0]
        z = np.stack((pool[i[use]],pool[j[use]]),axis=1)
    for v in (all_cost,easy_cost):
        cost = v[z[:,:,0],z[:,:,1]].sum(1)
        z = z[cost <= v[tuple(take.T)].sum()+EPS]
    return np.concatenate((take[None],z[np.any(z!=take,axis=(1,2))]))


def choose(sets, action_scores, bonuses=None):
    s = action_scores[sets[:,:,0],sets[:,:,1]].mean(1)
    return sets[(s if bonuses is None else s+bonuses).argmax()]
