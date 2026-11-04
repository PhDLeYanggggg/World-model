"""Fit-only upward harm refinement; causal prediction accepts no future labels."""
import hashlib
from pathlib import Path
import time

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from src.world_model import m3w_amplitude_cost as parent
from src.world_model.m3w_multiaction_cost import select_actions

base=parent.base.base
ARMS=('global','fixed_region','adaptive_region')


def digest(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def tree_hash(tree):
    h=hashlib.sha256(repr(sorted(tree.get_params().items())).encode())
    for key,value in sorted(tree.tree_.__getstate__().items()):
        h.update(key.encode())
        if isinstance(value,np.ndarray)and value.dtype.names:
            # Structured sklearn nodes contain padding bytes with no scientific meaning.
            for field in value.dtype.names:
                h.update(field.encode());h.update(digest(value[field]).encode())
        else:h.update(digest(value).encode())
    return h.hexdigest()


def check(causal,prediction):
    if set(causal)!=base.CAUSAL_KEYS:
        raise ValueError('Only the original exact past-only causal schema is accepted')
    p=np.asarray(prediction)
    n=len(causal['x'])
    if (p.shape!=(n,7,5) or causal['x'].shape!=(n,7,428)
            or not np.isfinite(p).all() or (p<0).any()
            or causal['allowed'].shape!=(n,7) or causal['allowed'].dtype!=bool
            or np.any(causal['scales']<=0)):
        raise ValueError('Aligned finite causal bank and frozen cost prediction required')
    return p


def layout(causal):
    n=len(causal['x']);q=np.full(n,-1,np.int64)
    for i,at in enumerate(causal['groups']):
        if np.any(q[at]>=0):raise ValueError('Query rows must not overlap')
        q[at]=i
    if np.any(q<0):raise ValueError('Every causal row must belong to a query')
    rows,actions=np.nonzero(causal['allowed'])
    return rows,actions,q


def features(causal,prediction,preprocess,indices=None):
    p=check(causal,prediction)
    rows,actions,q=layout(causal)if indices is None else indices
    x=np.empty((len(rows),387),np.float32)
    x[:,:380]=np.clip((causal['x'][rows,actions,:380]-preprocess['mean'][:380])/
        preprocess['std'][:380],-preprocess['clip'],preprocess['clip'])
    x[:,380:385]=np.clip(p[rows,actions]/causal['scales'][rows,None]/preprocess['rms'],0,20)
    x[:,385]=np.asarray((0.,*parent.FRACTIONS))[actions]
    counts=np.bincount(q,minlength=len(causal['groups']))
    x[:,386]=np.log1p(counts[q[rows]])
    if not np.isfinite(x).all():raise ValueError('Finite causal refinement features required')
    return x


def weights(causal,choice,arm,multiplier,indices=None):
    if arm not in ARMS or multiplier<1:raise ValueError('Fixed registered arm and multiplier required')
    rows,actions,q=layout(causal)if indices is None else indices
    choice=np.asarray(choice)
    if choice.shape!=(len(q),):raise ValueError('One causal choice per row required')
    query_mass=np.zeros(len(causal['groups']))
    seen=np.zeros(len(query_mass),int)
    for source in causal['source_groups']:
        if len(source)==0:raise ValueError('Nonempty source query roster required')
        query_mass[source]=1/(len(causal['source_groups'])*len(source));seen[source]+=1
    if not np.all(seen==1):raise ValueError('Every query must belong to exactly one source')
    legal_n=causal['allowed'].sum(1)
    local=1/np.maximum(legal_n[rows],1)
    if arm!='global':local*=np.where(actions==choice[rows],multiplier,1.)
    total=np.bincount(q[rows],weights=local,minlength=len(query_mass))
    return local*query_mass[q[rows]]/total[q[rows]]


def update(prediction,correction,envelope,allowed):
    p=np.asarray(prediction);d=np.asarray(correction)
    if (d.shape!=(*p.shape[:2],2) or not np.isfinite(d).all()
            or not np.isfinite(p).all() or not np.isfinite(envelope).all()
            or np.any(envelope<0)):
        raise ValueError('Finite two-component harm correction and causal bounds required')
    out=p.copy();positive=np.maximum(d,0)
    eh=np.minimum(envelope,p[:,:,4]+positive[:,:,1])
    harm=np.minimum(envelope,np.maximum(p[:,:,1]+positive[:,:,0],eh))
    out[:,:,1]=np.where(allowed,harm,p[:,:,1])
    out[:,:,4]=np.where(allowed,np.minimum(eh,harm),p[:,:,4])
    out[:,:,0]=np.where(allowed,np.minimum.reduce((p[:,:,0],p[:,:,2],np.maximum(envelope-harm,0))),p[:,:,0])
    return out


def apply_tree(prediction,causal,preprocess,tree,rate,indices=None,x=None):
    indices=layout(causal)if indices is None else indices
    rows,actions,_=indices
    if x is None:x=features(causal,prediction,preprocess,indices)
    else:
        x[:,380:385]=np.clip(prediction[rows,actions]/causal['scales'][rows,None]/preprocess['rms'],0,20)
    if not len(rows):return prediction.copy()
    correction=np.zeros((*prediction.shape[:2],2))
    correction[rows,actions]=rate*tree.predict(x)*causal['scales'][rows,None]*preprocess['rms'][[1,4]]
    return update(prediction,correction,causal['envelope'],causal['allowed'])


def predict(state,causal,frozen_prediction):
    p=check(causal,frozen_prediction).copy()
    indices=layout(causal)
    x=features(causal,p,state['preprocess'],indices)
    for tree in state['trees']:
        p=apply_tree(p,causal,state['preprocess'],tree,state['settings']['learning_rate'],indices,x)
    return p


def fit(data,frozen_prediction,*,arm,seed,settings,identity,path,resume=False,
        stop_at=None,heartbeat=lambda **_:None,guard=lambda:None):
    c,pr=data['causal'],data['preprocess'];frozen=check(c,frozen_prediction)
    indices=layout(c);rows,actions,_=indices
    if not len(rows):raise ValueError('No legal fitting pairs; unsupported is not a successful zero fit')
    if arm not in ARMS:raise ValueError('Registered harm refinement arm required')
    truth=data['truth_by_arm']['bound_aware'].numpy().astype(float)*c['scales'][:,None,None]
    binding=dict(identity=identity,arm=arm,seed=seed,settings=settings,preprocess=pr,
        frozen_prediction_sha256=digest(frozen),fit_ids_sha256=digest(c['ids']),
        feature_sha256=digest(c['x']),truth_sha256=digest(truth),legal_sha256=digest(c['allowed']),
        real_tree_fitting=True,new_neural_training=False,feature_dimension=387,
        legal_pairs=len(rows),queries=len(c['groups']),future_features=False)
    state=dict(binding,trees=[],trace=[],round=0,seconds=0.)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not resume:raise ValueError('Explicit unchanged-binding resume required')
        state=base.read_checkpoint(path)
        for k,v in binding.items():base.exact(state[k],v)
    p=predict(state,c,frozen)
    if state['round'] and digest(p)!=state['prediction_sha256']:
        raise ValueError('Resume replay does not reproduce saved fitting prediction')
    limit=settings['rounds']if stop_at is None else min(settings['rounds'],stop_at)
    if not 0<=state['round']<=limit or limit<=0:raise ValueError('No rollback or empty fitting budget')
    initial_choice=select_actions(frozen,c['allowed'],arm='physical_neural')[0]
    x=features(c,p,pr,indices);denominator=c['scales'][rows,None]*pr['rms'][[1,4]]
    start=time.monotonic();elapsed=state['seconds']
    def save():
        state['seconds']=elapsed+time.monotonic()-start
        state['prediction_sha256']=digest(p);state['tree_hashes']=[tree_hash(t)for t in state['trees']]
        guard();base.save_checkpoint(path,state)
    try:
        while state['round']<limit:
            guard();choice=select_actions(p,c['allowed'],arm='physical_neural')[0]
            w=weights(c,initial_choice if arm=='fixed_region'else choice,arm,settings['region_multiplier'],indices)
            target=(truth[rows,actions][:,[1,4]]-p[rows,actions][:,[1,4]])/denominator
            tree=DecisionTreeRegressor(max_depth=settings['max_depth'],splitter='random',
                max_features=settings['max_features'],min_samples_leaf=settings['min_samples_leaf'],
                random_state=seed+7919*(state['round']+1))
            tree.fit(x,target,sample_weight=w)
            p=apply_tree(p,c,pr,tree,settings['learning_rate'],indices,x)
            state['trees'].append(tree);state['round']+=1
            residual=(truth[rows,actions][:,[1,4]]-p[rows,actions][:,[1,4]])/denominator
            next_choice=select_actions(p,c['allowed'],arm='physical_neural')[0]
            state['trace'].append(dict(round=state['round'],weighted_normalized_harm_MSE=float(
                np.sum(w*np.mean(residual**2,axis=1))/w.sum()),selected_rows=int((next_choice>=0).sum()),
                tree_nodes=tree.tree_.node_count,tree_hash=tree_hash(tree),
                selection_sha256=digest(next_choice),prediction_sha256=digest(p)))
            save();heartbeat(state='harm_tree_fitted',arm=arm,**state['trace'][-1])
            # The next fitting round must see its own current causal cost features.
            x[:,380:385]=np.clip(p[rows,actions]/c['scales'][rows,None]/pr['rms'],0,20)
    except (KeyboardInterrupt,TimeoutError):
        save();raise
    return base.read_checkpoint(path)


def exact_refit(a,b):
    for key in a:
        if key not in ('trees','seconds'):base.exact(a[key],b[key])
    if [tree_hash(t)for t in a['trees']]!=[tree_hash(t)for t in b['trees']]:
        raise ValueError('Scientific tree states differ across deterministic refit/resume')

