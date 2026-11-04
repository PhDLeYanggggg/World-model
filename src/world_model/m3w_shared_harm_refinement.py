"""Matched 435-column local-zero/shared cost refiners; no new trajectory model."""
from pathlib import Path
import time

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from src.world_model import m3w_harm_refinement as previous

ARMS=('local_zero','shared_motion')
DIMENSION=435
SCHEMA='local387_then_C_standardized_shared48_v1'
base=previous.base


def features(causal,prediction,preprocess,arm,indices=None):
    if arm not in ARMS:raise ValueError('Registered local-zero/shared feature arm required')
    common=previous.features(causal,prediction,preprocess,indices)
    rows,actions,_=previous.layout(causal)if indices is None else indices
    out=np.zeros((len(rows),DIMENSION),np.float32)
    # Keep the previous dynamic cost columns380:385 unchanged; append context.
    out[:,:387]=common
    if arm=='shared_motion':
        if (np.shape(preprocess['mean'])!=(428,)or np.shape(preprocess['std'])!=(428,)
                or np.any(preprocess['std'][380:]<=0)):
            raise ValueError('Original C-fitted shared48 normalization required')
        out[:,387:]=np.clip((causal['x'][rows,actions,380:]-preprocess['mean'][380:])/
            preprocess['std'][380:],-preprocess['clip'],preprocess['clip'])
    if not np.isfinite(out).all():raise ValueError('Finite past/current-only features required')
    return out


def predict(state,causal,frozen_prediction):
    if (state.get('feature_dimension')!=DIMENSION or state.get('feature_schema')!=SCHEMA
            or state.get('arm')not in ARMS):raise ValueError('Exact shared-refinement schema required')
    p=previous.check(causal,frozen_prediction).copy();indices=previous.layout(causal)
    x=features(causal,p,state['preprocess'],state['arm'],indices)
    for tree in state['trees']:
        if tree.n_features_in_!=DIMENSION:raise ValueError('Both fitted arms must use435columns')
        p=previous.apply_tree(p,causal,state['preprocess'],tree,state['settings']['learning_rate'],indices,x)
    return p


def fit(data,frozen_prediction,*,arm,seed,settings,identity,path,resume=False,
        stop_at=None,heartbeat=lambda **_:None,guard=lambda:None):
    if arm not in ARMS:raise ValueError('Registered information arm required')
    c,pr=data['causal'],data['preprocess'];frozen=previous.check(c,frozen_prediction)
    indices=previous.layout(c);rows,actions,_=indices
    if not len(rows):raise ValueError('Unsupported fitting cohort is not a successful zero fit')
    truth=data['truth_by_arm']['bound_aware'].numpy().astype(float)*c['scales'][:,None,None]
    initial_x=features(c,frozen,pr,arm,indices)
    binding=dict(identity=identity,arm=arm,seed=seed,settings=settings,preprocess=pr,
        frozen_prediction_sha256=previous.digest(frozen),fit_ids_sha256=previous.digest(c['ids']),
        raw_feature_sha256=previous.digest(c['x']),initial_effective_feature_sha256=previous.digest(initial_x),
        truth_sha256=previous.digest(truth),legal_sha256=previous.digest(c['allowed']),
        feature_dimension=DIMENSION,feature_schema=SCHEMA,cost_region='adaptive_region',
        legal_pairs=len(rows),queries=len(c['groups']),real_tree_fitting=True,new_neural_training=False,
        no_new_normalizer_fit=True,future_features=False)
    state=dict(binding,trees=[],trace=[],round=0,seconds=0.)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not resume:raise ValueError('Explicit unchanged-definition resume required')
        state=base.read_checkpoint(path)
        for k,v in binding.items():base.exact(state[k],v)
    p=predict(state,c,frozen)
    if state['round']and previous.digest(p)!=state['prediction_sha256']:
        raise ValueError('Resume must exactly reconstruct the saved fitting prediction')
    limit=settings['rounds']if stop_at is None else min(settings['rounds'],stop_at)
    if not 0<=state['round']<=limit or limit<=0:raise ValueError('Nondecreasing complete budget required')
    x=initial_x if not state['round']else features(c,p,pr,arm,indices)
    del initial_x
    denominator=c['scales'][rows,None]*pr['rms'][[1,4]];elapsed=state['seconds'];start=time.monotonic()
    def save():
        state['seconds']=elapsed+time.monotonic()-start
        state['prediction_sha256']=previous.digest(p)
        state['tree_hashes']=[previous.tree_hash(t)for t in state['trees']]
        guard();base.save_checkpoint(path,state)
    try:
        while state['round']<limit:
            guard();choice=previous.select_actions(p,c['allowed'],arm='physical_neural')[0]
            weights=previous.weights(c,choice,'adaptive_region',settings['region_multiplier'],indices)
            target=(truth[rows,actions][:,[1,4]]-p[rows,actions][:,[1,4]])/denominator
            tree=DecisionTreeRegressor(max_depth=settings['max_depth'],splitter='random',
                max_features=settings['max_features'],min_samples_leaf=settings['min_samples_leaf'],
                random_state=seed+7919*(state['round']+1))
            tree.fit(x,target,sample_weight=weights)
            if tree.n_features_in_!=DIMENSION:raise ValueError('Unmatched input width')
            p=previous.apply_tree(p,c,pr,tree,settings['learning_rate'],indices,x)
            state['trees'].append(tree);state['round']+=1
            residual=(truth[rows,actions][:,[1,4]]-p[rows,actions][:,[1,4]])/denominator
            selected=previous.select_actions(p,c['allowed'],arm='physical_neural')[0]
            state['trace'].append(dict(round=state['round'],weighted_normalized_harm_MSE=float(
                np.sum(weights*np.mean(residual**2,axis=1))/weights.sum()),selected_rows=int((selected>=0).sum()),
                tree_nodes=tree.tree_.node_count,tree_hash=previous.tree_hash(tree),
                selection_sha256=previous.digest(selected),prediction_sha256=previous.digest(p)))
            save();heartbeat(state='shared_cost_tree_fitted',arm=arm,**state['trace'][-1])
            x[:,380:385]=np.clip(p[rows,actions]/c['scales'][rows,None]/pr['rms'],0,20)
    except(KeyboardInterrupt,TimeoutError):save();raise
    return base.read_checkpoint(path)


exact_refit=previous.exact_refit
