"""Source-excluded reference support and mean probes, not a deployment policy."""
import hashlib
from pathlib import Path
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.metrics import roc_auc_score,average_precision_score
from src.world_model import m3w_harm_refinement as old

DIMENSION=382
TASKS=('direct','event','positive')
base=old.base


def inputs(causal):
    if set(causal)!=base.CAUSAL_KEYS or causal['x'].shape!=(len(causal['ids']),7,428):
        raise ValueError('Exact past-only causal schema required')
    counts=np.zeros(len(causal['ids']),int)
    for group in causal['groups']:
        if np.any(counts[group]):raise ValueError('Disjoint current queries required')
        counts[group]=len(group)
    if (counts<=0).any()or np.any(causal['scales']<=0):raise ValueError('Complete queries and positive causal scales required')
    x=np.column_stack((causal['x'][:,6,:380],np.log1p(counts),np.log1p(causal['scales']))).astype(np.float32)
    if x.shape[1]!=DIMENSION or not np.isfinite(x).all():raise ValueError('Finite382causal descriptors required')
    return x


def inner_folds(sites):
    names=sorted(set(map(str,sites)),key=lambda s:hashlib.sha256(('reference_inner_v1|'+s).encode()).hexdigest())
    if len(names)<3:raise ValueError('At least three whole training sources required')
    return {name:i%3 for i,name in enumerate(names)}


def weights(causal,use):
    use=np.asarray(use,bool)
    if use.shape!=(len(causal['ids']),):raise ValueError('Explicit supervised fitting subset required')
    names=sorted(set(causal['sites'][use]));out=np.zeros(len(use))
    if not names:raise ValueError('No supported fitting labels')
    for site in names:
        groups=[at[use[at]]for at in causal['groups']if str(causal['sites'][at[0]])==str(site)and use[at].any()]
        for at in groups:out[at]=1/(len(names)*len(groups)*len(at))
    if not np.isclose(out.sum(),1):raise ValueError('Source/query balanced loss weights required')
    return out


def model_hash(model):
    h=hashlib.sha256(repr((type(model).__name__,model.n_features_in_,getattr(model,'classes_',None))).encode())
    for tree in model.estimators_:h.update(old.tree_hash(tree).encode())
    return h.hexdigest()


def predict_task(state,x):
    if x.shape[1]!=DIMENSION or not np.isfinite(x).all():raise ValueError('Finite registered feature width required')
    model=state['model'];model.set_params(n_jobs=1)
    if state['task']=='event':
        if not np.array_equal(model.classes_,np.array([0,1])):raise ValueError('Both actual event classes required')
        result=model.predict_proba(x)[:,1]
    else:result=model.predict(x)
    if not np.isfinite(result).all()or (result<0).any():raise ValueError('Finite nonnegative reference predictions required')
    return result


def fit(causal,target,known,fit_sources,*,task,settings,seed,identity,path,resume=False,stop_at=None,
        checkpoint_every=8,beat=lambda **_:None,guard=lambda:None):
    if task not in TASKS:raise ValueError('Registered reference task required')
    x=inputs(causal);target=np.asarray(target,float);known=np.asarray(known)
    if (target.shape!=(len(x),)or known.shape!=target.shape or known.dtype!=bool
            or not np.isfinite(target).all()or (target<0).any()):raise ValueError('Separate finite reference labels and known mask required')
    legal=causal['allowed'][:,6];use=np.isin(causal['sites'],fit_sources)&known&legal
    w=weights(causal,use)
    if task=='positive':
        # Condition the original source/query measure; do not rebalance positive queries.
        use&=target>0;w[~use]=0
        if w.sum()<=0:raise ValueError('No observed positive reference support')
        w/=w.sum()
    at=np.flatnonzero(use)
    y=(target[at]>0).astype(int)if task=='event'else target[at]
    if task=='event'and not np.array_equal(np.unique(y),[0,1]):raise ValueError('Missing positive or negative support-event training class')
    binding=dict(task=task,settings=settings,seed=seed,identity=identity,fit_sources=sorted(map(str,fit_sources)),
        feature_schema='raw_local380_log_query_size_log_causal_scale_no_learned_preprocessing_v1',
        fit_ids_sha256=old.digest(causal['ids'][at]),fit_feature_sha256=old.digest(x[at]),
        target_sha256=old.digest(y),weight_sha256=old.digest(w[at]),rows=len(at),dimension=DIMENSION,
        checkpoint_every=checkpoint_every,new_reference_training=True,future_features=False)
    cls=ExtraTreesClassifier if task=='event'else ExtraTreesRegressor
    model=cls(n_estimators=0,max_depth=settings['max_depth'],min_samples_leaf=settings['min_samples_leaf'],
        max_features=settings['max_features'],bootstrap=False,random_state=seed,n_jobs=1,warm_start=True)
    state=dict(binding,model=model,trees=0,seconds=0.,trace=[]);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not resume:raise ValueError('Explicit unchanged-reference resume required')
        state=base.read_checkpoint(path)
        for key,value in binding.items():base.exact(state[key],value)
        model=state['model']
        if state['trees']and model_hash(model)!=state['model_sha256']:raise ValueError('Saved forest changed')
    limit=settings['trees'][task]if stop_at is None else min(settings['trees'][task],stop_at)
    if not 0<=state['trees']<=limit or limit<=0:raise ValueError('Nondecreasing complete tree budget required')
    elapsed=state['seconds'];start=time.monotonic();xx=x[at];ww=w[at]*len(at)
    while state['trees']<limit:
        guard();model.set_params(n_estimators=min(limit,state['trees']+checkpoint_every))
        model.fit(xx,y,sample_weight=ww)
        state['trees']=model.n_estimators;state['model_sha256']=model_hash(model)
        prediction=predict_task(state,xx)
        row=dict(trees=state['trees'],training_loss=float(np.sum(w[at]*(prediction-y)**2)),
            loss='Brier'if task=='event'else'normalized_reference_MSE',model_sha256=state['model_sha256'],
            prediction_sha256=old.digest(prediction),tree_nodes=sum(t.tree_.node_count for t in model.estimators_))
        state['trace'].append(row);state['seconds']=elapsed+time.monotonic()-start
        guard();base.save_checkpoint(path,state);beat(task=task,**row)
    return base.read_checkpoint(path)


def exact_refit(a,b):
    for key in a:
        if key not in ('model','seconds'):base.exact(a[key],b[key])
    if model_hash(a['model'])!=model_hash(b['model']):raise ValueError('Scientific forest state differs')


def predictions(states,causal):
    if set(states)!=set(TASKS):raise ValueError('Complete direct/event/positive reference models required')
    x=inputs(causal);direct,event,positive=(predict_task(states[k],x)for k in TASKS)
    return dict(direct=direct,event=event,positive=positive,hurdle=event*positive)


def metrics(target,known,causal,prediction,*,sources=None):
    use=np.asarray(known,bool)&causal['allowed'][:,6]
    if sources is not None:use&=np.isin(causal['sites'],sources)
    w=weights(causal,use);at=np.flatnonzero(use);y=np.asarray(target)[at]
    result={k:dict(MSE=float(np.sum(w[at]*(np.asarray(prediction[k])[at]-y)**2)),
        MAE=float(np.sum(w[at]*np.abs(np.asarray(prediction[k])[at]-y))))for k in ('prior','direct','hurdle')}
    event=(y>0).astype(int);p=np.asarray(prediction['event'])[at];prior=np.asarray(prediction['prior_event'])[at]
    if not np.isfinite(p).all()or np.any((p<0)|(p>1)):raise ValueError('Finite event probabilities required')
    q=np.clip(p,1e-12,1-1e-12)
    result['event']=dict(Brier=float(np.sum(w[at]*(p-event)**2)),prior_Brier=float(np.sum(w[at]*(prior-event)**2)),
        positive_rate=float(np.sum(w[at]*event)),AUROC=float(roc_auc_score(event,p,sample_weight=w[at]))if len(np.unique(event))==2 else None,
        AUPRC=float(average_precision_score(event,p,sample_weight=w[at]))if event.any()else None,
        log_loss=float(np.sum(w[at]*(-event*np.log(q)-(1-event)*np.log1p(-q)))))
    result['inventory']=dict(required_rows=len(use),known_eligible=int(use.sum()),unknown_rows=int((~np.asarray(known)).sum()),
        positive_labels=int((event>0).sum()),source_groups=len(set(causal['sites'][use])),
        target='available_label_positive_easy_reference_not_complete_future_or_physical_safety')
    return result
