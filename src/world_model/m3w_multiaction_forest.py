"""Sample-matched five-moment ExtraTrees control for the current causal bank."""
import hashlib
import os
from pathlib import Path
import time
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
import torch
from src.world_model import m3w_multiaction_cost as base
from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_source_forest import project_moments, exact_forest

OBJECTIVE='sample_matched_query_averaged_standard_five_moment_quadratic'


def fingerprint(value):
    value=np.ascontiguousarray(value)
    digest=hashlib.sha256(str((value.shape,value.dtype.str)).encode())
    digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def replay_draws(data,settings,seed):
    rng=torch.Generator().manual_seed(seed+7919)
    counts=np.zeros(len(data['truth']),np.int64);digest='0'*64
    for _ in range(settings['steps']):
        rows=draw_batch(data['groups'],settings['batch_size'],rng)
        np.add.at(counts,rows,1)
        digest=hashlib.sha256(bytes.fromhex(digest)+rows.tobytes()).hexdigest()
    return counts,digest


def feature_rows(z,envelope,rows,actions):
    return np.column_stack((z[rows,actions],np.eye(7,dtype=np.float32)[actions],
                            np.log1p(envelope[rows,actions]))).astype(np.float32)


def training_matrix(data,arm,counts,chunk_size=2048):
    if arm not in base.ARMS or np.shape(counts)!=(len(data['truth']),) or (counts<0).any():
        raise ValueError('Registered bank and original nonnegative query draw counts required')
    allowed=data['allowed'].numpy().copy()
    allowed[:,sorted(set(range(7))-set(base.ARMS[arm]))]=False
    allowed &= counts[:,None]>0
    rows,actions=np.nonzero(allowed);number=allowed.sum(1)
    if not len(rows):raise ValueError('Positive sampled fitting actions required')
    x=np.empty((len(rows),data['z'].shape[-1]+8),np.float32)
    y=np.empty((len(rows),5),np.float32)
    weight=counts[rows].astype(float)/number[rows]
    rms=data['preprocess']['rms'];envelope=data['envelope'].numpy()
    # Expand only the current sampled fitting matrix,not trajectory episodes.
    for start in range(0,len(data['truth']),chunk_size):
        stop=min(start+chunk_size,len(data['truth']))
        left,right=np.searchsorted(rows,(start,stop));at=rows[left:right]
        if left==right:continue
        z=data['z'][np.arange(start,stop)].numpy()
        x[left:right]=feature_rows(z,envelope[start:stop],at-start,actions[left:right])
        y[left:right]=data['truth'][at,actions[left:right]].numpy()/rms
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y<0).any():
        raise ValueError('Finite known fitting moments and causal features required')
    weight_mean=float(weight.mean());weight/=weight_mean
    return dict(x=x,y=y,weight=weight,positive_candidate_rows=len(rows),
        positive_query_rows=int(np.unique(rows).size),query_weight_rescale=weight_mean,
        matrix_bytes=x.nbytes+y.nbytes+weight.nbytes,
        input_hashes={k:fingerprint(v) for k,v in dict(features=x,targets=y,weight=weight,counts=counts).items()})


def fit(matrix,*,preprocess,arm,settings,seed,identity,path,draw_hash,
        heartbeat=lambda **_:None,guard=lambda:None,resume=False,stop_at=None):
    binding=dict(preprocess=preprocess,arm=arm,settings=settings,seed=seed,identity=identity,
        draw_hash=draw_hash,input_hashes=matrix['input_hashes'],objective=OBJECTIVE,
        sklearn=sklearn.__version__,feature_dimension=matrix['x'].shape[1],
        positive_candidate_rows=matrix['positive_candidate_rows'],positive_query_rows=matrix['positive_query_rows'],
        query_weight_rescale=matrix['query_weight_rescale'],matrix_bytes=matrix['matrix_bytes'])
    model=ExtraTreesRegressor(n_estimators=1,criterion='squared_error',max_depth=settings['max_depth'],
        min_samples_leaf=settings['min_samples_leaf'],max_features=settings['max_features'],
        bootstrap=False,random_state=seed,n_jobs=settings['fit_threads'],warm_start=True)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    trees,elapsed,trace=0,0.,[]
    if path.exists():
        if not resume:raise ValueError('Explicit same-forest resume required')
        state=joblib.load(path)
        for key,value in binding.items():base.exact(state[key],value)
        model,elapsed,trace=state['model'],state['seconds'],state['trace'];trees=len(model.estimators_)
    limit=settings['trees'] if stop_at is None else stop_at
    if not 0<=trees<=limit<=settings['trees'] or limit==0:raise ValueError('Fixed nondecreasing tree budget required')
    monitor=np.linspace(0,len(matrix['x'])-1,min(4096,len(matrix['x'])),dtype=int)
    started=time.monotonic()
    def save():
        state=dict(**binding,trees=trees,seconds=elapsed+time.monotonic()-started,trace=trace,model=model,
            nodes=sum(t.tree_.node_count for t in model.estimators_),inner_held_used_for_training=False,
            unknown_rows_sampled=0)
        temporary=path.with_suffix('.tmp');joblib.dump(state,temporary,compress=3);os.replace(temporary,path)
    while trees<limit:
        guard();trees=min(limit,(trees//settings['checkpoint_every']+1)*settings['checkpoint_every'])
        heartbeat(state='tree_block_start',arm=arm,trees=trees,total=settings['trees'])
        model.set_params(n_estimators=trees,n_jobs=settings['fit_threads'])
        with joblib.parallel_backend('threading'):
            model.fit(matrix['x'],matrix['y'],sample_weight=matrix['weight'])
        model.set_params(n_jobs=1)
        if trees%settings['checkpoint_every']==0 or trees==settings['trees']:
            p=model.predict(matrix['x'][monitor]);err=np.square(p-matrix['y'][monitor]).mean(1)
            trace.append(dict(trees=trees,fixed_fitting_monitor_MSE=float(np.average(err,weights=matrix['weight'][monitor])),
                monitor_candidates=len(monitor),nodes=sum(t.tree_.node_count for t in model.estimators_)))
        save();heartbeat(state='tree_block_saved',arm=arm,trees=trees,total=settings['trees'])
    return joblib.load(path)


def predict(state,x,scene,envelope,scales,chunk_size=1024):
    combined=np.concatenate((x,scene),axis=-1)
    base.validate_arrays(combined,envelope,np.ones(np.shape(envelope),bool))
    scales=np.asarray(scales)
    if scales.shape!=(len(x),) or not np.isfinite(scales).all() or (scales<=0).any():
        raise ValueError('Frozen positive fitting scales required')
    model=state['model'];model.set_params(n_jobs=1);parts=[]
    for start in range(0,len(x),chunk_size):
        stop=min(start+chunk_size,len(x));z=base.standardized(combined[start:stop],state['preprocess'])
        env=(envelope[start:stop]/scales[start:stop,None]).astype(np.float32)
        rows=np.repeat(np.arange(stop-start),7);actions=np.tile(np.arange(7),stop-start)
        estimates=model.predict(feature_rows(z,env,rows,actions))*state['preprocess']['rms']
        estimates=project_moments(estimates,env.ravel()).reshape(-1,7,5)
        parts.append(estimates*scales[start:stop,None,None])
    return np.concatenate(parts)
