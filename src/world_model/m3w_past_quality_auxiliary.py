"""Train-only, centered quality regressions on frozen cost-forest leaves."""
import numpy as np
from src.world_model import m3w_source_forest as forest
from src.world_model.m3w_component_calibration import eligible, matched
from src.world_model.m3w_unknown_outcome_bounds import completion_bounds
from src.world_model.m3w_forest_projection import interval

FEATURES = ('line_residual_over_width', 'last_fd_ols8_disagreement_over_width',
            'width_range_over_width', 'reversal_fraction', 'raw_prefix_frame_presence',
            'raw_prefix_mean_detector_confidence', 'partial_nearest_neighbors')
ARMS = ('quality', 'placebo')


def quality_matrix(fields):
    if set(fields) != set(FEATURES):
        raise ValueError('Exactly the seven frozen past-only diagnostics required')
    q = np.column_stack([fields[k] for k in FEATURES]).astype(float)
    if not np.isfinite(q).all(): raise ValueError('Finite past quality required')
    return q


def standardize(q, mean, std):
    q = np.asarray(q, float)
    if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all():
        raise ValueError('Seven finite past-quality features required')
    return np.clip((q-mean)/std, -8, 8)


def within_recording_permutation(recordings, seed):
    r = np.asarray(recordings); order = np.arange(len(r))
    rng = np.random.default_rng(seed+61009)
    for recording in np.unique(r):
        ix = np.flatnonzero(r == recording); order[ix] = rng.permutation(ix)
    np.testing.assert_array_equal(r, r[order])
    return order


def leaf_regression(leaves, q, target, weights, penalty):
    """Weighted conditional slopes, with zero mean correction in each train leaf."""
    q, y, w = np.asarray(q,float), np.asarray(target,float), np.asarray(weights,float)
    if (q.shape != (len(w),7) or y.shape != (len(w),5) or len(leaves) != len(w)
            or not len(w) or not np.isfinite(q).all() or not np.isfinite(y).all()
            or not np.isfinite(w).all() or (w <= 0).any() or penalty <= 0):
        raise ValueError('Finite known training rows with positive weights and ridge required')
    nodes, inv = np.unique(leaves,return_inverse=True)
    mass = np.bincount(inv,weights=w)
    wnorm = w/mass[inv]
    def avg(v): return np.bincount(inv,weights=wnorm*v,minlength=len(nodes))
    qm = np.column_stack([avg(q[:,i]) for i in range(7)])
    ym = np.column_stack([avg(y[:,i]) for i in range(5)])
    z, t = q-qm[inv], y-ym[inv]
    gram = np.empty((len(nodes),7,7)); rhs = np.empty((len(nodes),7,5))
    for i in range(7):
        for j in range(i,7): gram[:,i,j] = gram[:,j,i] = avg(z[:,i]*z[:,j])
        for j in range(5): rhs[:,i,j] = avg(z[:,i]*t[:,j])
    coef = np.linalg.solve(gram+penalty*np.eye(7),rhs)
    delta = np.einsum('ni,nij->nj',z,coef[inv])
    zero_mean = np.column_stack([avg(delta[:,j]) for j in range(5)])
    np.testing.assert_allclose(zero_mean,0,atol=1e-9)
    before = float(w@np.mean(t*t,1)/w.sum())
    after = float(w@np.mean((t-delta)**2,1)/w.sum())
    assert after <= before+1e-9
    return dict(nodes=nodes,means=qm,coef=coef,original=ym,mass=mass,
                loss_before=before,loss_after=after,max_centering_error=float(np.abs(zero_mean).max()))


def fit(state, x, env, y, q, sites, recordings, frames, *, arm, seed, penalty=1.):
    if arm not in ARMS: raise ValueError('Registered auxiliary arm required')
    pr = state['preprocess']
    forest.core.exact(pr,forest.core.preprocess(x,env,y,sites,recordings,frames,training_site=pr['training_site']))
    z,_ = forest.causal_inputs(x,env,pr)
    known = np.isfinite(y).all(1); w,_ = forest.core.weights(sites,recordings,frames,known)
    inputs = dict(features=forest.fingerprint(z[known]),
        targets=forest.fingerprint(forest.transformed_targets(y[known],pr)),
        weights=forest.fingerprint(w[known]),known_mask=forest.fingerprint(known))
    assert inputs == state['input_hashes'], 'Frozen training membership/targets changed'
    q = np.asarray(q,float)
    if q.shape != (len(y),7) or not np.isfinite(q).all(): raise ValueError('Past features must align')
    mean = w@q; std = np.sqrt(w@((q-mean)**2)).clip(1e-6)
    qq = standardize(q,mean,std)[known]
    order = np.arange(len(qq)) if arm == 'quality' else within_recording_permutation(np.asarray(recordings)[known],seed)
    qq = qq[order]
    output_scale = pr['scale']*pr['rms'][:5]
    yy,ww,zz = y[known]/output_scale,w[known],z[known]
    chunks = dict(nodes=[],means=[],coef=[]); offsets = [0]; before=[]; after=[]; centering=[]
    for tree in state['model'].estimators_:
        a = leaf_regression(tree.apply(zz),qq,yy,ww,penalty)
        raw = tree.tree_.value[a['nodes'],:,0]/forest.FACTORS*pr['rms']*pr['scale']
        np.testing.assert_allclose(a['original']*output_scale,raw[:,:5],rtol=2e-6,atol=1e-7)
        np.testing.assert_allclose(a['mass'],tree.tree_.weighted_n_node_samples[a['nodes']],rtol=2e-6,atol=1e-10)
        for key in chunks: chunks[key].append(a[key])
        offsets.append(offsets[-1]+len(a['nodes']))
        before.append(a['loss_before']);after.append(a['loss_after']);centering.append(a['max_centering_error'])
    return dict(**{k:np.concatenate(v) for k,v in chunks.items()},offsets=np.array(offsets,dtype=np.int64),
        quality_mean=mean,quality_std=std,output_scale=output_scale,
        arm=arm,seed=seed,penalty=penalty,input_hashes=inputs,
        past_quality_hash=forest.fingerprint(q),known_permutation_hash=forest.fingerprint(order),
        known_train_rows=int(known.sum()),unknown_train_rows=int((~known).sum()),
        original_leaves_reconstructed=offsets[-1],training_zero_mean_max=max(centering),
        mean_leaf_training_loss_before=float(np.mean(before)),
        mean_leaf_training_loss_after=float(np.mean(after)))


def predict(state, fit, x, env, q):
    z,support = forest.causal_inputs(x,env,state['preprocess'])
    v = standardize(q,fit['quality_mean'],fit['quality_std'])
    if len(v) != len(z): raise ValueError('Aligned inference rows required')
    pr=state['preprocess']; state['model'].set_params(n_jobs=1)
    raw=state['model'].predict(z)/forest.FACTORS*pr['rms']*pr['scale']
    old=forest.project_moments(raw[:,:5],env)
    delta=np.zeros((len(x),5))
    for j,tree in enumerate(state['model'].estimators_):
        start,end=fit['offsets'][j:j+2]; nodes=fit['nodes'][start:end]
        at=np.searchsorted(nodes,tree.apply(z))
        assert (at<len(nodes)).all()
        np.testing.assert_array_equal(nodes[at],tree.apply(z))
        at+=start
        delta += np.einsum('ni,nij->nj',v-fit['means'][at],fit['coef'][at])
    delta *= fit['output_scale']/len(state['model'].estimators_)
    new=forest.project_moments(np.maximum(raw[:,:5]+delta,0),env)
    return old,new,support,dict(correction_RMS=float(np.sqrt(np.mean(delta**2))),
                              negative_raw_coordinates=int((raw[:,:5]+delta<0).sum()))


def evaluate(state, predictions, y, env, moving, support, sites, rec, frames, ids):
    if set(predictions) != {'original','quality','placebo'}: raise ValueError('All fixed controls required')
    actions={k:eligible(p,moving,support) for k,p in predictions.items()}
    for other in ('original','placebo'):
        a,b=matched(actions[other],actions['quality'],forest.core.signed(predictions[other])[:,0],
                    forest.core.signed(predictions['quality'])[:,0],rec,frames,ids)
        actions[other+'_matched_quality']=a;actions['quality_matched_'+other]=b
        _,inv=np.unique(np.rec.fromarrays([rec,frames]),return_inverse=True)
        np.testing.assert_array_equal(np.bincount(inv,weights=a),np.bincount(inv,weights=b))
    policies={k:completion_bounds(y,a,env) for k,a in actions.items()}
    scores={k:forest.signed_score(p,y,state['preprocess'],sites,rec,frames) for k,p in predictions.items()}
    den=float(y[np.isfinite(y).all(1),2].sum()); contrasts={}
    for other in ('original','placebo'):
        contrasts['quality_minus_'+other+'_signed_MSE']=scores['quality']-scores[other]
        for mode in ('full','matched'):
            a=policies[other if mode=='full' else other+'_matched_quality']
            b=policies['quality' if mode=='full' else 'quality_matched_'+other]
            contrasts['quality_minus_'+other+'_'+mode+'_utility_percent']=100*(b['selected_net_gain_lower_mass']-a['selected_net_gain_lower_mass'])/den if den>0 else None
    return dict(scores=scores,policies=policies,contrasts=contrasts,full_known_reference_mass=den),actions


def summarize(groups,cfg):
    out=dict(groups=len(groups),auxiliary_fits=len(groups)*2,new_tree_splits=0,new_neural_updates=0,
             independent_confirmation=False,transfer_evaluated=False,deployment_changed=False)
    for name in groups[0]['result']['policies']:
        r=[g['result']['policies'][name] for g in groups]
        risk=[v['easy_selected_risk_upper'] for v in r if v['easy_selected_risk_upper'] is not None]
        out[name]=dict(selected=sum(v['selected_count'] for v in r),unknown_selected=sum(v['selected_unknown'] for v in r),
            complete_support=sum(v['finite_completion_supported'] for v in r),defined_easy_risk=len(risk),
            violations=sum(v>.02+1e-12 for v in risk),worst_easy_upper=max(risk) if risk else None)
    for key in groups[0]['result']['contrasts']:
        out[key]=interval([(g['source'],g['result']['contrasts'][key]) for g in groups],cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
    conditions=[out['quality_minus_'+o+'_signed_MSE']['CI95'][1]<0 for o in ('original','placebo')]
    conditions += [out['quality_minus_'+o+'_'+m+'_utility_percent']['CI95'][0]>0 for o in ('original','placebo') for m in ('full','matched')]
    conditions += [out['quality']['complete_support']>=out['original']['complete_support'],
                   out['quality']['violations']<=out['original']['violations'],
                   out['quality']['worst_easy_upper'] is not None and out['original']['worst_easy_upper'] is not None
                   and out['quality']['worst_easy_upper']<=out['original']['worst_easy_upper']]
    out['advance_to_transfer']=bool(all(conditions))
    return out
