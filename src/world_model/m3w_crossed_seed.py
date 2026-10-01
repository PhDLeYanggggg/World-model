"""Causal actions and offline accounting for a fixed-upstream seed control."""
import numpy as np
from src.world_model import m3w_inner_separability as core

SEEDS=(17,29,43)


def actions(predictions,moving,support,recordings,frames,ids,source_pass):
    if set(predictions)!=set(SEEDS) or set(source_pass)!=set(SEEDS):
        raise ValueError('All registered seeds are required, with no winner selection')
    if any(type(v) not in (bool,np.bool_) for v in source_pass.values()):
        raise ValueError('Source-only Boolean screen required')
    out={}
    for s in (17,29):
        pair=core.decisions(dict(affine=predictions[43],nonlinear=predictions[s]),
                            moving,support,recordings,frames,ids)
        out['seed'+str(s)]=pair['nonlinear']
        out['seed'+str(s)+'_matched']=pair['nonlinear_matched']
        out['ref43_matched'+str(s)]=pair['affine_matched']
        if 'seed43' in out:
            np.testing.assert_array_equal(out['seed43'],pair['affine'])
        out['seed43']=pair['affine']
    for s in SEEDS:
        out['screen'+str(s)]=out['seed'+str(s)].copy() if source_pass[s] else np.zeros(len(ids),bool)
    return out


def component_accounting(target,prediction,action):
    """Outcome-conditioned descriptive sums; never used by causal actions."""
    y,p,a=np.asarray(target,float),np.asarray(prediction,float),np.asarray(action)
    if (y.ndim!=2 or y.shape[1]!=5 or p.shape!=y.shape or a.shape!=(len(y),)
            or a.dtype!=bool or not np.isfinite(p).all() or (p<0).any()):
        raise ValueError('Aligned moments and fixed Boolean actions required')
    known=np.isfinite(y).all(1)
    if not (known | np.isnan(y).all(1)).all() or (y[known]<0).any():
        raise ValueError('Known nonnegative moments or wholly unknown rows required')
    take=known&a;actual=y[take].sum(0);predicted=p[take].sum(0)
    predicted_excess=predicted[4]-.02*predicted[3]
    actual_excess=actual[4]-.02*actual[3]
    harm_error=actual[4]-predicted[4]
    reference_error=-.02*(actual[3]-predicted[3])
    np.testing.assert_allclose(actual_excess-predicted_excess,harm_error+reference_error,rtol=1e-10,atol=1e-8)
    return dict(known_selected=int(take.sum()),unknown_selected=int((a&~known).sum()),
        predicted_known_selected_moments=predicted.tolist(),actual_known_selected_moments=actual.tolist(),
        predicted_all_selected_moments=p[a].sum(0).tolist(),
        predicted_easy_budget_excess= float(predicted_excess),actual_easy_budget_excess=float(actual_excess),
        easy_harm_error_contribution=float(harm_error),easy_reference_error_contribution=float(reference_error),
        actual_easy_risk=float(actual[4]/actual[3]) if actual[3]>0 else None,
        predicted_easy_risk=float(predicted[4]/predicted[3]) if predicted[3]>0 else None,
        used_for_inference=False)


def risk_status(ratio):
    if ratio is None:
        return 'undefined_not_pass'
    if not np.isfinite(ratio) or ratio<0:
        raise ValueError('Nonnegative finite risk or explicitly undefined required')
    return 'violating' if ratio>.02+1e-10 else 'defined_within_budget'


def assert_shared_fit(original,new):
    for k in ('preprocess','settings','input_hashes','sklearn','objective',
              'known_training_rows','unknown_training_rows'):
        core.exact(original[k],new[k])
    assert original['seed']==43 and new['seed'] in (17,29)
    assert len(original['model'].estimators_)==len(new['model'].estimators_)==128
