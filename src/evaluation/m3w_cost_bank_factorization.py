"""Causal frozen-expert controls separate bank fitting from action competition."""
import numpy as np
from src.world_model.m3w_multiaction_cost import ARMS,select_actions

FAILURES=('nonpositive_gain','zero_reference','zero_easy_reference','all_harm_budget','easy_harm_budget')


def validate(p):
    p=np.asarray(p)
    if p.ndim!=3 or p.shape[1:]!=(7,5) or not np.isfinite(p).all() or (p<0).any():
        raise ValueError('Seven finite nonnegative predicted cost vectors required')
    return p


def factorized(physical,neural):
    physical,neural=validate(physical),validate(neural)
    if physical.shape!=neural.shape:raise ValueError('Aligned current-query costs and common units required')
    out=physical.copy();out[:,6]=neural[:,6]
    return out


def gate6(p,allowed):
    p=validate(p);allowed=np.asarray(allowed)
    if allowed.shape!=p.shape[:2] or allowed.dtype!=bool:
        raise ValueError('Past-only Boolean action availability required')
    b,h,r,er,eh=p[:,6].T
    failures=dict(zip(FAILURES,(b<=h,r<=0,er<=0,h>.02*r,eh>.02*er)))
    eligible=allowed[:,6];remaining=eligible.copy();exclusive={}
    for name,mask in failures.items():
        exclusive[name]=int((remaining & mask).sum());remaining &= ~mask
    exclusive['passed']=int(remaining.sum())
    bits=sum((1<<i)*mask.astype(np.int64) for i,mask in enumerate(failures.values()))
    values,counts=np.unique(bits[eligible],return_counts=True)
    summary=dict(queries=len(p),eligible=int(eligible.sum()),unavailable=int((~eligible).sum()),
        passing=int(remaining.sum()),overlapping_failures={k:int((eligible & m).sum()) for k,m in failures.items()},
        exclusive_failures=exclusive,failure_bit_patterns={str(v):int(n) for v,n in zip(values,counts)},
        exclusive_order=list(FAILURES),bit_pattern_order=list(FAILURES))
    return remaining,failures,summary


def policies(predictions,allowed):
    if set(predictions)!=set(ARMS):raise ValueError('All three frozen original bank estimators required')
    predictions={k:validate(v) for k,v in predictions.items()}
    if len({p.shape for p in predictions.values()})!=1:raise ValueError('Aligned frozen bank predictions required')
    choices,ranks={},{}
    for name,p in predictions.items():choices[name],ranks[name]=select_actions(p,allowed,arm=name)
    combined=factorized(predictions['physical_only'],predictions['neural_only'])
    choices['factorized'],ranks['factorized']=select_actions(combined,allowed,arm='physical_neural')
    choices['mixed_neural_restricted'],ranks['mixed_neural_restricted']=select_actions(
        predictions['physical_neural'],allowed,arm='neural_only')
    neutral=factorized(predictions['physical_neural'],predictions['physical_neural'])
    choices['neutral'],ranks['neutral']=select_actions(neutral,allowed,arm='physical_neural')
    np.testing.assert_array_equal(neutral,predictions['physical_neural'])
    np.testing.assert_array_equal(choices['neutral'],choices['physical_neural'])
    np.testing.assert_array_equal(ranks['neutral'],ranks['physical_neural'])
    mixed_pass,mixed_fail,mixed=gate6(predictions['physical_neural'],allowed)
    neural_pass,neural_fail,dedicated=gate6(predictions['neural_only'],allowed)
    np.testing.assert_array_equal(mixed_pass,choices['mixed_neural_restricted']==6)
    np.testing.assert_array_equal(neural_pass,choices['neural_only']==6)
    suppressed=neural_pass & ~mixed_pass
    competition=mixed_pass & (choices['physical_neural']!=6)
    choices['suppressed_neural_expert']=np.where(suppressed,6,-1)
    ranks['suppressed_neural_expert']=np.where(suppressed,ranks['neural_only'],0.)
    choices['competition_lost']=np.where(competition,6,-1)
    ranks['competition_lost']=np.where(competition,ranks['mixed_neural_restricted'],0.)
    choices['floor']=np.full(len(mixed_pass),-1,np.int64);ranks['floor']=np.zeros(len(mixed_pass))
    diagnostics=dict(mixed=mixed,dedicated=dedicated,
        neural_expert_pass_mixed_gate_reject=int(suppressed.sum()),
        mixed_gate_pass_physical_competition_lost=int(competition.sum()),
        mixed_gate_pass_and_neural_selected=int((choices['physical_neural']==6).sum()),
        neural_expert_pass_mixed_reject_components={k:int((suppressed & mask).sum()) for k,mask in mixed_fail.items()},
        mixed_gate_pass_neural_expert_reject=int((mixed_pass & ~neural_pass).sum()),
        label_oracle_used_for_choices=False,neutral_predictions_and_choices_exact=True)
    return choices,ranks,{**predictions,'factorized':combined},diagnostics
