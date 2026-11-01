"""Learned utility with frozen two-percent causal cost constraints and all seven actions."""
from collections import Counter
import math
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import coo_array,csc_array,vstack
from src.world_model import m3w_frame_action_allocation as base
from src.world_model.m3w_selected_query_critic import digest

ARMS=('self_context','selected_context')
NAMES=tuple(a+'_'+b for a in ('zero','self','selected') for b in ('independent','joint','matched'))


def valid_actions(p,allowed,utility):
    p,ok=base.validate(p,allowed);u=np.asarray(utility,float)
    if u.shape!=ok.shape or not np.isfinite(u).all():raise ValueError('Finite seven-action native routing utility required')
    return base.positive_actions(p,ok)&(u>0)


def independent(p,allowed,utility):
    legal=valid_actions(p,allowed,utility)&(p[:,:,1]<=.02*p[:,:,2])&(p[:,:,4]<=.02*p[:,:,3])
    best=np.where(legal,utility,-np.inf).argmax(1)
    return np.where(legal.any(1),best,-1).astype(np.int64)


def feasible(p,allowed,u,choice):
    if not base.feasible(p,allowed,choice):return False
    at=np.flatnonzero(choice>=0)
    return bool(valid_actions(p,allowed,u)[at,choice[at]].all())


def objective(u,choice):
    at=np.flatnonzero(choice>=0);return math.fsum(np.asarray(u)[at,choice[at]].tolist())


def solve(p,allowed,u,parent,*,count,seconds):
    legal=valid_actions(p,allowed,u);parent=np.asarray(parent)
    if (not feasible(p,allowed,u,parent) or not isinstance(count,int) or count!=int((parent>=0).sum())
            or not np.isfinite(seconds) or seconds<=0):raise ValueError('Feasible exact-count anchor and fixed runtime required')
    anchor=objective(u,parent)
    def result(choice,reason,optimal):
        if not feasible(p,allowed,u,choice) or int((choice>=0).sum())!=count:raise AssertionError('No constraint/count violation may escape')
        return choice.copy(),dict(reason=reason,optimal=optimal,selected=count,
            routing_objective=objective(u,choice),predicted_budget_excess=base.terms(p,choice)[1],labels_used=False)
    if count==0:return result(parent,'zero_count_exact',True)
    best=np.where(legal,u,-np.inf).argmax(1);candidate=np.where(legal.any(1),best,-1)
    at=np.flatnonzero(candidate>=0);keep=at[np.argsort(-u[at,candidate[at]],kind='stable')[:count]]
    mask=np.zeros(len(p),bool);mask[keep]=True;candidate[~mask]=-1
    if feasible(p,allowed,u,candidate):return result(candidate,'unconstrained_optimum_feasible',True)
    if len(p)==1:return result(parent,'singleton_parent_exact',True)
    row,action=np.nonzero(legal);m=len(row)
    per_agent=coo_array((np.ones(m),(row,np.arange(m))),shape=(len(p),m)).tocsc()
    risk=np.stack((p[row,action,1]-.02*p[row,action,2],p[row,action,4]-.02*p[row,action,3]))
    scale=np.max(np.abs(risk),axis=1).clip(1e-300)
    matrix=vstack((per_agent,csc_array(risk/scale[:,None]),csc_array(np.ones((1,m)))),format='csc')
    low=np.r_[np.full(len(p)+2,-np.inf),count];high=np.r_[np.ones(len(p)),np.zeros(2),count]
    value=milp(c=-u[row,action]/u[row,action].max(),integrality=np.ones(m),bounds=Bounds(np.zeros(m),np.ones(m)),
        constraints=LinearConstraint(matrix,low,high),options=dict(presolve=True,mip_rel_gap=0.,time_limit=seconds))
    if value.status!=0 or value.x is None:return result(parent,f'solver_status_{value.status}_parent_fallback',False)
    selected=np.flatnonzero(value.x>.5);out=np.full(len(p),-1,np.int64);out[row[selected]]=action[selected]
    if (len(np.unique(row[selected]))!=len(selected) or int((out>=0).sum())!=count or not feasible(p,allowed,u,out)
            or objective(u,out)<anchor):return result(parent,'postcheck_failed_parent_fallback',False)
    return result(out,'exact_milp',True)


def allocate(p,allowed,utilities,recordings,frames,ids,zero_control,*,seconds,checkpoint=lambda _:None,heartbeat=lambda **_:None,resume_state=None):
    base.validate(p,allowed)
    if set(utilities)!=set(ARMS):raise ValueError('Both matched learned-utility arms required')
    for u in utilities.values():valid_actions(p,allowed,u)
    groups=base.query_groups(recordings,frames,ids);zero=p[:,:,0]-p[:,:,1]
    utility=dict(zero=zero,self=utilities['self_context'],selected=utilities['selected_context'])
    choices={a:np.full(len(ids),-1,np.int64) for a in NAMES}
    for prefix,u in utility.items():choices[prefix+'_independent']=independent(p,allowed,u)
    np.testing.assert_array_equal(choices['zero_independent'],zero_control['independent'])
    choices['zero_joint']=np.asarray(zero_control['joint']).copy()
    for g in groups:
        if not feasible(p[g],allowed[g],zero[g],choices['zero_joint'][g]):raise ValueError('Frozen zero-gate teacher is not feasible')
        if (choices['zero_joint'][g]>=0).sum()!=(choices['zero_independent'][g]>=0).sum():raise ValueError('Original per-query count changed')
    binds={k:digest(v) for k,v in dict(prediction=p,allowed=allowed,self_utility=utility['self'],selected_utility=utility['selected'],
        recordings=recordings,frames=frames,ids=ids,zero_joint=choices['zero_joint']).items()};binds.update(seconds=seconds,version=1)
    start=0;calls=0;reasons={a:Counter() for a in NAMES if a.endswith(('joint','matched'))};nonoptimal=Counter()
    if resume_state is not None:
        if resume_state['bindings']!=binds:raise ValueError('Same frozen inputs/actor utilities required for resume')
        start=resume_state['completed_queries']
        if not 0<=start<=len(groups):raise ValueError('Invalid resume query position')
        for a in ('zero_independent','zero_joint','self_independent','selected_independent'):
            np.testing.assert_array_equal(resume_state['choices'][a],choices[a])
        choices={a:np.asarray(v).copy() for a,v in resume_state['choices'].items()}
        reasons={a:Counter(v) for a,v in resume_state['reasons'].items()};nonoptimal=Counter(resume_state['nonoptimal']);calls=resume_state['solver_calls']
        for g in groups[:start]:
            for name,c in choices.items():
                if not feasible(p[g],allowed[g],utility[name.split('_')[0]][g],c[g]):raise ValueError('Invalid completed query checkpoint')
    def save(completed):
        checkpoint(dict(bindings=binds,completed_queries=completed,choices={a:v.copy() for a,v in choices.items()},
            reasons={a:dict(v) for a,v in reasons.items()},nonoptimal=dict(nonoptimal),solver_calls=calls))
    for index in range(start,len(groups)):
        g=groups[index];counts={a:int((choices[a+'_independent'][g]>=0).sum()) for a in utility};common=min(counts.values())
        for prefix,u in utility.items():
            anchor=choices[prefix+'_independent'][g]
            if prefix=='zero':own=choices['zero_joint'][g];info=dict(reason='cached_verified_zero_gate_full_policy',optimal=False)
            else:
                own,info=solve(p[g],allowed[g],u[g],anchor,count=counts[prefix],seconds=seconds);calls+=1
                choices[prefix+'_joint'][g]=own;nonoptimal[prefix+'_joint']+=int(not info['optimal'])
            reasons[prefix+'_joint'][info['reason']]+=1
            if common==counts[prefix]:matched,mi=own,info
            else:
                at=np.flatnonzero(anchor>=0);keep=at[np.argsort(-u[g][at,anchor[at]],kind='stable')[:common]]
                trimmed=np.full(len(g),-1,np.int64);trimmed[keep]=anchor[keep]
                matched,mi=solve(p[g],allowed[g],u[g],trimmed,count=common,seconds=seconds);calls+=1
            choices[prefix+'_matched'][g]=matched;reasons[prefix+'_matched'][mi['reason']]+=1
            if mi['reason']!='cached_verified_zero_gate_full_policy':nonoptimal[prefix+'_matched']+=int(not mi['optimal'])
        if index==0 or (index+1)%250==0 or index+1==len(groups):save(index+1);heartbeat(queries=index+1,total_queries=len(groups))
    if any(len({int((choices[a+'_matched'][g]>=0).sum()) for a in utility})!=1 for g in groups):raise AssertionError('All count controls must match each query')
    ranks={}
    for name,c in choices.items():
        u=utility[name.split('_')[0]];at=np.flatnonzero(c>=0);rank=np.zeros(len(ids));rank[at]=u[at,c[at]];ranks[name]=rank
    return choices,ranks,dict(queries=len(groups),solver_calls=calls,reason_counts={a:dict(v) for a,v in reasons.items()},
        nonoptimal_queries=dict(nonoptimal),same_count_controls_every_query=True,frozen_moments=True,labels_used=False,
        zero_full_control='cached_verified_not_reoptimized',routing_scores_are_not_calibrated_costs=True,
        full_original_recording_risk_unchanged=True,matched_role='utility_only_not_deployment_safety')
