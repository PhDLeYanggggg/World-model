"""Label-aware necessary easy-anchor feasibility under frozen query budgets."""
from collections import Counter
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import coo_array,csc_array,vstack
from src.world_model import m3w_reference_action_support as support


def query_witness(prediction,allowed,known_easy,*,seconds):
    p,ok=support.original.validate(prediction,allowed);easy=np.asarray(known_easy)
    if easy.shape!=(len(p),) or easy.dtype!=bool or not np.isfinite(seconds) or seconds<=0:
        raise ValueError('Aligned diagnostic easy-label mask and fixed positive runtime required')
    legal=support.reference_actions(p,ok);anchors=legal&easy[:,None]
    fallback=np.full(len(p),-1,np.int64)
    def answer(choice,status,reason,solver_status=None):
        selected=np.flatnonzero(choice>=0)
        if status=='feasible':
            if not support.feasible(p,ok,choice) or not easy[selected].any():raise AssertionError('Exact original predicted budgets and known easy support required')
        return dict(status=status,reason=reason,solver_status=solver_status,
            selected=int(len(selected)),selected_known_easy=int(easy[selected].sum()),
            witness_uses_future_easy_label=True,not_a_deployable_policy=True),choice
    if not anchors.any():return answer(fallback,'infeasible','no_legal_known_easy_action')
    simple=support.pointwise_actions(p,ok)&easy[:,None]
    if simple.any():
        i,a=np.argwhere(simple)[0];out=fallback.copy();out[i]=a
        return answer(out,'feasible','single_pointwise_anchor')
    if len(p)==1:return answer(fallback,'infeasible','singleton_exact_no_anchor')
    risk=np.stack((p[:,:,1]-.02*p[:,:,2],p[:,:,4]-.02*p[:,:,3]),-1)
    # Nonnegative query-wide costs cannot offset an anchor's positive excess.
    blocked=np.zeros_like(anchors)
    for k in range(2):
        if (risk[:,:,k][legal]>=0).all():blocked|=anchors&(risk[:,:,k]>0)
    if (blocked|~anchors).all():return answer(fallback,'infeasible','nonnegative_cost_sign_certificate')
    rows,actions=np.nonzero(legal);n=len(rows)
    per_agent=coo_array((np.ones(n),(rows,np.arange(n))),shape=(len(p),n)).tocsc()
    costs=risk[rows,actions].T;scale=np.max(np.abs(costs),axis=1).clip(1e-300)
    # Binary support avoids an arbitrary epsilon on tiny native reference values.
    matrix=vstack((per_agent,csc_array(costs/scale[:,None]),csc_array(easy[rows][None,:].astype(float))),format='csc')
    low=np.r_[np.full(len(p)+2,-np.inf),1.];high=np.r_[np.ones(len(p)),np.zeros(2),np.inf]
    result=milp(c=np.zeros(n),integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),
        constraints=LinearConstraint(matrix,low,high),options=dict(presolve=True,time_limit=seconds,mip_rel_gap=0.))
    if result.status==2:return answer(fallback,'infeasible','milp_status2_numerical_infeasibility',2)
    if result.status!=0 or result.x is None:return answer(fallback,'inconclusive','solver_not_optimal',int(result.status))
    selected=np.flatnonzero(result.x>.5);out=fallback.copy();out[rows[selected]]=actions[selected]
    if (len(np.unique(rows[selected]))!=len(selected) or not support.feasible(p,ok,out)
            or not easy[np.flatnonzero(out>=0)].any()):
        return answer(fallback,'inconclusive','original_exact_postcheck_failed',0)
    return answer(out,'feasible','exact_budget_postchecked_milp',0)


def recording_diagnosis(packet,prediction,*,seconds,heartbeat=lambda **_:None):
    p,ok=support.original.validate(prediction,packet['allowed']);y=packet['y']
    easy=np.isfinite(y).all((1,2))&(y[:,0,3]>0)
    counts=Counter();source_count=0;queries=support.original.query_groups(packet['recordings'],packet['frames'],packet['ids'])
    witnessed=[];details=[]
    for index,g in enumerate(queries):
        if easy[g].any():
            result,choice=query_witness(p[g],ok[g],easy[g],seconds=seconds);counts[result['reason']]+=1
            details.append(result)
            if result['status']=='feasible':witnessed.append(index)
        else:source_count+=1
        if index==0 or (index+1)%25==0 or index+1==len(queries):heartbeat(queries=index+1,total_queries=len(queries))
    conclusive=bool(witnessed) or not any(r['status']=='inconclusive' for r in details)
    algebraic=sum(r['reason'] in ('singleton_exact_no_anchor','nonnegative_cost_sign_certificate','no_legal_known_easy_action') for r in details)
    return dict(queries=len(queries),queries_without_known_easy_rows=source_count,queries_with_known_easy_rows=len(details),
        feasible_known_easy_anchor_queries=len(witnessed),inconclusive_queries=sum(r['status']=='inconclusive' for r in details),
        infeasible_queries=sum(r['status']=='infeasible' for r in details),reason_counts=dict(counts),
        algebraic_or_singleton_infeasible_queries=algebraic,
        necessary_query_easy_support=bool(witnessed),necessary_condition_conclusive=conclusive,
        complete_easy_support_obstruction=conclusive and not witnessed,
        obstruction_algebraic_or_singleton_only=conclusive and not witnessed and algebraic==len(details),
        label_aware_necessary_condition_only=True,positive_actual_safe_gain_proven=False,
        current_query_targets_not_all_visible_agents=True,original2percent_budgets_unchanged=True,
        raw_choices_or_geometry_exported=False)
