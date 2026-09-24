"""Versioned numerical repair; the frozen V1 functions remain unchanged."""
from dataclasses import replace
import numpy as np
from src.world_model.m3w_easy_allocation import selected_budget, ARMS
from src.world_model.m3w_interaction_controls import decompose_pair_objective, _describe
from src.world_model.m3w_scaled_risk_controls import solve_scaled_risk_control as solve_control
from src.world_model.m3w_native_joint_controls import make_problem
from src.evaluation.m3w_native_scene_alignment import restore

def allocate(problem, q, r, target, *, cost_scale, pointwise, strict, rho=.02, seconds=5.):
    """No observed errors, labels, future masks or test statistics in this API."""
    problem.validate()
    q, r = np.asarray(q, float), np.asarray(r, float)
    target, pointwise, strict = map(np.asarray, (target, pointwise, strict))
    n = len(q)
    if (r.shape != (n,) or problem.supported.shape != (n,)
            or not np.isfinite(q).all() or not np.isfinite(r).all() or (q<0).any() or (r<0).any()
            or any(v.shape!=(n,) or v.dtype!=bool for v in (target,pointwise,strict))
            or not target.any() or np.any(problem.supported & ~target)
            or np.any(pointwise & ~problem.supported) or np.any(strict & ~problem.supported)
            or np.any(q[~target]!=0) or np.any(r[~target]!=0)
            or not np.isfinite(cost_scale) or cost_scale<=0 or rho!=.02
            or not np.isfinite(seconds) or seconds<=0):
        raise ValueError('Aligned causal query moments and fixed tolerance required')
    budget = rho*float(r.sum())
    gain = np.maximum(problem.expected_gain,0)
    p = replace(problem, expected_gain=gain, expected_harm=q/cost_scale,
        max_mean_predicted_harm=budget/(n*cost_scale), max_interventions=int(problem.supported.sum()))
    p.validate()
    parts = decompose_pair_objective(p)
    chosen = dict(floor=np.zeros(n,bool), net_stop=p.supported.copy(),
                  strict_stop=strict.copy(), pointwise=pointwise.copy())
    chosen['aggregate_selected'], ok, reason = selected_budget(
        gain, q/cost_scale, r/cost_scale, p.supported, rho, seconds)
    info = dict(selected_optimal=ok, selected_reason=reason, population_budget=budget,
                denominator_target_rows=int(target.sum()))
    # Positive-risk candidates individually exceeding the entire budget are infeasible.
    feasible_support = p.supported & (q<=budget)
    reduced = replace(p, supported=feasible_support, max_interventions=int(feasible_support.sum()))
    if q[feasible_support].sum()<=budget:
        ref = dict(switch=feasible_support.copy(), solver_optimal=True,
                   reason='all_individually_feasible_candidates_fit', **_describe(p, feasible_support, parts))
    else:
        ref = solve_control(reduced, objective_kind='independent', time_limit_seconds=seconds)
    chosen['aggregate_population'] = ref['switch']
    k = int(ref['switch'].sum())
    feasible = bool(np.all(~target | p.supported) and q[target].sum()<=budget)
    chosen['scene_uniform'] = target.copy() if feasible else np.zeros(n,bool)
    unique = k in (0, int(feasible_support.sum()))
    if unique:
        unary = dict(ref, switch=ref['switch'].copy(), reason='unique_assignment_at_reference_count')
    else:
        unary = solve_control(reduced, objective_kind='unary_geometry', exact_interventions=k,
                              time_limit_seconds=seconds)
    active = feasible_support[p.edges].all(1) & (parts['product_objective']!=0)
    if unique or k<=1 or not active.any():
        joint = dict(unary, switch=unary['switch'].copy(), reason='joint_same_feasible_objective')
    else:
        joint = solve_control(reduced, objective_kind='joint', exact_interventions=k,
                              time_limit_seconds=seconds)
    chosen['aggregate_unary'], chosen['aggregate_joint'] = unary['switch'], joint['switch']
    matched = all(v['solver_optimal'] and int(v['switch'].sum())==k
                  and float(q@v['switch'])<=budget+1e-8 for v in (ref,unary,joint))
    info.update(population_optimal=ref['solver_optimal'], unary_optimal=unary['solver_optimal'],
        joint_optimal=joint['solver_optimal'], matched=bool(matched), reference_count=k,
        nonadditive_edges=int(active.sum()) if k>=2 else 0,
        joint_unary_changed_agents=int(np.count_nonzero(joint['switch']!=unary['switch'])))
    info['policies']={name:dict(selected=int(bits.sum()),
        predicted_easy_harm=float(q@bits), selected_easy_denominator=float(r@bits),
        population_budget_satisfied=bool(q@bits<=budget+1e-8),
        expected_gain=float(np.sum(gain*bits)),
        pair_proxy=float(p.pair_cost[np.arange(len(p.edges)),bits[p.edges[:,0]].astype(int),bits[p.edges[:,1]].astype(int)].mean()) if len(p.edges) else 0.)
        for name,bits in chosen.items()}
    return chosen, info

def query(rows,c,ids_to_view,values,prediction,data,scale,cut,action,cfg):
    target=c['context_target_rows'][rows]>=0
    tid=c['context_target_rows'][rows][target];loc=ids_to_view[tid]
    assert (loc>=0).all()
    n=len(rows);valid=c['context_cv_valid'][rows].all(1)
    b=c['context_cv_rollout'][rows].copy();candidate=b.copy()
    b[target]=restore(data['geometry'][tid,332:356].reshape(-1,12,2),data['origin'][tid],data['rotation'][tid],data['scale'][tid])
    candidate[target]=restore(prediction[loc],data['origin'][tid],data['rotation'][tid],data['scale'][tid])
    valid[target]=True
    scores=values[action+'__score'][loc];moment=values[action+'__moments'][loc]
    support=np.zeros(n,bool);support[target]=values[action+'__net_stop'][loc]
    benefit=np.zeros(n);harm=np.zeros(n);benefit[target],harm[target]=scores[:,0],scores[:,1]
    p,geo=make_problem(baseline=b,candidate=candidate,current=c['context_xy'][rows],forecast_valid=valid,
        target_mask=target,eligible=support,benefit=benefit,harm=harm,scale=scale,past_target_scales=data['scale'][tid])
    q=np.zeros(n);r=np.zeros(n);q[target]=moment[:,0]*values[action+'__distance'][loc];r[target]=moment[:,1]*cut
    point=np.zeros(n,bool);strict=point.copy()
    point[target]=values[action+'__joint_easy_moment'][loc];strict[target]=values[action+'__strict_stop'][loc]
    choices,report=allocate(p,q,r,target,cost_scale=scale,pointwise=point,strict=strict,
                            rho=cfg['easy_rho'],seconds=cfg['solver_seconds'])
    report['geometry']=geo
    return tid,np.column_stack([choices[k][target] for k in ARMS]),report

