"""Matched native-utility allocation for two newly fitted continuous cost models."""
from collections import Counter
import numpy as np
from src.world_model import m3w_frame_action_allocation as base
from src.world_model.m3w_selected_query_critic import digest

ARMS=('locked_references','learned_references')
NAMES=tuple(a+'_'+b for a in ('locked','learned') for b in ('independent','joint','matched'))


def allocate(predictions,allowed,recordings,frames,ids,parent_joint,*,seconds,checkpoint=lambda _:None,heartbeat=lambda **_:None,resume_state=None):
    if set(predictions)!=set(ARMS):raise ValueError('Both frozen fit endpoints required')
    p={a:base.validate(v,allowed)[0] for a,v in predictions.items()};groups=base.query_groups(recordings,frames,ids)
    choices={a:np.full(len(ids),-1,np.int64) for a in NAMES};choices['parent_joint']=np.asarray(parent_joint).copy()
    for prefix,arm in zip(('locked','learned'),ARMS):choices[prefix+'_independent'],_=base.select_actions(p[arm],allowed,arm='physical_neural')
    bindings={a:digest(v) for a,v in dict(**predictions,allowed=allowed,recordings=recordings,frames=frames,ids=ids,parent_joint=parent_joint).items()};bindings['seconds']=seconds
    reasons={a:Counter() for a in NAMES if a.endswith(('joint','matched'))};start=0;calls=0;nonoptimal=Counter()
    if resume_state is not None:
        if resume_state['bindings']!=bindings:raise ValueError('Frozen refit allocation inputs changed')
        for a in ('locked_independent','learned_independent','parent_joint'):np.testing.assert_array_equal(resume_state['choices'][a],choices[a])
        choices={a:v.copy() for a,v in resume_state['choices'].items()};start=resume_state['completed_queries'];calls=resume_state['solver_calls']
        reasons={a:Counter(v) for a,v in resume_state['reasons'].items()};nonoptimal=Counter(resume_state['nonoptimal'])
        if not 0<=start<=len(groups):raise ValueError('Valid completed query count required')
        for g in groups[:start]:
            for prefix,arm in zip(('locked','learned'),ARMS):
                for mode in ('independent','joint','matched'):
                    if not base.feasible(p[arm][g],allowed[g],choices[prefix+'_'+mode][g]):raise ValueError('Invalid completed query checkpoint')
    def save(count):checkpoint(dict(bindings=bindings,completed_queries=count,choices={a:v.copy() for a,v in choices.items()},solver_calls=calls,reasons={a:dict(v) for a,v in reasons.items()},nonoptimal=dict(nonoptimal)))
    for i in range(start,len(groups)):
        g=groups[i];counts={a:int((choices[a+'_independent'][g]>=0).sum()) for a in ('locked','learned')};common=min(counts.values())
        for prefix,arm in zip(('locked','learned'),ARMS):
            q=p[arm][g];anchor=choices[prefix+'_independent'][g]
            own,info=base.solve_frame(q,allowed[g],anchor,exact_count=counts[prefix],seconds=seconds);calls+=1
            choices[prefix+'_joint'][g]=own;reasons[prefix+'_joint'][info['reason']]+=1;nonoptimal[prefix+'_joint']+=int(not info['optimal'])
            if counts[prefix]==common:matched,mi=own,info
            else:
                at=np.flatnonzero(anchor>=0);gain=q[at,anchor[at],0]-q[at,anchor[at],1]
                keep=at[np.argsort(-gain,kind='stable')[:common]];trimmed=np.full(len(g),-1,np.int64);trimmed[keep]=anchor[keep]
                matched,mi=base.solve_frame(q,allowed[g],trimmed,exact_count=common,seconds=seconds);calls+=1
            choices[prefix+'_matched'][g]=matched;reasons[prefix+'_matched'][mi['reason']]+=1;nonoptimal[prefix+'_matched']+=int(not mi['optimal'])
        if i==0 or (i+1)%250==0 or i+1==len(groups):save(i+1);heartbeat(queries=i+1,total_queries=len(groups))
    if any((choices['locked_matched'][g]>=0).sum()!=(choices['learned_matched'][g]>=0).sum() for g in groups):raise AssertionError('Each query must match counts')
    ranks={a:np.zeros(len(ids)) for a in choices}
    for name,c in choices.items():
        if name=='parent_joint':continue
        arm=ARMS[0] if name.startswith('locked') else ARMS[1];at=np.flatnonzero(c>=0);ranks[name][at]=p[arm][at,c[at],0]-p[arm][at,c[at],1]
    return choices,ranks,dict(queries=len(groups),solver_calls=calls,reason_counts={a:dict(v) for a,v in reasons.items()},nonoptimal=dict(nonoptimal),same_count_controls_every_query=True,labels_used=False)
