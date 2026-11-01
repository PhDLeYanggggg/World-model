"""Unbounded routing scores separate from frozen continuous cost moments."""
import copy
import hashlib
from pathlib import Path
import time

from src.world_model import m3w_decision_cost_critic as prior
from src.world_model import m3w_frame_action_allocation as allocation

np, torch, nn = prior.np, prior.torch, prior.nn
parent = prior.parent
ARMS = ('self_context', 'selected_context')
CAUSAL_KEYS = frozenset(('features', 'prediction', 'allowed', 'physical_allowed', 'native_gain',
    'scales', 'temperature', 'preprocess', 'teacher', 'sites', 'recordings', 'frames', 'ids',
    'groups', 'source_groups', 'keys'))


def preprocessing(raw, sites):
    mean, second = np.zeros(raw[ARMS[0]].shape[-1]), np.zeros(raw[ARMS[0]].shape[-1])
    names = sorted(set(sites))
    for site in names:
        at = np.flatnonzero(sites == site)
        for start in range(0,len(at),512):
            ix = at[start:start+512]
            for arm in ARMS:
                x = raw[arm][ix,:6].astype(float)
                mean += x.sum((0,1))/(len(names)*len(at)*6*len(ARMS))
                second += np.square(x).sum((0,1))/(len(names)*len(at)*6*len(ARMS))
    return dict(mean=mean,std=np.sqrt(np.maximum(second-mean*mean,0)).clip(1e-6),clip=8.,
        source_names=names,physical_actions_only=True,known_outcome_mask_used=False)


def causal_inputs(c, cost_state, prediction, teacher, *, preprocess=None, fitting=False):
    if set(c)!=parent.CAUSAL_KEYS: raise ValueError('Declared causal cost inputs only')
    p,teacher = np.asarray(prediction,float),np.asarray(teacher)
    allocation.validate(p,c['allowed'])
    if (teacher.shape!=c['ids'].shape or p.shape!=(len(teacher),7,5)
            or not np.issubdtype(teacher.dtype,np.integer) or ((teacher < -1)|(teacher >= 7)).any()):
        raise ValueError('Aligned frozen causal teacher required')
    model=prior.base.CoupledCritic(cost_state['dimension'],cost_state['settings']['width'],cost_state['seed'],'released_benefit')
    model.load_state_dict(cost_state['model']);model.eval();model.requires_grad_(False)
    n=len(teacher);width=cost_state['settings']['width'];raw={a:np.empty((n,7,2*width+5),np.float32) for a in ARMS}
    temperature=float(np.sqrt(cost_state['rms'][0]**2+cost_state['rms'][1]**2))
    if not np.isfinite(temperature) or temperature<=0: raise ValueError('Frozen fitting-only cost temperature required')
    pending=[];size=0
    def flush():
        rows=np.concatenate(pending);seg=np.repeat(np.arange(len(pending)),[len(g) for g in pending])
        x,b,env,_,segments,count=parent.batch(c,rows,seg,len(pending))
        onehot=torch.eye(7,dtype=x.dtype).expand(len(x),-1,-1)
        with torch.no_grad():
            h=model.encoder(torch.cat((x,torch.log1p(b.clamp_max(2980)).float(),onehot,
                torch.log1p(env.clamp_max(2980)).float()[:,:,None]),-1))
            peer,selected=parent.selected_context(h,torch.from_numpy(teacher[rows]),segments,count)
            sizes=torch.bincount(segments,minlength=count).float()[segments]
            meta=torch.stack((torch.log1p(selected),selected/sizes),1)[:,None].expand(-1,7,-1)
            q=p[rows];scale=c['scales'][rows,None]
            diag=np.stack((np.log1p(np.maximum(q[:,:,0]-q[:,:,1],0)/scale/temperature),
                np.log1p(q[:,:,1]/q[:,:,2].clip(1e-150)),np.log1p(q[:,:,4]/q[:,:,3].clip(1e-150))),-1)
            for arm in ARMS:
                context=h if arm=='self_context' else peer[:,None].expand_as(h)
                raw[arm][rows]=np.concatenate((h.numpy(),context.numpy(),meta.numpy(),diag.astype(np.float32)),-1)
    for g in c['groups']:
        if pending and size+len(g)>1024:flush();pending=[];size=0
        pending.append(g);size+=len(g)
    if pending:flush()
    if preprocess is None:
        if not fitting:raise ValueError('Inference cannot fit normalization statistics')
        preprocess=preprocessing(raw,c['sites'])
    if fitting and preprocess['source_names']!=sorted(set(c['sites'])):raise ValueError('Fitting-only preprocessing roles changed')
    features={a:np.clip((v.astype(float)-preprocess['mean'])/preprocess['std'],-preprocess['clip'],preprocess['clip']).astype(np.float32)
        for a,v in raw.items()}
    if any(not np.isfinite(v).all() for v in features.values()):raise ValueError('Finite causal gate features required')
    return dict(features=features,prediction=p,allowed=allocation.positive_actions(p,c['allowed']),physical_allowed=c['allowed'],
        native_gain=p[:,:,0]-p[:,:,1],scales=c['scales'],temperature=temperature,preprocess=preprocess,teacher=teacher,
        **{k:c[k] for k in ('sites','recordings','frames','ids','groups','source_groups','keys')})


def training_data(c, source):
    if set(c)!=CAUSAL_KEYS:raise ValueError('No outcome labels in actor causal schema')
    np.testing.assert_array_equal(c['ids'],source['causal']['ids'])
    layout=prior.training_layout(source);qi=np.empty(len(c['ids']),np.int64)
    for i,g in enumerate(c['groups']):qi[g]=i
    qscale=np.bincount(qi,weights=c['scales'])
    p=c['prediction'];qrisk=np.stack((p[:,:,1]-.02*p[:,:,2],p[:,:,4]-.02*p[:,:,3]),-1)
    hashes={a:parent.digest(x) for a,x in c['features'].items()}
    hashes.update({k:parent.digest(c[k]) for k in ('prediction','allowed','teacher','ids','scales')})
    hashes.update(source['input_hashes'])
    # Actor-specific hashes are separate: SOURCE cost inputs have a different prediction array.
    hashes['actor_prediction']=parent.digest(c['prediction']);hashes['actor_allowed']=parent.digest(c['allowed'])
    return dict(causal=c,layout=layout,query_index=qi,query_scales=qscale,predicted_query_risk=qrisk,
        known=source['known'],input_hashes=hashes,original_SOURCE_input_hashes=source['input_hashes'])


class AllocationActor(nn.Module):
    def __init__(self,dimension,width,seed):
        super().__init__();torch.manual_seed(seed)
        self.network=nn.Sequential(nn.Linear(dimension,width),nn.SiLU(),nn.Linear(width,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
    def forward(self,x):return self.network(x).squeeze(-1).double()


def probabilities(offset,c,rows):
    if set(c)!=CAUSAL_KEYS:raise ValueError('Only causal actor fields accepted')
    score=torch.from_numpy(c['native_gain'][rows]/c['scales'][rows,None]/c['temperature'])+offset
    score=score.masked_fill(~torch.from_numpy(c['allowed'][rows]),-torch.inf)
    return torch.cat((torch.zeros_like(score[:,:1]),score),1).softmax(1)


def losses(offset,data,rows,segments,count,record_dual,query_dual):
    c=data['causal'];layout=data['layout'];rows=np.asarray(rows);p=probabilities(offset,c,rows)[:,1:]
    rec=torch.from_numpy(layout['record'][rows]);qid=torch.from_numpy(data['query_index'][rows])
    gain=-(p*torch.from_numpy(layout['gain'][rows])).sum(1)*torch.from_numpy(layout['objective_weight'][rows])
    record=(p[:,:,None]*torch.from_numpy(layout['risk'][rows])*record_dual.detach()[rec,None]).sum((1,2))*torch.from_numpy(layout['risk_weight'][rows])
    query=(p[:,:,None]*torch.from_numpy(data['predicted_query_risk'][rows])*query_dual.detach()[qid,None]).sum((1,2))/torch.from_numpy(data['query_scales'][data['query_index'][rows]])
    total=p.new_zeros(count).index_add(0,segments,gain+record+query).mean()
    return dict(total=total,expected_gain_cost=gain.sum()/count,record_risk_cost=record.sum()/count,
        causal_query_risk_cost=query.sum()/count,unknown_bound_rows=int((~data['known'][rows]).sum()),unknown_supervised_labels=0)


def full_risk(model,data,arm,chunk_rows=2048):
    c=data['causal'];lay=data['layout'];rr=np.zeros((len(lay['record_keys']),3));qr=np.zeros((len(c['groups']),2));mass=0.
    for start in range(0,len(c['ids']),chunk_rows):
        rows=np.arange(start,min(start+chunk_rows,len(c['ids'])))
        with torch.no_grad():p=probabilities(model(torch.from_numpy(c['features'][arm][rows])),c,rows)[:,1:].numpy()
        np.add.at(rr,lay['record'][rows],(p[:,:,None]*lay['risk'][rows]).sum(1))
        np.add.at(qr,data['query_index'][rows],(p[:,:,None]*data['predicted_query_risk'][rows]).sum(1));mass+=float(p.sum())
    rr[:,2]-=lay['easy_constant']
    return rr/lay['record_scales'][:,None],qr/data['query_scales'][:,None],mass/len(c['ids'])


def fit(data,*,arm,seed,settings,decision,identity,path,heartbeat=lambda **_:None,guard=lambda:None,resume=False,stop_at=None):
    if arm not in ARMS:raise ValueError('Registered context arm required')
    c=data['causal'];model=AllocationActor(c['features'][arm].shape[-1],settings['width'],seed)
    initial=copy.deepcopy(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    rng=torch.Generator().manual_seed(seed+7919);monitor=parent.draw_queries(c['groups'],c['source_groups'],settings['query_batch_size'],torch.Generator().manual_seed(seed+9137))
    rd=torch.zeros((len(data['layout']['record_keys']),3),dtype=torch.float64);qd=torch.zeros((len(c['groups']),2),dtype=torch.float64)
    binding=dict(arm=arm,seed=seed,settings=settings,decision=decision,identity=identity,input_hashes=data['input_hashes'],
        initial_model=initial,preprocess=c['preprocess'],temperature=c['temperature'],dimension=c['features'][arm].shape[-1],parameters=sum(p.numel() for p in model.parameters()),
        query_keys=c['keys'],frozen_cost_moments=True,training_outcomes_are_not_inputs=True)
    step,seconds,trace,draw_hash,row_draws,unknown_draws=0,0.,[],'0'*64,0,0;path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not resume:raise ValueError('Explicit same-definition resume required')
        old=parent.read_checkpoint(path)
        for k,v in binding.items():parent.exact(old[k],v)
        model.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);rng.set_state(old['sampler_rng']);torch.set_rng_state(old['torch_rng'])
        step,seconds,trace,draw_hash,row_draws,unknown_draws,rd,qd=(old[k] for k in ('step','seconds','trace','draw_hash','row_draws','unknown_bound_rows_drawn','record_dual','query_dual'))
    limit=settings['steps'] if stop_at is None else min(settings['steps'],stop_at)
    if not 0<=step<=limit or limit<=0:raise ValueError('Fixed nondecreasing update budget required')
    began=time.monotonic()
    def diagnostic():
        rows,seg,_=monitor
        with torch.no_grad():v=losses(model(torch.from_numpy(c['features'][arm][rows])),data,rows,torch.from_numpy(seg),settings['query_batch_size'],rd,qd)
        return {k:float(v) if isinstance(v,torch.Tensor) else v for k,v in v.items()}
    def save():
        guard();parent.save_checkpoint(path,dict(**binding,step=step,seconds=seconds+time.monotonic()-began,trace=trace,
            model=model.state_dict(),optimizer=opt.state_dict(),sampler_rng=rng.get_state(),torch_rng=torch.get_rng_state(),
            draw_hash=draw_hash,row_draws=row_draws,unknown_bound_rows_drawn=unknown_draws,unknown_rows_with_supervised_labels=0,
            record_dual=rd,query_dual=qd))
    if not trace:trace.append(dict(step=0,monitor=diagnostic()))
    while step<limit:
        guard();rows,seg,qids=parent.draw_queries(c['groups'],c['source_groups'],settings['query_batch_size'],rng)
        opt.zero_grad(set_to_none=True);terms=losses(model(torch.from_numpy(c['features'][arm][rows])),data,rows,torch.from_numpy(seg),settings['query_batch_size'],rd,qd)
        if not torch.isfinite(terms['total']) or terms['unknown_supervised_labels']:raise FloatingPointError('Invalid cost-aware routing objective')
        terms['total'].backward();norm=nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True);opt.step()
        step+=1;row_draws+=len(rows);unknown_draws+=int((~data['known'][rows]).sum())
        draw_hash=hashlib.sha256(bytes.fromhex(draw_hash)+qids.tobytes()+rows.tobytes()).hexdigest();full=None
        if step%decision['dual_every']==0:
            r,q,mass=full_risk(model,data,arm)
            rd=(rd+decision['record_dual_learning_rate']*torch.from_numpy(r)).clamp(0,decision['dual_maximum'])
            qd=(qd+decision['query_dual_learning_rate']*torch.from_numpy(q)).clamp(0,decision['dual_maximum'])
            full=dict(positive_expected_record_constraints=(r>0).sum(0).tolist(),max_expected_record_constraints=r.max(0).tolist(),
                positive_predicted_query_constraints=(q>0).sum(0).tolist(),max_predicted_query_constraints=q.max(0).tolist(),
                soft_intervention_fraction=mass,maximum_record_dual=float(rd.max()),maximum_query_dual=float(qd.max()),all_records_and_queries_included=True)
        if step==1 or step%settings['heartbeat_every']==0 or step==settings['steps']:
            row=dict(step=step,loss=float(terms['total'].detach()),gradient_norm=float(norm),monitor=diagnostic(),full_pass=full);trace.append(row);heartbeat(state='free_actor_training',arm=arm,**row)
        if step%settings['checkpoint_every']==0 or step==limit:save()
    return parent.read_checkpoint(path)


def predict(state,c):
    if set(c)!=CAUSAL_KEYS:raise ValueError('Inference rejects future labels,endpoints and outcome masks')
    parent.exact(state['preprocess'],c['preprocess']);parent.exact(state['temperature'],c['temperature'])
    model=AllocationActor(state['dimension'],state['settings']['width'],state['seed'])
    model.load_state_dict(state['model']);model.eval();value=np.empty_like(c['native_gain'])
    for start in range(0,len(value),2048):
        at=slice(start,start+2048)
        with torch.no_grad():off=model(torch.from_numpy(c['features'][state['arm']][at])).numpy()
        value[at]=c['native_gain'][at]+off*c['scales'][at,None]*c['temperature']
    if not np.isfinite(value).all():raise FloatingPointError('Finite routing utility required')
    return value


def assert_matched(left,right):
    for k in ('seed','settings','decision','input_hashes','initial_model','preprocess','dimension','parameters','query_keys',
              'draw_hash','row_draws','unknown_bound_rows_drawn','sampler_rng','torch_rng','step'):
        parent.exact(left[k],right[k])
