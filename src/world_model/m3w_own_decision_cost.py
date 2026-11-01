"""Matched coherent critics with fixed-choice versus own-choice query losses."""
import copy
import hashlib
from pathlib import Path
import time

from src.world_model import m3w_reference_cost_refit as previous
from src.world_model import m3w_refit_cost_allocation as allocation

np,torch,nn=previous.np,previous.torch,previous.nn
parent=previous.parent
ARMS=('fixed_choice','own_choice')
NAMES=tuple(a+'_'+b for a in ('fixed','own') for b in ('independent','joint','matched'))


def coherent_native(raw,allowed):
    if raw.ndim!=3 or raw.shape[1:]!=(7,5) or allowed.shape!=raw.shape[:2] or allowed.dtype!=torch.bool:
        raise ValueError('Aligned seven-candidate moments and causal availability required')
    counts=allowed.sum(1);ix=torch.arange(len(raw));first=allowed.to(torch.int64).argmax(1);shared={}
    for k in (2,3):
        anchor=raw[ix,first,k]
        shared[k]=anchor+torch.where(allowed,raw[:,:,k]-anchor[:,None],0.).sum(1)/counts.clamp_min(1)
    shared[3]=torch.minimum(shared[3],shared[2])
    reference=torch.where(allowed,shared[2][:,None],raw[:,:,2])
    easy=torch.where(allowed,shared[3][:,None],raw[:,:,3])
    benefit=torch.where(allowed,torch.minimum(raw[:,:,0],reference),raw[:,:,0])
    return torch.stack((benefit,raw[:,:,1],reference,easy,raw[:,:,4]),-1)


class CoherentCritic(previous.ReferenceCostCritic):
    def __init__(self,dimension,width,seed):
        super().__init__(dimension,width,seed,'learned_references')

    def native(self,args,native_anchor,scales,allowed):
        p=super().forward(*args);anchor=args[1]
        raw=torch.stack(tuple(native_anchor[:,:,k]+(p[:,:,k]-anchor[:,:,k])*scales[:,None]
            if k in (0,2,3) else p[:,:,k]*scales[:,None] for k in range(5)),-1)
        return coherent_native(raw,allowed)


def initialize(warm):
    if warm.get('arm')!='learned_references':raise ValueError('Same verified learned-reference warm model required')
    model=CoherentCritic(warm['dimension'],warm['settings']['width'],warm['seed'])
    model.load_state_dict(warm['model'],strict=True)
    return model


def batch(causal,rows,segments,count):
    args=parent.batch(causal,rows,segments,count)
    return args,torch.from_numpy(causal['prediction'][rows]),torch.from_numpy(causal['scales'][rows]),torch.from_numpy(causal['allowed'][rows])


def own_choices(native,allowed,segments,*,seconds,heartbeat=lambda **_:None):
    """Stop-gradient native allocation; accepts no outcome or label-validity input."""
    p,ok=allocation.base.validate(native,allowed);segments=np.asarray(segments)
    if segments.shape!=(len(p),) or not np.issubdtype(segments.dtype,np.integer):raise ValueError('Aligned causal query segments required')
    selected,_=allocation.base.select_actions(p,ok,arm='physical_neural');out=selected.copy();reasons={}
    queries=np.unique(segments)
    for i,segment in enumerate(queries):
        at=np.flatnonzero(segments==segment);anchor=selected[at]
        out[at],info=allocation.base.solve_frame(p[at],ok[at],anchor,exact_count=int((anchor>=0).sum()),seconds=seconds)
        reasons[info['reason']]=reasons.get(info['reason'],0)+1
        heartbeat(queries=i+1,total_queries=len(queries))
    return out,dict(queries=len(np.unique(segments)),solver_reason_counts=reasons,
        selected_rows=int((out>=0).sum()),labels_used=False)


def query_error(prediction,data,rows,segments,count,choices):
    c=data['causal'];known=torch.from_numpy(data['known'][rows]);scales=torch.from_numpy(c['scales'][rows])
    target=torch.from_numpy(np.where(data['known'][rows,None,None],data['targets'][rows]/c['scales'][rows,None,None],0.))
    choices=torch.as_tensor(choices,dtype=torch.int64);selected=choices>=0;ix=torch.arange(len(rows));action=choices.clamp_min(0)
    pred,truth=prediction[ix,action],target[ix,action]
    signed=pred[:,[1,4]]-.02*pred[:,[2,3]]-truth[:,[1,4]]+.02*truth[:,[2,3]]
    total=prediction.new_zeros((count,2)).index_add(0,segments,signed*scales[:,None]*selected[:,None])
    mass=prediction.new_zeros(count).index_add(0,segments,scales*selected)
    selected_n=prediction.new_zeros(count).index_add(0,segments,selected.double())
    missing=prediction.new_zeros(count).index_add(0,segments,(selected&~known).double())
    complete=(selected_n>0)&(missing==0)
    return total/mass.clamp_min(1e-150)[:,None],complete,dict(empty_selected_queries=int((selected_n==0).sum()),
        unknown_selected_queries=int((missing>0).sum()),complete_selected_queries=int(complete.sum()),
        selected_rows=int(selected.sum()))


def losses(prediction,data,rows,segments,count,normalizer,choices):
    terms=previous.losses(prediction,data,rows,segments,count,normalizer)
    residual,complete,counts=query_error(prediction,data,rows,segments,count,choices)
    rms=torch.as_tensor(normalizer['rms'],dtype=prediction.dtype)
    risk=((residual/rms).square().mean(1)*complete).mean()
    return dict(terms,total=terms['total']+.5*(risk-terms['selected_query_score']),
        fixed_query_score=terms['selected_query_score'],selected_query_score=risk,**counts)


def fit(data,*,warm,arm,settings,identity,path,heartbeat=lambda **_:None,guard=lambda:None,resume=False,stop_at=None):
    if arm not in ARMS:raise ValueError('Registered fixed/own choice arm required')
    parent.exact(warm['input_hashes'],data['input_hashes']);c=data['causal'];model=initialize(warm)
    if warm['settings']['width']!=settings['width']:raise ValueError('Warm architecture must remain unchanged')
    initial=copy.deepcopy(model.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    sampler=previous.CyclicQueries(c['groups'],c['source_groups'],settings['query_batch_size'],warm['seed'])
    normalizer=previous.base.units.query_normalizer(data);budget=previous.steps_for(c,settings)
    monitor=parent.draw_queries(c['groups'],c['source_groups'],settings['query_batch_size'],torch.Generator().manual_seed(warm['seed']+9137))
    binding=dict(arm=arm,settings=settings,identity=identity,input_hashes=data['input_hashes'],initial_model=initial,
        seed=warm['seed'],dimension=warm['dimension'],rms=data['rms'],query_normalizer=normalizer,query_keys=c['keys'],
        total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        budget=budget,source_query_counts=list(map(len,c['source_groups'])),training_role='exposed_SOURCE_fitting',
        causal_inference_only=True,new_trajectory_training=False,optimizer_reused=False)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    step=0;seconds=0.;trace=[];draw_hash='0'*64;unknown_draws=0;row_draws=0;solver_calls=0;solver_seconds=0.;coverage_counts={};reason_counts={}
    if path.exists():
        if not resume:raise ValueError('Explicit unchanged-definition resume required')
        old=parent.read_checkpoint(path)
        for k,v in binding.items():parent.exact(old[k],v)
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);sampler.restore(old['sampler']);torch.set_rng_state(old['torch_rng'])
        step,seconds,trace,draw_hash,unknown_draws,row_draws,solver_calls,solver_seconds,coverage_counts,reason_counts=(old[k] for k in
            ('step','seconds','trace','draw_hash','unknown_rows_drawn','row_draws','training_solver_calls','training_solver_seconds','selected_query_counts','solver_reason_counts'))
    limit=budget if stop_at is None else min(stop_at,budget)
    if not 0<=step<=limit or limit<=0:raise ValueError('Nondecreasing coverage-complete budget required')
    start=time.monotonic();last_live=[start]
    def live(**kw):
        if time.monotonic()-last_live[0]>=60:
            guard();heartbeat(state='own_decision_live_allocation',arm=arm,step=step,budget=budget,**kw);last_live[0]=time.monotonic()
    def objective(ix,seg,count):
        args,anchor,scale,allowed=batch(c,ix,seg,count);native=model.native(args,anchor,scale,allowed);selection=c['initial'][ix]
        info=dict(queries=0,solver_reason_counts={});elapsed=0.
        if arm=='own_choice':
            begin=time.monotonic();selection,info=own_choices(native.detach().numpy(),allowed.numpy(),seg,seconds=settings['solver_seconds_per_query'],heartbeat=live)
            elapsed=time.monotonic()-begin
        terms=losses(native/scale[:,None,None],data,ix,args[-2],count,normalizer,selection)
        return terms,info,elapsed
    def diagnostic():
        ix,seg,_=monitor
        with torch.no_grad():terms,_,_=objective(ix,seg,settings['query_batch_size'])
        return {k:float(v) if isinstance(v,torch.Tensor) else v for k,v in terms.items()}
    def save():
        guard();parent.save_checkpoint(path,dict(**binding,step=step,seconds=seconds+time.monotonic()-start,trace=trace,
            model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.state(),torch_rng=torch.get_rng_state(),draw_hash=draw_hash,
            unknown_rows_drawn=unknown_draws,unknown_rows_with_supervised_labels=0,row_draws=row_draws,
            training_solver_calls=solver_calls,training_solver_seconds=solver_seconds,selected_query_counts=coverage_counts,solver_reason_counts=reason_counts))
    if not trace:trace.append(dict(step=0,monitor=diagnostic()))
    while step<limit:
        guard();ix,seg,qids=sampler.draw();optimizer.zero_grad(set_to_none=True)
        terms,info,elapsed=objective(ix,seg,settings['query_batch_size'])
        if not torch.isfinite(terms['total']) or terms['unknown_supervised_rows']:raise FloatingPointError('Invalid own-decision cost objective')
        terms['total'].backward();norm=nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        step+=1;row_draws+=len(ix);unknown_draws+=int((~data['known'][ix]).sum());solver_calls+=info['queries'];solver_seconds+=elapsed
        for k in ('empty_selected_queries','unknown_selected_queries','complete_selected_queries','selected_rows'):coverage_counts[k]=coverage_counts.get(k,0)+terms[k]
        for k,v in info['solver_reason_counts'].items():reason_counts[k]=reason_counts.get(k,0)+v
        draw_hash=hashlib.sha256(bytes.fromhex(draw_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step==1 or step%settings['heartbeat_every']==0 or step==budget:
            record=dict(step=step,budget=budget,loss=float(terms['total'].detach()),gradient_norm=float(norm),monitor=diagnostic(),
                minimum_query_visits=int(sampler.counts.min()),never_drawn_queries=int((sampler.counts==0).sum()),
                training_solver_calls=solver_calls,training_solver_seconds=solver_seconds,selected_query_counts=dict(coverage_counts))
            trace.append(record);heartbeat(state='coherent_own_decision_training',arm=arm,**record)
        if step%settings['checkpoint_every']==0 or step==limit:save()
    result=parent.read_checkpoint(path)
    if step==budget and int(result['sampler']['counts'].min())<settings['largest_source_cycles']:raise AssertionError('Every query must receive the registered coverage')
    return result


def predict(state,causal):
    if set(causal)!=parent.CAUSAL_KEYS:raise ValueError('Inference accepts declared causal fields only')
    model=CoherentCritic(state['dimension'],state['settings']['width'],state['seed']);model.load_state_dict(state['model']);model.eval()
    result=np.empty_like(causal['prediction']);pending=[];size=0
    def flush():
        rows=np.concatenate(pending);segments=np.repeat(np.arange(len(pending)),[len(g) for g in pending]);args,anchor,scale,allowed=batch(causal,rows,segments,len(pending))
        with torch.no_grad():result[rows]=model.native(args,anchor,scale,allowed).numpy()
    for g in causal['groups']:
        if pending and size+len(g)>1024:flush();pending=[];size=0
        pending.append(g);size+=len(g)
    if pending:flush()
    if not np.isfinite(result).all() or (result<0).any():raise FloatingPointError('Finite nonnegative coherent costs required')
    return result


def assert_matched(a,b):
    for k in ('settings','identity','input_hashes','initial_model','seed','dimension','rms','query_normalizer','query_keys',
            'total_parameters','trainable_parameters','budget','source_query_counts','sampler','draw_hash','unknown_rows_drawn','row_draws','step','torch_rng'):
        parent.exact(a[k],b[k])


def assert_replay(a,b):
    def scientific(value):
        if isinstance(value,dict):return {k:scientific(v) for k,v in value.items() if k not in ('seconds','training_solver_seconds')}
        if isinstance(value,list):return [scientific(v) for v in value]
        return value
    parent.exact(scientific(a),scientific(b))


def allocate(predictions,*args,**kwargs):
    if set(predictions)!=set(ARMS):raise ValueError('Both registered learned endpoints required')
    values={old:predictions[new] for old,new in zip(allocation.ARMS,ARMS)}
    choices,ranks,info=allocation.allocate(values,*args,**kwargs)
    def rename(name):
        return 'frozen_projected_joint' if name=='parent_joint' else name.replace('locked_','fixed_').replace('learned_','own_')
    info=dict(info,reason_counts={rename(k):v for k,v in info['reason_counts'].items()},nonoptimal={rename(k):v for k,v in info['nonoptimal'].items()})
    return {rename(k):v for k,v in choices.items()},{rename(k):v for k,v in ranks.items()},info
