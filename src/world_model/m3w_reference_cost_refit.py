"""Warm-started causal cost learning with coverage-complete query sampling."""
import copy
import hashlib
from pathlib import Path
import time
from src.world_model import m3w_coupled_benefit_risk as base

np,torch,nn=base.np,base.torch,base.nn
parent=base.parent
ARMS=('locked_references','learned_references')


class CyclicQueries:
    def __init__(self,groups,source_groups,batch_size,seed):
        if not source_groups or batch_size<=0 or batch_size%len(source_groups):raise ValueError('Equal source allocation required')
        self.groups,self.sources=groups,source_groups;self.each=batch_size//len(source_groups)
        if any(not len(g) for g in source_groups):raise ValueError('No empty source in a coverage-complete sampler')
        if sorted(np.concatenate(source_groups).tolist())!=list(range(len(groups))):raise ValueError('Every query belongs to exactly one source')
        self.rng=torch.Generator().manual_seed(seed+7919)
        self.order=[None]*len(source_groups);self.position=[0]*len(source_groups);self.completed=[0]*len(source_groups)
        self.counts=np.zeros(len(groups),np.int64)
    def draw(self):
        values=[]
        for i,source in enumerate(self.sources):
            need=self.each
            while need:
                if self.order[i] is None:
                    self.order[i]=source[torch.randperm(len(source),generator=self.rng).numpy()];self.position[i]=0
                n=min(need,len(source)-self.position[i]);values.append(self.order[i][self.position[i]:self.position[i]+n])
                self.position[i]+=n;need-=n
                if self.position[i]==len(source):self.completed[i]+=1;self.order[i]=None
        qids=np.concatenate(values);np.add.at(self.counts,qids,1);blocks=[self.groups[i] for i in qids]
        return np.concatenate(blocks),np.repeat(np.arange(len(qids)),[len(x) for x in blocks]),qids
    def state(self):
        return dict(rng=self.rng.get_state(),order=copy.deepcopy(self.order),position=list(self.position),completed=list(self.completed),counts=self.counts.copy())
    def restore(self,state):
        self.rng.set_state(state['rng']);self.order=copy.deepcopy(state['order']);self.position=list(state['position']);self.completed=list(state['completed']);self.counts=state['counts'].copy()


def steps_for(c,settings):
    each=settings['query_batch_size']//len(c['source_groups'])
    return max(settings['minimum_updates'],settings['largest_source_cycles']*((max(map(len,c['source_groups']))+each-1)//each))


class ReferenceCostCritic(base.CoupledCritic):
    def __init__(self,dimension,width,seed,arm):
        if arm not in ARMS:raise ValueError('Registered reference arm required')
        super().__init__(dimension,width,seed,'released_benefit');self.reference_arm=arm
        rng=torch.get_rng_state();self.reference_output=nn.Linear(2*width+2,2)
        nn.init.zeros_(self.reference_output.weight);nn.init.zeros_(self.reference_output.bias);torch.set_rng_state(rng)
        if arm=='locked_references':self.reference_output.requires_grad_(False)
    def forward(self,x,anchor,envelope,initial,segments,count):
        old=super().forward(x,anchor,envelope,initial,segments,count)
        if self.reference_arm=='locked_references':return old
        onehot=torch.eye(7,dtype=x.dtype).expand(len(x),-1,-1)
        h=self.encoder(torch.cat((x,torch.log1p(anchor.clamp_max(2980)).float(),onehot,torch.log1p(envelope.clamp_max(2980)).float()[:,:,None]),-1))
        context,selected=parent.selected_context(h,initial,segments,count)
        sizes=torch.bincount(segments,minlength=count).float()[segments]
        features=torch.cat((h[:,:6].mean(1),context,torch.stack((torch.log1p(selected),selected/sizes),1)),-1)
        raw=self.reference_output(features).double().clamp(-20.,20.)
        reference=old[:,:,2]*torch.exp(raw[:,0,None])
        easy_reference=torch.minimum(old[:,:,3]*torch.exp(raw[:,1,None]),reference)
        return torch.stack((old[:,:,0],old[:,:,1],reference,easy_reference,old[:,:,4]),-1)


def initialize(warm,arm):
    model=ReferenceCostCritic(warm['dimension'],warm['settings']['width'],warm['seed'],arm)
    result=model.load_state_dict(warm['model'],strict=False)
    if set(result.missing_keys)!={'reference_output.weight','reference_output.bias'} or result.unexpected_keys:raise ValueError('Only the new zero reference branch may be missing')
    return model


def losses(prediction,data,rows,segments,count,normalizer):
    result=base.losses(prediction,data,rows,segments,count,normalizer);c=data['causal']
    known=torch.from_numpy(data['known'][rows]);legal=torch.from_numpy(c['allowed'][rows])&known[:,None]
    target=torch.from_numpy(np.where(data['known'][rows,None,None],data['targets'][rows]/c['scales'][rows,None,None],0.))
    error=((prediction[:,:,[2,3]]-target[:,:,[2,3]])/torch.from_numpy(data['rms'][[2,3]])).square().mean(2)
    row=(error*legal).sum(1)/legal.sum(1).clamp_min(1)
    total=prediction.new_zeros(count).index_add(0,segments,row);support=prediction.new_zeros(count).index_add(0,segments,legal.any(1).double())
    ref=(total/support.clamp_min(1)).mean()
    return dict(result,total=result['total']+ref,reference_MSE=ref,unknown_supervised_rows=int((legal&~known[:,None]).sum()))


def fit(data,*,warm,arm,settings,identity,path,heartbeat=lambda **_:None,guard=lambda:None,resume=False,stop_at=None):
    parent.exact(warm['input_hashes'],data['input_hashes']);c=data['causal'];model=initialize(warm,arm);initial=copy.deepcopy(model.state_dict())
    if warm['settings']['width']!=settings['width']:raise ValueError('Warm architecture unchanged')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings['learning_rate'],weight_decay=.0001)
    sampler=CyclicQueries(c['groups'],c['source_groups'],settings['query_batch_size'],warm['seed']);normalizer=base.units.query_normalizer(data)
    budget=steps_for(c,settings);monitor=parent.draw_queries(c['groups'],c['source_groups'],settings['query_batch_size'],torch.Generator().manual_seed(warm['seed']+9137))
    binding=dict(arm=arm,settings=settings,identity=identity,input_hashes=data['input_hashes'],initial_model=initial,
        seed=warm['seed'],dimension=warm['dimension'],query_keys=c['keys'],rms=data['rms'],query_normalizer=normalizer,
        total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        budget=budget,source_query_counts=list(map(len,c['source_groups'])),warm_optimizer_reused=False,
        causal_inference_only=True,training_role='exposed_SOURCE_fitting',new_trajectory_training=False)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);step=0;seconds=0.;trace=[];draw_hash='0'*64;unknown_draws=0;row_draws=0
    if path.exists():
        if not resume:raise ValueError('Explicit same-definition resume required')
        old=parent.read_checkpoint(path)
        for k,v in binding.items():parent.exact(old[k],v)
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);sampler.restore(old['sampler']);torch.set_rng_state(old['torch_rng'])
        step,seconds,trace,draw_hash,unknown_draws,row_draws=(old[k] for k in ('step','seconds','trace','draw_hash','unknown_rows_drawn','row_draws'))
    limit=budget if stop_at is None else min(stop_at,budget)
    if not 0<=step<=limit or limit<=0:raise ValueError('Nondecreasing registered coverage budget required')
    start=time.monotonic()
    def diagnostic():
        ix,seg,_=monitor;args=parent.batch(c,ix,seg,settings['query_batch_size'])
        with torch.no_grad():terms=losses(model(*args),data,ix,args[-2],settings['query_batch_size'],normalizer)
        return {k:float(v) if isinstance(v,torch.Tensor) else v for k,v in terms.items()}
    def save():
        guard();parent.save_checkpoint(path,dict(**binding,step=step,seconds=seconds+time.monotonic()-start,trace=trace,
            model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.state(),torch_rng=torch.get_rng_state(),draw_hash=draw_hash,
            unknown_rows_drawn=unknown_draws,unknown_rows_with_supervised_labels=0,row_draws=row_draws))
    if not trace:trace.append(dict(step=0,monitor=diagnostic()))
    while step<limit:
        guard();ix,seg,qids=sampler.draw();args=parent.batch(c,ix,seg,settings['query_batch_size']);optimizer.zero_grad(set_to_none=True)
        terms=losses(model(*args),data,ix,args[-2],settings['query_batch_size'],normalizer)
        if not torch.isfinite(terms['total']) or terms['unknown_supervised_rows']:raise FloatingPointError('Invalid reference cost fit')
        terms['total'].backward();norm=nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        step+=1;row_draws+=len(ix);unknown_draws+=int((~data['known'][ix]).sum())
        draw_hash=hashlib.sha256(bytes.fromhex(draw_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step==1 or step%settings['heartbeat_every']==0 or step==budget:
            record=dict(step=step,budget=budget,loss=float(terms['total'].detach()),gradient_norm=float(norm),monitor=diagnostic(),
                minimum_query_visits=int(sampler.counts.min()),never_drawn_queries=int((sampler.counts==0).sum()),source_completed_cycles=list(sampler.completed))
            trace.append(record);heartbeat(state='reference_cost_training',arm=arm,**record)
        if step%settings['checkpoint_every']==0 or step==limit:save()
    result=parent.read_checkpoint(path)
    if step==budget and int(result['sampler']['counts'].min())<settings['largest_source_cycles']:raise AssertionError('Every causal query must receive the registered minimum visits')
    return result


def predict(state,causal):
    if set(causal)!=parent.CAUSAL_KEYS:raise ValueError('Inference accepts causal fields only,not labels or endpoints')
    model=ReferenceCostCritic(state['dimension'],state['settings']['width'],state['seed'],state['arm']);model.load_state_dict(state['model']);model.eval()
    result=causal['prediction'].copy();pending=[];size=0
    def flush():
        rows=np.concatenate(pending);segments=np.repeat(np.arange(len(pending)),[len(g) for g in pending]);args=parent.batch(causal,rows,segments,len(pending))
        with torch.no_grad():p=model(*args).numpy()
        scale=causal['scales'][rows,None];anchor=args[1].numpy()
        for k in (0,2,3):result[rows,:,k]+=(p[:,:,k]-anchor[:,:,k])*scale
        for k in (1,4):result[rows,:,k]=p[:,:,k]*scale
    for g in causal['groups']:
        if pending and size+len(g)>1024:flush();pending=[];size=0
        pending.append(g);size+=len(g)
    if pending:flush()
    if not np.isfinite(result).all() or (result<0).any():raise FloatingPointError('Finite nonnegative native costs required')
    return result


def assert_matched(a,b):
    for k in ('settings','identity','input_hashes','initial_model','seed','dimension','query_keys','rms','query_normalizer',
              'total_parameters','budget','source_query_counts','sampler','draw_hash','unknown_rows_drawn','row_draws','step','torch_rng'):
        parent.exact(a[k],b[k])
