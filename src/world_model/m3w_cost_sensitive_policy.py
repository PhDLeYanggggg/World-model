"""Cost-sensitive primal repair; fixed hard constraints, no safety theorem."""
import copy
from pathlib import Path
import time

from src.world_model import m3w_whole_record_policy as whole

base, torch = whole.base, whole.torch


def cost_surrogate(logit, coefficient, temperature):
    if temperature <= 0 or not base.np.isfinite(temperature):
        raise ValueError('Positive scheduled temperature required')
    c = coefficient.detach()
    return c.abs()*torch.nn.functional.softplus(c.sign()*logit/temperature)


def full_pass(model, data, loc, dual, temperature, chunk_size, backward=False):
    n = len(loc['sizes'])
    gain = torch.zeros(n,dtype=torch.float64)
    risk = torch.zeros((n,2),dtype=torch.float64)
    actions = torch.zeros(n,dtype=torch.float64)
    loss_value, hard_value = 0.,0.
    for start in range(0,len(loc['index']),chunk_size):
        ix = loc['index'][start:start+chunk_size]; rec = loc['record'][start:start+chunk_size]
        with torch.set_grad_enabled(backward):
            z = model(data['z'][ix],data['anchor'][ix])
            coefficient = -data['gain'][ix]+(dual.detach()[rec]*data['risk'][ix]/data['record_scales'][rec]).sum(1)
            weight = loc['weight'][start:start+chunk_size]
            loss = (cost_surrogate(z,coefficient,temperature)*weight).sum()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite cost-sensitive objective')
            if backward: loss.backward()
        a = (z.detach()>0).double()
        gain.index_add_(0,rec,a*data['gain'][ix].double())
        risk.index_add_(0,rec,a[:,None]*data['risk'][ix].double())
        actions.index_add_(0,rec,a)
        loss_value += float(loss.detach())
        hard_value += float((a*coefficient.detach().double()*weight.double()).sum())
    size = loc['sizes'].double()
    return dict(surrogate_objective=loss_value,hard_objective=hard_value,
                gain=gain/size,risk=risk/size[:,None],action=actions/size)


def gradient_diagnostic(model, data, loc, dual, temperature=.25):
    stats = dict(rows=0,wrong=0,wrong_cost_mass=0.,cost_mass=0.,
        wrong_STE_gradient_mass=0.,wrong_cost_gradient_mass=0.,
        wrong_saturated_cost_mass=0.,wrong_saturated_rows=0)
    with torch.no_grad():
        for start in range(0,len(loc['index']),8192):
            ix,rec = loc['index'][start:start+8192],loc['record'][start:start+8192]
            z = model(data['z'][ix],data['anchor'][ix]).double()
            c = (-data['gain'][ix]+(dual[rec]*data['risk'][ix]/data['record_scales'][rec]).sum(1)).double()
            w = loc['weight'][start:start+8192].double()
            wrong = ((z>0)!=(c<0)) & (c!=0)
            p = torch.sigmoid(z/temperature)
            ste = c.abs()*p*(1-p)/temperature
            cost = c.abs()*torch.sigmoid(c.sign()*z/temperature)/temperature
            saturated = wrong & (p*(1-p)<1e-4)
            stats['rows'] += len(ix); stats['wrong'] += int(wrong.sum())
            stats['cost_mass'] += float((w*c.abs()).sum())
            stats['wrong_cost_mass'] += float((w*c.abs()*wrong).sum())
            stats['wrong_STE_gradient_mass'] += float((w*ste*wrong).sum())
            stats['wrong_cost_gradient_mass'] += float((w*cost*wrong).sum())
            stats['wrong_saturated_cost_mass'] += float((w*c.abs()*saturated).sum())
            stats['wrong_saturated_rows'] += int(saturated.sum())
    stats['interpretation'] = 'logit_derivative_at_frozen_TRAIN_dual_not_parameter_gradient_or_proof_of_cause'
    return stats


def fit(data, *, settings, seed, identity, path, heartbeat=lambda **_:None,
        resume=False, stop_at=None, guard=lambda:None):
    loc = whole.layout(data)
    model = base.Policy(data['z'].shape[1],settings['width'],seed)
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    dual = torch.zeros((len(data['groups']),2))
    binding = dict(arm='cost_sensitive',settings=settings,seed=seed,identity=identity,
        initial_model=initial,input_hashes=data['input_hashes'],source_preprocess=data['source_preprocess'],
        record_keys=data['record_keys'],record_sizes=data['sizes'],record_scales=data['record_scales'],
        empty_training_records=data['empty_training_records'],dimension=data['z'].shape[1],
        parameters=sum(p.numel() for p in model.parameters()))
    step,seconds,trace = 0,0.,[]
    path = Path(path)
    if path.exists():
        if not resume: raise ValueError('Explicit resume required')
        state = base.core.read_checkpoint(path)
        for k,v in binding.items(): base.core.exact(state[k],v)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        step,seconds,trace,dual = (state[k] for k in ('step','seconds','trace','dual'))
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0<=step<=limit<=settings['steps'] or limit<=0: raise ValueError('Fixed update budget required')
    began = time.monotonic()
    def save():
        guard();path.parent.mkdir(parents=True,exist_ok=True)
        base.core.save_checkpoint(path,dict(**binding,step=step,optimizer_steps=step,
            seconds=seconds+time.monotonic()-began,trace=trace,dual=dual,model=model.state_dict(),
            optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
            full_gradient_rows=step*len(loc['index']),new_forecaster=False,held_used_for_training=False))
    if step==0: save()
    while step<limit:
        temperature = 1.-.75*step/max(1,settings['steps']-1)
        opt.zero_grad(set_to_none=True)
        stats = full_pass(model,data,loc,dual,temperature,settings['chunk_size'],True)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
        opt.step()
        dual = (dual+settings['dual_learning_rate']*(stats['risk']/data['record_scales']).float()).clamp(0,settings['dual_maximum'])
        step += 1
        if step==1 or step%settings['heartbeat_every']==0 or step==settings['steps']:
            row = dict(step=step,surrogate_objective=stats['surrogate_objective'],
                hard_objective=stats['hard_objective'],gradient_norm=float(norm),
                full_gain_lower=float((data['weights']*stats['gain']).sum()),
                action_fraction=float(stats['action'].mean()),dual=dual.tolist(),
                exact_record_risk=stats['risk'].tolist())
            trace.append(row);heartbeat(state='training',**row)
        if step%settings['checkpoint_every']==0 or step==limit: save()
    return base.core.read_checkpoint(path)
