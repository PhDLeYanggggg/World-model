"""Matched base learning with cyclic complete-teacher-region moment gradients."""
import copy
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from src.world_model import m3w_policy_region_cost as old
from src.world_model.m3w_exact_region_cost import exact_region_pass


def fit(data, *, arm, settings, seed, identity, path, regions, region_weight,
        chunk_size=512, heartbeat=lambda **_:None, guard=lambda:None, resume=False, stop_at=None):
    if arm not in old.base.ARMS or region_weight < 0 or not len(regions['keys']):
        raise ValueError('A registered action bank and nonempty fixed training regions are required')
    pr=data['preprocess']; actions=torch.tensor(old.base.ARMS[arm])
    model=old.SceneCostHead(data['z'].shape[-1],settings['width'],seed,pr['bias'])
    initial=copy.deepcopy(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    rng=torch.Generator().manual_seed(seed+7919)
    order=np.random.default_rng(seed+104729).permutation(len(regions['keys'])).tolist()
    binding=dict(arm=arm,settings=settings,seed=seed,identity=identity,preprocess=pr,
        input_hashes=data['input_hashes'],initial_model=initial,dimension=data['z'].shape[-1],
        parameters=sum(p.numel() for p in model.parameters()),region_weight=region_weight,
        region_keys=regions['keys'],selected_known_region_rows=regions['selected_known_rows'],
        region_arrays={key:hashlib.sha256(regions[key].numpy().tobytes()).hexdigest()
                       for key in ('pool','local_action','target')},
        exact_region_order=order,exact_region_chunk_size=chunk_size,
        auxiliary_schedule='cyclic_complete_fixed_teacher_record_regions')
    step,elapsed,trace,draw_hash,row_draws,extra_rows=0,0.,[],'0'*64,0,0
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not resume: raise ValueError('Explicit same-checkpoint resume required')
        state=old.base.read_checkpoint(path)
        for key,value in binding.items(): old.base.exact(state[key],value)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']);torch.set_rng_state(state['torch_rng'])
        step,elapsed,trace,draw_hash,row_draws,extra_rows=(state[k] for k in
            ('step','seconds','trace','draw_hash','row_draws','auxiliary_teacher_rows'))
    limit=settings['steps'] if stop_at is None else stop_at
    if not 0<=step<=limit<=settings['steps'] or limit==0: raise ValueError('Nondecreasing fixed budget required')
    started=time.monotonic();rms=torch.from_numpy(pr['rms'])
    def save():
        guard();old.base.save_checkpoint(path,dict(**binding,step=step,seconds=elapsed+time.monotonic()-started,
            trace=trace,model=model.state_dict(),optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(),torch_rng=torch.get_rng_state(),draw_hash=draw_hash,row_draws=row_draws,
            auxiliary_teacher_rows=extra_rows,auxiliary_forward_rows=2*extra_rows,
            unknown_rows_sampled=0,inner_held_used_for_training=False))
    if step==0: save()
    while step<limit:
        rows=old.draw_batch(data['groups'],settings['batch_size'],rng)
        p=model(data['z'][rows][:,actions],data['envelope'][rows][:,actions],actions)
        terms=old.base.continuous_loss(p,data['truth'][rows][:,actions],data['allowed'][rows][:,actions],rms,
            underharm_weight=settings['underharm_weight'],ranking_weight=settings['ranking_weight'])
        if not torch.isfinite(terms['total']): raise FloatingPointError('Nonfinite base loss')
        optimizer.zero_grad(set_to_none=True);terms['total'].backward()
        extra=None
        if region_weight:
            if region_weight!=1.: raise ValueError('Primary exact-region weight is fixed at1')
            extra=exact_region_pass(model,data,regions,actions,rms,pool_id=order[step%len(order)],
                chunk_size=chunk_size,backward=True,zero_grad=False)
            extra_rows+=extra['rows']
        norm=nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();step+=1;row_draws+=len(rows)
        draw_hash=hashlib.sha256(bytes.fromhex(draw_hash)+rows.tobytes()).hexdigest()
        if step==1 or step%settings['heartbeat_every']==0 or step==settings['steps']:
            row=dict(step=step,loss=float(terms['total'].detach())+(extra['loss'] if extra else 0.),
                regression=float(terms['regression'].detach()),ranking=float(terms['ranking'].detach()),gradient_norm=float(norm))
            if extra:
                row.update(region_moment_loss=extra['loss'],region_pool_index=extra['pool_ids'][0],region_rows=extra['rows'])
            trace.append(row);heartbeat(state='training',arm=arm,**row)
        if step%settings['checkpoint_every']==0 or step==limit: save()
    return old.base.read_checkpoint(path)
