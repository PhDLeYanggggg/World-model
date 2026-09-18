"""Training-only output-conditioning diagnostic; not a forecasting benchmark."""
import os
import time

import numpy as np
import torch

from src.world_model.m3w_source_cost_dynamics import SourceDynamics, restore, dynamics_loss
from src.world_model.m3w_source_visual_start import VisualStart

DECODERS = ('context_radius', 'training_cost_scale')


def select_cohorts(ids, tracks, cv, maximum_target, radius, support, *, count=16, seed=104729):
    ids,tracks,cv,maximum_target,radius,support=map(np.asarray,(ids,tracks,cv,maximum_target,radius,support))
    if any(v.shape!=ids.shape for v in (tracks,cv,maximum_target,radius,support)) or ids.ndim!=1:
        raise ValueError('Aligned training-only cohort arrays required')
    eligible=support & (radius>0) & (maximum_target<.8*radius)
    order=np.random.default_rng(seed).permutation(len(ids));used=set();positive=[];zero=[]
    for i in order:
        if eligible[i] and cv[i]>0 and tracks[i] not in used:
            positive.append(int(ids[i]));used.add(tracks[i])
            if len(positive)==count:break
    for i in order:
        if eligible[i] and cv[i]==0 and tracks[i] not in used:
            zero.append(int(ids[i]));used.add(tracks[i])
            if len(zero)==count:break
    if len(positive)!=count or len(zero)!=count:
        raise ValueError('Insufficient distinct training tracks for the fixed microfit')
    return dict(nonzero_only=np.array(positive),mixed_zero=np.array(positive+zero))


class MicrofitDynamics(SourceDynamics):
    def trajectory(self, features, frame, decoder, training_cost_scale):
        if decoder not in DECODERS or training_cost_scale <= 0 or not np.isfinite(training_cost_scale):
            raise ValueError('Registered decoder and fixed positive training-only scale required')
        if decoder == 'context_radius':
            return restore(self(*features, 'past_rgb'), *frame)
        radius, rotation, support = frame
        if torch.any(radius[support] <= 0):
            raise ValueError('Positive supported radius required')
        raw = VisualStart.forward(self, *features, 'past_rgb').reshape(-1, 12, 2)
        divisor = torch.where(support, radius, torch.ones_like(radius))
        local = raw * (training_cost_scale / divisor)[:, None, None]
        local = local / (1 + torch.linalg.vector_norm(local, dim=-1, keepdim=True))
        return restore(local, radius, rotation, support)


def metrics(prediction, target, scale):
    with torch.no_grad():
        error = torch.linalg.vector_norm(prediction-target, dim=-1).mean(1)
        cv = torch.linalg.vector_norm(target, dim=-1).mean(1)
        zero, moving = cv == 0, cv > 0
        return dict(ade=float(error.mean()), cv_ade=float(cv.mean()),
            normalized_ade=float(error.mean()/scale),
            gain_percent=100*float(1-error.mean()/cv.mean()),
            moving_ade=float(error[moving].mean()) if moving.any() else None,
            moving_cv_ade=float(cv[moving].mean()) if moving.any() else None,
            zero_rows=int(zero.sum()),
            easy_absolute_harm=float(error[zero].mean()) if zero.any() else None,
            easy_relative_degradation=None,
            max_prediction_norm=float(torch.linalg.vector_norm(prediction, dim=-1).max()))


def fit_micro(model, features, frame, target, *, decoder, scale, config, identity,
              checkpoint, heartbeat, stop_at=None):
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])
    first = step = 0
    seconds = 0.
    trace = []; clipped_updates=0; max_gradient=0.
    if checkpoint.exists():
        cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if cp['identity'] != identity or cp['config'] != config or cp['decoder'] != decoder or cp['scale'] != scale:
            raise ValueError('Microfit checkpoint identity changed')
        model.load_state_dict(cp['model']); opt.load_state_dict(cp['optimizer'])
        torch.set_rng_state(cp['torch_rng'])
        first = step = cp['step']; seconds = cp['fit_seconds']; trace = cp['trace']
        clipped_updates=cp['clipped_updates'];max_gradient=cp['max_gradient']
    if step == 0:
        with torch.no_grad():
            initial = metrics(model.trajectory(features, frame, decoder, scale), target, scale)
        trace.append(dict(step=0, preclip_gradient=None, **initial))
    limit = config['updates'] if stop_at is None else min(config['updates'], stop_at)
    if limit < step or limit <= 0:
        raise ValueError('Positive nondecreasing fixed budget required')
    model.train(); started = time.monotonic()
    try:
        while step < limit:
            opt.zero_grad(set_to_none=True)
            prediction = model.trajectory(features, frame, decoder, scale)
            loss, _ = dynamics_loss(prediction, target, scale, 'ade')
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite microfit loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            clipped_updates+=int(float(grad)>5);max_gradient=max(max_gradient,float(grad))
            opt.step(); step += 1
            if step == 1 or step % 100 == 0:
                with torch.no_grad():
                    value = metrics(model.trajectory(features, frame, decoder, scale), target, scale)
                trace.append(dict(step=step, preclip_gradient=float(grad), **value))
            if step == limit or step % config['checkpoint_every'] == 0:
                value = dict(identity=identity, config=config, decoder=decoder, scale=scale,
                    step=step, model=model.state_dict(), optimizer=opt.state_dict(),
                    torch_rng=torch.get_rng_state(), fit_seconds=seconds+time.monotonic()-started,
                    trace=trace, examples_per_step=len(target),clipped_updates=clipped_updates,max_gradient=max_gradient)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                tmp = checkpoint.with_suffix('.tmp'); torch.save(value, tmp); os.replace(tmp, checkpoint)
                heartbeat(step=step, state='training', normalized_loss=float(loss.detach()),
                          fit_seconds=value['fit_seconds'])
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', last_completed_step=step)
        raise
    model.eval()
    with torch.no_grad():
        prediction = model.trajectory(features, frame, decoder, scale)
    return prediction.numpy(), dict(step=step, complete=step == config['updates'],
        new_updates=step-first, fit_seconds=seconds+time.monotonic()-started,
        trace=trace,clipped_updates=clipped_updates,max_gradient=max_gradient,
        final=metrics(prediction, target, scale))
