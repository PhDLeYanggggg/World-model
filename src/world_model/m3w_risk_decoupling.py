"""Matched continuation: conditional moment estimation versus action coupling."""
import copy
import hashlib
from pathlib import Path
import time

from src.evaluation import m3w_query_conditioning as audit

api, torch, np = audit.api, audit.torch, audit.np
ARMS = ('coupled', 'estimation')


def prepared(packets, origin):
    data = {}
    for source, args in sorted(packets.items()):
        x, env, y, _, sites, recs, frames, pr = args
        api.core.exact(pr, origin['anchor_state']['source_preprocess'][source])
        for name, a in zip(('x', 'envelope', 'primary', 'temporal', 'sites', 'recordings', 'frames'), args[:-1]):
            if api.fingerprint(a) != origin['input_hashes'][source][name]:
                raise ValueError('Changed original fitting packet')
        api.parent.previous.old.parent.forest.causal_inputs(x, env, pr)
        z, _, _ = api.context_inputs(x, recs, frames, pr)
        known = np.isfinite(y).all(1)
        groups, sg, _ = api.core.sampling.query_groups(sites, recs, frames, known)
        generator = torch.Generator(); generator.set_state(origin['samplers'][source]['rng'])
        data[source] = dict(z=torch.from_numpy(z), envelope=torch.from_numpy((env/pr['scale']).astype(np.float32)),
            y=torch.from_numpy(np.where(known[:, None], y/pr['scale'], 0).astype(np.float32)),
            known=known, eligible=torch.from_numpy(api.parent.previous.prior.eligible(args)),
            groups=groups, sg=sg, generator=generator,
            draw_hash=origin['samplers'][source]['draw_hash'], row_draws=origin['samplers'][source]['row_draws'])
    return data


def fit(packets, *, origin, origin_sha256, objective_arm, updates, path,
        registration_sha256, heartbeat, guard=lambda: None, resume=False, stop_at=None):
    if objective_arm not in ARMS or origin['arm'] != 'self' or updates <= 0:
        raise ValueError('Registered continuation from frozen self head required')
    api.check_protection(origin)
    data = prepared(packets, origin)
    model = audit.model_from(origin); model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if not all(p.requires_grad for p in model.risk_encoder.parameters()):
        raise ValueError('Copied risk encoder must train')
    optimizer = torch.optim.AdamW(params, lr=origin['settings']['learning_rate'], weight_decay=.0001)
    optimizer.load_state_dict(copy.deepcopy(origin['optimizer']))
    torch.set_rng_state(origin['torch_rng'])
    identity = dict(origin_sha256=origin_sha256, objective_arm=objective_arm, updates=updates,
                    registration_sha256=registration_sha256, origin_step=origin['step'])
    current, seconds, trace = 0, 0., []
    dual = origin['dual'].clone(); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not resume: raise ValueError('Explicit continuation resume required')
        state = api.core.read_checkpoint(path); api.core.exact(state['continuation'], identity)
        for key in ('anchor_state', 'settings', 'objective_config', 'input_hashes', 'arm'):
            api.core.exact(state[key], origin[key])
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']); dual = state['dual'].clone()
        current, seconds, trace = state['continuation_step'], state['continuation_seconds'], state['continuation_trace']
        for source, d in data.items():
            saved = state['samplers'][source]; d['generator'].set_state(saved['rng'])
            d['draw_hash'], d['row_draws'] = saved['draw_hash'], saved['row_draws']
    limit = updates if stop_at is None else stop_at
    if not 0 <= current <= limit <= updates or limit <= 0:
        raise ValueError('Fixed nondecreasing update budget required')
    began = time.monotonic()
    rms = torch.from_numpy(origin['anchor_state']['preprocess']['rms'])
    config, settings = origin['objective_config'], origin['settings']
    scales = origin['anchor_state']['scales']
    temperature = api.parent.previous.old.temperature(origin['step'], settings, config)

    def save():
        guard()
        state = dict(origin, continuation=identity, continuation_step=current,
            continuation_seconds=seconds+time.monotonic()-began, continuation_trace=trace,
            model=model.state_dict(), optimizer=optimizer.state_dict(), dual=dual,
            step=origin['step']+current, torch_rng=torch.get_rng_state(),
            samplers={s: dict(rng=d['generator'].get_state(), draw_hash=d['draw_hash'], row_draws=d['row_draws'])
                      for s, d in data.items()}, held_scored=False, new_forecaster=False)
        api.check_protection(state); api.core.save_checkpoint(path, state)
    if current == 0: save()
    while current < limit:
        optimizer.zero_grad(set_to_none=True)
        primary, regret, excess = [], [], []
        for source, d in data.items():
            ix, seg, qids = api.core.sampling.draw_queries(d['groups'], d['sg'], settings['query_batch_size'], d['generator'])
            if not d['known'][ix].all(): raise ValueError('Unknown outcome sampled for supervision')
            segment = torch.from_numpy(seg)
            prediction = model(d['z'][ix], d['envelope'][ix], d['z'][ix])
            primary.append(api.core.losses(prediction, d['y'][ix], segment, settings['query_batch_size'], rms)['total'])
            terms = api.parent.previous.old.decision.terms(prediction, d['y'][ix], segment,
                settings['query_batch_size'], d['eligible'][ix], rms, temperature)
            regret.append(terms['regret']); excess.append(terms['risk_excess'])
            d['draw_hash'] = hashlib.sha256(bytes.fromhex(d['draw_hash'])+qids.tobytes()+ix.tobytes()).hexdigest()
            d['row_draws'] += len(ix)
        primary = torch.stack(primary).mean(); regret = torch.stack(regret).mean()
        excess = torch.stack(excess)/scales
        decision = regret+(dual.detach()*excess).sum()/len(data)
        loss = primary+decision if objective_arm == 'coupled' else primary
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite continuation loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        if objective_arm == 'coupled':
            dual = (dual+config['dual_learning_rate']*excess.detach()).clamp(0, config['dual_maximum'])
        current += 1
        if current == 1 or current % settings['heartbeat_every'] == 0 or current == updates:
            row = dict(step=current, loss=float(loss.detach()), primary=float(primary.detach()),
                       decision=float(decision.detach()), gradient_norm=float(norm), dual=dual.tolist())
            trace.append(row); heartbeat(state='training', objective_arm=objective_arm, **row)
        if current % settings['checkpoint_every'] == 0 or current == limit: save()
    state = api.core.read_checkpoint(path)
    if any(int(v['step']) != origin['step']+current for v in state['optimizer']['state'].values()):
        raise ValueError('Optimizer history not continued exactly')
    return state


def matched(a, b):
    if [s['continuation']['objective_arm'] for s in (a, b)] != list(ARMS):
        raise ValueError('Complete coupled/estimation pair required')
    for key in ('anchor_state', 'settings', 'objective_config', 'input_hashes', 'samplers',
                'step', 'continuation_step', 'trainable_parameters', 'total_parameters'):
        api.core.exact(a[key], b[key])
    for key in ('origin_sha256', 'updates', 'registration_sha256', 'origin_step'):
        api.core.exact(a['continuation'][key], b['continuation'][key])
    api.check_protection(a); api.check_protection(b)
