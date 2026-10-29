"""Exact empirical recording constraints; no population-safety guarantee."""
import copy
import hashlib
from pathlib import Path
import time

from src.world_model import m3w_direct_policy as base

torch, np = base.torch, base.np
ARMS = ('exact_dual', 'exact_gradient')


def layout(data):
    sizes = torch.tensor([len(g) for g in data['groups']], dtype=torch.long)
    index = torch.cat(data['groups'])
    if len(index.unique()) != len(index) or (sizes <= 0).any():
        raise ValueError('Nonempty disjoint recording groups required')
    if sizes.tolist() != data['sizes'] or int(index.max()) >= len(data['z']):
        raise ValueError('Aligned row sizes required')
    record = torch.repeat_interleave(torch.arange(len(sizes)), sizes)
    return dict(index=index, record=record, sizes=sizes,
                weight=data['weights'][record]/sizes[record].float())


def full_pass(model, data, loc, dual, temperature, chunk_size, backward=False):
    """One exact finite-population objective, chunked without optimizer steps."""
    n = len(loc['sizes'])
    gain = torch.zeros(n, dtype=torch.float64)
    risk = torch.zeros((n, 2), dtype=torch.float64)
    selected = torch.zeros(n, dtype=torch.float64)
    loss_value, gap = 0., 0.
    for start in range(0, len(loc['index']), chunk_size):
        ix = loc['index'][start:start+chunk_size]; rec = loc['record'][start:start+chunk_size]
        with torch.set_grad_enabled(backward):
            logit = model(data['z'][ix], data['anchor'][ix])
            action = base.decision(logit, True, temperature)
            coefficient = (-data['gain'][ix]+(dual.detach()[rec]*data['risk'][ix]
                                             /data['record_scales'][rec]).sum(1))
            loss = (action*coefficient*loc['weight'][start:start+chunk_size]).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite full-record objective')
            if backward: loss.backward()
        a = action.detach().double()
        gain.index_add_(0, rec, a*data['gain'][ix].double())
        risk.index_add_(0, rec, a[:, None]*data['risk'][ix].double())
        selected.index_add_(0, rec, a)
        loss_value += float(loss.detach())
        gap += float((torch.sigmoid(logit.detach()/temperature)-a).abs().sum())
    sizes = loc['sizes'].double()
    return dict(loss=loss_value, gain=gain/sizes, risk=risk/sizes[:, None],
        action=selected/sizes, soft_hard_gap=gap/len(loc['index']))


def sampled_loss(model, data, ix, dual, temperature):
    logits = model(data['z'][ix.flatten()], data['anchor'][ix.flatten()]).reshape(ix.shape)
    action = base.decision(logits, True, temperature)
    gain = (action*data['gain'][ix]).mean(1)
    risk = (action[..., None]*data['risk'][ix]).mean(1)/data['record_scales']
    return -(data['weights']*gain).sum()+(data['weights'][:, None]*dual.detach()*risk).sum()


def sampling_diagnostic(terms, count, replicates, seed):
    if terms.ndim != 2 or terms.shape[1] != 2 or not len(terms):
        raise ValueError('Nonempty all/easy term matrix required')
    rng = torch.Generator().manual_seed(seed)
    terms = terms.double()
    ix = torch.randint(len(terms), (replicates, count), generator=rng)
    sampled = terms[ix].mean(1); positive = (terms > 0).sum(0)
    return dict(exact_mean=terms.mean(0).tolist(), positive_count=positive.tolist(),
        miss_positive_probability=((1-positive.double()/len(terms))**count).tolist(),
        nonpositive_sample_fraction=(sampled <= 0).double().mean(0).tolist(),
        sample_mean=sampled.mean(0).tolist(), sample_std=sampled.std(0, unbiased=False).tolist(),
        rows=len(terms), rows_per_draw=count, replicates=replicates)


def diagnose(data, state, chunk_size=8192):
    model = base.Policy(state['dimension'], state['settings']['width'], state['seed'])
    model.load_state_dict(state['model']); loc = layout(data)
    stats = full_pass(model, data, loc, state['dual'], .25, chunk_size, backward=True)
    full_gradient = torch.cat([p.grad.flatten() for p in model.parameters()])
    records = []
    for i, ix in enumerate(data['groups']):
        with torch.no_grad():
            action = model(data['z'][ix], data['anchor'][ix]) > 0
        terms = action[:, None]*data['risk'][ix]
        report = sampling_diagnostic(terms, state['settings']['rows_per_record'], 256, state['seed']+i+104729)
        records.append(dict(record=data['record_keys'][i], **report,
            selected=int(action.sum()), dual=state['dual'][i].tolist()))
    rng = torch.Generator().manual_seed(state['seed']+15485863); cosine = []
    for _ in range(16):
        ix = torch.stack([g[torch.randint(len(g), (64,), generator=rng)] for g in data['groups']])
        model.zero_grad(set_to_none=True)
        sampled_loss(model, data, ix, state['dual'], .25).backward()
        grad = torch.cat([p.grad.flatten() for p in model.parameters()])
        den = float(grad.norm()*full_gradient.norm())
        cosine.append(float(torch.dot(grad, full_gradient))/den if den > 0 else None)
    return dict(records=records, exact_objective=stats['loss'],
        full_gradient_norm=float(full_gradient.norm()), sample_full_gradient_cosines=cosine,
        no_optimizer_step=True, held_read=False, new_training=False)


def fit(data, *, arm, settings, seed, identity, path, heartbeat=lambda **_: None,
        resume=False, stop_at=None, guard=lambda: None):
    if arm not in ARMS: raise ValueError('Registered estimator required')
    loc = layout(data); model = base.Policy(data['z'].shape[1], settings['width'], seed)
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    dual = torch.zeros((len(data['groups']), 2)); step, seconds, trace = 0, 0., []
    draw_hash, row_draws, forward_rows = '0'*64, 0, 0
    binding = dict(arm=arm, settings=settings, seed=seed, identity=identity, initial_model=initial,
        input_hashes=data['input_hashes'], source_preprocess=data['source_preprocess'],
        record_keys=data['record_keys'], record_sizes=data['sizes'], record_scales=data['record_scales'],
        empty_training_records=data['empty_training_records'], dimension=data['z'].shape[1],
        parameters=sum(p.numel() for p in model.parameters()))
    path = Path(path)
    if path.exists():
        if not resume: raise ValueError('Explicit resume required')
        state = base.core.read_checkpoint(path)
        for k, v in binding.items(): base.core.exact(state[k], v)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, seconds, trace, dual, draw_hash, row_draws, forward_rows = (state[k] for k in
            ('step', 'seconds', 'trace', 'dual', 'draw_hash', 'row_draws', 'whole_record_forward_rows'))
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0 <= step <= limit <= settings['steps'] or limit <= 0:
        raise ValueError('Fixed nondecreasing update budget required')
    began = time.monotonic()

    def save():
        guard(); path.parent.mkdir(parents=True, exist_ok=True)
        base.core.save_checkpoint(path, dict(**binding, step=step, optimizer_steps=step,
            seconds=seconds+time.monotonic()-began, trace=trace, dual=dual,
            model=model.state_dict(), optimizer=opt.state_dict(), sampler_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(), draw_hash=draw_hash, row_draws=row_draws,
            whole_record_forward_rows=forward_rows,
            full_gradient_rows=forward_rows if arm == 'exact_gradient' else 0,
            new_forecaster=False, held_used_for_training=False))

    if step == 0: save()
    while step < limit:
        temperature = 1.-.75*step/max(1, settings['steps']-1)
        opt.zero_grad(set_to_none=True)
        if arm == 'exact_dual':
            ix = torch.stack([g[torch.randint(len(g), (settings['rows_per_record'],), generator=rng)]
                              for g in data['groups']])
            loss = sampled_loss(model, data, ix, dual, temperature)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite sampled objective')
            loss.backward(); row_draws += ix.numel()
            draw_hash = hashlib.sha256(bytes.fromhex(draw_hash)+ix.flatten().numpy().tobytes()).hexdigest()
        stats = full_pass(model, data, loc, dual, temperature, settings['chunk_size'],
                          backward=arm == 'exact_gradient')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        opt.step()
        risk = (stats['risk']/data['record_scales']).float()
        dual = (dual+settings['dual_learning_rate']*risk).clamp(0, settings['dual_maximum'])
        step += 1; forward_rows += len(loc['index'])
        # Trace scheduling must not change when a checkpoint stops between heartbeats.
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, full_objective=stats['loss'], gradient_norm=float(norm),
                full_gain_lower=float((data['weights']*stats['gain']).sum()),
                full_action_fraction=float(stats['action'].mean()),
                exact_record_risk=stats['risk'].tolist(), dual=dual.tolist(),
                soft_hard_gap=stats['soft_hard_gap'])
            trace.append(row); heartbeat(state='training', arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit: save()
    return base.core.read_checkpoint(path)
