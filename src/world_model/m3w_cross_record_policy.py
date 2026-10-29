"""TRAIN-only cross-record meta-updates; neither calibration nor new test data."""
import copy
import hashlib
import json
from pathlib import Path
import time

from src.world_model import m3w_whole_record_policy as whole

base, torch, np = whole.base, whole.torch, whole.np
ARMS = ('same_record_meta', 'cross_record_meta')


def partition(record_keys, seed, step):
    if len(record_keys) < 2 or len(set(record_keys)) != len(record_keys):
        raise ValueError('At least two distinct source-qualified recordings required')
    groups = {}
    for i, key in enumerate(record_keys):
        groups.setdefault(json.loads(key)[0], []).append(i)
    def rank(key):
        return hashlib.sha256(f'cross-record-episode-v1|{seed}|{step}|{key}'.encode()).hexdigest()
    a = torch.zeros(len(record_keys), dtype=torch.bool); single = []
    for source, ids in sorted(groups.items()):
        if len(ids) == 1:
            single += ids
            continue
        ordered = sorted(ids, key=lambda i: rank(record_keys[i]))
        first = int(rank(source), 16) % 2
        for j, i in enumerate(ordered): a[i] = (j+first) % 2 == 0
    for j, i in enumerate(sorted(single, key=lambda i: rank(record_keys[i]))):
        a[i] = (j+int(rank('singletons'), 16)) % 2 == 0
    if not a.any() or not (~a).any():
        raise ValueError('Both TRAIN episode sides need recordings')
    return a


def adapt(params, loss, learning_rate, clip):
    grad = torch.autograd.grad(loss, tuple(params.values()), create_graph=True)
    norm = torch.sqrt(sum(g.square().sum() for g in grad).clamp_min(1e-24))
    scale = (clip/norm).clamp(max=1.)
    return {k:p-learning_rate*scale*g for (k,p),g in zip(params.items(),grad)}, norm, scale


def fold_loss(model, params, data, loc, dual, temperature, side, chunk_size, collect=False):
    active = side[loc['record']]
    ix_all, rec_all, weight = loc['index'][active], loc['record'][active], loc['weight'][active]
    mass = data['weights'][side].sum()
    if mass <= 0: raise ValueError('Positive episode weight required')
    loss = next(iter(params.values())).new_zeros(())
    stats = dict(gain=torch.zeros(len(side), dtype=torch.float64),
        risk=torch.zeros((len(side), 2), dtype=torch.float64),
        action=torch.zeros(len(side), dtype=torch.float64), gap=0.) if collect else None
    for start in range(0, len(ix_all), chunk_size):
        ix, rec = ix_all[start:start+chunk_size], rec_all[start:start+chunk_size]
        logits = torch.func.functional_call(model, params, (data['z'][ix], data['anchor'][ix]), strict=True)
        action = base.decision(logits, True, temperature)
        coefficient = -data['gain'][ix]+(dual.detach()[rec]*data['risk'][ix]/data['record_scales'][rec]).sum(1)
        loss = loss+(action*coefficient*weight[start:start+chunk_size]/mass).sum()
        if collect:
            a = action.detach().double()
            stats['gain'].index_add_(0, rec, a*data['gain'][ix].double())
            stats['risk'].index_add_(0, rec, a[:, None]*data['risk'][ix].double())
            stats['action'].index_add_(0, rec, a)
            stats['gap'] += float((torch.sigmoid(logits.detach()/temperature)-a).abs().sum())
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite episode objective')
    return loss, stats


def meta_loss(model, data, loc, dual, temperature, side, arm, settings):
    if arm not in ARMS: raise ValueError('Fixed registered arm required')
    params = dict(model.named_parameters()); chunk = settings['chunk_size']
    la, sa = fold_loss(model, params, data, loc, dual, temperature, side, chunk, True)
    lb, sb = fold_loss(model, params, data, loc, dual, temperature, ~side, chunk, True)
    pa, na, ca = adapt(params, la, settings['inner_learning_rate'], settings['inner_gradient_clip'])
    pb, nb, cb = adapt(params, lb, settings['inner_learning_rate'], settings['inner_gradient_clip'])
    wa, wb = data['weights'][side].sum(), data['weights'][~side].sum()
    qa, _ = fold_loss(model, pa, data, loc, dual, temperature,
                      ~side if arm == 'cross_record_meta' else side, chunk)
    qb, _ = fold_loss(model, pb, data, loc, dual, temperature,
                      side if arm == 'cross_record_meta' else ~side, chunk)
    original = wa*la+wb*lb
    query = wb*qa+wa*qb if arm == 'cross_record_meta' else wa*qa+wb*qb
    beta = settings['meta_weight']; loss = (1-beta)*original+beta*query
    sizes = loc['sizes'].double()
    info = dict(loss=float(original.detach()), query_loss=float(query.detach()),
        gain=(sa['gain']+sb['gain'])/sizes, risk=(sa['risk']+sb['risk'])/sizes[:, None],
        action=(sa['action']+sb['action'])/sizes,
        soft_hard_gap=(sa['gap']+sb['gap'])/len(loc['index']),
        inner_gradient_norms=[float(na.detach()),float(nb.detach())],
        inner_clip_factors=[float(ca.detach()),float(cb.detach())],
        side_a_records=int(side.sum()), side_a_weight=float(wa))
    return loss, info


def fit(data, *, arm, settings, seed, identity, path, heartbeat=lambda **_: None,
        resume=False, stop_at=None, guard=lambda: None):
    if arm not in ARMS: raise ValueError('Registered arm required')
    if not 0 <= settings['meta_weight'] <= 1 or settings['inner_learning_rate'] < 0:
        raise ValueError('Valid fixed meta objective required')
    loc = whole.layout(data); model = base.Policy(data['z'].shape[1], settings['width'], seed)
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    binding = dict(arm=arm, settings=settings, seed=seed, identity=identity, initial_model=initial,
        input_hashes=data['input_hashes'], source_preprocess=data['source_preprocess'],
        record_keys=data['record_keys'], record_sizes=data['sizes'], record_scales=data['record_scales'],
        empty_training_records=data['empty_training_records'], dimension=data['z'].shape[1],
        parameters=sum(p.numel() for p in model.parameters()))
    dual = torch.zeros((len(data['groups']), 2)); step, seconds, trace, episode_hash = 0, 0., [], '0'*64
    path = Path(path)
    if path.exists():
        if not resume: raise ValueError('Explicit resume required')
        saved = base.core.read_checkpoint(path)
        for k,v in binding.items(): base.core.exact(saved[k], v)
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        step, seconds, trace, dual, episode_hash = (saved[k] for k in ('step','seconds','trace','dual','episode_hash'))
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0 <= step <= limit <= settings['steps'] or limit <= 0:
        raise ValueError('Nondecreasing fixed update budget required')
    began = time.monotonic()
    def save():
        guard(); path.parent.mkdir(parents=True, exist_ok=True)
        base.core.save_checkpoint(path, dict(**binding, step=step, optimizer_steps=step,
            seconds=seconds+time.monotonic()-began, trace=trace, dual=dual, episode_hash=episode_hash,
            model=model.state_dict(), optimizer=opt.state_dict(), torch_rng=torch.get_rng_state(),
            meta_forward_rows=2*step*len(loc['index']), second_order_gradient=True,
            new_forecaster=False, held_used_for_training=False, test_time_adaptation=False))
    if step == 0: save()
    while step < limit:
        side = partition(data['record_keys'], seed, step)
        temperature = 1.-.75*step/max(1, settings['steps']-1)
        opt.zero_grad(set_to_none=True)
        loss, stats = meta_loss(model, data, loc, dual, temperature, side, arm, settings)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        opt.step()
        dual = (dual+settings['dual_learning_rate']*(stats['risk']/data['record_scales']).float()).clamp(0, settings['dual_maximum'])
        episode_hash = hashlib.sha256(bytes.fromhex(episode_hash)+side.numpy().tobytes()).hexdigest()
        step += 1
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, meta_objective=float(loss.detach()), full_objective=stats['loss'],
                query_objective=stats['query_loss'], gradient_norm=float(norm),
                full_gain_lower=float((data['weights']*stats['gain']).sum()),
                full_action_fraction=float(stats['action'].mean()), exact_record_risk=stats['risk'].tolist(),
                dual=dual.tolist(), soft_hard_gap=stats['soft_hard_gap'],
                inner_gradient_norms=stats['inner_gradient_norms'], inner_clip_factors=stats['inner_clip_factors'],
                side_a_records=stats['side_a_records'], side_a_weight=stats['side_a_weight'])
            trace.append(row); heartbeat(state='training', arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit: save()
    return base.core.read_checkpoint(path)
