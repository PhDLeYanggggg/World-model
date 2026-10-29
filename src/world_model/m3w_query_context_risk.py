"""Matched risk heads with or without contemporaneous, leave-self-out context."""
import copy
import hashlib
from pathlib import Path
import time

from src.world_model import m3w_rational_harm as parent

np, torch, core = parent.np, parent.torch, parent.core
ARMS = ('self', 'peer')
fingerprint = parent.previous.old.parent.forest.fingerprint


def peer_context(z, recordings, frames):
    """Pool the complete current query roster, never labels or a known mask."""
    z, recs, frames = map(np.asarray, (z, recordings, frames))
    if (z.ndim != 2 or not len(z) or not np.isfinite(z).all()
            or recs.shape != (len(z),) or frames.shape != (len(z),)
            or not np.isfinite(frames).all()):
        raise ValueError('Finite causal rows and aligned recording/frame keys required')
    keys = np.rec.fromarrays([recs.astype(str), frames], names='recording,frame')
    _, inverse, sizes = np.unique(keys, return_inverse=True, return_counts=True)
    sums = np.zeros((len(sizes), z.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, z)
    counts = sizes[inverse]
    out = (sums[inverse] - z) / np.maximum(counts-1, 1)[:, None]
    out[counts == 1] = z[counts == 1]
    return out.astype(np.float32), counts


class QueryHarmBranch(torch.nn.Module):
    def __init__(self, base, width):
        super().__init__()
        self.base = base
        self.risk_encoder = copy.deepcopy(base.primary.encoder)
        self.risk_output = torch.nn.Linear(2*width, 2)
        torch.nn.init.zeros_(self.risk_output.weight)
        torch.nn.init.zeros_(self.risk_output.bias)
        for p in self.base.parameters(): p.requires_grad_(False)

    def forward(self, z, envelope, context):
        p, _ = self.base(z, envelope)
        raw = self.risk_output(torch.cat((self.risk_encoder(z),
                                         self.risk_encoder(context)), 1)).double()
        h = p[:, 1].double()
        cap = (envelope-p[:, 0]).clamp_min(0).double()
        fraction = torch.minimum(h, cap)/cap.clamp_min(1e-150)
        q = (p[:, 4].double()/h.clamp_min(1e-150)).clamp(0, 1)
        changed = cap*parent.bounded_fraction(fraction, raw[:, 0])
        easy = changed*parent.bounded_fraction(q, raw[:, 1])
        return torch.stack((p[:, 0], changed.to(p.dtype), p[:, 2], p[:, 3], easy.to(p.dtype)), 1)


def initialize(anchor, arm):
    if arm not in ARMS: raise ValueError('Registered context arm required')
    return QueryHarmBranch(parent.initialize(anchor, 'isolated_risk').base, anchor['settings']['width'])


def check_protection(state):
    for k, v in state['anchor_state']['model'].items(): core.exact(v, state['model']['base.'+k])


def context_inputs(x, recs, frames, pr):
    z = np.clip((x-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
    peer, counts = peer_context(z, recs, frames)
    return z, peer, dict(rows=len(z), multiagent_rows=int((counts > 1).sum()),
        singleton_rows=int((counts == 1).sum()), maximum_query_size=int(counts.max()),
        peer_sha256=fingerprint(peer), counts_sha256=fingerprint(counts))


def fit(packets, *, anchor, anchor_sha256, arm, settings, objective_config, identity,
        experiment_sha256, path, heartbeat, resume=False, stop_at=None, guard=None):
    parent.previous.old.validate_objective(objective_config)
    if settings != anchor['settings'] or set(packets) != set(anchor['source_preprocess']):
        raise ValueError('Unchanged anchor settings and source roster required')
    data, hashes, context_stats = {}, {}, {}
    for name in sorted(packets):
        x, env, y, _, sites, recs, frames, pr = packets[name]
        core.exact(pr, anchor['source_preprocess'][name])
        hashes[name] = {k: fingerprint(a) for k, a in zip(
            ('x', 'envelope', 'primary', 'temporal', 'sites', 'recordings', 'frames'), packets[name][:-1])}
        core.exact(hashes[name], anchor['input_hashes'][name])
        z, peer, context_stats[name] = context_inputs(x, recs, frames, pr)
        known = np.isfinite(y).all(1)
        groups, sg, _ = core.sampling.query_groups(sites, recs, frames, known)
        rng = torch.Generator(); rng.set_state(anchor['samplers'][name]['rng'])
        data[name] = dict(z=torch.from_numpy(z), context=torch.from_numpy(z if arm == 'self' else peer),
            env=torch.from_numpy((env/pr['scale']).astype(np.float32)),
            y=torch.from_numpy(np.where(known[:, None], y/pr['scale'], 0).astype(np.float32)),
            eligible=torch.from_numpy(parent.previous.prior.eligible(packets[name])),
            groups=groups, sg=sg, rng=rng, draw_hash='0'*64, row_draws=0)
    model = initialize(anchor, arm)
    initial = copy.deepcopy(model.state_dict())
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rate'], weight_decay=.0001)
    binding = dict(arm=arm, settings=settings, objective_config=objective_config, identity=identity,
        experiment_sha256=experiment_sha256, anchor_sha256=anchor_sha256, anchor_state=anchor,
        initial_model=initial, input_hashes=hashes, context_stats=context_stats,
        trainable_parameters=sum(p.numel() for p in parameters),
        total_parameters=sum(p.numel() for p in model.parameters()))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, seconds, trace, dual = 0, 0., [], torch.zeros((4, 2))
    if path.exists():
        if not resume: raise ValueError('Explicit resume required')
        old = core.read_checkpoint(path)
        for k, v in binding.items(): core.exact(old[k], v)
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        torch.set_rng_state(old['torch_rng'])
        step, seconds, trace, dual = (old[k] for k in ('step', 'seconds', 'trace', 'dual'))
        for name, d in data.items():
            d['rng'].set_state(old['samplers'][name]['rng'])
            d.update({k: old['samplers'][name][k] for k in ('draw_hash', 'row_draws')})
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0 <= step <= limit <= settings['steps'] or limit <= 0: raise ValueError('Fixed update budget required')
    began = time.monotonic()
    rms, scales = torch.from_numpy(anchor['preprocess']['rms']), anchor['scales']

    def save():
        if guard: guard()
        state = dict(**binding, step=step, seconds=seconds+time.monotonic()-began, trace=trace,
            dual=dual, model=model.state_dict(), optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
            samplers={n: dict(rng=d['rng'].get_state(), draw_hash=d['draw_hash'], row_draws=d['row_draws'])
                      for n, d in data.items()}, inner_held_scored=False, new_forecaster=False)
        check_protection(state); core.save_checkpoint(path, state)
    if step == 0: save()
    while step < limit:
        optimizer.zero_grad(set_to_none=True)
        primary, regret, risk, soft = [], [], [], []
        for name, d in data.items():
            ix, seg, qids = core.sampling.draw_queries(d['groups'], d['sg'], settings['query_batch_size'], d['rng'])
            p = model(d['z'][ix], d['env'][ix], d['context'][ix]); seg = torch.from_numpy(seg)
            primary.append(core.losses(p, d['y'][ix], seg, settings['query_batch_size'], rms)['total'])
            t = parent.previous.old.decision.terms(p, d['y'][ix], seg, settings['query_batch_size'],
                d['eligible'][ix], rms, parent.previous.old.temperature(step, settings, objective_config))
            regret.append(t['regret']); risk.append(t['risk_excess']); soft.append(t['action_mean'])
            d['draw_hash'] = hashlib.sha256(bytes.fromhex(d['draw_hash'])+qids.tobytes()+ix.tobytes()).hexdigest()
            d['row_draws'] += len(ix)
        excess = torch.stack(risk)/scales
        loss = torch.stack(primary).mean()+torch.stack(regret).mean()+(dual.detach()*excess).sum()/4
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite matched objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        dual = (dual+objective_config['dual_learning_rate']*excess.detach()).clamp(0, objective_config['dual_maximum'])
        step += 1
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm),
                primary_loss=float(torch.stack(primary).mean().detach()), regret=float(torch.stack(regret).mean().detach()),
                mean_soft_action=float(torch.stack(soft).mean().detach()), dual=dual.tolist())
            trace.append(row); heartbeat(state='training', arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit: save()
    return core.read_checkpoint(path)


def assert_matched(a, b):
    if (a['arm'], b['arm']) != ARMS: raise ValueError('Self and peer matched pair required')
    for k in ('settings', 'objective_config', 'identity', 'anchor_sha256', 'anchor_state',
              'initial_model', 'input_hashes', 'context_stats', 'samplers', 'step',
              'trainable_parameters', 'total_parameters'):
        core.exact(a[k], b[k])
    check_protection(a); check_protection(b)


def predict(state, source, x, envelope, recordings, frames, *, batch_size=4096):
    anchor = state['anchor_state']
    if source not in anchor['source_preprocess']: raise ValueError('Unknown source mapping is not inferred')
    pr = anchor['source_preprocess'][source]
    parent.previous.old.parent.forest.causal_inputs(x, envelope, pr)
    check_protection(state)
    z, peer, _ = context_inputs(x, recordings, frames, pr)
    context = z if state['arm'] == 'self' else peer
    model = initialize(anchor, state['arm']); model.load_state_dict(state['model']); model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            sl = slice(start, start+batch_size)
            p = model(torch.from_numpy(z[sl]), torch.from_numpy((envelope[sl]/pr['scale']).astype(np.float32)),
                      torch.from_numpy(context[sl]))
            out.append(p.numpy().astype(float)*pr['scale'])
    return np.concatenate(out)
