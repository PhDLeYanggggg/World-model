"""Warm-started harm learning with an immutable benefit/reference floor."""
import copy
import hashlib
from pathlib import Path
import time

from src.world_model import m3w_decoder_risk_factorial as previous

np, torch, core = previous.np, previous.torch, previous.old.core
ARMS = ('utility_continue', 'joint_risk', 'isolated_risk')


class HarmBranch(torch.nn.Module):
    def __init__(self, base, width, isolate):
        super().__init__()
        self.base = base
        self.risk_encoder = copy.deepcopy(base.primary.encoder)
        self.risk_output = torch.nn.Linear(width, 2)
        torch.nn.init.zeros_(self.risk_output.weight)
        torch.nn.init.zeros_(self.risk_output.bias)
        for name, param in self.base.named_parameters():
            param.requires_grad_(not isolate and not name.startswith('auxiliary.'))

    def forward(self, z, envelope):
        p, _ = self.base(z, envelope)
        delta = 5*torch.tanh(self.risk_output(self.risk_encoder(z))/5)
        h, easy = p[:, 1].double(), p[:, 4].double()
        cap = (envelope-p[:, 0]).clamp_min(0).double()
        h0 = torch.minimum(h, cap)
        u, v = torch.expm1(delta.double()).unbind(1)
        tiny = torch.finfo(p.dtype).tiny
        # Float64 ratios avoid squared-denominator underflow at zero/tiny envelopes.
        # Odds shifts written as increments preserve the zero-branch identity.
        changed = h0+h0*(cap-h0)*u/(cap+h0*u).clamp_min(tiny)
        changed = torch.minimum(changed.clamp_min(0), cap)
        q = (easy/h.clamp_min(tiny)).clamp(0, 1)
        changed_easy = easy+(changed-h)*q+changed*q*(1-q)*v/(1+q*v)
        changed_easy = torch.minimum(changed_easy.clamp_min(0), changed)
        return torch.stack((p[:, 0], changed.to(p.dtype), p[:, 2], p[:, 3], changed_easy.to(p.dtype)), 1)


def initialize(anchor, arm):
    if arm not in ARMS or anchor['arm'] != 'shared_utility':
        raise ValueError('Fixed shared-utility anchor and registered continuation arm required')
    base = previous.initialize(anchor['preprocess'], anchor['settings'], anchor['seed'],
                               'shared_utility', anchor['source_preprocess'])
    base.load_state_dict(anchor['model'])
    return HarmBranch(base, anchor['settings']['width'], arm == 'isolated_risk')


def check_protection(state):
    if state['arm'] == 'isolated_risk':
        for key, tensor in state['anchor_state']['model'].items():
            core.exact(tensor, state['model']['base.'+key])


def fit(packets, *, anchor, anchor_sha256, arm, settings, objective_config, identity,
        experiment_sha256, path, heartbeat, resume=False, stop_at=None, guard=None):
    previous.old.validate_objective(objective_config)
    if settings != anchor['settings'] or set(packets) != set(anchor['source_preprocess']):
        raise ValueError('Same settings and four anchor sources required')
    if guard:
        guard()
    data, hashes = {}, {}
    for name in sorted(packets):
        x, env, y, _, sites, recs, frames, pr = packets[name]
        core.exact(pr, anchor['source_preprocess'][name])
        hashes[name] = {k: previous.old.parent.forest.fingerprint(a) for k, a in zip(
            ('x', 'envelope', 'primary', 'temporal', 'sites', 'recordings', 'frames'), packets[name][:-1])}
        core.exact(hashes[name], anchor['input_hashes'][name])
        known = np.isfinite(y).all(1)
        groups, sg, _ = core.sampling.query_groups(sites, recs, frames, known)
        rng = torch.Generator()
        rng.set_state(anchor['samplers'][name]['rng'])
        data[name] = dict(z=torch.from_numpy(np.clip((x-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)),
            env=torch.from_numpy((env/pr['scale']).astype(np.float32)),
            y=torch.from_numpy(np.where(known[:, None], y/pr['scale'], 0).astype(np.float32)),
            eligible=torch.from_numpy(previous.prior.eligible(packets[name])), groups=groups, sg=sg,
            rng=rng, draw_hash='0'*64, row_draws=0)
    model = initialize(anchor, arm)
    initial = copy.deepcopy(model.state_dict())
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rate'], weight_decay=.0001)
    dual = torch.zeros((4, 2))
    binding = dict(arm=arm, settings=settings, objective_config=objective_config, identity=identity,
        experiment_sha256=experiment_sha256, anchor_sha256=anchor_sha256, anchor_state=anchor,
        initial_model=initial, input_hashes=hashes,
        trainable_parameters=sum(p.numel() for p in parameters),
        total_parameters=sum(p.numel() for p in model.parameters()))
    path = Path(path)
    step, seconds, trace = 0, 0., []
    if path.exists():
        if not resume:
            raise ValueError('Explicit resume required')
        old = core.read_checkpoint(path)
        for key, value in binding.items():
            core.exact(old[key], value)
        model.load_state_dict(old['model'])
        optimizer.load_state_dict(old['optimizer'])
        torch.set_rng_state(old['torch_rng'])
        step, seconds, trace, dual = (old[k] for k in ('step', 'seconds', 'trace', 'dual'))
        for name, d in data.items():
            d['rng'].set_state(old['samplers'][name]['rng'])
            d.update({k: old['samplers'][name][k] for k in ('draw_hash', 'row_draws')})
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0 <= step <= limit <= settings['steps'] or limit <= 0:
        raise ValueError('Fixed continuation budget required')
    started = time.monotonic()
    rms, scales = torch.from_numpy(anchor['preprocess']['rms']), anchor['scales']

    def save():
        if guard:
            guard()
        value = dict(**binding, step=step, seconds=seconds+time.monotonic()-started, trace=trace,
            dual=dual, model=model.state_dict(), optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
            samplers={n: dict(rng=d['rng'].get_state(), draw_hash=d['draw_hash'], row_draws=d['row_draws'])
                      for n, d in data.items()}, inner_held_scored=False, new_forecaster=False)
        check_protection(value)
        core.save_checkpoint(path, value)

    path.parent.mkdir(parents=True, exist_ok=True)
    if step == 0:
        save()
    while step < limit:
        optimizer.zero_grad(set_to_none=True)
        primary, regret, risk, soft = [], [], [], []
        for name, d in data.items():
            ix, seg, qids = core.sampling.draw_queries(d['groups'], d['sg'], settings['query_batch_size'], d['rng'])
            p = model(d['z'][ix], d['env'][ix])
            seg = torch.from_numpy(seg)
            primary.append(core.losses(p, d['y'][ix], seg, settings['query_batch_size'], rms)['total'])
            t = previous.old.decision.terms(p, d['y'][ix], seg, settings['query_batch_size'], d['eligible'][ix], rms,
                previous.old.temperature(step, settings, objective_config))
            regret.append(t['regret']); risk.append(t['risk_excess']); soft.append(t['action_mean'])
            d['draw_hash'] = hashlib.sha256(bytes.fromhex(d['draw_hash'])+qids.tobytes()+ix.tobytes()).hexdigest()
            d['row_draws'] += len(ix)
        excess = torch.stack(risk)/scales
        loss = torch.stack(primary).mean()+torch.stack(regret).mean()+(dual.detach()*excess).sum()/4
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite continuation objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        if arm != 'utility_continue':
            dual = (dual+objective_config['dual_learning_rate']*excess.detach()).clamp(0, objective_config['dual_maximum'])
        step += 1
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm),
                primary_loss=float(torch.stack(primary).mean().detach()), regret=float(torch.stack(regret).mean().detach()),
                mean_soft_action=float(torch.stack(soft).mean().detach()), dual=dual.tolist())
            trace.append(row)
            heartbeat(state='training', arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    return core.read_checkpoint(path)


def assert_matched(states):
    if [s['arm'] for s in states] != list(ARMS):
        raise ValueError('All continuation controls required')
    for other in states[1:]:
        for key in ('settings', 'objective_config', 'identity', 'experiment_sha256', 'anchor_sha256',
                    'anchor_state', 'initial_model', 'input_hashes', 'samplers', 'step', 'total_parameters'):
            core.exact(states[0][key], other[key])
    check_protection(states[-1])
    if bool((states[0]['dual'] != 0).any()):
        raise ValueError('Utility continuation must not receive a risk penalty')


def predict(state, source, x, envelope):
    anchor = state['anchor_state']
    if source not in anchor['source_preprocess']:
        raise ValueError('No inferred mapping for an unknown source')
    pr = anchor['source_preprocess'][source]
    previous.old.parent.forest.causal_inputs(x, envelope, pr)
    check_protection(state)
    model = initialize(anchor, state['arm'])
    model.load_state_dict(state['model'])
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            z = np.clip((x[start:start+4096]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
            env = (envelope[start:start+4096]/pr['scale']).astype(np.float32)
            out.append(model(torch.from_numpy(z), torch.from_numpy(env)).numpy().astype(float)*pr['scale'])
    return np.concatenate(out)
