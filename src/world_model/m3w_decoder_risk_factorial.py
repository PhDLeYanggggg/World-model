"""Matched source-decoder versus risk-objective experiment inside TRAIN."""
import copy
import hashlib
from pathlib import Path
import time

from src.world_model import m3w_shared_train_risk as prior

old, np, torch = prior.old, prior.np, prior.torch
ARMS = ('shared_risk', 'shared_utility', 'source_risk', 'source_utility')


class SourceDecoder(torch.nn.Module):
    def __init__(self, base, sources):
        super().__init__()
        self.encoder = base.primary.encoder
        self.outputs = torch.nn.ModuleDict({name: copy.deepcopy(base.primary.output) for name in sources})
        self.auxiliary = base.auxiliary

    def forward(self, z, envelope, source):
        if source not in self.outputs:
            raise ValueError('A registered training source is required; no zero-shot mapping')
        encoded = self.encoder(z)
        raw = self.outputs[source](encoded)
        fraction = torch.softmax(torch.cat((raw[:, :2], torch.zeros_like(raw[:, :1])), 1), 1)
        benefit, harm = envelope*fraction[:, 0], envelope*fraction[:, 1]
        reference = torch.nn.functional.softplus(raw[:, 2])
        prediction = torch.stack((benefit, harm, reference,
            reference*torch.sigmoid(raw[:, 3]), harm*torch.sigmoid(raw[:, 4])), 1)
        return prediction, self.auxiliary(encoded).reshape(-1, 12, 2)


def initialize(pr, settings, seed, arm, sources):
    if arm not in ARMS or len(sources) != 4 or len(set(sources)) != 4:
        raise ValueError('Four registered sources and factorial arm required')
    base = old.parent.initialize(pr, settings, seed)
    return SourceDecoder(base, sorted(sources)) if arm.startswith('source_') else base


def forward(model, arm, source, z, envelope):
    return model(z, envelope, source) if arm.startswith('source_') else model(z, envelope)


def fit(packets, *, arm, settings, objective_config, seed, identity,
        experiment_sha256, path, heartbeat, resume=False, stop_at=None, guard=None):
    if guard:
        guard()
    old.validate_objective(objective_config)
    names = sorted(packets)
    pr = prior.shared_preprocess(packets)
    data, hashes, norms, scales = {}, {}, {}, []
    for name in names:
        x, env, y, series, sites, recs, frames, p = packets[name]
        known = np.isfinite(y).all(1)
        groups, sg, keys = old.core.sampling.query_groups(sites, recs, frames, known)
        z = np.clip((x-p['mean'])/p['std'], -p['clip'], p['clip']).astype(np.float32)
        data[name] = dict(z=torch.from_numpy(z), env=torch.from_numpy((env/p['scale']).astype(np.float32)),
            truth=torch.from_numpy(np.where(known[:, None], y/p['scale'], 0).astype(np.float32)),
            eligible=torch.from_numpy(prior.eligible(packets[name])), groups=groups, sg=sg, keys=keys,
            rng=torch.Generator().manual_seed(seed+7919), draw_hash='0'*64, row_draws=0)
        hashes[name] = {k: old.parent.forest.fingerprint(a) for k, a in zip(
            ('x', 'envelope', 'primary', 'temporal', 'sites', 'recordings', 'frames'), packets[name][:-1])}
        norms[name] = p
        scales.append(prior.local.scaling.train_budget_scales(y, sites, recs, frames, p['scale']))
    scale = torch.from_numpy(np.stack(scales))
    model = initialize(pr, settings, seed, arm, names)
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    dual = torch.zeros((4, 2))
    path = Path(path)
    step, seconds, trace = 0, 0., []
    binding = dict(arm=arm, settings=settings, objective_config=objective_config, seed=seed,
        identity=identity, experiment_sha256=experiment_sha256, preprocess=pr,
        source_preprocess=norms, input_hashes=hashes, initial_model=initial, scales=scale,
        parameter_count=sum(p.numel() for p in model.parameters()),
        primary_parameter_count=sum(p.numel() for n, p in model.named_parameters() if not n.startswith('auxiliary.')))
    if path.exists():
        if not resume:
            raise ValueError('Explicit resume required')
        state = old.core.read_checkpoint(path)
        for k, v in binding.items():
            old.core.exact(state[k], v)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        step, seconds, trace, dual = (state[k] for k in ('step', 'seconds', 'trace', 'dual'))
        for name in names:
            data[name]['rng'].set_state(state['samplers'][name]['rng'])
            for key in ('draw_hash', 'row_draws'):
                data[name][key] = state['samplers'][name][key]
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0 <= step <= limit <= settings['steps'] or limit <= 0:
        raise ValueError('Fixed nondecreasing update budget required')
    began = time.monotonic()
    rms = torch.from_numpy(pr['rms'])
    def save():
        if guard:
            guard()
        old.core.save_checkpoint(path, dict(**binding, step=step, seconds=seconds+time.monotonic()-began,
            trace=trace, dual=dual, model=model.state_dict(), optimizer=opt.state_dict(), torch_rng=torch.get_rng_state(),
            samplers={name: dict(rng=d['rng'].get_state(), draw_hash=d['draw_hash'], row_draws=d['row_draws'])
                      for name, d in data.items()}, new_forecaster=False, inner_held_scored=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    if step == 0:
        save()
    while step < limit:
        opt.zero_grad(set_to_none=True)
        primary, regret, risks, action = [], [], [], []
        for name in names:
            d = data[name]
            ix, seg, qids = old.core.sampling.draw_queries(d['groups'], d['sg'], settings['query_batch_size'], d['rng'])
            p, _ = forward(model, arm, name, d['z'][ix], d['env'][ix])
            seg = torch.from_numpy(seg)
            primary.append(old.core.losses(p, d['truth'][ix], seg, settings['query_batch_size'], rms)['total'])
            task = old.decision.terms(p, d['truth'][ix], seg, settings['query_batch_size'], d['eligible'][ix], rms,
                                     old.temperature(step, settings, objective_config))
            regret.append(task['regret'])
            risks.append(task['risk_excess'])
            action.append(task['action_mean'])
            d['row_draws'] += len(ix)
            d['draw_hash'] = hashlib.sha256(bytes.fromhex(d['draw_hash'])+qids.tobytes()+ix.tobytes()).hexdigest()
        risk = torch.stack(risks)
        excess = risk/scale
        penalty = (dual.detach()*excess).sum()/4
        loss = torch.stack(primary).mean()+torch.stack(regret).mean()+penalty
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite factorial objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        opt.step()
        if arm.endswith('_risk'):
            dual = (dual.detach()+objective_config['dual_learning_rate']*excess.detach()).clamp(0, objective_config['dual_maximum'])
        step += 1
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm),
                primary_loss=float(torch.stack(primary).mean().detach()), regret=float(torch.stack(regret).mean().detach()),
                mean_soft_action=float(torch.stack(action).mean().detach()),
                source_raw_excess=risk.detach().tolist(), dual=dual.tolist())
            trace.append(row)
            heartbeat(state='training', arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    return old.core.read_checkpoint(path)


def assert_previous_control(state, previous):
    if state['arm'] != 'shared_risk' or previous['mode'] != 'shared_source':
        raise ValueError('Exact previous source-risk control required')
    for key in ('settings', 'objective_config', 'seed', 'preprocess', 'source_preprocess',
                'input_hashes', 'initial_model', 'scales', 'step', 'trace', 'dual', 'model',
                'optimizer', 'torch_rng', 'samplers'):
        old.core.exact(state[key], previous[key])


def assert_matched(states):
    if [s['arm'] for s in states] != list(ARMS):
        raise ValueError('Complete fixed factorial required')
    for state in states[1:]:
        for key in ('settings', 'objective_config', 'seed', 'preprocess', 'source_preprocess',
                    'input_hashes', 'scales', 'step', 'samplers'):
            old.core.exact(states[0][key], state[key])
    for a, b in ((0, 1), (2, 3)):
        old.core.exact(states[a]['initial_model'], states[b]['initial_model'])
    shared, source = states[0]['initial_model'], states[2]['initial_model']
    for key in ('0.weight', '0.bias'):
        old.core.exact(shared['primary.encoder.'+key], source['encoder.'+key])
    for name in states[2]['source_preprocess']:
        for key in ('weight', 'bias'):
            old.core.exact(shared['primary.output.'+key], source['outputs.'+name+'.'+key])


def predict(state, source, x, envelope):
    if source not in state['source_preprocess']:
        raise ValueError('No registered source mapping for inference')
    p = state['source_preprocess'][source]
    old.parent.forest.causal_inputs(x, envelope, p)
    model = initialize(state['preprocess'], state['settings'], state['seed'], state['arm'], state['source_preprocess'])
    model.load_state_dict(state['model'])
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            z = np.clip((x[start:start+4096]-p['mean'])/p['std'], -p['clip'], p['clip']).astype(np.float32)
            env = (envelope[start:start+4096]/p['scale']).astype(np.float32)
            pred, _ = forward(model, state['arm'], source, torch.from_numpy(z), torch.from_numpy(env))
            result.append(pred.numpy().astype(float)*p['scale'])
    return np.concatenate(result)
