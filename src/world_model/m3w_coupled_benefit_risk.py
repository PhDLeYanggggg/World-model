"""Matched selected-set critics with protected references and coupled B/H capacity."""
import copy
import hashlib
from pathlib import Path
import time
from src.world_model import m3w_query_unit_critic as units

parent = units.parent
np, torch, nn = parent.np, parent.torch, parent.nn
ARMS = ('protected_benefit', 'released_benefit')


class CoupledCritic(parent.QueryCostCritic):
    def __init__(self, dimension, width, seed, mode):
        if mode not in ARMS: raise ValueError('Registered benefit constraint required')
        super().__init__(dimension, width, seed)
        rng = torch.get_rng_state()
        self.benefit_output = nn.Linear(2*width+2, 1)
        nn.init.zeros_(self.benefit_output.weight); nn.init.zeros_(self.benefit_output.bias)
        torch.set_rng_state(rng)
        self.mode = mode

    def forward(self, x, base, envelope, initial, segments, count):
        protected = super().forward(x, base, envelope, initial, segments, count, arm='selected_set')
        if self.mode == 'protected_benefit': return protected
        onehot = torch.eye(7, dtype=x.dtype, device=x.device).expand(len(x), -1, -1)
        h = self.encoder(torch.cat((x, torch.log1p(base.clamp_max(2980)).to(x.dtype),
            onehot, torch.log1p(envelope.clamp_max(2980)).to(x.dtype)[:, :, None]), -1))
        context, selected = parent.selected_context(h, initial, segments, count)
        size = torch.bincount(segments, minlength=count).to(x.dtype)[segments]
        metadata = torch.stack((torch.log1p(selected), selected/size), 1)[:, None].expand(-1, 7, -1)
        features = torch.cat((h, context[:, None].expand_as(h), metadata), -1)
        offsets = self.output(features).double()
        shift = self.benefit_output(features).squeeze(-1).double()
        b = base.double()
        cap0 = torch.maximum((envelope.double()-b[:, :, 0]).clamp_min(0), b[:, :, 1])
        # The parent anchor's floating-roundoff protection is retained,not a risk tolerance.
        total = b[:, :, 0]+cap0; fraction = b[:, :, 0]/total.clamp_min(1e-150)
        benefit = b[:, :, 0]+total*(parent.bounded_fraction(fraction, shift)-fraction)
        benefit = torch.minimum(benefit.clamp_min(0), total)
        cap = (cap0+(b[:, :, 0]-benefit)).clamp_min(0)
        harm_fraction = torch.where(cap0 > 0, protected[:, :, 1]/cap0.clamp_min(1e-150),
            parent.bounded_fraction(torch.zeros_like(cap0), offsets[:, :, 0]))
        harm = protected[:, :, 1]+(cap-cap0)*harm_fraction
        harm = torch.minimum(harm.clamp_min(0), cap)
        easy_fraction = torch.where(protected[:, :, 1] > 0,
            protected[:, :, 4]/protected[:, :, 1].clamp_min(1e-150),
            parent.bounded_fraction(torch.zeros_like(cap), offsets[:, :, 1]))
        easy = protected[:, :, 4]+(harm-protected[:, :, 1])*easy_fraction
        easy = torch.minimum(easy.clamp_min(0), harm)
        return torch.stack((benefit, harm, b[:, :, 2], b[:, :, 3], easy), -1)


def losses(prediction, data, rows, segments, count, normalizer):
    old = units.ORIGINAL_LOSS(prediction, data, rows, segments, count)
    error, complete = units.query_error(prediction, data, rows, segments, count)
    norm = torch.as_tensor(normalizer['rms'], dtype=prediction.dtype)
    query = ((error/norm).square().mean(1)*complete).mean()
    c = data['causal']; known = torch.from_numpy(data['known'][rows])
    target = torch.from_numpy(np.where(data['known'][rows, None],
        data['targets'][rows, :, 0]/c['scales'][rows, None], 0.))
    legal = torch.from_numpy(c['allowed'][rows]) & known[:, None]
    err = ((prediction[:, :, 0]-target)/data['rms'][0]).square()
    row = (err*legal).sum(1)/legal.sum(1).clamp_min(1)
    mass = prediction.new_zeros(count).index_add(0, segments, row)
    support = prediction.new_zeros(count).index_add(0, segments, legal.any(1).to(prediction.dtype))
    benefit = (mass/support.clamp_min(1)).mean()
    return dict(old, total=.5*(old['pointwise_risk']+query+benefit),
        selected_query_score=query, benefit_MSE=benefit,
        selected_query_score_original_units=old['selected_query_score'])


def fit(data, *, mode, seed, settings, identity, path, heartbeat=lambda **_: None,
        guard=lambda: None, resume=False, stop_at=None):
    c = data['causal']; model = CoupledCritic(c['features'].shape[-1], settings['width'], seed, mode)
    initial = copy.deepcopy(model.state_dict()); normalizer = units.query_normalizer(data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    monitor = parent.draw_queries(c['groups'], c['source_groups'], settings['query_batch_size'],
        torch.Generator().manual_seed(seed+9137))
    binding = dict(mode=mode, seed=seed, settings=settings, identity=identity, input_hashes=data['input_hashes'],
        rms=data['rms'], query_normalizer=normalizer, initial_model=initial, dimension=c['features'].shape[-1],
        parameters=sum(p.numel() for p in model.parameters()), query_keys=c['keys'],
        original_references_frozen=True, original_benefit_frozen=mode == 'protected_benefit',
        inference_uses_labels=False, all_queries_include_unknown_rows=True)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, seconds, trace, draw_hash, row_draws, unknown_draws = 0, 0., [], '0'*64, 0, 0
    if path.exists():
        if not resume: raise ValueError('Explicit same-model resume required')
        old = parent.read_checkpoint(path)
        for k, v in binding.items(): parent.exact(old[k], v)
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        rng.set_state(old['sampler_rng']); torch.set_rng_state(old['torch_rng'])
        step, seconds, trace, draw_hash, row_draws, unknown_draws = (old[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'row_draws', 'unknown_rows_drawn'))
    limit = settings['steps'] if stop_at is None else min(settings['steps'], stop_at)
    if not 0 <= step <= limit or limit <= 0: raise ValueError('Nondecreasing registered update budget required')
    started = time.monotonic()

    def diagnostic():
        ix, seg, _ = monitor; args = parent.batch(c, ix, seg, settings['query_batch_size'])
        with torch.no_grad(): terms = losses(model(*args), data, ix, args[-2], settings['query_batch_size'], normalizer)
        return {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in terms.items()}

    def save():
        guard()
        parent.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=seconds+time.monotonic()-started, trace=trace, draw_hash=draw_hash, row_draws=row_draws,
            unknown_rows_drawn=unknown_draws, unknown_rows_with_loss=0,
            known_supervised_rows=data['known_supervised_rows'], unknown_rows=data['unknown_rows']))
    if not trace: trace.append(dict(step=0, monitor=diagnostic()))
    while step < limit:
        guard(); ix, seg, qids = parent.draw_queries(c['groups'], c['source_groups'], settings['query_batch_size'], rng)
        args = parent.batch(c, ix, seg, settings['query_batch_size']); optimizer.zero_grad(set_to_none=True)
        terms = losses(model(*args), data, ix, args[-2], settings['query_batch_size'], normalizer)
        if not torch.isfinite(terms['total']) or terms['unknown_supervised_rows']:
            raise FloatingPointError('Nonfinite training or future-unknown supervision')
        terms['total'].backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1; row_draws += len(ix); unknown_draws += int((~data['known'][ix]).sum())
        draw_hash = hashlib.sha256(bytes.fromhex(draw_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, loss=float(terms['total'].detach()), gradient_norm=float(norm), monitor=diagnostic())
            trace.append(row); heartbeat(state='coupled_critic_training', mode=mode, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit: save()
    return parent.read_checkpoint(path)


def assert_old_control(state, old):
    if state['mode'] != 'protected_benefit': raise ValueError('Only protected arm is the old control')
    for k in ('seed', 'settings', 'input_hashes', 'rms', 'dimension', 'query_keys', 'draw_hash',
              'row_draws', 'unknown_rows_drawn', 'sampler_rng', 'torch_rng', 'step'):
        parent.exact(state[k], old[k])
    for field in ('model', 'initial_model'):
        parent.exact({k: state[field][k] for k in old[field]}, old[field])
    if any(v.abs().count_nonzero() for k, v in state['model'].items() if k.startswith('benefit_output.')):
        raise ValueError('Protected benefit branch changed')


def assert_matched(left, right):
    for k in ('seed', 'settings', 'input_hashes', 'rms', 'query_normalizer', 'initial_model', 'dimension',
              'parameters', 'query_keys', 'draw_hash', 'row_draws', 'unknown_rows_drawn', 'sampler_rng', 'torch_rng', 'step'):
        parent.exact(left[k], right[k])
    if left['unknown_rows_with_loss'] or right['unknown_rows_with_loss']: raise ValueError('Unknown supervision')


def predict(state, causal, *, chunk_rows=1024):
    if set(causal) != parent.CAUSAL_KEYS: raise ValueError('Inference takes declared causal fields only')
    model = CoupledCritic(state['dimension'], state['settings']['width'], state['seed'], state['mode'])
    model.load_state_dict(state['model']); model.eval(); result = causal['prediction'].copy(); pending, size = [], 0
    def flush():
        rows = np.concatenate(pending); seg = np.repeat(np.arange(len(pending)), [len(g) for g in pending])
        args = parent.batch(causal, rows, seg, len(pending))
        with torch.no_grad(): p = model(*args).numpy()
        scale = causal['scales'][rows, None]
        result[rows, :, 0] += (p[:, :, 0]-args[1].numpy()[:, :, 0])*scale
        for k in (1, 4): result[rows, :, k] = p[:, :, k]*scale
    for group in causal['groups']:
        if pending and size+len(group) > chunk_rows: flush(); pending, size = [], 0
        pending.append(group); size += len(group)
    if pending: flush()
    return result
