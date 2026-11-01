"""Matched ADE regression and published logistic-deferral loss on current inputs."""
import copy
import hashlib
from pathlib import Path
import time

from src.world_model import m3w_own_decision_cost as parent
from src.world_model import m3w_regression_deferral as published

np, torch, nn = parent.np, parent.torch, parent.nn
base = parent.parent
ARMS = ('ade', 'tdef')


class QueryScores(nn.Module):
    def __init__(self, dimension, width, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(nn.Linear(dimension + 13, width), nn.SiLU())
        self.action_output = nn.Linear(2 * width + 2, 1)
        self.floor_output = nn.Linear(2 * width + 2, 1)
        for head in (self.action_output, self.floor_output):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x, anchor, envelope, initial, segments, count, allowed):
        if allowed.dtype != torch.bool or allowed.shape != x.shape[:2]:
            raise ValueError('Causal candidate availability required')
        onehot = torch.eye(7, dtype=x.dtype).expand(len(x), -1, -1)
        h = self.encoder(torch.cat((x, torch.log1p(anchor.clamp_max(2980)).float(),
            onehot, torch.log1p(envelope.clamp_max(2980)).float()[:, :, None]), -1))
        context, selected = base.selected_context(h, initial, segments, count)
        size = torch.bincount(segments, minlength=count).float()[segments]
        metadata = torch.stack((torch.log1p(selected), selected / size), 1)
        peer = context[:, None].expand_as(h)
        score = self.action_output(torch.cat((h, peer, metadata[:, None].expand(-1, 7, -1)), -1)).squeeze(-1)
        # The floor uses the same analytic pool used by the frozen reference branch.
        floor = self.floor_output(torch.cat((h[:, :6].mean(1), context, metadata), -1))
        score = torch.where(allowed, score, floor.expand_as(score))
        return torch.cat((floor, score), 1).double()


def initialize(warm):
    model = QueryScores(warm['dimension'], warm['settings']['width'], warm['seed'])
    model.encoder.load_state_dict({k.removeprefix('encoder.'): v for k, v in warm['model'].items() if k.startswith('encoder.')}, strict=True)
    return model


def training_data(data, packets, ranges):
    c = data['causal']
    costs = np.full((len(c['ids']), 8), np.nan)
    for name, p in packets.items():
        at = ranges[name]
        np.testing.assert_array_equal(c['ids'][at], p['ids'])
        np.testing.assert_array_equal(c['allowed'][at], p['allowed'])
        candidate = np.where(p['allowed'], p['ade'], p['floor_ade'][:, None])
        costs[at] = np.column_stack((p['floor_ade'], candidate))
    known = np.isfinite(costs).all(1)
    if (not np.array_equal(known, data['known'])
            or not (known | np.isnan(costs).all(1)).all()
            or (costs[known] < 0).any()):
        raise ValueError('Aligned native partial-ADE labels; unknown stays NaN')
    return dict(data, costs=costs,
        input_hashes=dict(data['input_hashes'], canonical_ADE_costs=base.digest(costs)))


def loss(scores, costs, known, segments, count, arm):
    if arm not in ARMS or scores.shape != costs.shape or scores.shape[1] != 8:
        raise ValueError('Eight matched experts and registered loss required')
    terms = []
    for q in range(count):
        take = (segments == q) & known
        if not take.any():
            terms.append(scores.sum() * 0.)
        elif arm == 'tdef':
            terms.append(published.deferral_loss(scores[take], costs[take]))
        else:
            terms.append((torch.nn.functional.softplus(scores[take]) - costs[take]).square().mean())
    return torch.stack(terms).mean()


def fit(data, *, warm, arm, settings, identity, path, heartbeat=lambda **_: None,
        guard=lambda: None, resume=False, stop_at=None):
    if arm not in ARMS:
        raise ValueError('Registered ADE or published loss required')
    for k, value in warm['input_hashes'].items():
        base.exact(data['input_hashes'][k], value)
    c = data['causal']; model = initialize(warm)
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = parent.previous.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], warm['seed'])
    budget = parent.previous.steps_for(c, settings)
    monitor = base.draw_queries(c['groups'], c['source_groups'], settings['query_batch_size'], torch.Generator().manual_seed(warm['seed'] + 9137))
    binding = dict(arm=arm, seed=warm['seed'], settings=settings, identity=identity,
        input_hashes=data['input_hashes'], initial_model=copy.deepcopy(model.state_dict()),
        dimension=warm['dimension'], parameters=sum(p.numel() for p in model.parameters()),
        query_keys=c['keys'], budget=budget, source_query_counts=list(map(len,c['source_groups'])),
        warm_encoder_only=True, optimizer_reused=False)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step = 0; elapsed = 0.; trace = []; draws = '0' * 64; unknown = 0; row_draws = 0
    if path.exists():
        if not resume:
            raise ValueError('Explicit unchanged-definition resume required')
        old = base.read_checkpoint(path)
        for k, v in binding.items():
            base.exact(old[k], v)
        model.load_state_dict(old['model']); opt.load_state_dict(old['optimizer'])
        sampler.restore(old['sampler']); torch.set_rng_state(old['torch_rng'])
        step, elapsed, trace, draws, unknown, row_draws = (old[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'unknown_rows_drawn', 'row_draws'))
    limit = budget if stop_at is None else min(stop_at, budget)
    if not 0 <= step <= limit or limit <= 0:
        raise ValueError('Nondecreasing full-coverage budget required')
    start = time.monotonic()

    def objective(ix, seg):
        args = base.batch(c, ix, seg, settings['query_batch_size'])
        scores = model(*args, torch.from_numpy(c['allowed'][ix]))
        costs = torch.from_numpy(data['costs'][ix] / c['scales'][ix, None])
        return loss(scores, costs, torch.from_numpy(data['known'][ix]), args[-2], settings['query_batch_size'], arm)

    def save():
        guard()
        base.save_checkpoint(path, dict(**binding, step=step, seconds=elapsed + time.monotonic() - start,
            model=model.state_dict(), optimizer=opt.state_dict(), sampler=sampler.state(), torch_rng=torch.get_rng_state(),
            trace=trace, draw_hash=draws, unknown_rows_drawn=unknown, row_draws=row_draws,
            unknown_rows_with_supervised_labels=0))

    if not trace:
        with torch.no_grad():
            trace.append(dict(step=0, monitor_loss=float(objective(monitor[0], monitor[1]))))
    while step < limit:
        guard(); ix, seg, qids = sampler.draw(); opt.zero_grad(set_to_none=True)
        value = objective(ix, seg)
        if not torch.isfinite(value):
            raise FloatingPointError('Nonfinite continuous-cost objective')
        value.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        opt.step(); step += 1; row_draws += len(ix); unknown += int((~data['known'][ix]).sum())
        draws = hashlib.sha256(bytes.fromhex(draws) + qids.tobytes() + ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
            with torch.no_grad():
                record = dict(step=step, budget=budget, loss=float(value.detach()), gradient_norm=float(norm),
                    monitor_loss=float(objective(monitor[0], monitor[1])), minimum_query_visits=int(sampler.counts.min()))
            trace.append(record); heartbeat(state='current_bank_deferral_training', arm=arm, **record)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    result = base.read_checkpoint(path)
    if step == budget and int(result['sampler']['counts'].min()) < settings['largest_source_cycles']:
        raise AssertionError('Complete query coverage required')
    return result


def predict(state, causal):
    if set(causal) != base.CAUSAL_KEYS:
        raise ValueError('Causal inputs only, no target/endpoint/label-mask fields')
    model = QueryScores(state['dimension'], state['settings']['width'], state['seed'])
    model.load_state_dict(state['model']); model.eval()
    result = np.empty((len(causal['ids']), 8)); pending = []; size = 0

    def flush():
        ix = np.concatenate(pending); seg = np.repeat(np.arange(len(pending)), [len(g) for g in pending])
        args = base.batch(causal, ix, seg, len(pending))
        with torch.no_grad():
            score = model(*args, torch.from_numpy(causal['allowed'][ix]))
            result[ix] = (-torch.nn.functional.softplus(score) if state['arm'] == 'ade' else score).numpy()

    for g in causal['groups']:
        if pending and size + len(g) > 1024:
            flush(); pending = []; size = 0
        pending.append(g); size += len(g)
    if pending:
        flush()
    if not np.isfinite(result).all():
        raise FloatingPointError('Finite expert scores required')
    return result


def choices(scores, allowed, frozen_risk):
    risk, ok = parent.allocation.base.validate(frozen_risk, allowed)
    guard = ok & (risk[:, :, 0] > risk[:, :, 1]) & (risk[:, :, 2] > 0) & (risk[:, :, 3] > 0)
    guard &= (risk[:, :, 1] <= .02 * risk[:, :, 2]) & (risk[:, :, 4] <= .02 * risk[:, :, 3])
    return {mode: published.choices(scores, ok, arm='physical_neural', risk_allowed=guard if mode == 'guarded' else None)
            for mode in ('raw', 'guarded')}


def assert_matched(a, b):
    for k in ('settings', 'identity', 'input_hashes', 'initial_model', 'seed', 'dimension', 'parameters',
              'query_keys', 'budget', 'sampler', 'draw_hash', 'unknown_rows_drawn', 'row_draws', 'step', 'torch_rng'):
        base.exact(a[k], b[k])


def assert_replay(a, b):
    for k in a:
        if k != 'seconds':
            base.exact(a[k], b[k])
