"""Coherent conditional cost moments with optional identified event supervision."""
import copy
import hashlib
from pathlib import Path
import time
from src.world_model import m3w_own_decision_cost as old

base = old.parent
np, torch, nn = old.np, old.torch, old.nn
ARMS = ('marginal', 'identified')


def factor_moments(probability, conditional):
    if (probability.ndim != 1 or conditional.shape != (len(probability), 7, 2, 3)
            or not torch.isfinite(probability).all() or not torch.isfinite(conditional).all()
            or (probability < 0).any() or (probability > 1).any() or (conditional < 0).any()):
        raise ValueError('Finite event probability and two conditional B/H/R states required')
    q = probability[:, None, None]
    mean = (1-q)*conditional[:, :, 0]+q*conditional[:, :, 1]
    return torch.stack((mean[:, :, 0], mean[:, :, 1], mean[:, :, 2],
        probability[:, None]*conditional[:, :, 1, 2],
        probability[:, None]*conditional[:, :, 1, 1]), -1)


def prepare(data, packets, ranges, easy_cut):
    event = np.full(len(data['known']), np.nan)
    class_probability = []; branch_fractions = [[], []]; branch_reference = [[], []]
    c = data['causal']
    for name, packet in sorted(packets.items()):
        at = ranges[name]; known = data['known'][at]
        np.testing.assert_array_equal(np.isfinite(packet['cv_ade']), known)
        e = (packet['cv_ade'] > 0) & (packet['cv_ade'] <= easy_cut)
        event[at] = np.where(known, e.astype(float), np.nan)
        y = data['targets'][at]; legal = c['allowed'][at] & known[:, None]
        np.testing.assert_array_equal(y[known, :, 3], (e[:, None]*y[:, :, 2])[known])
        np.testing.assert_array_equal(y[known, :, 4], (e[:, None]*y[:, :, 1])[known])
        if not known.any(): raise ValueError('Fitting source needs observed cost labels')
        class_probability.append(float(e[known].mean()))
        for status in (0, 1):
            take = known & (e == status)
            active = legal & take[:, None]
            if active.any():
                env = c['envelope'][at]/c['scales'][at, None]
                cost = y/c['scales'][at, None, None]
                ratio = cost[:, :, :2]/env.clip(1e-150)[:, :, None]
                branch_fractions[status].append(ratio[active].mean(0))
                branch_reference[status].append(float(cost[take, 0, 2].mean()))
    p = float(np.mean(class_probability))
    if not 0 < p < 1 or any(not v for v in branch_reference):
        raise ValueError('Both fitting event strata need real supervised support')
    fractions = [np.mean(v, axis=0).tolist() for v in branch_fractions]
    reference = [float(np.mean(v)) for v in branch_reference]
    priors = dict(probability=p, fractions=fractions, references=reference,
        class_weights=[.5/(1-p), .5/p], source_names=sorted(packets),
        source_equal_initialization=True, fitting_labels_only=True)
    return dict(data, event=event, priors=priors,
        input_hashes=dict(data['input_hashes'], fitting_event_labels=base.digest(event)))


class ConditionalCritic(nn.Module):
    def __init__(self, dimension, width, seed, priors, encoder_state=None):
        super().__init__(); torch.manual_seed(seed)
        self.encoder = nn.Sequential(nn.Linear(dimension+13, width), nn.SiLU())
        self.conditional = nn.Linear(2*width+2, 4)
        self.shared = nn.Linear(2*width+2, 3)
        if encoder_state is not None: self.encoder.load_state_dict(encoder_state, strict=True)
        with torch.no_grad():
            self.conditional.weight.zero_(); self.shared.weight.zero_()
            fractions = np.asarray(priors['fractions'])
            levels = np.column_stack((fractions, 1-fractions.sum(1))).clip(1e-6)
            self.conditional.bias.copy_(torch.tensor(np.log(levels[:, :2]/levels[:, 2:]).reshape(-1), dtype=torch.float32))
            reference = np.maximum(priors['references'], 1e-6)
            p = priors['probability']
            self.shared.bias.copy_(torch.tensor(np.r_[np.log(p/(1-p)), reference+np.log(-np.expm1(-reference))], dtype=torch.float32))

    def components(self, x, anchor, envelope, initial, segments, count):
        onehot = torch.eye(7, dtype=x.dtype).expand(len(x), -1, -1)
        h = self.encoder(torch.cat((x, torch.log1p(anchor.clamp_max(2980)).float(),
            onehot, torch.log1p(envelope.clamp_max(2980)).float()[:, :, None]), -1))
        context, selected = base.selected_context(h, initial, segments, count)
        sizes = torch.bincount(segments, minlength=count).float()[segments]
        metadata = torch.stack((torch.log1p(selected), selected/sizes), 1)
        features = torch.cat((h, context[:, None].expand_as(h), metadata[:, None].expand(-1, 7, -1)), -1)
        shared = self.shared(torch.cat((h[:, :6].mean(1), context, metadata), -1)).double()
        probability = torch.sigmoid(shared[:, 0])
        reference = torch.nn.functional.softplus(shared[:, 1:])
        raw = self.conditional(features).double().reshape(len(x), 7, 2, 2)
        fraction = torch.softmax(torch.cat((raw, torch.zeros_like(raw[..., :1])), -1), -1)
        benefit = torch.minimum(envelope[:, :, None]*fraction[..., 0], reference[:, None])
        harm = envelope[:, :, None]*fraction[..., 1]
        conditional = torch.stack((benefit, harm, reference[:, None].expand(-1, 7, -1)), -1)
        return dict(moments=factor_moments(probability, conditional), probability=probability,
            event_logit=shared[:, 0], conditional=conditional)


def reduced(row, valid, segments, count):
    total = row.new_zeros(count).index_add(0, segments, row*valid)
    support = row.new_zeros(count).index_add(0, segments, valid.double())
    return (total/support.clamp_min(1)).mean()


def losses(parts, data, rows, segments, count, normalizer, arm):
    if arm not in ARMS: raise ValueError('Registered marginal or identified objective required')
    c = data['causal']; known = torch.from_numpy(data['known'][rows])
    legal = torch.from_numpy(c['allowed'][rows]) & known[:, None]
    target = torch.from_numpy(np.where(data['known'][rows, None, None], data['targets'][rows]/c['scales'][rows, None, None], 0.))
    rms = torch.from_numpy(data['rms']); prediction = parts['moments']
    error = ((prediction-target)/rms).square().mean(2)
    row_error = (error*legal).sum(1)/legal.sum(1).clamp_min(1)
    point = reduced(row_error, legal.any(1), segments, count)
    query_error, complete, counts = old.query_error(prediction, data, rows, segments, count, c['initial'][rows])
    qrms = torch.as_tensor(normalizer['rms'], dtype=prediction.dtype)
    query = ((query_error/qrms).square().mean(1)*complete).mean()
    event = torch.from_numpy(np.where(data['known'][rows], data['event'][rows], 0.))
    occurrence = reduced(torch.nn.functional.binary_cross_entropy_with_logits(parts['event_logit'], event, reduction='none'), known, segments, count)
    status = event.long(); index = torch.arange(len(rows))
    conditional = parts['conditional'][index, :, status]
    conditional_error = ((conditional-target[:, :, :3])/rms[:3]).square().mean(2)
    row_conditional = (conditional_error*legal).sum(1)/legal.sum(1).clamp_min(1)
    weight = torch.as_tensor(data['priors']['class_weights'], dtype=prediction.dtype)[status]
    conditional_loss = reduced(row_conditional*weight, legal.any(1), segments, count)
    anchor = .5*(point+query)
    auxiliary = occurrence+conditional_loss
    return dict(total=anchor+(auxiliary if arm == 'identified' else 0*auxiliary),
        anchor_objective=anchor, point_moment_MSE=point, selected_query_score=query,
        occurrence_BCE=occurrence, conditional_magnitude_MSE=conditional_loss,
        auxiliary_objective=auxiliary, unknown_supervised_rows=int((legal & ~known[:, None]).sum()), **counts)


def initialize(data, warm, settings):
    encoder = {k.removeprefix('encoder.'): v for k, v in warm['model'].items() if k.startswith('encoder.')}
    return ConditionalCritic(data['causal']['features'].shape[-1], settings['width'], warm['seed'], data['priors'], encoder)


def fit(data, *, warm, arm, settings, identity, path, heartbeat=lambda **_: None, guard=lambda: None, resume=False, stop_at=None):
    if arm not in ARMS: raise ValueError('Registered objective required')
    for k, v in warm['input_hashes'].items(): base.exact(v, data['input_hashes'][k])
    c = data['causal']; model = initialize(data, warm, settings); initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = old.previous.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], warm['seed'])
    normalizer = old.previous.base.units.query_normalizer(data); budget = old.previous.steps_for(c, settings)
    monitor = base.draw_queries(c['groups'], c['source_groups'], settings['query_batch_size'], torch.Generator().manual_seed(warm['seed']+9137))
    binding = dict(arm=arm, settings=settings, identity=identity, input_hashes=data['input_hashes'], initial_model=initial,
        seed=warm['seed'], dimension=c['features'].shape[-1], rms=data['rms'], priors=data['priors'], query_normalizer=normalizer,
        query_keys=c['keys'], budget=budget, parameters=sum(p.numel() for p in model.parameters()),
        source_query_counts=list(map(len, c['source_groups'])), causal_inference_only=True,
        source_role='exposed_SOURCE_fitting_resubstitution_only', optimizer_reused=False, new_trajectory_training=False)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step=0; seconds=0.; trace=[]; draw_hash='0'*64; unknown_draws=0; row_draws=0
    if path.exists():
        if not resume: raise ValueError('Explicit unchanged-definition resume required')
        saved = base.read_checkpoint(path)
        for k, v in binding.items(): base.exact(saved[k], v)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); sampler.restore(saved['sampler']); torch.set_rng_state(saved['torch_rng'])
        step, seconds, trace, draw_hash, unknown_draws, row_draws = (saved[k] for k in ('step','seconds','trace','draw_hash','unknown_rows_drawn','row_draws'))
    limit = budget if stop_at is None else min(stop_at, budget)
    if not 0 <= step <= limit or limit <= 0: raise ValueError('Nondecreasing full-budget continuation required')
    start = time.monotonic()
    def objective(ix, seg, count):
        args = base.batch(c, ix, seg, count)
        return losses(model.components(*args), data, ix, args[-2], count, normalizer, arm)
    def diagnostic():
        with torch.no_grad(): terms = objective(monitor[0], monitor[1], settings['query_batch_size'])
        return {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in terms.items()}
    def save():
        guard(); base.save_checkpoint(path, dict(**binding, step=step, seconds=seconds+time.monotonic()-start, trace=trace,
            model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.state(), torch_rng=torch.get_rng_state(),
            draw_hash=draw_hash, unknown_rows_drawn=unknown_draws, unknown_rows_with_supervised_labels=0, row_draws=row_draws))
    if not trace: trace.append(dict(step=0, monitor=diagnostic()))
    while step < limit:
        guard(); ix, seg, qids = sampler.draw(); optimizer.zero_grad(set_to_none=True)
        terms = objective(ix, seg, settings['query_batch_size'])
        if not torch.isfinite(terms['total']) or terms['unknown_supervised_rows']: raise FloatingPointError('Finite proper-target losses and no unknown supervision required')
        terms['total'].backward(); norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True); optimizer.step()
        step+=1; unknown_draws+=int((~data['known'][ix]).sum()); row_draws+=len(ix)
        draw_hash=hashlib.sha256(bytes.fromhex(draw_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
            record=dict(step=step, budget=budget, loss=float(terms['total'].detach()), gradient_norm=float(norm), monitor=diagnostic(),
                minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts==0).sum()))
            trace.append(record); heartbeat(state='conditional_moment_training', arm=arm, **record)
        if step % settings['checkpoint_every'] == 0 or step == limit: save()
    result = base.read_checkpoint(path)
    if step == budget and result['sampler']['counts'].min() < 4: raise ValueError('Every query needs original full coverage')
    return result


def predict(state, causal, *, components=False):
    if set(causal) != base.CAUSAL_KEYS: raise ValueError('Inference takes causal inputs only, not the realized easy label')
    model = ConditionalCritic(state['dimension'], state['settings']['width'], state['seed'], state['priors'])
    model.load_state_dict(state['model']); model.eval()
    means = np.empty_like(causal['prediction']); prob = np.empty(len(means)); pending=[]; size=0
    def flush():
        rows=np.concatenate(pending); seg=np.repeat(np.arange(len(pending)), [len(g) for g in pending])
        with torch.no_grad(): parts=model.components(*base.batch(causal, rows, seg, len(pending)))
        means[rows]=parts['moments'].numpy()*causal['scales'][rows,None,None]; prob[rows]=parts['probability'].numpy()
    for group in causal['groups']:
        if pending and size+len(group)>1024: flush(); pending=[]; size=0
        pending.append(group); size+=len(group)
    if pending: flush()
    if not np.isfinite(means).all() or (means<0).any(): raise FloatingPointError('Finite coherent native moments required')
    return (means, prob) if components else means


def assert_matched(a, b):
    for k in ('settings','identity','input_hashes','initial_model','seed','dimension','rms','priors','query_normalizer',
              'query_keys','budget','parameters','source_query_counts','sampler','draw_hash','unknown_rows_drawn','row_draws','step','torch_rng'):
        base.exact(a[k], b[k])


def assert_replay(a, b):
    base.exact({k:v for k,v in a.items() if k!='seconds'}, {k:v for k,v in b.items() if k!='seconds'})
