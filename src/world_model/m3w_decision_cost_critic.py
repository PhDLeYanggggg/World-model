"""Seven-action decision-cost training; SOURCE empirical duals are not safety guarantees."""
import copy
import hashlib
from pathlib import Path
import time

from src.world_model import m3w_coupled_benefit_risk as base

np, torch, nn = base.np, base.torch, base.nn
parent = base.parent


def training_layout(data):
    c = data['causal']; y = data['targets']; known = data['known']
    keys = sorted(set(zip(c['sites'], c['recordings'])))
    lookup = {key: i for i, key in enumerate(keys)}
    rec = np.array([lookup[key] for key in zip(c['sites'], c['recordings'])])
    count = len(keys); scale = np.bincount(rec, weights=c['scales'], minlength=count)
    truth = np.where(known[:, None, None], y, 0.)
    benefit, harm, reference, easy_reference, easy_harm = np.moveaxis(truth, -1, 0)
    # Wholly unknown outcomes contribute causal worst-case envelopes, not fake labels.
    gain = np.where(known[:, None], benefit-harm, -c['envelope'])
    risk = np.stack((np.where(known[:, None], harm-.02*reference, c['envelope']),
        np.where(known[:, None], easy_harm-.02*easy_reference, c['envelope']),
        np.where(known[:, None], easy_harm-benefit*(easy_reference > 0), c['envelope'])), -1)
    easy_constant = .02*np.bincount(rec, weights=truth[:, 0, 3], minlength=count)
    objective_weight = np.zeros(len(rec)); risk_weight = np.zeros(len(rec)); record_weights = np.zeros(count)
    for site, groups in zip(sorted(set(c['sites'])), c['source_groups']):
        mask = c['sites'] == site; rids = np.unique(rec[mask]); q = len(groups)
        ref = float(truth[mask, 0, 2].sum())
        if ref <= 0: raise ValueError('Positive known original SOURCE reference required')
        objective_weight[mask] = q/ref
        risk_weight[mask] = q/(len(rids)*scale[rec[mask]])
        record_weights[rids] = 1/(len(c['source_groups'])*len(rids))
    gain_rms = float(np.sqrt(data['rms'][0]**2+data['rms'][1]**2))
    return dict(record_keys=keys, record=rec, record_scales=scale, record_weights=record_weights,
        gain=gain, risk=risk, easy_constant=easy_constant,
        objective_weight=objective_weight, risk_weight=risk_weight, temperature=max(gain_rms, 1e-6))


def probabilities(prediction, allowed, temperature):
    if temperature <= 0 or prediction.ndim != 3 or prediction.shape[1:] != (7, 5):
        raise ValueError('Positive fitting-only temperature and seven moments required')
    score = (prediction[..., 0]-prediction[..., 1])/temperature
    score = score.masked_fill(~allowed, -torch.inf)
    logits = torch.cat((torch.zeros_like(score[:, :1]), score), 1)
    return logits.softmax(1)


def losses(prediction, data, rows, segments, count, normalizer, layout, dual, *, decision_weight):
    terms = base.losses(prediction, data, rows, segments, count, normalizer)
    c = data['causal']; ix = np.asarray(rows)
    prob = probabilities(prediction, torch.from_numpy(c['allowed'][ix]), layout['temperature'])[:, 1:]
    scale = torch.from_numpy(c['scales'][ix])
    gain = torch.from_numpy(layout['gain'][ix]); risk = torch.from_numpy(layout['risk'][ix])
    rec = torch.from_numpy(layout['record'][ix])
    gain_term = -(prob*gain).sum(1)*torch.from_numpy(layout['objective_weight'][ix])
    risk_term = (prob[:, :, None]*risk*dual.detach()[rec, None]).sum((1, 2))*torch.from_numpy(layout['risk_weight'][ix])
    row = gain_term+risk_term
    # SOURCE-uniform/query-uniform draws: these weights reproduce whole-record expectations.
    value = prediction.new_zeros(count).index_add(0, segments, row).mean()
    direct = value + prediction.sum()*0
    return dict(terms, total=terms['total']+decision_weight*direct,
        decision_cost=direct, expected_gain_cost=gain_term.sum()/count,
        record_risk_cost=risk_term.sum()/count,
        unknown_bound_rows=int((~data['known'][ix]).sum()), unknown_supervised_rows=0)


def full_risk(model, data, layout, chunk_rows=1024):
    c = data['causal']; totals = np.zeros((len(layout['record_keys']), 3)); pending = []; size = 0
    action_mass = 0.
    def flush():
        rows = np.concatenate(pending); seg = np.repeat(np.arange(len(pending)), [len(g) for g in pending])
        args = parent.batch(c, rows, seg, len(pending))
        with torch.no_grad():
            p = probabilities(model(*args), torch.from_numpy(c['allowed'][rows]), layout['temperature'])[:, 1:].numpy()
        np.add.at(totals, layout['record'][rows], (p[:, :, None]*layout['risk'][rows]).sum(1))
        return float(p.sum())
    for g in c['groups']:
        if pending and size+len(g) > chunk_rows: action_mass += flush(); pending = []; size = 0
        pending.append(g); size += len(g)
    if pending: action_mass += flush()
    totals[:, 2] -= layout['easy_constant']
    return totals/layout['record_scales'][:, None], action_mass/len(c['ids'])


def fit(data, *, seed, settings, decision, identity, path, heartbeat=lambda **_: None,
        guard=lambda: None, resume=False, stop_at=None, regression_control=False):
    c = data['causal']; model = base.CoupledCritic(c['features'].shape[-1], settings['width'], seed, 'released_benefit')
    initial = copy.deepcopy(model.state_dict()); normalizer = base.units.query_normalizer(data); layout = training_layout(data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    monitor = parent.draw_queries(c['groups'], c['source_groups'], settings['query_batch_size'], torch.Generator().manual_seed(seed+9137))
    dual = torch.zeros((len(layout['record_keys']), 3), dtype=torch.float64)
    binding = dict(mode='released_benefit', seed=seed, settings=settings, decision=decision,
        regression_control=regression_control, identity=identity, input_hashes=data['input_hashes'],
        rms=data['rms'], query_normalizer=normalizer, initial_model=initial, dimension=c['features'].shape[-1],
        parameters=sum(p.numel() for p in model.parameters()), query_keys=c['keys'],
        record_keys=layout['record_keys'], record_scales=layout['record_scales'],
        probability_temperature=layout['temperature'], inference_uses_labels=False,
        all_queries_include_unknown_rows=True, unknown_values_are_envelope_bounds_not_supervised_labels=True)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, seconds, trace, draw_hash, row_draws, unknown_draws = 0, 0., [], '0'*64, 0, 0
    if path.exists():
        if not resume: raise ValueError('Existing head requires explicit same-definition resume')
        old = parent.read_checkpoint(path)
        for k, v in binding.items(): parent.exact(old[k], v)
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        rng.set_state(old['sampler_rng']); torch.set_rng_state(old['torch_rng'])
        step, seconds, trace, draw_hash, row_draws, unknown_draws, dual = (old[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'row_draws', 'unknown_rows_drawn', 'dual'))
    limit = settings['steps'] if stop_at is None else min(settings['steps'], stop_at)
    if not 0 <= step <= limit or limit <= 0: raise ValueError('Fixed nondecreasing training budget required')
    began = time.monotonic()
    weight = 0. if regression_control else decision['cost_weight']
    def diagnostic():
        ix, seg, _ = monitor; args = parent.batch(c, ix, seg, settings['query_batch_size'])
        with torch.no_grad(): t = losses(model(*args), data, ix, args[-2], settings['query_batch_size'], normalizer, layout, dual, decision_weight=weight)
        return {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
    def save():
        guard(); parent.save_checkpoint(path, dict(**binding, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), step=step,
            seconds=seconds+time.monotonic()-began, trace=trace, draw_hash=draw_hash, row_draws=row_draws,
            unknown_rows_drawn=unknown_draws, unknown_rows_with_loss=0, dual=dual,
            known_supervised_rows=data['known_supervised_rows'], unknown_rows=data['unknown_rows']))
    if not trace: trace.append(dict(step=0, monitor=diagnostic()))
    while step < limit:
        guard(); ix, seg, qids = parent.draw_queries(c['groups'], c['source_groups'], settings['query_batch_size'], rng)
        args = parent.batch(c, ix, seg, settings['query_batch_size']); optimizer.zero_grad(set_to_none=True)
        terms = losses(model(*args), data, ix, args[-2], settings['query_batch_size'], normalizer, layout, dual, decision_weight=weight)
        if not torch.isfinite(terms['total']) or terms['unknown_supervised_rows']: raise FloatingPointError('Invalid direct objective')
        terms['total'].backward(); norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1; row_draws += len(ix); unknown_draws += int((~data['known'][ix]).sum())
        draw_hash = hashlib.sha256(bytes.fromhex(draw_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
        full = None
        if not regression_control and step % decision['dual_every'] == 0:
            risk, rate = full_risk(model, data, layout)
            dual = (dual+decision['dual_learning_rate']*torch.from_numpy(risk)).clamp(0, decision['dual_maximum'])
            full = dict(max_expected_record_risk=risk.max(0).tolist(), positive_expected_record_constraints=(risk > 0).sum(0).tolist(),
                soft_intervention_fraction=rate, maximum_dual=float(dual.max()), all_records_included=True)
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, loss=float(terms['total'].detach()), gradient_norm=float(norm), monitor=diagnostic(), full_record_pass=full)
            trace.append(row); heartbeat(state='decision_critic_training', **row)
        if step % settings['checkpoint_every'] == 0 or step == limit: save()
    return parent.read_checkpoint(path)


def predict(state, causal):
    if state['regression_control']: raise ValueError('Use registered original-control predictor')
    return base.predict(state, causal)


def assert_matched(state, control):
    for k in ('seed', 'settings', 'input_hashes', 'rms', 'initial_model', 'dimension', 'parameters',
              'query_keys', 'draw_hash', 'row_draws', 'unknown_rows_drawn', 'sampler_rng', 'torch_rng', 'step'):
        parent.exact(state[k], control[k])
    if state['unknown_rows_with_loss'] or control['unknown_rows_with_loss']:
        raise AssertionError('No unknown supervised targets allowed')
