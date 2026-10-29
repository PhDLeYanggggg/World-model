"""Fit-only conditioning of frozen query heads; no persistent optimization."""
import copy

from src.world_model import m3w_query_context_risk as api
from src.world_model.m3w_constraint_scale import gradient_vector, cosine
from src.evaluation import m3w_harm_jacobian_audit as previous

np, torch = api.np, api.torch
COHORTS = ('eligible', 'anchor', 'self', 'peer')


def model_from(state):
    api.check_protection(state)
    model = api.initialize(state['anchor_state'], state['arm'])
    model.load_state_dict(state['model']); model.eval()
    return model


def collect(state, source, x, envelope, recordings, frames, *, chunk=4096):
    pr = state['anchor_state']['source_preprocess'][source]
    api.parent.previous.old.parent.forest.causal_inputs(x, envelope, pr)
    z, peer, context_stats = api.context_inputs(x, recordings, frames, pr)
    context = z if state['arm'] == 'self' else peer
    model = model_from(state); captured = {}
    handles = [model.risk_output.register_forward_hook(lambda _m, _i, o: captured.update(raw=o)),
               model.base.register_forward_hook(lambda _m, _i, o: captured.update(base=o[0]))]
    values = {k: [] for k in ('prediction', 'anchor', 'jacobian', 'raw', 'cap', 'anchor_h', 'anchor_q')}
    try:
        for start in range(0, len(x), chunk):
            sl = slice(start, start+chunk)
            env = torch.from_numpy((envelope[sl]/pr['scale']).astype(np.float32))
            p = model(torch.from_numpy(z[sl]), env, torch.from_numpy(context[sl]))
            raw, base = captured['raw'], captured['base']
            # Context is already computed. Each output row depends on its own raw logits.
            jac = torch.stack([torch.autograd.grad(p[:, col].sum(), raw, retain_graph=col == 1)[0]
                               for col in (1, 4)], 1)
            cap = (env-base[:, 0]).clamp_min(0)
            hh = torch.minimum(base[:, 1], cap)
            qq = (base[:, 4].double()/base[:, 1].double().clamp_min(1e-150)).clamp(0, 1)
            for k, v in dict(prediction=p, anchor=base, jacobian=jac, raw=raw, cap=cap,
                             anchor_h=hh, anchor_q=qq).items():
                values[k].append(v.detach().numpy().astype(float))
    finally:
        for h in handles: h.remove()
    result = {k: np.concatenate(v) for k, v in values.items()}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('Nonfinite frozen diagnostic')
    result['prediction'] *= pr['scale']; result['anchor'] *= pr['scale']
    return result, context_stats


def account(values, y, cohort, scale):
    raw = previous.account(values, y, cohort, scale, 'isolated_odds')
    result = {k: raw[k] for k in ('rows', 'known', 'unknown')}
    known = np.asarray(cohort, bool) & np.isfinite(y).all(1)
    p, t = values['prediction'][known]/scale, y[known]/scale
    for name, col, ref in (('all', 1, 2), ('easy', 4, 3)):
        part = {k: v for k, v in raw[name].items() if not k.startswith('underprediction_hard_')}
        predicted = p[:, col]-.02*p[:, ref]; observed = t[:, col]-.02*t[:, ref]
        residual = observed-predicted
        part.update(predicted_excess=float(predicted.sum()), observed_excess=float(observed.sum()),
                    residual_sum=float(residual.sum()), residual_square_sum=float((residual**2).sum()),
                    positive_residual=float(np.maximum(residual, 0).sum()),
                    false_safe_rows=int(((predicted <= 0) & (observed > 1e-12)).sum()))
        result[name] = part
    previous.validate(result)
    return result


def prepare_probes(packets, state, count=8):
    """Fixed next query draws; outcome availability is used for supervised loss only."""
    batches = [dict() for _ in range(count)]
    for source, args in sorted(packets.items()):
        x, env, y, _, sites, recs, frames, pr = args
        api.core.exact(pr, state['anchor_state']['source_preprocess'][source])
        z, peer, _ = api.context_inputs(x, recs, frames, pr)
        groups, sg, _ = api.core.sampling.query_groups(sites, recs, frames, np.isfinite(y).all(1))
        generator = torch.Generator(); generator.set_state(state['samplers'][source]['rng'])
        eligible = api.parent.previous.prior.eligible(args)
        for batch in batches:
            ix, seg, qids = api.core.sampling.draw_queries(groups, sg, state['settings']['query_batch_size'], generator)
            batch[source] = dict(z=torch.from_numpy(z[ix]), peer=torch.from_numpy(peer[ix]),
                envelope=torch.from_numpy((env[ix]/pr['scale']).astype(np.float32)),
                y=torch.from_numpy((y[ix]/pr['scale']).astype(np.float32)), segments=torch.from_numpy(seg),
                eligible=torch.from_numpy(eligible[ix]), row_ids=ix, query_ids=qids,
                query_sha256=api.fingerprint(qids), rows_sha256=api.fingerprint(ix))
    return batches


def objective_parts(model, state, batch):
    rms = torch.from_numpy(state['anchor_state']['preprocess']['rms'])
    primary, regret, all_risk, easy_risk = [], [], [], []
    predictions = {}
    temperature = api.parent.previous.old.temperature(state['step'], state['settings'], state['objective_config'])
    for i, (source, d) in enumerate(sorted(batch.items())):
        context = d['z'] if state['arm'] == 'self' else d['peer']
        p = model(d['z'], d['envelope'], context); predictions[source] = p
        primary.append(api.core.losses(p, d['y'], d['segments'], state['settings']['query_batch_size'], rms)['total'])
        term = api.parent.previous.old.decision.terms(p, d['y'], d['segments'],
            state['settings']['query_batch_size'], d['eligible'], rms, temperature)
        regret.append(term['regret'])
        weighted = state['dual'][i].detach()*term['risk_excess']/state['anchor_state']['scales'][i]
        all_risk.append(weighted[0]); easy_risk.append(weighted[1])
    return {k: torch.stack(v).mean() for k, v in dict(primary=primary, regret=regret,
            dual_all=all_risk, dual_easy=easy_risk).items()}, predictions


def fixed_score_mse(predictions, batch, masks):
    result = {}
    for name, col, ref in (('all', 1, 2), ('easy', 4, 3)):
        sums, count = 0., 0
        for source, p in predictions.items():
            mask = masks[source]
            residual = (p[:, col]-.02*p[:, ref])-(batch[source]['y'][:, col]-.02*batch[source]['y'][:, ref])
            sums += float(residual[mask].detach().double().square().sum()); count += int(mask.sum())
        result[name] = sums/count if count else None
    return result


def additivity_error(gradients, total, *, relative_limit):
    magnitude = sum(g.abs() for g in gradients.values())
    error = sum(gradients.values())-total
    if not torch.isfinite(magnitude).all() or not torch.isfinite(total).all():
        raise FloatingPointError('Nonfinite component gradient')
    denominator = float(magnitude.norm())
    relative = float(error.norm())/denominator if denominator else float(error.norm())
    if relative > relative_limit:
        raise AssertionError('Gradient decomposition exceeds precision-aware norm bound')
    return dict(relative_component_norm_error=relative, max_abs_error=float(error.abs().max()),
                component_magnitude_norm=denominator, relative_limit=relative_limit)


def verify_additivity(model, state, batch, gradients, total):
    # Separate backward reductions differ in float32, especially after cancellation.
    native = additivity_error(gradients, total, relative_limit=32*torch.finfo(torch.float32).eps)
    clone = copy.deepcopy(model).double()
    precise = {s: {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point()
                   else v for k, v in d.items()} for s, d in batch.items()}
    parts, _ = objective_parts(clone, state, precise)
    params = [p for p in clone.parameters() if p.requires_grad]
    gs = {k: gradient_vector(v, params).double() for k, v in parts.items()}
    reference = additivity_error(gs, gradient_vector(sum(parts.values()), params).double(),
                                 relative_limit=1e-10)
    return dict(native_float32=native, reference_float64=reference)


def gradient_probe(state, batch, common_masks):
    """Two one-step disposable copies share the exact saved AdamW state."""
    model = model_from(state); parameters = [p for p in model.parameters() if p.requires_grad]
    parts, prediction = objective_parts(model, state, batch)
    gradients = {k: gradient_vector(v, parameters).double() for k, v in parts.items()}
    total = gradient_vector(sum(parts.values()), parameters).double()
    additivity = verify_additivity(model, state, batch, gradients, total)
    before = {k: float(v.detach()) for k, v in parts.items()}; before['total'] = sum(before.values())
    risk, data = gradients['dual_all']+gradients['dual_easy'], gradients['primary']+gradients['regret']
    actual = {}
    for mode in ('as_trained', 'unclipped'):
        clone = copy.deepcopy(model); params = [p for p in clone.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=state['settings']['learning_rate'], weight_decay=.0001)
        optimizer.load_state_dict(copy.deepcopy(state['optimizer'])); optimizer.zero_grad(set_to_none=True)
        terms, _ = objective_parts(clone, state, batch); sum(terms.values()).backward()
        if mode == 'as_trained':
            torch.nn.utils.clip_grad_norm_(params, state['settings']['gradient_clip'], error_if_nonfinite=True)
        elif not all(torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError('Nonfinite counterfactual gradient')
        optimizer.step()
        with torch.no_grad():
            after, pp = objective_parts(clone, state, batch)
        after = {k: float(v) for k, v in after.items()}; after['total'] = sum(after.values())
        delta = torch.cat([(q-p).detach().double().reshape(-1) for p, q in zip(parameters, params)])
        steps = [int(v['step']) for v in optimizer.state.values()]
        if set(steps) != {state['step']+1}:
            raise ValueError('Counterfactual optimizer did not preserve history')
        actual[mode] = dict(after=after, change={k: after[k]-before[k] for k in before},
            parameter_step_norm=float(delta.norm()), first_order_total=float(total@delta),
            fixed_self_score_MSE=fixed_score_mse(pp, batch, common_masks), optimizer_step=steps[0])
    api.core.exact(model.state_dict(), state['model'])
    return dict(before=before, additivity=additivity,
        gradient_norms={k: float(v.norm()) for k, v in gradients.items()},
        total_gradient_norm=float(total.norm()), data_risk_cosine=cosine(data, risk),
        weighted_risk_to_data_norm=float(risk.norm()/data.norm()) if data.norm() > 0 else None,
        clip_factor=min(1., state['settings']['gradient_clip']/(float(total.norm())+1e-6)),
        fixed_self_score_MSE_before=fixed_score_mse(prediction, batch, common_masks), virtual_steps=actual,
        rows={s: len(d['z']) for s, d in batch.items()},
        draw_hashes={s: {k: d[k] for k in ('query_sha256', 'rows_sha256')} for s, d in batch.items()},
        persistent_model_changed=False, source_packets_changed=False)


def summarize(rows, probes):
    if len(rows) != 72 or len({(r['context'], r['source'], r['seed']) for r in rows}) != 72:
        raise ValueError('All72 fitting views required')
    if len(probes) != 36 or len({(p['context'], p['seed'], p['arm']) for p in probes}) != 36:
        raise ValueError('All36 gradient probe groups required')
    result = {}
    for arm in api.ARMS:
        cohorts = {}
        for cohort in COHORTS:
            entries = [r['result'][arm][cohort] for r in rows]
            value = {k: sum(r[k] for r in entries) for k in ('rows', 'known', 'unknown')}
            for target in ('all', 'easy'):
                value[target] = {k: sum(r[target][k] for r in entries) for k in entries[0][target]}
            previous.validate(value); cohorts[cohort] = value
        pp = [x for p in probes if p['arm'] == arm for x in p['probes']]
        def distribution(values):
            values = [v for v in values if v is not None]
            return dict(defined=len(values), median=float(np.median(values)) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)
        comparisons = {}
        for measure in ('total', 'primary', 'regret', 'dual_all', 'dual_easy'):
            changes = [p['virtual_steps']['unclipped']['after'][measure]-p['virtual_steps']['as_trained']['after'][measure] for p in pp]
            tolerances = [max(1e-10, 1e-6*abs(p['before'][measure])) for p in pp]
            comparisons[measure] = dict(unclipped_minus_as_trained=distribution(changes),
                better=sum(v < -e for v, e in zip(changes, tolerances)),
                worse=sum(v > e for v, e in zip(changes, tolerances)),
                tied=sum(abs(v) <= e for v, e in zip(changes, tolerances)))
        result[arm] = dict(cohorts=cohorts, probes=len(pp),
            clip_active=sum(p['clip_factor'] < 1 for p in pp),
            clip_factor=distribution([p['clip_factor'] for p in pp]),
            data_risk_cosine=distribution([p['data_risk_cosine'] for p in pp]),
            weighted_risk_to_data_norm=distribution([p['weighted_risk_to_data_norm'] for p in pp]),
            virtual_comparison=comparisons)
    return dict(arms=result, fit_views=72, gradient_probe_groups=36,
        disposable_optimizer_steps=sum(len(p['probes'])*2 for p in probes),
        no_persistent_training=True, held_inference=False, independent_roles_read=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
