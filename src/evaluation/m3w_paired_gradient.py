"""Own-fitting paired-loss gradients and disposable saved-AdamW updates."""
import copy
from src.world_model import m3w_paired_decision_risk as api

torch, np = api.torch, api.np
TERMS = ('risk', 'decision', 'joint', 'point0', 'rank0', 'point1999', 'rank1999')
VARIANTS = ('base', 'rank', 'risk_detached_rank', 'risk_protected_rank')


def weighted_terms(predictions, target, allowed, observed, segments, query_mass):
    risk, decision = predictions
    count = len(query_mass)
    supported = allowed.any(1)
    membership = api.parent.F.one_hot(segments, count).to(risk.dtype).T
    sizes = membership @ supported.float()
    weights = query_mass[segments]/sizes[segments].clamp_min(1)

    def reduce(error):
        row = (error.mean(-1)*allowed).sum(1)/allowed.sum(1).clamp_min(1)
        return (row.double()*weights*supported).sum()

    terms = dict(risk=.5*reduce((risk-target).square()),
        decision=.5*reduce(torch.where(observed[:, None, None],
            api.parent.signed_cost(decision, target), (decision-target).square())),
        joint=.5*reduce(torch.where(observed[:, None, None],
            api.previous.joint_cost(risk, decision, target), 0.)))
    score = torch.maximum(risk, decision).amax(-1)
    for step in (0, 1999):
        for mode in ('point', 'rank'):
            value = score.sum()*0
            for group, p, n, w in api.pairs(target, allowed, observed, segments, count, step):
                value = value+query_mass[group]*api.contrast(
                    score[p[:, 0], p[:, 1]], score[n[:, 0], n[:, 1]], w, 'paired_'+mode).mean()
            terms[mode+str(step)] = .5*value
    return terms


def full(state, data, guard=lambda: None):
    c = data['causal']; z = api.parent.validate(data, state); model = api.model_from(state)
    target = (api.parent.kin.parent.signed(data['truth_by_arm']['bound_aware']) /
        torch.from_numpy(api.parent.kin.parent.normalizer(data['preprocess']))).detach()
    params = list(model.parameters()); names = [n for n, _ in model.named_parameters()]
    values = {key: 0. for key in TERMS}
    gradients = {key: [torch.zeros_like(p, dtype=torch.float64) for p in params] for key in TERMS}
    counts = state['sampler']['counts']; cursor = 0
    for ix, seg in api.parent.batches(c):
        guard(); n = int(seg.max())+1
        mass = torch.tensor(counts[cursor:cursor+n]/counts.sum(), dtype=torch.float64)
        terms = weighted_terms(model(z[ix], torch.from_numpy(seg), n), target[ix],
            torch.from_numpy(c['allowed'][ix]), torch.from_numpy(data['observed_future'][ix]),
            torch.from_numpy(seg), mass)
        for i, key in enumerate(TERMS):
            if not torch.isfinite(terms[key]):
                raise FloatingPointError('Finite diagnostic objective required')
            values[key] += float(terms[key].detach())
            gs = torch.autograd.grad(terms[key], params, retain_graph=i < len(TERMS)-1, allow_unused=True)
            for total, g in zip(gradients[key], gs):
                if g is not None:
                    total.add_(g.detach().double())
        cursor += n
    if cursor != len(counts):
        raise ValueError('All original fitting queries required')
    api.base.exact(model.state_dict(), state['model'])
    return values, gradients, names


def geometry(a, b):
    x, y = (torch.cat([v.flatten().double() for v in g]) for g in (a, b))
    nx, ny, dot = float(x.norm()), float(y.norm()), float(x@y)
    return dict(anchor_norm=nx, auxiliary_norm=ny, dot=dot,
        cosine=dot/(nx*ny) if nx and ny else None,
        auxiliary_anchor_norm_ratio=ny/nx if nx else None,
        first_order_conflict=bool(dot < 0) if nx and ny else None)


def combine(gradients, names, variant):
    if variant not in VARIANTS:
        raise ValueError('Fixed disposable variants required')
    base = [sum(g) for g in zip(*(gradients[k] for k in ('risk', 'decision', 'joint')))]
    extra = [g.clone() for g in gradients['rank1999']]
    risk_ids = [i for i, name in enumerate(names) if name.startswith('risk.')]
    if not risk_ids:
        raise ValueError('Explicit separate risk parameters required')
    anchor = [gradients['risk'][i] for i in risk_ids]
    pair = [extra[i] for i in risk_ids]
    relation = geometry(anchor, pair)
    if variant == 'risk_detached_rank':
        for i in risk_ids:
            extra[i].zero_()
    if variant == 'risk_protected_rank' and relation['first_order_conflict']:
        square = sum(float(g.square().sum()) for g in anchor)
        for i in risk_ids:
            extra[i] -= relation['dot']/square*gradients['risk'][i]
    result = base if variant == 'base' else [a+b for a, b in zip(base, extra)]
    return result, relation


def virtual(state, gradients, names, variant):
    model = api.model_from(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=state['settings']['learning_rate'], weight_decay=.0001)
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    gs, relation = combine(gradients, names, variant)
    for p, g in zip(model.parameters(), gs):
        p.grad = g.to(p.dtype).clone()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), state['settings']['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Finite disposable step required')
    changed = dict(state, model=copy.deepcopy(model.state_dict()))
    return changed, dict(risk_pair_relation=relation, gradient_norm_before_clip=float(norm))
