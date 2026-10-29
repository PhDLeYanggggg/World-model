"""Compare the actual minibatch estimator with complete fixed-region gradients."""
import hashlib
import numpy as np
import torch
from src.world_model import m3w_exact_region_cost as exact
from src.world_model import m3w_policy_region_cost as old


def sampling(data, regions, settings, seed, probes=64):
    generator = torch.Generator().manual_seed(seed+7919)
    moments = set(np.linspace(0, settings['steps']-1, probes, dtype=int).tolist())
    visits = np.zeros(len(regions['keys']), np.int64)
    counts = np.zeros_like(visits); batches = []; empty = 0; digest = '0'*64
    for step in range(settings['steps']):
        rows = old.draw_batch(data['groups'], settings['batch_size'], generator)
        digest = hashlib.sha256(bytes.fromhex(digest)+rows.tobytes()).hexdigest()
        pools = regions['pool'][rows].numpy(); positive = pools[pools >= 0]
        if not len(positive):
            empty += 1
        visits[np.unique(positive)] += 1
        counts += np.bincount(positive, minlength=len(counts))
        if step in moments:
            batches.append(rows)
    return dict(batches=batches, visits=visits.tolist(), row_draws_per_region=counts.tolist(),
                zero_region_steps=empty, draw_hash=digest)


def diagnose(model, data, regions, actions, rms, batches):
    full = exact.exact_region_pass(model, data, regions, actions, rms, backward=True)
    eh = exact.exact_region_pass(model, data, regions, actions, rms, backward=True, channels=(4,))
    region, base = [], []
    parameters = list(model.parameters())
    def vector(loss, retain):
        values = torch.autograd.grad(loss, parameters, retain_graph=retain, allow_unused=True)
        return np.concatenate([np.zeros(p.numel()) if v is None else
            v.detach().double().numpy().ravel() for p,v in zip(parameters, values)])
    for rows in batches:
        p = exact.prediction(model, data, actions, rows)
        loss = old.region_loss(p, rows, regions, rms)
        region.append(vector(loss, True))
        terms = old.base.continuous_loss(p, data['truth'][rows][:, actions], data['allowed'][rows][:, actions], rms,
            underharm_weight=data['diagnostic_settings']['underharm_weight'],
            ranking_weight=data['diagnostic_settings']['ranking_weight'])
        base.append(vector(terms['total'], False))
    region, base = np.asarray(region), np.asarray(base)
    avg, ordinary = region.mean(0), base.mean(0)
    full_gradient, eh_gradient = full.pop('gradient'), eh.pop('gradient')
    return dict(exact_regions=full, exact_easy_harm=eh, minibatches=len(batches),
        gradient_norm=dict(exact=float(np.linalg.norm(full_gradient)),
            mean_minibatch=float(np.linalg.norm(avg)), base=float(np.linalg.norm(ordinary)),
            exact_EH=float(np.linalg.norm(eh_gradient))),
        minibatch_mean_vs_exact_cosine=exact.cosine(avg, full_gradient),
        minibatch_mean_relative_error=float(np.linalg.norm(avg-full_gradient)/max(np.linalg.norm(full_gradient),1e-30)),
        batch_vs_exact_cosines=[exact.cosine(v,full_gradient) for v in region],
        base_vs_exact_cosine=exact.cosine(ordinary,full_gradient),
        base_vs_EH_cosine=exact.cosine(ordinary,eh_gradient),
        other_moments_vs_EH_cosine=exact.cosine(4*full_gradient-eh_gradient,eh_gradient),
        no_optimizer_updates=True)
