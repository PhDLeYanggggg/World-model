"""Training-only gradient accounting for fixed source dynamics checkpoints."""
import numpy as np
import torch


def flat_gradient(loss, parameters, *, retain_graph=False):
    values = torch.autograd.grad(loss, parameters, retain_graph=retain_graph,
                                 allow_unused=True)
    result = torch.cat([
        (torch.zeros_like(p) if g is None else g).detach().reshape(-1).double()
        for p, g in zip(parameters, values)
    ]).cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite model gradient')
    return result


def gradient_relation(vector, reference):
    a, b = np.asarray(vector, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite([a, b]).all():
        raise ValueError('Aligned finite gradient vectors required')
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return dict(norm=na, reference_norm=nb,
                cosine=float(a @ b / (na * nb)) if na and nb else None,
                norm_ratio=na / nb if nb else None,
                projection_ratio=float(a @ b / (nb * nb)) if nb else None,
                relative_error=float(np.linalg.norm(a-b) / nb) if nb else None)


def clipped_gradient(gradient, cap):
    g = np.asarray(gradient, dtype=float)
    if g.ndim != 1 or not np.isfinite(g).all() or not np.isfinite(cap) or cap <= 0:
        raise ValueError('Finite gradient and positive cap required')
    return g * min(1., cap / (np.linalg.norm(g) + 1e-6))


def layer_energy(vector, named_parameters):
    offset, values = 0, {}
    for name, parameter in named_parameters:
        size = parameter.numel()
        group = 'output_head' if name.startswith('head.2.') else name.split('.')[0]
        values[group] = values.get(group, 0.) + float(vector[offset:offset+size] @ vector[offset:offset+size])
        offset += size
    if offset != len(vector):
        raise ValueError('Parameter layout changed')
    total = sum(values.values())
    return {k: dict(norm=float(np.sqrt(v)), squared_norm_share=v/total if total else 0.)
            for k, v in values.items()}


def output_shrink_curve(prediction, target, native_scale, loss_scale, alphas):
    p, y = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    native = np.asarray(native_scale, dtype=float)
    if (p.shape != y.shape or p.ndim != 3 or p.shape[1:] != (12, 2)
            or native.shape != (len(p),) or not len(p) or loss_scale <= 0
            or not np.isfinite(p).all() or not np.isfinite(y).all()
            or not np.isfinite(native).all() or np.any(native <= 0)
            or not np.isfinite(loss_scale)):
        raise ValueError('Complete aligned training forecasts and scales required')
    moving = np.any(y != 0, axis=(1, 2))
    floor = np.linalg.norm(y, axis=-1).mean(1)
    results = []
    for alpha in alphas:
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('Diagnostic interpolation to zero only')
        error = np.linalg.norm(alpha*p-y, axis=-1).mean(1)
        results.append(dict(alpha=float(alpha), uniform_loss=float(error.mean()/loss_scale),
            gain_percent=float(100*(1-error.mean()/floor.mean())) if floor.mean() else None,
            native_pixel_ade=float((error*native).mean()),
            static_pixel_harm=float((error[~moving]*native[~moving]).mean()) if (~moving).any() else None,
            nonzero_gain_percent=float(100*(1-error[moving].sum()/floor[moving].sum())) if moving.any() else None))
    return results
