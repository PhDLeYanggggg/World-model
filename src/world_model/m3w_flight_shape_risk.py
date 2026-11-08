"""Differentiable training terms for the unchanged recording risk ratios."""
import torch

ARMS = ('flight_point', 'flight_budget', 'flight_relative', 'flight_noncancel')


def quantities(pred, incumbent, target, valid, cv_ade, scale, radius):
    """Inputs are CV-centered; scale converts residual errors to release units."""
    label = torch.where(valid[..., None], target, 0.)
    count = valid.sum(1).clamp_min(1)

    def signed(path):
        errors = torch.linalg.vector_norm(path-label, dim=-1)
        return torch.where(valid, errors, 0.).sum(1)/count*scale-cv_ade

    known = valid.any(1)
    signed_new, signed_old = signed(pred), signed(incumbent)
    new = torch.where(known, signed_new.relu(), 0.)
    old = torch.where(known, signed_old.relu(), 0.)
    debit = radius[~known].sum()
    return new, old, signed_new, debit


def penalty(new, old, signed, debit, easy, all_mass, easy_mass, full_easy_mass, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    zero = new.sum()*0.
    if arm == 'flight_point':
        return zero

    def hinge(numerator, mass):
        # Undefined risk is left undefined in evaluation, not optimized as zero risk.
        return (numerator/mass-.02).relu() if mass > 0 else zero

    absolute = (hinge(new.sum()+debit, all_mass)
                + hinge(new[easy].sum()+debit, easy_mass)
                + hinge(signed[easy].sum(), full_easy_mass))
    if arm == 'flight_budget' or easy_mass <= 0:
        return absolute
    increase = new[easy]-old[easy]
    relative = increase.relu().sum() if arm == 'flight_noncancel' else increase.sum().relu()
    return absolute+relative/easy_mass
