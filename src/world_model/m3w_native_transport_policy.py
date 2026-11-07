"""Frozen native-8/12 transport choices from one past query at a time."""
import numpy as np

from src.world_model import m3w_record_joint_actions as prior


MATCHED = ('independent', 'scene', 'pair_additive', 'joint')
ARMS = (*MATCHED, 'gated_joint')


def gate(choice, gain, all_excess, easy_excess, predicted_easy_reference):
    selected = np.flatnonzero(choice >= 0)
    at = selected, choice[selected]
    ok = (len(selected) > 0 and gain[at].sum() > 0. and all_excess[at].sum() <= 0.
          and easy_excess[at].sum() <= 0. and predicted_easy_reference[at].sum() > 0.)
    return choice.copy() if ok else np.full_like(choice, -1)


def decide(history, bank, floor, ols, crowd, coefficients):
    if (len(history) != len(bank) or not np.isfinite(history).all() or not np.isfinite(bank).all()
            or not np.isfinite(floor).all() or np.any(crowd <= 0)):
        raise ValueError('Require aligned finite past inputs and candidate paths')
    x, scale = prior.descriptors(history, bank, floor, crowd)
    gain, all_excess, easy_excess, legal = prior.decision_costs(x, coefficients, bank, floor, scale)
    proposed = prior.choose_query(history, floor, bank, ols, scale, gain, legal)[:4]
    expected = np.einsum('nad,adc->nac', x, coefficients)
    easy_ref = np.minimum(expected[..., 4].clip(0.), expected[..., 3].clip(0.))
    guarded = gate(proposed[3], gain, all_excess, easy_excess, easy_ref*scale[:, None])
    out = dict(zip(ARMS, (*proposed, guarded)))
    if len({int((z >= 0).sum()) for z in proposed}) != 1:
        raise ValueError('Matched action counts changed')
    for z in out.values():
        selected = np.flatnonzero(z >= 0)
        if not legal[selected, z[selected]].all():
            raise ValueError('Identity forecast counted as an intervention')
    return out
