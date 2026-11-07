"""Full-bank comparison with the original chronological risk ledger."""
import numpy as np

from src.world_model import m3w_causal_record_ledger as ledger


def score_arms(u, e, easy_reference, scale):
    if u.shape!=e.shape or e.shape!=easy_reference.shape or scale.shape!=(len(u),):
        raise ValueError('Aligned causal costs required')
    return (u/scale[:,None],(u+.25*easy_reference)/scale[:,None],
            (u-.25*e)/scale[:,None])


def bound(groups, a, e, legal, witness, actual_a, actual_e, actual_easy_reference):
    """Necessary observed conditions at the original predicted prefix allowance."""
    slack = np.zeros(2)
    minima = np.zeros(2)
    support = 0.
    active = 0
    for g in groups:
        w = witness[g]
        n = int((w>=0).sum())
        if not n:
            continue
        active += 1
        ca,ce,ok = a[g],e[g],legal[g]
        values = (-actual_a[g],-actual_e[g],np.broadcast_to(actual_easy_reference[g,None],ca.shape))
        totals = []
        for v in values:
            c = ledger.query_choice(v,ca,ce,ok,slack,count=n)
            if c is None:
                raise ValueError('Original predicted-prefix witness must remain feasible')
            totals.append(ledger.totals(c,v,ca,ce)[0])
        minima -= totals[:2]
        support += totals[2]
        slack -= ledger.totals(w,ca,ca,ce)[1:]
    failed = [key for key,test in (('no_selected_query',not active),
        ('no_easy_reference_in_any_option',support<=0.),
        ('all_budget_lower_bound_positive',minima[0]>1e-9),
        ('easy_budget_lower_bound_positive',minima[1]>1e-9)) if test]
    return dict(failed_necessary_conditions=failed,active=active,maximum_easy_reference=float(support),
                minimum_excess=minima.tolist())
