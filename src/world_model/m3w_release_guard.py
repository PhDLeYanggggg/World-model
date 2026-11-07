"""Donor-calibrated easy-harm guard for reopened actions."""
import numpy as np

from src.world_model import m3w_causal_record_ledger as ledger

GRIDS = {'learned': (None,.01,.025,.05,.10,.20,.40,1.),
         'radius': (None,.10,.25,.50,1.,2.,4.,None)}


def roles(sources, held):
    roster = sorted(sources)
    i = roster.index(held)
    calibration = tuple(roster[(i+j)%len(roster)] for j in (1,2,3))
    fit = tuple(s for s in roster if s not in (*calibration,held))
    if len(calibration)!=3 or not fit:
        raise ValueError('Require separate held, calibration and fitting sources')
    return fit, calibration


def keep(name,index,score,released):
    if index==0:
        return np.zeros_like(released)
    cut = GRIDS[name][index]
    return released.copy() if cut is None else released & (score<=cut)


def choose_threshold(entries):
    """Only calibration outcomes may be supplied here."""
    best = 0
    best_gain = 0.
    for i, e in enumerate(entries):
        if e['preserved'] and e['lower_pp'] is not None and e['lower_pp']>best_gain:
            best, best_gain = i, e['lower_pp']
    return best


def preserves(original, candidate):
    easy = original['known_easy_net_degradation']
    new = candidate['known_easy_net_degradation']
    if easy is not None and (new is None or new>max(.02,easy)+1e-12):
        return False
    if original['completion_easy_risk_upper'] is not None and candidate['completion_easy_risk_upper'] is None:
        return False
    return not original['supported'] or candidate['supported']


def allocation(groups,value,all_cost,easy_cost,legal,witness,revised_a,revised_e,accepted):
    count = [int((witness[g]>=0).sum()) for g in groups]
    a = np.where(accepted,revised_a,all_cost)
    e = np.where(accepted,revised_e,easy_cost)
    return ledger.stream(groups,value,a,e,legal,counts=count,witness=witness)[0]
