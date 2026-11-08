"""Outcome-blind blend construction and nested alpha selection."""
import numpy as np


def prepare(d, neural):
    cv = d['cv']
    radii = np.linalg.norm(neural-cv,axis=-1).max(1)
    common = (radii>1e-9) & d['legal'].any(1)
    agreement = np.linalg.norm(d['ols']-cv,axis=-1).max(1)
    cap = np.minimum(1.,(.05+agreement)/radii.clip(1e-9))
    return common,cap


def apply(cv, candidate, multiplier):
    return cv + multiplier[:,None,None]*(candidate-cv)


def old_candidate(d, name):
    if name=='hindcast':
        merit = d['hindcast']
    elif name=='minimum':
        merit = -d['dist'].mean(2)
    else:
        raise ValueError(name)
    k = np.where(d['legal'],merit,-np.inf).argmax(1)
    return d['bank'][np.arange(len(k)),k]


def pick(trials):
    feasible = []
    for row in trials:
        records = row['sites']
        gains = [v['lower_pp'] for v in records]
        if (len(records)==6 and sum(v>0 for v in gains)>=4 and np.mean(gains)>0 and
            all(v['positive_easy_pct'] is not None and v['positive_easy_pct']<=2. for v in records)):
            feasible.append((float(np.mean(gains)),-row['alpha']))
    return -max(feasible)[1] if feasible else None
