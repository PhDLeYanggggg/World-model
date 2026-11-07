"""Fit candidate-specific easy-support events and test one causal opening action."""
import numpy as np

from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments
from src.world_model import m3w_record_joint_actions as original

THRESHOLDS = (.0, .1, .2, .35, .5, .65, .8)


def labels(floor, bank, target, valid, cut):
    m = [moments(floor,path,target,valid,cut) for path in bank.swapaxes(0,1)]
    stack = lambda k: np.column_stack([v[k] for v in m])
    easy = m[0]['easy']
    gain, radius = stack('diff'), stack('radius')
    all_cost = stack('all_excess')
    safe = (all_cost<=0.) & (radius>0.)
    joint = safe & easy[:,None]
    unsafe = (all_cost>0.) & (radius>0.)
    event = np.stack((joint,np.broadcast_to(easy[:,None],joint.shape),safe,unsafe),axis=-1)
    known = m[0]['known']
    harm = np.maximum(-gain,0.)+np.where(known[:,None],0.,radius)
    # Additive masses for the unchanged recording-level risk ratios.
    atoms = np.stack((np.broadcast_to(m[0]['base'][:,None],gain.shape),
                      np.broadcast_to(m[0]['ref_easy'][:,None],gain.shape),
                      harm,np.maximum(-gain,0.)*easy[:,None]+np.where(known[:,None],0.,radius),
                      stack('easy_signed'),stack('utility')),axis=-1)
    return event, atoms, float(m[0]['ref_easy'].sum())


def fitting(x,event,legal,crowd):
    grams, rhs = [], []
    for j,z in enumerate(x.swapaxes(0,1)):
        w = legal[:,j]/crowd
        w /= max(w.sum(),1e-12)
        grams.append(z.T@(w[:,None]*z))
        rhs.append(z.T@(w[:,None]*event[:,j]))
    return np.stack(grams),np.stack(rhs)


def fit(sources):
    # Keep the same regularization and source weighting as the direct head.
    return original.coefficients(sources)


def predict(x,coefs):
    j,e,s,b = np.einsum('nad,adc->nac',x,coefs).clip(0.,1.).transpose(2,0,1)
    return j-b, e*s-b


def prefix_options(groups,direct,values,legal):
    """Only until the first direct switch; no inference-time recording total."""
    out = []
    for g in groups:
        if (direct[g]>=0).any():
            break
        ok = legal[g]
        if len(g)<2 or not ok.any():
            continue
        i,k = np.unravel_index(np.where(ok,values[g],-np.inf).argmax(),ok.shape)
        out.append((float(values[g[i],k]),int(g[i]),int(k),g))
    return out


def first(options,threshold):
    if threshold is None:
        return None
    return next((z for z in options if z[0]>=threshold),None)


def risk(mass,full_easy):
    ref, easy, harm, easy_harm, signed, _ = mass
    return dict(supported=bool(ref>0. and easy>0. and full_easy>0. and
        harm/ref<=BUDGET and easy_harm/easy<=BUDGET and signed/full_easy<=BUDGET),
        easy_net_over=bool(full_easy>0. and signed/full_easy>BUDGET))


def calibrate(records):
    """Use full risk preservation and maximize additional passes on donor sources."""
    table = []
    for cut in THRESHOLDS:
        opened = recovered = lost = extra_easy = standalone = 0
        gains = []
        for rec in records:
            base = risk(rec['mass'],rec['easy'])
            opt = first(rec['options'],cut)
            if opt is None:
                continue
            _,i,j,_ = opt
            z = rec['atoms'][i,j]
            opened += 1
            after = risk(rec['mass']+z,rec['easy'])
            recovered += after['supported'] and not base['supported']
            lost += base['supported'] and not after['supported']
            extra_easy += after['easy_net_over'] and not base['easy_net_over']
            standalone += risk(z,rec['easy'])['supported']
            gains.append(float(z[-1]/rec['crowd'][i]))
        table.append(dict(threshold=cut,opened=opened,recovered=recovered,lost=lost,
                          new_easy_violations=extra_easy,standalone=standalone,gain=float(sum(gains))))
    ok = [r for r in table if r['recovered']>0 and not r['lost'] and not r['new_easy_violations'] and r['gain']>=0.]
    if not ok:
        return None,table
    chosen = max(ok,key=lambda r:(r['recovered'],r['standalone']-r['opened'],r['gain'],-r['opened'],r['threshold']))
    return chosen['threshold'],table
