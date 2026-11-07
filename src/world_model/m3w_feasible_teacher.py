"""Oracle-constrained donor demonstrations; no outcomes are used for inference."""
import numpy as np

from src.world_model import m3w_causal_record_ledger as solver
from src.evaluation.m3w_selection_forecast_readout import errors


def actual(bank, cv, y, valid, easy_cut):
    ref,_ = errors(cv,y,valid)
    known = valid.any(1)
    ref = np.where(known,ref,0.)
    easy = known & (ref>0.) & (ref<=easy_cut)
    utility, all_excess, easy_excess = [], [], []
    for k in range(bank.shape[1]):
        path = bank[:,k]
        radius = np.linalg.norm(path-cv,axis=-1).max(1)
        err,_ = errors(path,y,valid)
        gain = np.where(known,ref-err,0.)
        harm = np.maximum(-gain,0.)
        missing = np.where(known,0.,radius)
        utility.append(gain-missing)
        all_excess.append(harm+missing-.02*ref)
        easy_excess.append(easy*harm+missing-.02*easy*ref)
    return tuple(np.stack(v,axis=1) for v in (utility,all_excess,easy_excess))


def examples(groups, u, a, e, legal):
    labels = np.zeros((2,*u.shape),dtype=float)
    mask = np.zeros_like(legal)
    total = np.zeros((2,3))
    n_query = n_switch = changed = saturated = 0
    zero = np.zeros(2)
    for g in groups:
        ug,ag,eg,ok = (v[g] for v in (u,a,e,legal))
        constrained = solver.query_choice(ug,ag,eg,ok,zero)
        n = (constrained>=0).sum()
        if not n:
            continue
        nc = ok.sum()
        if nc<=n:
            saturated += 1
            continue
        ablated = np.zeros_like(ug)
        unconstrained = solver.query_choice(ug,ablated,ablated,ok,zero,count=n)
        mask[g] = ok
        n_query += 1
        n_switch += int(n)
        changed += int((constrained!=unconstrained).any())
        for j,choice in enumerate((unconstrained,constrained)):
            take = choice>=0
            labels[j,g[take],choice[take]] = nc/n
            total[j] += solver.totals(choice,ug,ag,eg)
        if np.any(solver.totals(constrained,ug,ag,eg)[1:]>solver.EPS):
            raise ValueError('Constrained donor teacher violated its observed budget')
    return labels,mask,dict(active_queries=n_query,switches=n_switch,changed_queries=changed,
                            saturated_queries=saturated,
                            utility_all_easy_excess=total.tolist())
