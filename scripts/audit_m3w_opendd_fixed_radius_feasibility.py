"""Optimistic same-radius feasibility bound; uses labels, never a policy."""
from collections import Counter
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_opendd_flight_shape_risk as run


def minimum_ade(residual, valid, radius):
    """Minimum observed ADE over arbitrary paths with max offset exactly r."""
    err = np.where(valid, np.linalg.norm(residual,axis=-1),0.)
    projected = np.maximum(err-radius[:,None],0.).sum(1)
    # Without a missing step, the cheapest radius witness is the largest true offset.
    witness = np.where(valid.all(1),np.maximum(radius-err.max(1),0.),0.)
    return (projected+witness)/valid.sum(1).clip(1)


def feasibility(d, radius, easy_cut):
    cv, y, valid, rec, ref = (d[k] for k in ('cv','y','valid','recordings','cv_ade'))
    err = minimum_ade(y-cv,valid,radius)
    known = valid.any(1)
    easy = known & (ref>0) & (ref<=easy_cut)
    picked = radius>0
    signed = np.where(picked & known,err-ref,0.)
    positive = signed.clip(0.)
    output = {}
    for name in sorted(set(rec)):
        at = rec==name
        debit = float(radius[at & ~known].sum())
        mass = float(ref[at & picked & known].sum())
        emass = float(ref[at & picked & easy].sum())
        full = float(ref[at & easy].sum())
        undefined = min(mass,emass,full)<=0.
        debit_over_budget = not undefined and (debit>.02*mass or debit>.02*emass)
        lower = [None if den<=0 else float(num/den) for num,den in (
            (positive[at].sum()+debit,mass),(positive[at & easy].sum()+debit,emass),
            (signed[at & easy].sum(),full))]
        reason = ('undefined' if undefined else 'debit_only_impossible' if debit_over_budget
                  else 'fixed_radius_impossible' if max(lower)>.02 else 'hindsight_feasible')
        output[name] = dict(status=reason,lower_risks=lower,unknown_debit=debit,selected_easy_mass=emass)
    return output


def main():
    if platform.system()!='Darwin' or platform.machine()!='arm64':
        raise RuntimeError('Use native arm64 .venv-pytorch on this host')
    import torch
    from src.world_model import m3w_opendd_velocity_candidate as model
    from src.world_model import m3w_radius_shape as shape
    from src.evaluation.m3w_selection_forecast_readout import policy

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg, _ = run.register()
    p = run.PUBLIC/'analysis.json'
    digest = run.prior.sha(p)
    raw = json.loads(p.read_text())
    receipt = json.loads((run.PUBLIC/'replay_execution.json').read_text())
    if raw['registration']!=reg or receipt['analysis_sha256']!=digest or not receipt['fresh_exact_replay']:
        raise ValueError('Require exact replay before the outcome-aware bound')
    nets = run.shape_run.frozen.load_networks(torch,model)
    sources = {s:run.shape_run.site_data(s,torch,model,shape) for s in run.SITES}
    rows, counts, comparison = {}, Counter(), Counter()
    for i,s in enumerate(run.SITES):
        d = sources[s]
        cuts = run.base.cuts_for([sources[k] for k in run.SITES if k!=s])
        rows[s] = []
        for view in raw['by_location'][s]:
            seed = view['seed']
            c = run.shape_run.candidate(d,nets[seed][i],torch,model)
            bound = feasibility(d,c['r'],cuts[0])
            incumbent = policy(c['inc'],d['cv'],d['cv'],d['y'],d['valid'],d['recordings'],
                               easy_cut=cuts[0],hard_cut=cuts[1])['recordings']
            for k,v in bound.items():
                old = incumbent[k]
                if old['supported'] and v['status']!='hindsight_feasible':
                    raise ValueError('Optimistic lower bound excludes a feasible incumbent')
                for lo,hi in zip(v['lower_risks'],(old['completion_all_risk_upper'],
                    old['completion_easy_risk_upper'],old['known_easy_net_degradation']),strict=True):
                    if lo is not None and (hi is None or lo>hi+1e-10):
                        raise ValueError('Bound does not lower-bound incumbent risk')
                counts[v['status']] += 1
                if not old['supported']:
                    comparison[v['status']] += 1
            rows[s].append(dict(seed=seed,recordings=bound))
    run.prior.write_json(run.PUBLIC/'fixed_radius_feasibility.json',dict(analysis_sha256=digest,
        posthoc_outcome_aware_bound=True,smoothness_constraints=False,no_policy_or_gate_changes=True,
        status_counts=counts,incumbent_failures_by_feasibility=comparison,by_location=rows),immutable=True)


if __name__=='__main__':
    main()
