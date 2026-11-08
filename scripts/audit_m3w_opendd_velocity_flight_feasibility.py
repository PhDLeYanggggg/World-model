"""Exact hindsight feasibility of the original risk algebra for a fixed path per row."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_opendd_velocity_calibration as run
from src.evaluation.m3w_selection_forecast_readout import errors


def disposition(easy, eligible, ref, harm):
    if not easy.any():
        return 'no_full_easy_reference'
    support = easy & eligible
    if not support.any():
        return 'no_eligible_easy_edit'
    if np.any(harm[support] <= .02*ref[support]):
        return 'hindsight_singleton_feasible'
    return 'every_eligible_easy_edit_exceeds_budget'


def audit(d,pred,cut):
    base,_ = errors(d['cv'],d['y'],d['valid'])
    changed,_ = errors(pred,d['y'],d['valid'])
    harm = np.maximum(changed-base,0.)
    easy = np.isfinite(base) & (base>0.) & (base<=cut)
    eligible = np.any(pred!=d['cv'],axis=(1,2))
    counts = Counter()
    for rec in sorted(set(d['recordings'])):
        at = d['recordings']==rec
        counts[disposition(easy[at],eligible[at],base[at],harm[at])] += 1
    return counts


def main():
    import torch
    from src.evaluation import m3w_opendd_velocity_calibration as calibrate
    from src.world_model import m3w_ols_replacement as motion, m3w_opendd_velocity_candidate as model

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    run.np,run.torch,run.calibrate,run.model = np,torch,calibrate,model
    run.parent.np,run.parent.torch,run.parent.motion = np,torch,motion
    run.velocity.np,run.velocity.torch,run.velocity.model = np,torch,model
    reg,frozen = run.register()
    path = run.PUBLIC/'analysis.json'
    result = json.loads(path.read_text())
    if result['registration']!=reg or len(result['fresh_inner_fits'])!=63:
        raise ValueError('Require complete source-bound evaluation')
    roster = {s:run.data(s,run.prior.sha(run.prior.PUBLIC/'registration.json')) for s in run.SITES}
    total = {arm:Counter() for arm in run.ARMS}
    for held,d in roster.items():
        cut = run.parent.easy_cut(roster,[s for s in run.SITES if s!=held])
        for view in result['sites'][held]:
            p = run.frozen_forecast(roster,held,view['seed'],frozen)
            for arm in run.ARMS:
                pred,_ = run.rollout(d,p,view['selected'][arm],arm)
                total[arm].update(audit(d,pred,cut))
    for arm,v in total.items():
        if sum(v.values())!=result['summary'][arm]['flight']['flights']:
            raise ValueError('Flight feasibility roster differs')
        if v['hindsight_singleton_feasible'] < result['summary'][arm]['flight']['pass_count']:
            raise ValueError('Feasibility count contradicts the original evaluator')
    report = dict(scope='posthoc_development_hindsight_only',analysis_sha256=run.prior.sha(path),
                  views={k:dict(v) for k,v in total.items()},certifies_deployment=False)
    run.prior.write_json(run.PUBLIC/'flight_feasibility.json',report,immutable=True)
    print(json.dumps(report,sort_keys=True))


if __name__=='__main__':
    main()
