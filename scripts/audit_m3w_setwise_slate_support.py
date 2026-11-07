"""Necessary observed-risk conditions on the same frozen causal slate."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_setwise_repair as study

run,np = study.run,study.np


def main():
    path = study.PUBLIC/'analysis.json'
    result = run.read(path)
    if run.read(study.PUBLIC/'replay_execution.json')['artifact']['sha256']!=run.sha(path):
        raise ValueError('Require the exact registered outcome readout')
    d,refs = run.population(),run.register()
    frames = run.old.prior.arrays(('frames',))['frames']
    by_record = study.prior.blocks(d,frames)
    y = run.old.prior.arrays(('target_eval','valid'))
    valid = y['valid']
    target = y['target_eval']-d['origin'][:,None]
    cut = run.cuts_by_source()
    heads = {c['seed']:c for c in run.read(run.PUBLIC/'freeze.json')['seeds']}
    records = []
    for cell in run.read(study.PUBLIC/'freeze.json')['cells']:
        seed = cell['seed']
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(cell['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            ch = z['choices'][0]
        with np.load(run.checked(heads[seed]['cache']),allow_pickle=False) as z:
            coef = dict(zip(z['names'],z['coefficients']))
        for ix,groups in by_record.values():
            s = d['sites'][ix[0]]
            c,b = cv[ix],bank[ix]
            x,scale = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
            u,a,e,legal = run.model.decision_costs(x,coef[s],b,c,scale)
            _,all_cost,easy_cost = study.nested.base.teacher.actual(b,c,target[ix],valid[ix],cut[s]['easy'])
            ref,_ = run.model.errors(c,target[ix],valid[ix])
            eligible_easy = (ref>0.)&(ref<=cut[s]['easy'])
            active = count = 0
            min_all = min_easy = 0.
            possible_easy = False
            for g in groups:
                original = ch[ix[g]]
                if not (original>=0).any():
                    continue
                z = study.model.slate(u[g]/scale[g,None],a[g],e[g],legal[g],original)
                rows,actions = g[z[:,:,0]],z[:,:,1]
                active += 1
                count += len(z)
                min_all += float(all_cost[rows,actions].sum(1).min())
                min_easy += float(easy_cost[rows,actions].sum(1).min())
                possible_easy |= bool(eligible_easy[rows].any())
            failed = [name for name,violated in (
                ('no_selected_query',not active),('no_easy_reference_in_any_option',not possible_easy),
                ('all_budget_lower_bound_positive',min_all>1e-9),('easy_budget_lower_bound_positive',min_easy>1e-9))
                      if violated]
            records.append(dict(options=count,active=active,failed_necessary_conditions=failed))
    if (len(records)!=102 or sum(r['options'] for r in records)!=result['slate']['options']
            or sum(r['active'] for r in records)!=result['slate']['active']):
        raise ValueError('The observed bound must use exactly the registered slate')
    summary = dict(source=run.artifact(path),views=len(records),
        proven_infeasible=sum(bool(r['failed_necessary_conditions']) for r in records),
        failed_conditions=dict(Counter(k for r in records for k in r['failed_necessary_conditions'])),
        not_ruled_out=sum(not r['failed_necessary_conditions'] for r in records),
        optimality_or_sufficiency_claim=False,no_outcome_aware_choices_exported=True,
        independent_outcomes_opened=False)
    run.once(study.PUBLIC/'slate_support.json',summary)
    print(summary)


if __name__=='__main__':
    main()
