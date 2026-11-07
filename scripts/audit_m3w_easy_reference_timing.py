"""Posthoc source of missing easy-reference support, without exporting actions."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_expanded_risk_support as study

run,np = study.run,study.np


def main():
    path = study.PUBLIC/'analysis.json'
    result = run.read(path)
    if run.read(study.PUBLIC/'replay_execution.json')['artifact']['sha256']!=run.sha(path):
        raise ValueError('Require exact replay of the frozen prefix experiment')
    limited = {(r['seed'],r['recording']) for r in result['records']
               if 'no_easy_reference_in_any_option' in r['fixed_prefix_bound']['failed_necessary_conditions']}
    d,refs = run.population(),run.register()
    frames = run.old.prior.arrays(('frames',))['frames']
    records = study.prior.blocks(d,frames)
    labels = run.old.prior.arrays(('target_eval','valid'))
    y,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    cuts = run.cuts_by_source()
    heads = {c['seed']:c for c in run.read(run.PUBLIC/'freeze.json')['seeds']}
    categories = Counter()
    inactive_support = inactive_joint_safe = 0
    pairs_any = pairs_inactive = pairs_inactive_observed = 0
    oracle_single = oracle_pair = blocked_but_oracle = 0
    for cell in run.read(study.PUBLIC/'freeze.json')['cells']:
        seed = cell['seed']
        cv,bank = run.candidates(d,refs,seed)
        with np.load(run.checked(cell['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            original = z['choices'][0]
        with np.load(run.checked(heads[seed]['cache']),allow_pickle=False) as z:
            coef = dict(zip(z['names'],z['coefficients']))
        for r,(ix,groups) in records.items():
            if (seed,r) not in limited:
                continue
            s = d['sites'][ix[0]]
            c,b = cv[ix],bank[ix]
            ref,_ = run.model.errors(c,y[ix],valid[ix])
            easy = (ref>0.)&(ref<=cuts[s]['easy'])
            if not easy.any():
                categories['no_observed_positive_easy_row'] += 1
                continue
            x,scale = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
            _,a,e,legal = run.model.decision_costs(x,coef[s],b,c,scale)
            _,real_a,real_e = study.nested.base.teacher.actual(b,c,y[ix],valid[ix],cuts[s]['easy'])
            slack = np.zeros(2)
            any_legal = can = inactive = joint = False
            paired = inactive_pair = observed_pair = False
            one_actual = two_actual = False
            for g in groups:
                w = original[ix[g]]
                option = legal[g]&easy[g,None]&(len(g)>1)
                any_legal |= bool(option.any())
                ok = option&(a[g]<=slack[0]+1e-10)&(e[g]<=slack[1]+1e-10)
                can |= bool(ok.any())
                one_actual |= bool((option&(real_a[g]<=1e-10)&(real_e[g]<=1e-10)).any())
                if not (w>=0).any():
                    inactive |= bool(ok.any())
                    joint |= bool((ok&(real_a[g]<=1e-10)&(real_e[g]<=1e-10)).any())
                if len(g)>2 and option.any():
                    value = np.broadcast_to(np.where(easy[g],ref[g],0.)[:,None],a[g].shape)
                    pair = study.prior.model.query_choice(value,a[g],e[g],legal[g],slack,count=2)
                    if pair is not None and study.prior.model.totals(pair,value,a[g],e[g])[0]>0.:
                        paired = True
                        if not (w>=0).any():
                            inactive_pair = True
                            actual = study.prior.model.totals(pair,value,real_a[g],real_e[g])[1:]
                            observed_pair |= bool((actual<=1e-10).all())
                    exact = study.prior.model.query_choice(value,real_a[g],real_e[g],legal[g],np.zeros(2),count=2)
                    if exact is not None:
                        two_actual |= bool(study.prior.model.totals(exact,value,real_a[g],real_e[g])[0]>0.)
                slack -= study.prior.model.totals(w,a[g],a[g],e[g])[1:]
            categories['no_legal_easy_single_switch' if not any_legal else
                       'prefix_admits_an_easy_single_switch' if can else
                       'predicted_prefix_blocks_all_easy_single_switches'] += 1
            inactive_support += inactive
            inactive_joint_safe += joint
            pairs_any += paired
            pairs_inactive += inactive_pair
            pairs_inactive_observed += observed_pair
            oracle_single += one_actual
            oracle_pair += two_actual
            blocked_but_oracle += not (can or paired) and (one_actual or two_actual)
    if sum(categories.values())!=len(limited):
        raise ValueError('Missing reference roster changed')
    summary = dict(source=run.artifact(path),missing_fixed_count_views=len(limited),
        categories=dict(categories),inactive_query_admits_easy=inactive_support,
        inactive_query_admits_easy_and_nonpositive_observed_excess=inactive_joint_safe,
        not_full_policy_feasibility=True,no_label_aware_choices_exported=True,independent_outcomes_opened=False)
    run.once(study.PUBLIC/'reference_timing.json',summary)
    run.once(study.PUBLIC/'paired_reference_timing.json',dict(source=run.artifact(path),
        missing_fixed_count_views=len(limited),any_query_admits_easy_pair=pairs_any,
        inactive_query_admits_easy_pair=pairs_inactive,
        inactive_max_reference_pair_also_passes_observed_excess=pairs_inactive_observed,
        not_full_policy_feasibility=True,no_label_aware_choices_exported=True))
    run.once(study.PUBLIC/'observed_reference_opportunities.json',dict(source=run.artifact(path),
        missing_fixed_count_views=len(limited),observed_zero_credit_easy_single=oracle_single,
        observed_zero_credit_easy_pair=oracle_pair,predicted_prefix_blocks_but_observed_option_exists=blocked_but_oracle,
        posthoc_development_only=True,not_full_policy_feasibility=True,no_label_aware_choices_exported=True))
    print(summary)


if __name__=='__main__':
    main()
