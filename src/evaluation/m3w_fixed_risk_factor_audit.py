"""Label-aware fixed-decision diagnostics, never a causal deployment policy."""
import numpy as np

FACTORS=('known_harm','known_reference','unknown_bound_harm','unknown_reference_credit')
BUDGET=.02


def factors(record):
    keys=('predicted_full_selected_moments','predicted_on_selected_known_moments',
        'realized_selected_known_moments','bounded_label_selected_moments')
    p,pk,a,y=(np.asarray(record[k],float)for k in keys)
    if any(x.shape!=(5,)or not np.isfinite(x).all()or (x<0).any()for x in (p,pk,a,y)):
        raise ValueError('Finite nonnegative B H R ER EH diagnostic masses required')
    if not(record['selected']==record['selected_known']+record['selected_unknown']):
        raise ValueError('Known/unknown selected inventory must be exhaustive')
    if not np.allclose(y[[2,3]],a[[2,3]],rtol=1e-12,atol=1e-10):
        raise ValueError('Original unknown reference contribution must remain zero')
    result={}
    for name,h,r in (('all',1,2),('easy',4,3)):
        pred=float(p[h]-BUDGET*p[r]);truth=float(y[h]-BUDGET*y[r])
        parts=np.array([a[h]-pk[h],BUDGET*(pk[r]-a[r]),
            (y[h]-a[h])-(p[h]-pk[h]),BUDGET*(p[r]-pk[r])])
        if not np.isclose(pred+parts.sum(),truth,rtol=1e-10,atol=1e-10):
            raise ValueError('Risk-excess decomposition must exactly account for the original moments')
        hypothetical=[float(pred+sum(parts[k]for k in range(4)if mask&(1<<k)))for mask in range(16)]
        positive=[mask for mask in range(1,16)if hypothetical[mask]>0]
        minimal=[mask for mask in positive if not any((sub&mask)==sub for sub in positive if sub!=mask)]
        ratio=lambda num,den:float(num/den)if den>0 else None
        pr,kr,br=ratio(p[h],p[r]),ratio(a[h],a[r]),ratio(y[h],y[r])
        target_bad=br is not None and br>BUDGET
        prediction_pass=pr is not None and pr<=BUDGET
        result[name]=dict(predicted_risk=pr,known_risk=kr,completion_risk_upper=br,
            predicted_excess=pred,known_excess=float(a[h]-BUDGET*a[r]),completion_excess=truth,
            components=dict(zip(FACTORS,map(float,parts))),
            counterfactual_excess_by_mask=hypothetical,minimal_positive_masks=minimal,
            active=record['selected']>0,known_reference_defined=bool(a[r]>0),
            bounded_violation=target_bad,known_violation=kr is not None and kr>BUDGET,
            predicted_feasible_but_bounded_violating=bool(target_bad and prediction_pass),
            zero_known_reference_positive_known_harm=bool(a[r]==0 and a[h]>0),
            predicted_positive_known_reference_absent=bool(pk[r]>0 and a[r]==0),
            predicted_reference_only_on_unknown=bool(p[r]>0 and pk[r]==0),
            no_observed_selected_reference=bool(record['selected']>0 and a[r]==0),
            unknown_selected=record['selected_unknown'],
            all_selected_rows_have_some_future_label=bool(record['selected_unknown']==0),
            all16_subset_corrections_are_label_aware_diagnostics_not_policies=True)
    return result


def aggregate(rows,kind):
    cells=[r['factors'][kind]for r in rows]
    failures=[r for r in cells if r['predicted_feasible_but_bounded_violating']]
    counts={key:sum(r[key]for r in cells)for key in ('active','known_reference_defined','bounded_violation',
        'known_violation','predicted_feasible_but_bounded_violating','zero_known_reference_positive_known_harm',
        'predicted_positive_known_reference_absent','predicted_reference_only_on_unknown','no_observed_selected_reference')}
    singles={name:sum(r['counterfactual_excess_by_mask'][1<<k]>0 for r in failures)for k,name in enumerate(FACTORS)}
    minimum=[]
    for r in failures:
        size=min((mask.bit_count()for mask in r['minimal_positive_masks']),default=None)
        minimum.append(size)
    return dict(required_repeated_record_views=len(rows),counts=counts,
        single_factor_flags_among_false_feasible=singles,
        minimal_required_factor_count={str(k):minimum.count(k)for k in (1,2,3,4,None)},
        active_no_observed_reference_with_unknown=sum(r['no_observed_selected_reference']and r['unknown_selected']>0 for r in cells),
        active_no_observed_reference_all_selected_have_some_label=sum(r['no_observed_selected_reference']and
            r['all_selected_rows_have_some_future_label']for r in cells),
        source_groups=len({r['site']for r in rows}),
        no_new_policy_or_safety_certificate=True)
