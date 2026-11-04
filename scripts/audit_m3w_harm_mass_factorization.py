"""Hash-verified arithmetic decomposition of existing selected fitting moments."""
from scripts.report_m3w_moving_harm_refinement import factorization,text_once,ROOT,source,np


def main():
    original=ROOT/'outputs/publication_readiness_2026_09/moving_amplitude_cost_v1'
    convex=ROOT/'outputs/publication_readiness_2026_09/moving_convex_amplitude_v1'
    public=ROOT/'outputs/publication_readiness_2026_09/moving_harm_refinement_v1'
    if (not source.read(original/'fitting_check/verify_execution.json')['all_scalars_exact']
            or not source.read(convex/'verification.json')['all36cost_outputs_and_policy_scalars_exact']):
        raise ValueError('Verified immutable earlier fitting and complete held replay required')
    old=source.read(original/'fitting_check/analysis.json')
    a=[r['scores']for r in old['rows']if r['arm']=='amplitude_bank'and r['region']=='amplitude_bank_independent']
    b=[];refs=[]
    for ref in source.read(convex/'training_freeze.json')['fits']:
        row=source.read(source.checked(ref));source.checked(row['checkpoint'])
        b.append(row['own_selected_costs']);refs.append(ref)
    if len(a)!=18 or len(b)!=18:raise ValueError('All18own fitting views required in each old family')
    result={}
    for arm,rows in (('unconstrained',a),('convex',b)):
        actual=np.sum([r['realized_selected_known_moments']for r in rows],0)
        predicted=np.sum([r['predicted_on_selected_known_moments']for r in rows],0)
        result[arm]=factorization(actual,predicted)
    receipt=source.stable(dict(result_source='cached_verified_prior_inference_fresh_arithmetic_no_new_predictions',
        rows=result,convex_fit_receipts=refs,bindings={str(p.relative_to(ROOT)):source.sha(p)for p in
            (original/'fitting_check/analysis.json',original/'fitting_check/verify_execution.json',
                convex/'training_freeze.json',convex/'verification.json')},
        same_rows_within_each_mass_ratio=True,same_selected_rows_between_families=False,
        independent_confirmation=False,new_training=False,deployment_changed=False))
    source.once(public/'prior_mass_factorization.json',receipt)
    lines=['# Selected fitting risk: numerator versus reference mismatch','',
        'Cached-verified prior inference; fresh hash checks and arithmetic, not new fitting or held predictions.',
        'Each family uses its own selected known rows. Rows match within each ratio, not across families.',
        '', '|Family|Actual/predicted H|Actual/predicted R|Actual/predicted EH|Actual/predicted ER|',
        '|---|---:|---:|---:|---:|']
    for arm,value in result.items():
        m=value['moments'];lines.append('|'+arm+'|'+ '|'.join(f"{m[k]['ratio']:.6f}"for k in ('H','R','EH','ER'))+'|')
    lines.extend(['','On these pooled fitting masses, reference inflation is not the dominant explanation: R is near the realized mass, and ER is underpredicted. Harm is underestimated by roughly40--43x, selected easy harm by202--245x. This is a selected-region magnitude problem already on fitting data, not solely a held-source shift.',
        '', 'The exact identity is risk-error factor = harm-error factor / reference-error factor. It does not prove every query reference is accurate, isolate a unique cause, or certify any risk. The already-complete convex decoder did not fix magnitudes or the failed primary; its negative result is retained.',
        '', 'The separately registered three-arm cost-refinement experiment tests a repair; these diagnostic ratios do not count as evidence that the repair works. Original24group primary remains not estimable. No new deployment, metric/seconds claim, Stage5C or SMC.'])
    text_once(public/'prior_mass_factorization.md','\n'.join(lines)+'\n')
    print({arm:{k:r['moments'][k]['ratio']for k in ('H','R','EH','ER')}for arm,r in result.items()})


if __name__=='__main__':main()
