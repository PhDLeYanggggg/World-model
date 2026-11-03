"""Zero-request support is explicit; no empty group becomes a favorable score."""


def fitting_design_with_empty_groups(original,data,requested,baseline_ade,empty_groups):
    actual=set(data['sites']);missing=set(requested)-actual
    if not missing<=set(empty_groups):raise ValueError('Unexpected missing fitting group')
    result=original(data,sorted(set(requested)&actual),baseline_ade)
    result['declared_but_structurally_empty_groups']=sorted(missing)
    return result


def qualify_supported_analysis(raw,declared_groups,empty_groups):
    result=dict(raw)
    result['supported_group_completion_lower_secondary']=raw['primary']
    result['supported_group_known_secondary']=raw['known']
    result['primary']=dict(status='not_estimable',point=None,CI95=None,
        reason='Declared source group has no eligible8_12query,not a zero prediction error',
        declared_groups=len(declared_groups),empty_groups=sorted(empty_groups))
    result.pop('known')
    result['declared_admission_groups']=sorted(declared_groups)
    result['structurally_empty_groups']=sorted(empty_groups)
    result['all_declared_group_primary_passed']=False
    result['scope']='prediction_supported_development_groups_secondary_not_full_roster_confirmation'
    result['zero_error_imputation_for_empty_groups']=False
    return result
