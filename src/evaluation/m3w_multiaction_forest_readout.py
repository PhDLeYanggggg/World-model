"""Supplementary known-outcome slices, never substitutes for full risk bounds."""
import numpy as np
from src.evaluation.m3w_multiaction_cost_readout import chosen_values


def subsets(packet, choices, easy_cut):
    known = np.isfinite(packet['y']).all((1, 2))
    masks = dict(hard=known & packet['hard'],
                 easy=known & (packet['cv_ade'] > 0) & (packet['cv_ade'] <= easy_cut))
    result = {}
    for subset, mask in masks.items():
        fde_mask = mask & np.isfinite(packet['floor_fde'])
        reference = float(packet['floor_ade'][mask].sum())
        fde_reference = float(packet['floor_fde'][fde_mask].sum())
        policies = {}
        for name, choice in choices.items():
            ade = chosen_values(packet['ade'], packet['floor_ade'], choice)
            fde = chosen_values(packet['fde'], packet['floor_fde'], choice)
            error, fde_error = float(ade[mask].sum()), float(fde[fde_mask].sum())
            policies[name] = dict(ADE_error_mass=error, FDE_error_mass=fde_error,
                ADE_gain_percent=100*(1-error/reference) if reference > 0 else None,
                FDE_gain_percent=100*(1-fde_error/fde_reference) if fde_reference > 0 else None,
                switched=int((mask & (choice >= 0)).sum()))
        result[subset] = dict(known_rows=int(mask.sum()), known_FDE_rows=int(fde_mask.sum()),
            reference_ADE_mass=reference, reference_FDE_mass=fde_reference, policies=policies,
            unknown_rows_not_imputed=True, primary_risk_replacement=False)
    return result


def pooled_subsets(rows):
    output = {}
    for subset in ('hard', 'easy'):
        groups = [row['result']['subsets'][subset] for row in rows]
        ref = sum(g['reference_ADE_mass'] for g in groups)
        fde_ref = sum(g['reference_FDE_mass'] for g in groups)
        output[subset] = dict(known_row_views=sum(g['known_rows'] for g in groups),
            required_views=len(groups), empty_known_views=sum(g['known_rows'] == 0 for g in groups),
            macro_status='not_estimable' if any(g['reference_ADE_mass'] <= 0 for g in groups) else 'see_primary_locality_protocol',
            aggregation='supplementary_pooled_known_error_mass_over_repeated_views_not_independent_samples',
            policies={name:dict(
                ADE_gain_percent=100*(1-sum(g['policies'][name]['ADE_error_mass'] for g in groups)/ref) if ref > 0 else None,
                FDE_gain_percent=100*(1-sum(g['policies'][name]['FDE_error_mass'] for g in groups)/fde_ref) if fde_ref > 0 else None,
                switched=sum(g['policies'][name]['switched'] for g in groups))
                for name in groups[0]['policies']})
    return output
