"""Describe all frozen fitting curves without manufacturing a safety claim."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_amplitude_shape_v1'


def read(path):
    return json.loads(path.read_text())


def aggregate(records):
    result=[]
    for arm in ('binary_only','amplitude_bank'):
        values=[r['scores']for r in records if r['arm']==arm]
        if len(values)!=18:
            raise ValueError('All18frozen fitting views required for each head family')
        active=sum(r['active_rows']for r in values)
        selected=sum(r['selected']for r in values)
        mass=sum(r['selected_known_EH_mass']for r in values)
        for comp in ('H','EH'):
            counts={k:sum(r['predicted'][comp][k]for r in values)
                for k in ('nonmonotone_rows','nonconvex_rows','ratio_order_violating_rows')}
            bad=sum(r['selected_ratio_inversion'][comp]['selected_rows']for r in values)
            bad_mass=sum(r['selected_ratio_inversion'][comp]['known_EH_mass']for r in values)
            result.append(dict(arm=arm,component=comp,active_repeated_rows=active,selected_repeated_rows=selected,
                selected_known_EH_mass=mass,**counts,
                selected_ratio_inversions=bad,selected_known_EH_mass_on_ratio_inversions=bad_mass,
                selected_known_EH_mass_share_on_ratio_inversions=bad_mass/mass if mass>0 else None,
                reference_max_relative_span=max(max(r['reference_max_relative_span'])for r in values)))
    return result


def main():
    verify=read(PUBLIC/'verify_execution.json')
    if not verify['all_scalars_exact']or not verify['all36fitting_prediction_hashes_exact']:
        raise ValueError('Complete separate frozen-prediction/scalar replay required')
    raw=read(PUBLIC/'analysis.json')
    rows=aggregate(raw['rows'])
    fmt=lambda x:'not estimable'if x is None else f'{100*x:.4f}%'
    lines=['# Cross-Amplitude Shape On Frozen Fitting Sources','','## Material Passport','',
        'Fresh_run: all 36 original fitting prediction hashes and curve diagnostics.',
        'Cached_verified: original data, forecasts and frozen cost-head weights.',
        'Separate exact replay checks the complete diagnostic again. No new model,',
        'held-policy evaluation, threshold, data role or reserved outcomes.','',
        'Counts repeat sources/producer/seed contexts. Ratios below pool repeated',
        'known cost mass; they are not independent macro risk scores, calibration',
        'or a causal explanation. Binary-only intermediate actions were not',
        'supervised and are a diagnostic comparator only.','',
        '| Head | Cost | Active repeated rows | Nonmonotone | Nonconvex | H/alpha order violated | Selected ratio inversions | Selected known EH mass on those inversions |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['arm']} | {r['component']} | {r['active_repeated_rows']} | {r['nonmonotone_rows']} | "
            f"{r['nonconvex_rows']} | {r['ratio_order_violating_rows']} | {r['selected_ratio_inversions']}/{r['selected_repeated_rows']} | "
            f"{fmt(r['selected_known_EH_mass_share_on_ratio_inversions'])} |")
    if any(v for r in raw['rows']for comp in r['scores']['bounded_labels'].values()for v in comp.values()):
        raise ValueError('True bounded-label shape check failed')
    lines+=['','True fixed-mask bounded-label H/EH curves have zero counted violations',
        'in every fitting view. Shared predicted R/ER have maximum relative span',
        f"{max(r['reference_max_relative_span']for r in rows):.8g}. The diagnostic tolerance is 1e-6 after",
        'normalization by the full causal discrepancy; it is numerical, not a',
        'new risk threshold. A ratio inversion means the chosen action predicts',
        'smaller cost/alpha than some smaller action. Convexity is a stronger',
        'condition than nondecreasing cost/alpha and is reported separately.','',
        '## Implication, Not Demonstrated Repair','',
        'If material violations occur in the supervised amplitude head, a matched',
        'cross-amplitude constrained head is a testable repair. Correct shape does',
        'not by itself ensure correct magnitude, selected-set reliability, finite',
        'reference support or independent-domain generalization. No projection,',
        'new fit, calibration, deployment or formal guarantee is measured here.',
        'The parent primary and complete recording risk remain failed.','',
        'Existing scalar-score monotone calibration is not this within-row shape',
        'constraint. Input-convex and monotone networks are established methods;',
        'their use alone is not a novelty or publication claim. See the separate',
        'primary-source note. Pixels/native annotation steps remain uncalibrated.',
        'Stage5C execution and SMC remain disabled.']
    (PUBLIC/'summary.json').write_text(json.dumps(dict(rows=rows),indent=2,sort_keys=True,allow_nan=False)+'\n')
    (PUBLIC/'readout.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
