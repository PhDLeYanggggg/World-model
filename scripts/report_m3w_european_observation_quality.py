"""Aggregate-only observation report; no prediction or deployment claims."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_m3w_european_squares import digest
from scripts.run_m3w_native_forecast import immutable_json

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_observation_quality_v1'


def main():
    doc = json.loads((PUBLIC/'audit.json').read_text())
    assert doc['source_rows'] == 318969 and not doc['future_labels_read']
    for ref in doc['receipts']: assert digest(ROOT/ref['path']) == ref['sha256']
    p = doc['pooled']; n = doc['source_rows']
    changed = round(n*p['neighbor_geometry_changed']['mean'])
    affected = round(n*p['has_partial_nearest_neighbor']['mean'])
    lines = ['# Source Observation Quality and Neighbor Coverage', '',
        '## Material Passport',
        'Fresh raw-source audit; cached inputs hash-verified. Source development only.',
        'No model training, future outcome scoring or independent confirmation.', '',
        '## Exact Source Checks',
        f"{n:,} complete-history target queries from 163 recordings and 12 source localities.",
        'All original packed geometry and 2,551,752 observed boxes match exactly.',
        'The raw archive checksum and each admitted member row hash are checked.',
        'Whole raw files are parsed for identity checks; per-query diagnostics use only',
        '[query-84, query]. Future label arrays and reserved recordings are not opened.', '',
        '## Measured Omission',
        f"{affected:,}/{n:,} queries ({100*affected/n:.2f}%) have a partial-history neighbor in the nearest eight currently visible agents.",
        f"The new neighbor geometry changes on {changed:,} queries ({100*changed/n:.2f}%).",
        f"Mean neighbor count: {p['legacy_neighbor_count']['mean']:.4f} legacy, {p['repaired_neighbor_count']['mean']:.4f} masked.",
        'The old source cache contains these visible agents, but the old model packer',
        'excludes them because target eligibility was also used for neighbor eligibility.',
        'This is a demonstrated representation omission, not proof of useful interaction lift.', '',
        '## Per-Locality Diagnostics',
        '| Locality | Queries | Partial neighbor % | Raw-prefix frame presence | Line RMS / width median | FD prefix error | OLS4 prefix error | OLS6 prefix error |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for site, d in sorted(doc['per_site'].items()):
        lines.append(f"| {site} | {d['path_over_width']['n']} | {100*d['has_partial_nearest_neighbor']['mean']:.2f} | "
            f"{d['raw_prefix_frame_presence']['mean']:.4f} | {d['line_residual_over_width']['median']:.4f} | "
            f"{d['observed_prefix_fd_error']['mean']:.4f} | {d['observed_prefix_ols4_error']['mean']:.4f} | {d['observed_prefix_ols6_error']['mean']:.4f} |")
    lines += ['', '## Motion and Detector Proxies',
        '| Diagnostic | Mean | Median | P90 | P99 |', '|---|---:|---:|---:|---:|']
    for key, values in p.items():
        lines.append('| '+key+' | '+' | '.join(f'{values[k]:.6g}' for k in ('mean','median','p90','p99'))+' |')
    lines += ['', '## Interpretation Boundaries',
        'Fixed prefix forecasts use observed steps 1-6 to predict observed steps 7-8.',
        'Errors are in current-box-width units with a one-coordinate-unit denominator floor.',
        'They are not future ADE/FDE, learned estimator gains, or metric/seconds claims.',
        'Each locality is shown, without choosing a favorable one. Windows overlap;',
        'these distribution summaries are descriptive, not independent confidence intervals.',
        'Box motion, reversal and line residuals may reflect real dynamics or detector noise.',
        'High raw frame presence does not prove identity correctness or annotation accuracy.',
        'Sparse event-label attribution was not rerun; no new label-driven filter was selected.', '',
        '## Repair and Next Test',
        'The versioned input repair retains all current-visible neighbor candidates,',
        'selects the nearest eight by current position, and zeros masked coordinates/times.',
        'Ego history, baseline rollout, target rows and raw future labels are unchanged.',
        'Do not deploy old weights on the new schema as if retraining had happened.',
        'A separate neural adapter also replaces complete-history attention eligibility',
        'and includes valid partial tokens in past-only conditioning. It retains the',
        'original parameter budget, zero-initial correction and motion-bounded output.',
        'Synthetic forward/backward checks are code tests, not trained model evidence.',
        'Next: one matched legacy-vs-masked-neighbor retraining experiment with the',
        'same full producer-chain exclusions, fixed training budget and three seeds.',
        'This isolates an observed input defect; it does not assume smoothing or more',
        'context works. Keep cap, losses, target rules and policy frozen for that contrast.',
        'Independent selection/calibration/confirmation stay closed. Stage5C and SMC stay off.', '']
    (PUBLIC/'results.md').write_text('\n'.join(lines))
    immutable_json(PUBLIC/'gates.json', dict(input_audit_complete=True, original_geometry_exact=True,
        raw_observation_match=True, partial_neighbor_omission_demonstrated=affected>0,
        versioned_repair_implemented=True, causal_invariance_unit_tests_required=True,
        prediction_lift='not_run', retraining='not_run', event_stratified_quality='not_run',
        independent_confirmation=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    print(json.dumps(dict(source_rows=n, partial_neighbor_rows=affected, geometry_changed_rows=changed)))


if __name__ == '__main__': main()
