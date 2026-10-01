"""Render verified accounting, keeping undefined coverage and dependent views visible."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_selected_pool_accounting as run


def number(value, percent=False):
    return 'undefined' if value is None else f'{value*(100 if percent else 1):.6f}'


def risk_counts(rows):
    risks = [r['statistics']['kept']['easy']['observed_risk'] for r in rows]
    defined = [v for v in risks if v is not None]
    return dict(views=len(rows), defined=len(defined), undefined=len(risks)-len(defined),
        violations=sum(v > .02 for v in defined), worst=max(defined) if defined else None,
        unknown_occurrences=sum(r['statistics']['kept']['unknown'] for r in rows))


def main():
    public = run.PUBLIC
    seal = json.loads((public/'verification.json').read_text())
    for name, value in seal['artifacts'].items(): assert run.digest(public/name) == value
    assert seal['source_and_transfer_exact_replay']
    rows = run.read_rows(); summary = json.loads((public/'summary.json').read_text())
    sources = {(r['group'],r['head_seed'],r['mode'],r['role']):r for r in rows if r['role'] != 'transfer'}
    screened = [r for r in rows if r['role']=='transfer' and r['mode']=='joint' and r['source_screen']]
    source_failures = [r for r in rows if r['role']=='source_oof' and r['mode']=='joint'
        and r['statistics']['easy_contrast']['new_known_violation']]
    transfer_failures = [r for r in screened if r['statistics']['kept']['easy']['observed_risk'] is not None
        and r['statistics']['kept']['easy']['observed_risk'] > .02]
    lines = ['# Frozen Selected-Pool Accounting: Source and Transfer', '',
        '## Evidence Status', '',
        'Fresh accounting of cached_verified frozen heads, calibrators and causal actions. '
        'All72 source groups replay exactly locally. All216 transferred head views '
        'complete and replay on CREATE, including exact parity with144 local partial outputs. '
        'No training, parameter updates, policy search or deployment change.', '',
        'The216 views and1080 role/mode rows are dependent repeated evaluations, not independent scenes. '
        'All12 localities are already-exposed development data. Independent selection, calibration '
        'and confirmation stay closed. Unknown outcomes remain unknown; zero risk denominators '
        'are undefined, not safety passes.', '',
        'Obs8/pred12, stride12 raw frames, image-local detector-silver. No metric, seconds, '
        'human-gold, true3D, foundation or physical-safety claim. Stage5C and SMC remain off.', '',
        '## Full Frozen Accounting', '',
        'These rows describe each adjustment before applying the whole-source screen. '
        'Screened joint transfer is reported separately below.', '',
        '| Role | Arm | Views | Easy violations / defined | Undefined | Risk increases | New violations |',
        '|---|---|---:|---:|---:|---:|---:|']
    for role in ('source_oof','source_resubstitution','transfer'):
        for mode in run.parent.api.MODES:
            g = summary['groups'][mode+'_'+role]; v = g['easy']
            lines.append(f"| {role} | {mode} | {g['views']} | {v['kept_violations']}/{v['defined_kept']} | "
                f"{g['views']-v['defined_kept']} | {v['risk_increased']} | {v['new_known_violations']} |")
    counts = risk_counts(screened)
    lines += ['', f"Source-screened joint transfer: {counts['violations']}/{counts['defined']} defined views violate2%; "
        f"{counts['undefined']}/{counts['views']} screened views have undefined risk. Worst defined risk "
        f"{number(counts['worst'],True)}%. Unknown retained occurrences across these repeated views: "
        f"{counts['unknown_occurrences']}. Source-rejected views are not scored as safety passes.", '',
        '## Registered Bias Diagnostic', '',
        'Signed bias is (actual harm - predicted harm + .02*(predicted reference - actual reference)) '
        'divided by the causal disagreement envelope. The retained-minus-raw contrast uses the same '
        'adjusted predictions on both pools. It is not an ADE improvement or percentage risk.', '',
        '| Role / arm | Defined contrasts / total | Strict all-view mean | Defined-only descriptive mean | Nominal95% interval | Localities in descriptive subset |',
        '|---|---:|---:|---:|---|---:|']
    for role in ('source_oof','source_resubstitution','transfer'):
        for mode in run.parent.api.MODES:
            z = summary['groups'][mode+'_'+role]['easy']['bias_shift']
            d = z['defined_only_descriptive'] or {}; ci = d.get('CI95')
            lines.append(f"| {role}/{mode} | {z['defined_views']}/{z['total_views']} | "
                f"{number(z['all_views']['mean'])} | {number(d.get('mean'))} | "
                f"{str([round(v,6) for v in ci]) if ci else 'undefined'} | {d.get('localities',0)} |")
    lines += ['', 'Undefined all-view means do not become estimable by silently dropping empty or unsupported pools. '
        'The defined-only column is a coverage-conditioned description with3000 locality bootstrap draws; '
        'it cannot establish a universal selection effect or independent confirmation. See summary.json '
        'for the predeclared supported-pool sensitivity and source-to-transfer contrasts.', '',
        '## Source Counterexamples and Their Limit', '',
        f'{len(source_failures)} joint OOF new-violation views occur in '
        f'{len({r["site"] for r in source_failures})} distinct source localities. '
        f'{sum(r["source_screen"] for r in source_failures)} pass the existing whole-source screen. '
        'They show that raw-pool margins can miscalibrate the retained pool within source data, '
        'but do not show that those same cases caused the surviving screened transfer failures.', '',
        '| Source view | Raw easy risk (%) | Kept easy risk (%) | Predicted kept risk (%) | Harm retained (%) | Reference retained (%) | Source screen |',
        '|---|---:|---:|---:|---:|---:|---|']
    cases = []
    for r in source_failures:
        s=r['statistics']; c=s['easy_contrast']
        lines.append(f"| {r['view']} | {number(s['raw']['easy']['observed_risk'],True)} | "
            f"{number(s['kept']['easy']['observed_risk'],True)} | {number(s['kept']['easy']['predicted_risk'],True)} | "
            f"{number(c['harm_retained'],True)} | {number(c['reference_retained'],True)} | {r['source_screen']} |")
        cases.append(r)
    lines += ['', 'Repeated heads and changed mixtures of recordings are not independent replications or proof '
        'that every recording becomes worse. Full-source resubstitution uses fitted coefficients on the same '
        'source data; its lower violation count is not independent safety evidence.', '',
        '## Surviving Screened Transfer Failures', '',
        '| View | Raw easy risk (%) | Kept (%) | Predicted kept (%) | Harm retained (%) | Reference retained (%) | Source OOF kept (%) | Source full-fit kept (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in transfer_failures:
        s=r['statistics']; c=s['easy_contrast']; source=[]
        for role in ('source_oof','source_resubstitution'):
            source.append(sources[r['group'],r['head_seed'],'joint',role]['statistics']['kept']['easy']['observed_risk'])
        lines.append(f"| {r['view']} | {number(s['raw']['easy']['observed_risk'],True)} | "
            f"{number(s['kept']['easy']['observed_risk'],True)} | {number(s['kept']['easy']['predicted_risk'],True)} | "
            f"{number(c['harm_retained'],True)} | {number(c['reference_retained'],True)} | "
            f"{number(source[0],True)} | {number(source[1],True)} |")
        cases.append(r)
    lines += ['', 'Risk changes obey the checked exact retained/removed harm/reference identity. '
        'Removing actions is not generally monotone in realized selected risk. Source-to-target '
        'contrasts change locality, composition, support and coefficient-fitting regime; they do '
        'not causally isolate domain shift. The unchanged parent policy/readout remains the comparison.', '',
        '## Verification and Decision', '',
        f"{summary['scalar_and_identity_checks']} independent scalar/partition/ratio checks; "
        f"{summary['parent_risk_checks']} parent risk comparisons; all1080 rows included. "
        'Scoped arithmetic, calibration and unknown-label tests are recorded in scoped_tests.txt. '
        'Additional portable/collector tests verify transport parity and reject corrupt/incomplete jobs. '
        'Full legacy test suite is not_run.', '',
        '**No deployment promotion.** This run identifies concrete conditional-composition failures '
        'and separates source OOF from transfer evidence; it does not repair the policy. '
        'Keep the original2% selected-reference budget and frozen floor. Do not train on the motivating '
        'transfer failures or select a favorable seed. Any next selected-policy risk estimator requires '
        'a separately registered source-only design, support audit and recording-disjoint evaluation.', '']
    (public/'results.md').write_text('\n'.join(lines))
    run.immutable(public/'case_details.json',dict(status='descriptive_not_policy_selection',rows=cases))
    run.immutable(public/'findings_verification.json',dict(numerical_seal_sha256=run.digest(public/'verification.json'),
        report_sha256=run.digest(public/'results.md'),case_sha256=run.digest(public/'case_details.json'),
        script_sha256=run.digest(__file__),source_screened_joint_counts=counts,
        deployment_changed=False,independent_confirmation=False))
    print(json.dumps(counts))


if __name__ == '__main__': main()
