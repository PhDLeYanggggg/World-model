"""Reconstruct SOURCE diagnosis from verified public sufficient statistics."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_calibration_source_fit import aggregate, FAMILIES, RATIOS

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_calibration_source_fit_v1'


def main():
    read = lambda p: json.loads(Path(p).read_text())
    complete, verification = read(PUBLIC/'complete.json'), read(PUBLIC/'verification.json')
    rows = []
    for ref in complete['cases']:
        p = ROOT/ref['path']
        if hashlib.sha256(p.read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError('Verified SOURCE case changed')
        rows.append(read(p))
    value = aggregate(rows, sorted({r['trial'] for r in rows}))
    if (value != read(PUBLIC/'summary.json') or not verification['all18_SOURCE_model_inferences_exact']
            or not verification['original72_source_control_groups_exact']
            or verification['summary_sha256'] != hashlib.sha256((PUBLIC/'summary.json').read_bytes()).hexdigest()):
        raise ValueError('Complete source replay and scalar reconstruction required')
    reg = read(PUBLIC/'registration_v2.json'); ref = reg['calibration_cost_diagnosis']; path = ROOT/ref['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('Existing calibration attribution changed')
    calibration = next(r for r in read(path)['rows'] if r['multiplier'] == 1 and r['arm'] == 'independent')
    lines = ['# SOURCE Cost-Fit Diagnosis: Complete', '',
        'Fresh_run: frozen Torch inference on all18cost states and separate complete',
        'replay. Cached_verified: weights, original SOURCE inputs and SOURCE-only',
        'normalization. All144site groups reproduce exactly, including all72original',
        'control groups. No training, policy or threshold change. These SOURCE rows',
        'were used to fit the critics: this is resubstitution, not validation.', '',
        '## Selected Record Medians', '',
        '|Cohort/family|H predicted/true|EH predicted/true|R predicted/true|ER predicted/true|',
        '|---|---:|---:|---:|---:|']
    fmt = lambda v:'N/A' if v is None else f'{v:.6g}'
    for family in FAMILIES:
        r = value['families'][family]['selected_record_ratio_summaries']
        lines.append('|SOURCE / '+family+'|'+'|'.join(fmt(r[k]['median']) for k in RATIOS)+'|')
    lines.append('|Calibration / mean_absolute independent lambda1|'+
        '|'.join(fmt(calibration['record_ratio_summaries'][k]['median']) for k in RATIOS)+'|')
    lines += ['', 'Medians condition on defined ratios; finite counts and unknown rows',
        'remain in the JSON. SOURCE and calibration rosters differ; this is not a',
        'paired transfer effect, a causal shift decomposition or a new operating-point',
        'selection. The calibration row uses the same independent policy definition.', '',
        '## SOURCE Pooled Cost Masses', '',
        '|Family/pool|H predicted/true|EH predicted/true|Known all risk %|Known easy risk %|',
        '|---|---:|---:|---:|---:|']
    for family in FAMILIES:
        for pool in ('all_legal', 'selected'):
            r = value['families'][family]['pooled'][pool]
            lines.append('|'+family+'/'+pool+'|'+fmt(r['harm_prediction_ratio'])+'|'+fmt(r['easy_harm_prediction_ratio'])
                +'|'+fmt(None if r['observed_all_risk'] is None else 100*r['observed_all_risk'])
                +'|'+fmt(None if r['observed_easy_risk'] is None else 100*r['observed_easy_risk'])+'|')
    lines += ['', 'These pool repeated action views, not independent people or recordings.',
        'All-legal candidate costs are not a deployment policy. Known-label risk',
        'does not replace original unknown-completion and finite-support checks.', '',
        '## Conclusion And Controlled Next Step', '',
        'The mean-cost repair has severe selected easy-harm error even on fitting',
        'SOURCE. New-locality amplification is also present. A pure domain-shift',
        'explanation is therefore insufficient. This does not prove that a particular',
        'loss or activation is the sole cause: selection, sparse supervision and',
        'finite representation/capacity remain plausible mechanisms.', '',
        'The historical stable easy-deviance experiment already failed. Its code',
        'uses stable logs; do not invent a clipping bug or call that method new.',
        'A narrow next test adapts the existing stable cost divergence to the current',
        'three-forecaster-seed mean_absolute framework, with easy-only and H+EH',
        'controls. Same model, complete query coverage, draws, initialization,',
        'SOURCE scope and 2% evaluation remain. Any subsequent selection-cohort',
        'readout is exposed development, not independent confirmation.', '',
        '## Runtime', '', '|Phase|PID|Seconds|Peak RSS bytes|', '|---|---:|---:|---:|']
    for name, r in (('pilot', read(PUBLIC/'pilot.json')), ('complete', complete), ('replay', verification)):
        lines.append(f'|{name}|{r["pid"]}|{r["seconds"]:.6f}|{r["peak_RSS_bytes"]}|')
    lines += ['', 'The primary run reuses two hash-checked current-run pilot cases and',
        'freshly computes sixteen; separate replay freshly computes all18. No large',
        'array or model is written by this diagnosis. Local native arm64CPU4, workers0.',
        'No CREATE job submitted. Calibration arrays, confirmation and DroneCrowd',
        'remain unopened by this operator. No deployment promotion or risk certificate.',
        'Obs8/pred12/rawstride12, local-image pixels, detector/tracker silver; available',
        'label ADE is not full12 ADE. Not metric/seconds, human gold, true3D or foundation.',
        'Stage5C and SMC remain off.', '']
    (PUBLIC/'readout.md').write_text('\n'.join(lines))
    print(json.dumps(dict(model_states=18, site_groups=144, public_summary_reconstructed_exact=True)))


if __name__ == '__main__':
    main()
