"""Verify completed probes and disclose raw-label boundary sensitivity."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.probe_m3w_source_box_motion import probability_metrics
from scripts.run_m3w_source_crossfit import immutable_json
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np


def main():
    folder = ROOT/'outputs/publication_readiness_2026_09/source_box_motion_probe_v1'
    private = ROOT/'data/stage_cvpr2027_experiments/source_box_motion_probe_v1'
    reg = ROOT/'configs/m3w_source_box_motion_probe_v1.json'
    plan = json.loads(reg.read_text()); report = json.loads((folder/'report.json').read_text())
    assert report['identity']['registration_sha256'] == file_digest(reg)
    for path, digest in plan['bindings'].items(): assert file_digest(ROOT/path) == digest
    replay = json.loads((folder/'replay.json').read_text()); assert replay['exact_probes'] == 16
    prep = json.loads((ROOT/'outputs/publication_readiness_2026_09/source_motion_quality_v1/preparation.json').read_text())
    path = ROOT/'data/stage_cvpr2027_experiments/source_motion_quality_v1/rows.npz'
    assert file_digest(path) == prep['rows_sha256']
    with np.load(path, allow_pickle=False) as a:
        ids = a['ids'].copy(); peak = np.linalg.norm(a['future_native'].astype(float),axis=-1).max(1)
    sensitivity = []; hashes = {}
    for trial in report['trials']:
        artifact = trial['artifact']; assert file_digest(ROOT/artifact['path']) == artifact['sha256']
        hashes[artifact['path']] = artifact['sha256']
        assert max(trial['iterations']) < plan['max_iter']
        if trial['label'] != 'over10_annotation_pixels': continue
        with np.load(ROOT/artifact['path'], allow_pickle=False) as a:
            where = np.searchsorted(ids,a['held_ids']); np.testing.assert_array_equal(ids[where],a['held_ids'])
            raw_label = (peak[where] > 10).astype(int); changed = raw_label != a['held_label']
            sensitivity.append(dict(site=trial['site'],arm=trial['arm'], raw_positives=int(raw_label.sum()),
                registered_positives=int(a['held_label'].sum()), label_differences=int(changed.sum()),
                max_raw_distance_from_boundary=float(np.max(np.abs(peak[where][changed]-10))) if changed.any() else 0,
                metrics=probability_metrics(raw_label,a['held_probability'])))
    for path in [*private.glob('*.json'),folder/'report.json',folder/'replay.json']:
        hashes[str(path.relative_to(ROOT))] = file_digest(path)
    child = subprocess.run([sys.executable,'scripts/probe_m3w_source_box_motion.py','--registration',str(reg)],
                           cwd=ROOT,text=True,capture_output=True,check=True)
    events = [json.loads(x) for x in child.stdout.splitlines() if x.startswith('{') and x.endswith('}')]
    assert not any(e.get('state')=='fit' for e in events)
    assert hashes == {p:file_digest(ROOT/p) for p in hashes}
    verification = dict(result_source='fresh_run_checks_cached_verified_raw_geometry',models=16,exact_coefficient_replays=16,
        completed_resume_new_fits=0,unchanged_artifacts=len(hashes),artifact_hashes=hashes,
        raw_geometry_sha256=prep['rows_sha256'],post_hoc_raw_label_sensitivity=sensitivity,
        sensitivity_does_not_refit_or_select_models=True,new_deployment=False)
    immutable_json(folder/'verification.json',verification)
    lines = ['# Observed Motion Probability Probes','',
        'Sixteen fixed logistic models completed; no threshold or model selection.',
        'Same four explored source sites and 15,430 windows; no main or external scoring.',
        'Source: fresh_run fitting, cached_verified source geometry, fresh coefficient replay.', '',
        '| Target | Contrast (positive means improvement) | Motion minus quality | Conditional four-site 95% CI |',
        '| --- | --- | ---: | --- |']
    for value in report['contrasts']:
        lines.append(f"| {value['target']} | {value['metric']} | {value['equal_site_difference']:+.7f} | {value['conditional_four_site_ci95']} |")
    lines += ['', '## Interpretation', '',
        'The prespecified Brier contrast does not establish a motion-information improvement.',
        'For the larger-excursion target, ranking improves modestly but Brier and log loss',
        'worsen on every held site. Its motion AUROC ranges from 0.407 to 0.556, so the',
        'positive difference partly improves a poor control rather than establishing a useful detector.',
        'Both arms have worse larger-excursion Brier than the train-prevalence constant predictor',
        'on all four sites. General nonzero-change discrimination does not improve consistently.',
        'No safety threshold, calibration guarantee, trajectory gain or deployment follows.', '',
        '## Numerical Boundary Disclosure', '',
        'The fixed probe uses the stored float32 normalized future target restored to annotation',
        'pixels. It yields 739 larger-excursion positives; the earlier exact raw-annotation',
        'geometry audit yields 728. All 11 label differences are at a raw excursion of exactly',
        '10 pixels, affected by normalization roundoff. Neither rows nor registered outcomes',
        'are removed or silently changed. A post-hoc sensitivity readout of the same frozen',
        'probabilities against exact raw labels is retained below. No refitting or selection.', '',
        '| Site | Arm | Raw positives | Boundary differences | Raw-label Brier | Raw-label AUROC |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for t in sensitivity:
        lines.append(f"| {t['site']} | {t['arm']} | {t['raw_positives']} | {t['label_differences']} | {t['metrics']['brier']:.7f} | {t['metrics']['auroc']:.6f} |")
    lines += ['', 'This threshold is an annotation-coordinate diagnostic, not a physical movement definition.',
        'The raw-label readout is supplementary, not a replacement selected for better scores.', '',
        '## Reproduction', '', '```sh',
        '.venv-pytorch/bin/python scripts/probe_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_probe_v1.json',
        '.venv-pytorch/bin/python scripts/probe_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_probe_v1.json --replay',
        '.venv-pytorch/bin/python scripts/report_m3w_source_box_motion_probe.py', '```', '',
        f"All models converged within the fixed budget; maximum {max(t['iterations'][0] for t in report['trials'])} iterations.",
        f"Summed fitting time {sum(t['seconds'] for t in report['trials']):.3f}s. Sixteen exact coefficient replays.",
        f"Completed resume adds zero fits; {len(hashes)} artifacts unchanged. Private coefficients/predictions stay local.",
        'The model is convex and fitted deterministically once per cell; no duplicated-seed evidence is claimed.',
        'Two thousand site bootstrap resamples remain conditional, not independent confirmation.',
        'Stage5C and SMC remain off. The M3W research goal remains unmet.']
    assert sum(x['label_differences'] for x in sensitivity if x['arm']=='motion') == 11
    assert all(x['max_raw_distance_from_boundary'] == 0 for x in sensitivity)
    (folder/'conclusions.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(models=16,exact_replays=16,raw_label_boundary_differences=11,new_fits_on_resume=0)))


if __name__ == '__main__': main()
