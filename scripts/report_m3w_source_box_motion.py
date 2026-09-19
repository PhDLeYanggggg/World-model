"""Render verified source-motion evidence without selecting a deployable model."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_experiment_contract import file_digest


def main():
    folder = ROOT/'outputs/publication_readiness_2026_09/source_box_motion_v1'
    analysis = json.loads((folder/'analysis.json').read_text())
    verification = json.loads((folder/'verification.json').read_text())
    report = json.loads((folder/'report.json').read_text())
    extraction = json.loads((folder/'extraction.json').read_text())
    for path, sha in verification['artifact_hashes'].items():
        if file_digest(ROOT/path) != sha: raise ValueError('Changed verified artifact: '+path)
    s = analysis['summary']; contrast = analysis['contrasts'][0]
    forecast_pass = s['motion']['conditional_four_site_ci95'][0] > 0
    representation_pass = contrast['conditional_four_site_ci95'][0] > 0
    verdict = ('positive_source_forecast_evidence_not_confirmation' if forecast_pass else
               'no_positive_source_motion_forecast_evidence')
    seconds = sum(t['fit']['fit_seconds'] for t in report['trials'])
    extraction_seconds = sum(t['seconds'] for t in extraction['records'])
    rows = []
    for name in ('geometry','centered','quality','motion'):
        a = s[name]; ci = a['conditional_four_site_ci95']
        rows.append(f"| {name} ({a['result_source']}) | {a['equal_site_gain_percent']:.8f}% | "
                    f"[{ci[0]:.8f}, {ci[1]:.8f}] | {a['static_pixel_harm']:.8f} | "
                    f"{a['nonzero_gain_percent']:.8f}% | {a['binary_oracle_equal_site_gain_percent']:.8f}% |")
    table = '\n'.join(rows)
    evidence = f"""# Observed Box-Motion Comparison

## Scope

Result source: fresh_run for flow extraction,24 new Torch heads and readout;
cached_verified for original images and24 preceding conditioned references.
All15,430 source windows,29 recordings,four explored sites,three seeds retained.
Eight supplied-annotation observations ->12 predictions at12-raw-frame stride.
No main/bookstore/external scoring; no selection or deployment.

## Results

| Arm/source | Equal-site ADE gain vs CV | Conditional four-site95%CI | Static harm, annotation px | Nonzero-target gain | Binary CV/candidate oracle |
| --- | ---: | --- | ---: | ---: | ---: |
{table}

Motion-minus-quality: {contrast['gain_difference_pp']:.8f} percentage points;
conditional95%CI {contrast['conditional_four_site_ci95']}.
Positive held fits: quality {s['quality']['positive_held_models']}/12;
motion {s['motion']['positive_held_models']}/12.
Verdict: **{verdict}**.

The exact same63,960-parameter head is trained on quality-only versus
quality+motion tokens. Fixed importance-corrected uniform-row ADE, training-only
normalization/readout gain, same sampler streams and240,000updates.
No learned visual encoder or end-to-end pixel training is claimed.

## Measurement

23,890 unique observed pairs supply108,010 overlapping pair uses. Box support
{extraction['summary']['box_supported']['mean']*100:.3f}%; surround support
{extraction['summary']['surround_supported']['mean']*100:.3f}%.
Rounded crop translation and per-axis video resize are restored before regional
flow pooling. The surrounding region is a background proxy, not verified camera
motion. Flow magnitude and support do not establish intent or forecasting value.
Annotation-box regions are not segmented bodies. All pair uses contain at least
one generated annotation; {extraction['summary']['occluded']['mean']*100:.3f}% contain an occlusion flag.
These are offline supplied annotations, not strict sensor-as-of observations.

51 zero-radius unsupported rows initially stopped the pilot before any update.
The pre-fit amendment retains them, zeros only unavailable motion channels and
preserves the original zero-forecast support rule. No retrospective row deletion.

## Compute And Verification

Nativearm64 CPU4/inter-op1/workers0. Fresh fitting {seconds:.3f}s including the
100-update resume pilot; flow extraction {extraction_seconds:.3f}s excluding
source-loading/hash verification.24 exact model replays and23,890 exact flow
pair replays.24 regenerated sampler streams match controls;12 arm pairs match.
Six OOF archives recomputed.32 future-target poison queries unchanged;24 illegal
training-role checks rejected. Completed resume adds0updates and preserves
{verification['immutable_artifacts']} artifacts. Checkpoints/caches remain local.

## Limits

The2,000 bootstrap resamples use four explored sites and shared training folds;
they are conditional uncertainty, not independent confirmation. Seeds are
averaged at the error level, not ensembled forecasts. Overlapping windows are
not independent. Static percentage degradation is undefined because CV error
is zero; absolute harm is reported, not converted to a2%pass.

No metric/seconds/true3D/foundation claim. Raw-frame t+50 is not rerun here.
Historical external selector gains remain exploratory. Stage5C and SMC are off.
The main research goal is active and unmet.
"""
    failure = f"""# Interpretation And Remaining Gap

The matched representation contrast is {contrast['gain_difference_pp']:.8f}pp.
Its conditional lower bound is {contrast['conditional_four_site_ci95'][0]:.8f}pp.
The motion arm's gain against CV is {s['motion']['equal_site_gain_percent']:.8f}%,
with {s['motion']['positive_held_models']} of12 individually positive held fits.

This distinguishes measured input variation, reduced forecast harm and actual
predictive improvement. Engineering support and exact replay cannot replace
positive forecasting evidence. A tiny positive hard-slice result alone cannot
justify intervention if aggregate and static-target behavior are worse.

Possible limitations remain distinct:32px low-pass crops; annotation regions
rather than segmentation; neighboring-object/camera/occlusion confounding;
scarce independent onset events; source shift; representation aggregation;
and the tested conditional decoder/objective. This experiment cannot identify
one of these as the sole causal failure. It does not prove all pixels are useless.

The prior native96px ETH/Hotel/Zara experiment was negative. Do not simply
claim that higher resolution will repair SDD or repeat a threshold sweep on
near-zero candidates. A separately registered probability probe has now fitted
16 logistic models. The motion features slightly improve larger-excursion
ranking (AUROC difference +0.01241), but worsen Brier by0.001513 on average and
on every held site. Absolute motion AUROC is only0.407-0.556. General nonzero
change prediction does not improve consistently. This does not provide a safe
gate or prove useful trajectory direction. Full probability results and the
11-row numerical label-boundary disclosure are in
[the complementary report](../source_box_motion_probe_v1/conclusions.md).

Eighteen of24 trajectory fits improve training ADE slightly but none transfer;
the other six do not even improve training ADE. This is evidence of limited
source predictability/transfer for these readouts, not proof of unlearnability.

The low-pass32px crop's median annotated box is9.34 by11.86pixels, smaller than
the fixed15px flow aggregation window along both axes. Measurement support
therefore cannot establish resolved body motion. That scale comparison is a
post-hoc limitation, not proof it caused the failure. Before a higher-resolution
training run, inspect whether independently defined past events retain body
motion at native resolution, with matched regional/quality controls and no
held-label selection. That native SDD measurement comparison is not_run.

Main/external confirmation and scene-level risk calibration remain unestablished.
No new model is deployed. Baseline rejection is a fallback, not neural success.
"""
    gates = f"""# Source Experiment Gates

| Check | Outcome |
| --- | --- |
| Bound observed inputs and unchanged cohort | pass |
| All24 heads and240,000updates complete | pass |
| Flow/model replay, train-only normalization, role and poison checks | pass |
| Motion-minus-quality conditional lower bound >0 | {'pass' if representation_pass else 'fail'} |
| Motion-versus-CV conditional lower bound >0 | {'pass' if forecast_pass else 'fail'} |
| Independent primary/external confirmation | not_run |
| Formal independent-scene risk calibration | not_run |
| New deployment | false |
| Stage5C executed | false |
| SMC enabled | false |

Internal engineering checks do not constitute world-model or submission gates.
Verdict:{verdict}. Goal remains active and unmet.
"""
    reproducibility = f"""# Reproduction

Run from the repository with nativearm64 `.venv-pytorch`. Data and the bound
OpenCV runtime must already exist locally. No source data or checkpoints are inGit.

```sh
.venv-pytorch/bin/python scripts/build_m3w_source_box_motion.py
.venv-pytorch/bin/python scripts/run_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_v1.json --replay
.venv-pytorch/bin/python scripts/analyze_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_v1.json
.venv-pytorch/bin/python scripts/verify_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_v1.json
.venv-pytorch/bin/python scripts/report_m3w_source_box_motion.py
```

Repeating the training command verifies receipts and resumes incomplete fixed
trials. Never delete an active checkpoint or start duplicate workers. Atomic
checkpoints every200updates. Private training.log/fit_heartbeat.json record PID,
step and elapsed fit time. Initial100-update pilot is part of the fixed budget.
Interrupted-resume equality is tested on the model; actual pilot is recovered.

Registration: {file_digest(ROOT/'configs/m3w_source_box_motion_v1.json')}
Analysis: {file_digest(folder/'analysis.json')}
Verification: {file_digest(folder/'verification.json')}

Full legacy test suite is not rerun because some integrations overwrite historical
reports. Scoped tests and real artifact checks are reported separately.
"""
    for name, text in [('conclusions.md',evidence),('failure_analysis.md',failure),
                       ('gates.md',gates),('reproducibility.md',reproducibility)]:
        (folder/name).write_text(text)
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/m3w_motion_mpl')
    os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/m3w_motion_xdg')
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['geometry','centered','quality','motion']
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    x = list(range(4)); colors=['#707070','#98704b','#327eae','#2b976a']
    axes[0].bar(x,[s[n]['equal_site_gain_percent'] for n in names],color=colors)
    point = [s[n]['equal_site_gain_percent'] for n in names]
    axes[0].errorbar(x, point, yerr=[
        [s[n]['equal_site_gain_percent']-s[n]['conditional_four_site_ci95'][0] for n in names],
        [s[n]['conditional_four_site_ci95'][1]-s[n]['equal_site_gain_percent'] for n in names]],
        fmt='none', ecolor='black', capsize=4, linewidth=1)
    axes[0].axhline(0,color='black',linewidth=.8)
    axes[0].set(ylabel='Equal-site ADE gain vs CV (%)',title='Four explored sites: conditional 95% CI')
    axes[1].bar(x,[s[n]['static_pixel_harm'] for n in names],color=colors)
    axes[1].set(ylabel='Static absolute error (annotation pixels)',title='Zero-error CV: percentage undefined')
    for ax in axes: ax.set_xticks(x,names,rotation=20)
    fig.savefig(folder/'comparison.svg')
    svg_path = folder/'comparison.svg'
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
    fig.savefig('/private/tmp/m3w_source_box_motion_comparison.png',dpi=160)
    plt.close(fig)
    print(json.dumps(dict(verdict=verdict, fit_seconds=seconds, extraction_seconds=extraction_seconds)))


if __name__ == '__main__': main()
