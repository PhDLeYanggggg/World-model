"""Render verified fixed-arm evidence without choosing a model or a threshold."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args()
    reg = json.loads(args.registration.read_text())
    out = ROOT/reg['reports']
    read = lambda name: json.loads((out/(name+'.json')).read_text())
    report, analysis, prep, check = [read(k) for k in ['report', 'analysis', 'preparation', 'verification']]
    assert report['models'] == check['exact_replayed_heads'] == 36
    assert report['optimizer_updates'] == 360000 and check['new_updates_on_resume'] == 0
    for name, digest in check['artifact_hashes'].items():
        assert file_digest(ROOT/name) == digest
    arms = ['geometry', 'current', 'sequence']
    names = {'geometry':'Geometry + coverage', 'current':'Current appearance', 'sequence':'Eight-frame appearance'}
    summary = analysis['summary']
    interval = lambda v: '['+', '.join(f'{x:+.3f}' for x in v)+']'
    table = ['| Arm | Equal-site ADE gain (%) | Conditional 95% CI | Window gain (%) | Easy harm (annotation px) | Positive held fits |',
             '| --- | ---: | --- | ---: | ---: | ---: |']
    for arm in arms:
        x = summary[arm]
        table.append(f"| {names[arm]} | {x['equal_site_gain_percent']:+.3f} | {interval(x['conditional_four_site_ci95'])} | "
                     f"{x['window_gain_percent']:+.3f} | {x['easy_pixel_harm']:.6f} | {x['positive_held_models']}/12 |")
    contrasts = ['| Fixed contrast | Gain difference (percentage points) | Conditional 95% CI |',
                 '| --- | ---: | --- |']
    for x in analysis['contrasts']:
        contrasts.append(f"| {names[x['candidate']]} minus {names[x['reference']]} | "
                         f"{x['gain_difference_pp']:+.3f} | {interval(x['conditional_four_site_ci95'])} |")
    sites = ['| Site | Arm | Train gain (%) | Held gain (%) | Held hard gain (%) | Seed held gains (%) |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    for site in reg['sites']:
        for arm in arms:
            x = next(v for v in summary[arm]['sites'] if v['site'] == site)
            seed = ', '.join(f'{v:+.3f}' for v in x['seed_gains'])
            sites.append(f"| {site} | {arm} | {x['training_gain_percent']:+.3f} | {x['held_gain_percent']:+.3f} | "
                         f"{x['hard_gain_percent']:+.3f} | {seed} |")
    native = ['| Arm | ADE (annotation px) | FDE (annotation px) | ADE p95 | ADE p99 | Nonzero-target gain (%) | Oracle gain (%) |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in arms:
        x = summary[arm]
        native.append(f"| {arm} | {x['native_pixel_ade']:.5f} | {x['native_pixel_fde']:.5f} | {x['tail_ade95']:.5f} | "
                      f"{x['tail_ade99']:.5f} | {x['nonzero_gain_percent']:+.3f} | {x['binary_oracle_equal_site_gain_percent']:+.3f} |")
    fit_seconds = sum(t['fit']['fit_seconds'] for t in report['trials'])
    params = sorted(set(t['parameters'] for t in report['trials']))
    all_negative = all(x['equal_site_gain_percent'] <= 0 for x in summary.values())
    verdict = ('All three fixed arms lose to stationary CV on the primary source cross-fit readout. '
               'This experiment does not establish deployable forecasting benefit.' if all_negative else
               'At least one fixed arm has a positive development point estimate. This alone does not '
               'establish independent transfer, safe deployment or submission readiness.')
    text = f"""# Frozen Appearance And Temporal Prediction: Completed Evidence

## Conclusion

{verdict}

All 36 fresh neural trajectory heads completed their registered 10,000 updates.
No model, seed, threshold, row subset or checkpoint was selected using held results.
The three arms are reported together. An oracle is a future-label diagnostic, not a policy.

## Scope And Provenance

- `fresh_run`: 25,300 frozen-image embeddings, 36 heads, 360,000 updates, all-arm analysis, replay and verification.
- `cached_verified`: supplied image/history corpus, registered source split, geometric normalizers and matched sampler controls.
- `not_run`: main benchmark, bookstore/outer scoring, final confirmation, Stage37 comparison, new deployment and strict sensor-as-of evaluation.
- 15,430 stationary-history queries, 29 recordings, 545 scoped agents, four historically explored source sites.
- Eight observed and twelve predicted annotation steps, stride 12 raw frames. This is not a raw-frame t+50 rerun.
- Coordinate errors remain annotation-pixel/local diagnostics. No seconds, metric, true 3D or foundation-model claim.
- The offline annotation contract allows supplied interpolated histories. It is not real-time perception certification or human motion gold.

## Fixed Comparison

All arms use the same geometry/coverage, projection/GRU/trajectory architecture,
three seeds and full-population sampling. The geometry arm zeroes visual features;
current appearance repeats the last image embedding; temporal appearance uses all
eight historical embeddings. ResNet18 remains frozen. Only the trajectory heads
are trained, each with parameter count {params}. Source crops are 32 x 32 before
upsampling; interpolation does not restore lost visual detail.

{chr(10).join(table)}

Primary gain is the ratio of equal-site mean errors, not a simple average of
site percentage gains. Seeds are averaged as errors, not prediction ensembles.
The 2,000 site-resamples are conditional on four explored sites with shared
training populations. These intervals are not independent confirmation.

{chr(10).join(contrasts)}

## Per-Site And Seed Results

{chr(10).join(sites)}

Hard subsets use training-complement error cutoffs. No future-defined slice is
an inference input. Easy targets have zero CV error, so percentage degradation
is undefined; absolute harm is reported and cannot be relabeled a 2% safety pass.

## Error Magnitudes And Oracle

{chr(10).join(native)}

Native-pixel aggregates are descriptive for this SDD subset, not cross-dataset
metric averages. Oracle gains assume access to future errors and are not deployable.

## Compute And Verification

Frozen feature extraction: {prep['extraction_seconds']:.3f} seconds summed across
{prep['chunks']} immutable chunks. Head fitting: {fit_seconds:.3f} summed seconds,
including the 100-update pilot. CPU four threads, inter-op one, DataLoader workers
zero in the native arm64 environment. Three exact encoder chunk replays and
36 exact train/held head replays passed. The completed-run resume added zero
updates and preserved {check['immutable_artifacts']} hashed artifacts.

Raw query/image alignment, training/held separation, target poisoning, matched
sampling streams, finite bounded outputs and out-of-fold scores were checked.
These checks establish implementation evidence, not predictive success.

## Research Boundary

No new deployment, Stage5C execution or SMC activation. Historical selector scores
remain exploratory under the lineage audit. A visual contrast alone does not prove
world dynamics or a scene-level intervention contribution. Remaining requirements
include a useful cross-site candidate, safe independently calibrated intervention,
full-population forecasting, external domains and unexposed confirmation.

See [the immutable registration](../source_pretrained_temporal_decision.md),
[machine-readable results](analysis.json), [verification](verification.json) and
[reproduction commands](reproducibility.md).
"""
    (out/'conclusions.md').write_text(text)
    (out/'reproducibility.md').write_text(f"""# Reproduction

Registration was committed as `06f97771` before extraction or fitting.
Registration SHA256: `{file_digest(args.registration)}`.
Use the existing native arm64 `.venv-pytorch`; CPU four threads, workers zero.
The supplied local corpus, private parent checkpoints and official frozen encoder
weights must be available and match all registered hashes. Missing private assets
are an explicit prerequisite, not a successful reproduction. No downloads or fits
are hidden in the analyzer. Existing completed fits are verified and reused.

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_pretrained_temporal.py --registration {args.registration} --phase prepare
.venv-pytorch/bin/python scripts/run_m3w_source_pretrained_temporal.py --registration {args.registration} --phase train
.venv-pytorch/bin/python scripts/run_m3w_source_pretrained_temporal.py --registration {args.registration} --phase replay
.venv-pytorch/bin/python scripts/analyze_m3w_source_pretrained_temporal.py --registration {args.registration}
.venv-pytorch/bin/python scripts/verify_m3w_source_pretrained_temporal.py --registration {args.registration}
.venv-pytorch/bin/python scripts/report_m3w_source_pretrained_temporal.py --registration {args.registration}
```

Interruptions recover from the last atomic 200-update checkpoint with optimizer,
sampler and Torch RNG state. The heartbeat records the current PID, trial and
step. Do not rerun completed fits under another identity or tune held results.
Images, image features, weights and checkpoints remain private and excluded from Git.
See `verification.json` for current artifact hashes and actual resume evidence.
""")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    colors = ['#50555C', '#087F8C', '#B74B29']
    for i, (arm, color) in enumerate(zip(arms, colors)):
        x = summary[arm]; lo, hi = x['conditional_four_site_ci95']; pt = x['equal_site_gain_percent']
        ax[0].errorbar(pt, i, xerr=[[pt-lo], [hi-pt]], fmt='o', color=color, capsize=5)
        fits = [t for t in report['trials'] if t['arm'] == arm]
        traces = [[v['objective_loss'] for v in t['fit']['trace']] for t in fits]
        steps = np.asarray([v['step'] for v in fits[0]['fit']['trace']])
        mean = np.mean(traces, axis=0)
        ax[1].plot(steps, mean, color=color, label=names[arm], alpha=.8)
    ax[0].set_yticks(range(3), [names[a] for a in arms]); ax[0].invert_yaxis()
    ax[0].axvline(0, color='black', lw=.8); ax[0].set_xlabel('Equal-site ADE gain vs CV (%)')
    ax[0].set_title('Conditional four-site bootstrap, n=2,000')
    ax[1].set_xlabel('Optimizer update'); ax[1].set_ylabel('Sampled normalized ADE training loss')
    ax[1].set_title('Matched batch streams; mean of 12 fits per arm')
    ax[1].legend(fontsize=8)
    for a in ax: a.grid(alpha=.2)
    fig.savefig(out/'comparison.svg', metadata={'Date':None})
    fig.savefig(ROOT/reg['output']/'comparison.png', dpi=140)
    plt.close(fig)
    print(json.dumps(dict(report=str(out/'conclusions.md'), fit_seconds=fit_seconds,
                          all_arms_nonpositive=all_negative, new_deployment=False)))


if __name__ == '__main__':
    main()
