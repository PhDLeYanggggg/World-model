"""Render the complete matched readout comparison, including negative outcomes."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_experiment_contract import file_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args(); reg = json.loads(args.registration.read_text())
    public, private = ROOT/reg['reports'], ROOT/reg['output']
    report, analysis, check = [json.loads((public/(n+'.json')).read_text())
                               for n in ('report', 'analysis', 'verification')]
    for path, digest in check['artifact_hashes'].items():
        assert file_digest(ROOT/path) == digest
    summary = analysis['summary']
    interval = lambda values: '['+', '.join(f'{v:+.5f}' for v in values)+']'
    table = ['| Arm | Source | Equal-site gain (%) | Conditional 95% interval | Static harm (px) | Positive held fits |',
             '| --- | --- | ---: | --- | ---: | ---: |']
    detail = ['| Arm | Excluded source site | Held gain (%) | Hard gain (%) | Seed gains (%) | Training seed gains (%) |',
              '| --- | --- | ---: | ---: | --- | --- |']
    errors = ['| Arm | Native ADE | Native FDE | Nonzero-target gain (%) | Binary oracle (%) |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name, x in summary.items():
        table.append(f"| {name} | {x['result_source']} | {x['equal_site_gain_percent']:+.5f} | {interval(x['conditional_four_site_ci95'])} | {x['static_pixel_harm']:.7f} | {x['positive_held_models']}/12 |")
        errors.append(f"| {name} | {x['native_pixel_ade']:.7f} | {x['native_pixel_fde']:.7f} | {x['nonzero_gain_percent']:+.5f} | {x['binary_oracle_equal_site_gain_percent']:.6f} |")
        for site in x['sites']:
            detail.append(f"| {name} | {site['site']} | {site['gain_percent']:+.5f} | {site['hard_gain_percent']:+.5f} | "+
                ', '.join(f'{v:+.5f}' for v in site['seed_gains'])+' | '+
                ', '.join(f'{v:+.5f}' for v in site['training_gains'])+' |')
    contrasts = ['| Fixed contrast | Gain difference (pp) | Conditional interval |', '| --- | ---: | --- |']
    for x in analysis['contrasts']:
        contrasts.append(f"| {x['candidate']} minus {x['reference']} | {x['gain_difference_pp']:+.5f} | {interval(x['conditional_four_site_ci95'])} |")
    negative = all(summary[arm+'_conditioned']['equal_site_gain_percent'] <= 0 for arm in ('geometry', 'centered'))
    verdict = ('Neither conditioned arm beats stationary CV. No new deployment.' if negative else
               'A source-development point estimate is positive; this is not independent confirmation or deployment certification.')
    seconds = sum(t['fit']['fit_seconds'] for t in report['trials'])
    fraction = [t['logged_gradient_clipped_fraction'] for t in analysis['diagnostics']]
    text = f"""# Train-Scale Readout Conditioning

## Conclusion

{verdict}

Twenty-four fresh Torch heads complete 240,000 updates. The 24 unconditioned
importance-corrected heads are cached_verified controls. Registration was
committed as 5aa189b1 before the included 100-update pilot. No held-driven
checkpoint, seed, threshold or early-stop selection.

## What Changed

The fixed training-gradient audit found severe output-layer concentration but
no large clipping-induced mean-direction reversal at the final checkpoints.
This trial changes only q to q/c before the existing radial bound. Each c is
the median training radius divided by the existing training-loss scale, floored
at one. It does not change the represented forecast class or per-row bounds.
It does change parameter conditioning and optimizer geometry, including effective
step sizes and weight decay. It is not a pure clipping ablation.

All 63,960 parameters, inputs, training complements, original loss, sample streams,
seeds, 10,000-update budgets, schedule and cap5 are retained. Frozen ResNet18
features remain frozen; these are trajectory readout heads, not end-to-end
visual world-model training.

## Primary Results

{chr(10).join(table)}

{chr(10).join(contrasts)}

The primary is the ratio of equal-site mean past-normalized ADEs. Seed errors
are averaged, not prediction-ensembled. The 2,000 bootstrap draws are conditional
on four explored source sites and shared training folds. They are not independent
confirmation or a guarantee of generalization. No main, outer or bookstore scores.

## Every Site and Seed

{chr(10).join(detail)}

## Error Magnitudes and Candidate Utility

{chr(10).join(errors)}

Native values are annotation pixels. Easy percentage degradation is undefined
on these zero-error static-CV targets, not a 2% gate pass. Binary oracles use
future labels only for diagnosis, not as inference inputs or learned results.

## Optimization and Verification

Logged gradient clipping fractions range {min(fraction):.4f} to {max(fraction):.4f},
versus 1.0 in all predecessor fits. Logs cover update1/every100, not every update.
This quantifies changed optimization behavior, not prediction success.

All 24 train/held forecasts replay exactly; draw streams regenerate and match
controls. Six OOF archives recompute. Thirty-two future-label poison queries and
24 forbidden training-role checks pass. Completed resume adds zero updates and
preserves {check['immutable_artifacts']} artifacts. Thirty-three scoped tests pass;
the full legacy suite was not rerun. CPUarm64, four compute threads, one inter-op,
zero workers. Summed fitting time {seconds:.3f}s including the pilot.

Fresh: readout fits, analysis and verification. Cached_verified: source data,
fixed embeddings, event mapping and previous control fits. Not_run: independent
confirmation, main/outer/external readout or a deployable intervention policy.
Offline supplied annotations may contain later interpolation controls; not
strict sensor-as-of. Eight observed/twelve predicted steps, stride12rawframes;
not metric or seconds. No true-3D/foundation claim, Stage5C or SMC.

See [fixed design](../source_conditioned_readout_decision.md), [analysis](analysis.json),
[verification](verification.json), [failure analysis](failure_analysis.md),
[gates](gates.md) and [reproduction](reproducibility.md).
"""
    (public/'conclusions.md').write_text(text)
    commands = '\n'.join([
        f'.venv-pytorch/bin/python scripts/{name}_m3w_source_conditioned_readout.py --registration {args.registration}'
        for name in ('run', 'analyze', 'verify', 'report')])
    (public/'reproducibility.md').write_text(f"""# Reproduction and Recovery

Registration SHA256: `{file_digest(args.registration)}`. Original hash-matching
source data, embeddings, event groups and controls must be present locally.
Missing private assets are not a completed reproduction.

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_conditioned_readout.py --registration {args.registration}
.venv-pytorch/bin/python scripts/run_m3w_source_conditioned_readout.py --registration {args.registration} --replay
{commands.split(chr(10), 1)[1]}
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_gradient_diagnostic.py tests/test_m3w_source_conditioned_readout.py tests/test_m3w_source_conditioned_resume.py tests/test_m3w_source_importance_sampling.py tests/test_m3w_source_episode_sampler.py tests/test_m3w_source_pretrained_temporal.py tests/test_m3w_source_temporal_centered.py
```

The included pilot uses `--trial coupa_geometry_seed17 --stop-at 100`. Full
execution resumes that checkpoint. Checkpoints every200steps retain model,
optimizer, RNGs, sampler, readout gain and identity. Inspect PID/heartbeat/log
before restarting; an observation timeout is not a stopped process. A completed
run must add zero updates and leave artifacts unchanged. Native arm64 CPU4,
inter-op1, workers0, no resource probing or DataLoader multiprocessing.

The gradient audit can be recomputed separately with
`scripts/audit_m3w_source_gradients.py --registration configs/m3w_source_gradient_diagnostic_v1.json --replay`.
Its gradients concern only predecessor training rows; it does not score a new
predictor. Large private vectors, caches, data and checkpoints are not in Git.
""")
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/m3w-conditioned-mpl')
    os.environ.setdefault('XDG_CACHE_HOME', '/tmp/m3w-conditioned-cache')
    Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['svg.hashsalt'] = 'm3w-source-conditioned-fixed-v1'
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout='constrained')
    names = list(summary)
    for i, name in enumerate(names):
        x = summary[name]; value = x['equal_site_gain_percent']; lo, hi = x['conditional_four_site_ci95']
        axes[0].errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt='o', capsize=4, color=f'C{i}')
        axes[1].barh(i, x['static_pixel_harm'], color=f'C{i}')
    for ax in axes:
        ax.set_yticks(np.arange(len(names)), names); ax.invert_yaxis(); ax.grid(alpha=.2, axis='x')
    axes[0].axvline(0, color='black', lw=.8)
    axes[0].set_xlabel('Equal-site ADE gain vs stationary CV (%)')
    axes[1].set_xlabel('Static-target absolute harm (annotation pixels)')
    fig.suptitle('Train-scale readout: conditional source comparison, not confirmation')
    fig.savefig(public/'comparison.svg', metadata={'Date': None})
    svg = public/'comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(private/'comparison.png', dpi=150); plt.close(fig)
    print(json.dumps(dict(verdict=verdict, fit_seconds=seconds, report=str(public/'conclusions.md'))))


if __name__ == '__main__':
    main()
