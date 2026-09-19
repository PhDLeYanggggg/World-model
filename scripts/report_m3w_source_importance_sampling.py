"""Render the complete fixed importance-correction comparison."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.evaluation.m3w_experiment_contract import file_digest


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registration',type=Path,required=True)
    args = p.parse_args(); reg = json.loads(args.registration.read_text())
    public,private = ROOT/reg['reports'],ROOT/reg['output']
    read = lambda n:json.loads((public/(n+'.json')).read_text())
    report,analysis,check = [read(n) for n in ('report','analysis','verification')]
    for path,digest in check['artifact_hashes'].items(): assert file_digest(ROOT/path) == digest
    summary = analysis['summary']; interval = lambda x:'['+', '.join(f'{v:+.4f}' for v in x)+']'
    table = ['| Arm | Source | Equal-site gain (%) | Conditional 95% interval | Static harm (annotation px) | Positive held fits |',
             '| --- | --- | ---: | --- | ---: | ---: |']
    site_table = ['| Arm | Site | Training gain (%) | Held gain (%) | Hard gain (%) | Held seeds (%) |',
                  '| --- | --- | ---: | ---: | ---: | --- |']
    tails = ['| Arm | Native ADE | Native FDE | Nonzero gain (%) | p95 ADE | p99 ADE | Binary oracle (%) |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,x in summary.items():
        table.append(f"| {arm} | {x['result_source']} | {x['equal_site_gain_percent']:+.4f} | {interval(x['conditional_four_site_ci95'])} | {x['easy_pixel_harm']:.6f} | {x['positive_held_models']}/12 |")
        for v in x['sites']:
            site_table.append(f"| {arm} | {v['site']} | {v['training_gain_percent']:+.4f} | {v['held_gain_percent']:+.4f} | {v['hard_gain_percent']:+.4f} | "+', '.join(f'{s:+.4f}' for s in v['seed_gains'])+' |')
        tails.append(f"| {arm} | {x['native_pixel_ade']:.6f} | {x['native_pixel_fde']:.6f} | {x['nonzero_gain_percent']:+.4f} | {x['tail_ade95']:.6f} | {x['tail_ade99']:.6f} | {x['binary_oracle_equal_site_gain_percent']:.6f} |")
    contrast = ['| Fixed contrast | Gain difference (pp) | Conditional interval |','| --- | ---: | --- |']
    for x in analysis['contrasts']:
        contrast.append(f"| {x['candidate']} minus {x['reference']} | {x['gain_difference_pp']:+.4f} | {interval(x['conditional_four_site_ci95'])} |")
    negative = all(summary[k]['equal_site_gain_percent'] <= 0 for k in ('geometry_corrected','centered_corrected'))
    verdict = ('Neither corrected arm beats stationary CV. No new deployment.' if negative else
               'A development point estimate is positive; independent confirmation and safe deployment remain unproved.')
    seconds = sum(t['fit']['fit_seconds'] for t in report['trials'])
    text = f"""# Importance-Corrected Exposure: Completed Comparison

## Conclusion

{verdict}

24 fresh heads complete 240,000 updates. Four original uniform/uncorrected arms
(48 heads) are cached_verified, not retrained. Registration b2276809 was pushed
before the included 100-update pilot. No held-driven stopping, checkpoint, seed,
threshold or cohort selection. All planned contrasts are retained.

## Intervention And Objective

Same episode sampler, draw streams, geometry/centered inputs, 63,960 parameters,
all rows, initializations, learning-rate schedule, normalizers and original metric.
Each sampled row loss is multiplied by 1/(N_train*p_train(row)); no clipping or
self-normalization of weights. Four training-fold fixed-offset probes verify
expected loss/unclipped-gradient equality. This is not an unbiased Adam-update
claim: clipping, adaptive optimization and finite-batch variance remain.

Same 15,430 source queries, four explored sites, three seeds, offline eight past
and twelve future annotation steps at stride 12 raw frames. Not main/t+50/external
evaluation. Past labels can have later interpolation controls; not sensor-as-of.
No verified metric/seconds scale, independent physical events or human-gold claim.

## Primary Results

{chr(10).join(table)}

{chr(10).join(contrast)}

Primary is the ratio of equal-site mean normalized errors. Seed errors are
averaged, not prediction-ensembled. The 2,000 site-bootstrap draws are conditional
on four explored sites/shared fitting folds, not independent confirmation.

## All Sites And Seeds

{chr(10).join(site_table)}

## Error Magnitude

{chr(10).join(tails)}

Static CV error is zero; percentage easy degradation is undefined, not a 2% pass.
Native errors are annotation-pixel descriptions. Binary oracles use targets
only for diagnosis and do not demonstrate a learned switching policy.

## Verification And Cost

All 24 predictions replay exactly and draw streams regenerate. All 24 match their
uncorrected controls' draws; twelve paired-arm streams agree. Factors match the
training-only propensities. Six OOF archives recompute. Zero-update resume
preserves {check['immutable_artifacts']} immutable artifacts. 37 scoped tests pass;
no full legacy-suite rerun. Native arm64 CPU, four threads, one inter-op thread,
zero workers. Summed fitting
{seconds:.3f}s including pilot. Frozen image encoder is not retrained.

Fresh: expectation checks, corrected fits, analysis and verification.
Cached_verified: old controls, event groups, image features and source provenance.
Not_run: independent calibration/confirmation, main/outer/external readout,
new intervention policy. No new deployment, Stage5C execution or SMC.

See [design](../source_importance_sampling_decision.md), [analysis](analysis.json),
[verification](verification.json), [failure analysis](failure_analysis.md),
[gates](gates.md) and [reproduction](reproducibility.md).
"""
    (public/'conclusions.md').write_text(text)
    entry = lambda name:f'.venv-pytorch/bin/python scripts/{name}_m3w_source_importance_sampling.py --registration {args.registration}'
    commands = '\n'.join([entry('run')+' --check-objective',entry('run'),entry('run')+' --replay',
                          entry('analyze'),entry('verify'),entry('report')])
    (public/'reproducibility.md').write_text(f"""# Reproduction

Registration SHA256: `{file_digest(args.registration)}`. Private hash-matching
source data, past image features, episode mapping and cached controls are needed.
Missing private assets are not a completed reproduction. Keep bound files frozen.

```bash
{commands}
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_importance_sampling.py tests/test_m3w_source_episode_sampler.py
```

The included pilot uses `--trial coupa_geometry_seed17 --stop-at 100` before
the full run. Interrupted runs resume from 200-step atomic checkpoints; never
restart a live process after an observation timeout. PID/heartbeat/training log
and all 24 checkpoints remain local. Completed resume performs zero updates.
No CUDA/MPS resource probing or multiprocessing. On Darwin reject non-arm64 before
Torch import. Training cost does not extrapolate to end-to-end encoder fitting.
The full legacy test suite can rewrite old artifacts and was not rerun here.
""")
    os.environ.setdefault('MPLCONFIGDIR','/tmp/m3w-importance-mpl')
    os.environ.setdefault('XDG_CACHE_HOME','/tmp/m3w-importance-cache')
    Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['svg.hashsalt'] = 'm3w-source-importance-fixed-v1'
    fig,axes = plt.subplots(2,1,figsize=(11,8),layout='constrained')
    panels = [('Uniform and corrected (zoomed)',[k for k in summary if not k.endswith('_event')]),
              ('All samplers (same units, wider range)',list(summary))]
    colors = {name:f'C{i}' for i,name in enumerate(summary)}
    for ax,(title,names) in zip(axes,panels):
        for i,name in enumerate(names):
            x = summary[name]; value=x['equal_site_gain_percent']; lo,hi=x['conditional_four_site_ci95']
            ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',capsize=4,color=colors[name])
        ax.set_yticks(np.arange(len(names)),names); ax.invert_yaxis()
        ax.axvline(0,color='black',lw=.8); ax.grid(alpha=.2); ax.set_title(title)
        ax.set_xlabel('Equal-site ADE gain vs stationary CV (%)')
    fig.suptitle('Importance correction: conditional four-site intervals, not confirmation')
    fig.savefig(public/'comparison.svg',metadata={'Date':None})
    svg = public/'comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(private/'comparison.png',dpi=150); plt.close(fig)
    print(json.dumps(dict(verdict=verdict,fit_seconds=seconds,report=str(public/'conclusions.md'))))


if __name__ == '__main__': main()
