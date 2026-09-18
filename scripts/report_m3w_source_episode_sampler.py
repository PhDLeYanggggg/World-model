"""Render every fixed sampler outcome, without changing scientific selection."""
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
    for path,sha in check['artifact_hashes'].items(): assert file_digest(ROOT/path) == sha
    summary = analysis['summary']
    interval = lambda x:'['+', '.join(f'{v:+.4f}' for v in x)+']'
    table = ['| Arm | Source | Equal-site gain (%) | Conditional site interval | Absolute static harm (px) | Positive held fits |',
             '| --- | --- | ---: | --- | ---: | ---: |']
    sites = ['| Arm | Site | Training gain (%) | Held gain (%) | Hard gain (%) | Three held seed gains (%) |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    tails = ['| Arm | Native ADE | Native FDE | Nonzero gain (%) | p95 ADE | p99 ADE | Binary oracle (%) |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,x in summary.items():
        table.append(f"| {arm} | {x['result_source']} | {x['equal_site_gain_percent']:+.4f} | {interval(x['conditional_four_site_ci95'])} | {x['easy_pixel_harm']:.6f} | {x['positive_held_models']}/12 |")
        for v in x['sites']:
            sites.append(f"| {arm} | {v['site']} | {v['training_gain_percent']:+.4f} | {v['held_gain_percent']:+.4f} | {v['hard_gain_percent']:+.4f} | "+', '.join(f'{y:+.4f}' for y in v['seed_gains'])+' |')
        tails.append(f"| {arm} | {x['native_pixel_ade']:.6f} | {x['native_pixel_fde']:.6f} | {x['nonzero_gain_percent']:+.4f} | {x['tail_ade95']:.6f} | {x['tail_ade99']:.6f} | {x['binary_oracle_equal_site_gain_percent']:.6f} |")
    contrast = ['| Fixed contrast | Difference (pp) | Conditional interval |','| --- | ---: | --- |']
    for x in analysis['contrasts']:
        contrast.append(f"| {x['candidate']} minus {x['reference']} | {x['gain_difference_pp']:+.4f} | {interval(x['conditional_four_site_ci95'])} |")
    negative = all(summary[k]['equal_site_gain_percent'] <= 0 for k in ('geometry_event','centered_event'))
    verdict = ('Equal-episode training still does not beat stationary CV. No new deployment.' if negative else
               'A development point estimate is positive; independent confirmation and deployment safety remain unproved.')
    seconds = sum(t['fit']['fit_seconds'] for t in report['trials'])
    text = f"""# Equal-Episode Exposure: Completed Comparison

## Conclusion

{verdict}

All 24 fresh heads completed 10,000 updates each. The two original uniform
controls (24 heads) are cached_verified, not retrained. No held-driven stopping,
seed, checkpoint, threshold or population selection. Registration commits
662dcbba/dd5c7128 precede fitting; the latter fixes trailing whitespace and hashes.

## What Changed

Only training row probabilities changed: equal mass per past-defined annotation
episode within each training complement, then equal mass among its rows.
All rows keep positive probability. Geometry/coverage, frozen visual features,
63,960 parameters, seeded initialization, all-target ADE, original normalizer,
training cost scale, hard cutoff and original evaluation are unchanged.
The geometry and centered arms share weighted draws; draws intentionally differ
from uniform controls. No static-gradient removal or future-label group key.

Same 15,430 source queries, four explored sites, three seeds, eight observed and
twelve predicted annotation steps at stride 12 raw frames. Not the main benchmark
or t+50 supplement. Past-indexed supplied annotations may use later interpolation
controls: offline, not strict sensor-as-of. No metric or seconds equivalence.

## Primary Results

{chr(10).join(table)}

{chr(10).join(contrast)}

Primary remains the ratio of equal-site mean normalized errors, not mean site
percentages or episode-reweighted evaluation. Seeds average errors, not predictions.
The 2,000 shared site-bootstrap draws remain conditional on four explored sites
and overlapping fitting complements. They are not independent confirmation.

## Every Site And Seed

{chr(10).join(sites)}

## Error Magnitude

{chr(10).join(tails)}

Native errors are annotation-pixel descriptions. Static baseline error is zero;
percentage easy degradation is undefined, not a 2% safety pass. Future-informed
binary oracles are label-only diagnostics, not a learned switching policy.

## Reproducibility

Fresh fitting: 240,000 updates, {seconds:.3f}s summed fitting including the pilot.
Native arm64 CPU, four compute threads, one inter-op thread, zero loader workers.
All 24 forecasts replay exactly; weighted draws are independently regenerated.
All training group masses verify, twelve paired-arm streams match, six OOF
archives recompute. Completed resume adds zero updates and preserves
{check['immutable_artifacts']} artifacts. No raw data, feature cache or checkpoints in Git.

Fresh: weighted fitting, analysis and artifact verification. Cached_verified:
episode identities, image features and uniform controls. Not_run: independent
confirmation, main/outer and external forecasts, new intervention policy.
No new deployment, Stage5C execution or SMC. Not submission-ready.

See [fixed design](../source_episode_sampler_decision.md), [analysis](analysis.json),
[verification](verification.json), [failure analysis](failure_analysis.md)
and [evidence gates](gates.md).
"""
    (public/'conclusions.md').write_text(text)
    commands = '\n'.join(f'.venv-pytorch/bin/python scripts/{name}_m3w_source_episode_sampler.py --registration {args.registration}'
        for name in ('run','analyze','verify','report'))
    commands = commands.replace('\n.venv-pytorch/bin/python scripts/analyze',
        f'\n.venv-pytorch/bin/python scripts/run_m3w_source_episode_sampler.py --registration {args.registration} --replay\n.venv-pytorch/bin/python scripts/analyze')
    (public/'reproducibility.md').write_text(f"""# Reproduction

Registration SHA256: `{file_digest(args.registration)}`.
Private hash-matching source features, episode audit and parent fits are required.
Missing assets do not constitute a completed reproduction.

```bash
{commands}
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_episode_sampler.py tests/test_m3w_source_event_support.py
```

Atomic checkpoints every 200 steps retain optimizer, Torch RNG, sampler RNG and
draw counts. Restart the same command to resume; completed receipts verify hashes
and skip fitting. The 100-update named pilot is included in the fixed budget.
PID/step/loss heartbeat and the private training log support long-run monitoring.
No final-test checkpoint selection. The full non-hermetic legacy test suite was
not rerun for this scoped intervention. This is local reproduction, not new data.
""")
    cache = private/'plot_cache'; cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache)); os.environ.setdefault('XDG_CACHE_HOME',str(cache))
    import matplotlib
    matplotlib.use('Agg'); matplotlib.rcParams['svg.hashsalt']=file_digest(args.registration)
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(9,4),layout='constrained')
    for i,(arm,x) in enumerate(summary.items()):
        val=x['equal_site_gain_percent']; lo,hi=x['conditional_four_site_ci95']
        ax.errorbar(val,i,xerr=[[val-lo],[hi-val]],fmt='o',capsize=5)
    ax.axvline(0,color='black',lw=.8); ax.set_yticks(range(len(summary)),list(summary)); ax.invert_yaxis()
    ax.set_xlabel('Equal-site ADE gain vs stationary CV (%)'); ax.grid(alpha=.2)
    ax.set_title('Training exposure repair: conditional four-site intervals')
    svg=public/'comparison.svg'; fig.savefig(svg,metadata={'Date':None})
    svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
    fig.savefig(private/'comparison.png',dpi=140); plt.close(fig)
    print(json.dumps(dict(verdict=verdict,fit_seconds=seconds,new_deployment=False)))


if __name__ == '__main__': main()
