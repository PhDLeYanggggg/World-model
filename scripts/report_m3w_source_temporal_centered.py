"""Render all verified centering outcomes; no scientific model selection."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True)
    args=p.parse_args();reg=json.loads(args.registration.read_text());out=ROOT/reg['reports']
    read=lambda n:json.loads((out/(n+'.json')).read_text())
    report,analysis,check=[read(n) for n in ('report','analysis','verification')]
    for name,digest in check['artifact_hashes'].items():assert file_digest(ROOT/name)==digest
    summary=analysis['summary'];arms=list(summary)
    table=['| Arm | Source | Equal-site ADE gain (%) | Conditional site CI | Easy harm (annotation px) | Positive held fits |',
           '| --- | --- | ---: | --- | ---: | ---: |']
    interval=lambda x:'['+', '.join(f'{v:+.3f}' for v in x)+']'
    for arm,x in summary.items():
        table.append(f"| {arm} | {x['result_source']} | {x['equal_site_gain_percent']:+.3f} | "
            f"{interval(x['conditional_four_site_ci95'])} | {x['easy_pixel_harm']:.6f} | {x['positive_held_models']}/12 |")
    contrast=['| Fixed contrast | Difference (pp) | Conditional CI |','| --- | ---: | --- |']
    for x in analysis['contrasts']:
        contrast.append(f"| {x['candidate']} minus {x['reference']} | {x['gain_difference_pp']:+.3f} | {interval(x['conditional_four_site_ci95'])} |")
    detail=['| Arm | Site | Train gain (%) | Held gain (%) | Hard gain (%) | Seed held gains (%) |',
            '| --- | --- | ---: | ---: | ---: | --- |']
    native=['| Arm | Native ADE | Native FDE | Nonzero-target gain (%) | ADE p95 | ADE p99 | Binary oracle (%) |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,x in summary.items():
        for v in x['sites']:
            detail.append(f"| {arm} | {v['site']} | {v['training_gain_percent']:+.3f} | {v['held_gain_percent']:+.3f} | "
                f"{v['hard_gain_percent']:+.3f} | "+', '.join(f'{g:+.3f}' for g in v['seed_gains'])+' |')
        native.append(f"| {arm} | {x['native_pixel_ade']:.6f} | {x['native_pixel_fde']:.6f} | {x['nonzero_gain_percent']:+.3f} | "
            f"{x['tail_ade95']:.6f} | {x['tail_ade99']:.6f} | {x['binary_oracle_equal_site_gain_percent']:.6f} |")
    seconds=sum(t['fit']['fit_seconds'] for t in report['trials'])
    verdict=('Both fixed repairs still lose to stationary CV; no deployable forecasting benefit is established.'
             if all(summary[a]['equal_site_gain_percent']<=0 for a in reg['arms']) else
             'At least one repair has a positive development point estimate; independent confirmation and deployment safety remain unproved.')
    (out/'conclusions.md').write_text(f"""# Temporal-Centering Repair: Completed Comparison

## Conclusion

{verdict}

All24new heads completed10,000updates each. There was no held-based arm, seed,
checkpoint, threshold or cohort selection. The36original matched control heads
are cached_verified and rescored, not newly trained.

## Scope

Same15,430stationary-history queries,29recordings,four explored source sites,
seeds17/29/43. Eight observed and twelve future annotation steps at stride12raw
frames. Not the main benchmark, raw-frame t+50 supplement or a Stage37 rerun.
Pixel/local annotations only, no metric or seconds equivalence. Supplied
histories may use later interpolation controls: offline, not sensor-as-of.

Centering removes each observation window's mean frozen appearance. The unit
variant also divides by observed RMS variation with fixed0.001floor. Both keep
geometry,coverage,loss,63,960parameters,initialization,sampler and update budget.
ResNet18 remains frozen; no end-to-end encoder training claim.

## Actual Forecasts And Paired Comparisons

{chr(10).join(table)}

{chr(10).join(contrast)}

Primary is ratio of equal-site mean normalized errors, not mean site percentages.
Seeds average errors, not predictions.2,000shared physical-site bootstrap draws
are conditional on four explored sites and overlapping fitting populations.
No independent confirmation or multiplicity-adjusted guarantee is claimed.
An improved contrast against a failing model is not itself positive forecasting.

## Every Site And Seed

{chr(10).join(detail)}

## Magnitude And Oracle

{chr(10).join(native)}

Native errors are annotation-pixel descriptions, not cross-dataset metric means.
Zero-target CV error is zero; percentage easy degradation is undefined, not a
2%pass. Hard cutoffs use training-complement labels. Future-oracle minima are
label-only diagnostics, not a deployable switch rule.

## Verification And Compute

Fresh training totals240,000updates and{seconds:.3f}s summed fitting, including
the100-update pilot. Native arm64,CPU4/inter-op1/workers0. All24train/held forecasts
replay exactly. All24sample streams match cached controls; all15,430real input
transforms pass shared-offset and batch-composition checks. Six OOF archives
recompute, and completed resume adds zero updates while preserving
{check['immutable_artifacts']}artifacts. Frozen image features and upstream
provenance are checked through the parent manifest. No raw data/checkpoints inGit.

Fresh work: input audit,24heads,analysis,verification. Cached_verified: image
embeddings and36controlheads. Not_run: main/outer forecasts,new policy,independent
confirmation,external eval. No new deployment,Stage5C orSMC. Not submission-ready.
See[registered design](../source_temporal_centered_decision.md),[analysis](analysis.json)
and[verification](verification.json).
""")
    (out/'reproducibility.md').write_text(f"""# Reproduction

Registration committed as4b5dadd9 before fitting. SHA256`{file_digest(args.registration)}`.
Requires hash-matching private parent data,feature store,old controls and official
image encoder weights. Missing assets are prerequisites,not completed reproduction.

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_temporal_centered.py --registration {args.registration}
.venv-pytorch/bin/python scripts/run_m3w_source_temporal_centered.py --registration {args.registration} --replay
.venv-pytorch/bin/python scripts/analyze_m3w_source_temporal_centered.py --registration {args.registration}
.venv-pytorch/bin/python scripts/verify_m3w_source_temporal_centered.py --registration {args.registration}
.venv-pytorch/bin/python scripts/report_m3w_source_temporal_centered.py --registration {args.registration}
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_temporal_information.py tests/test_m3w_source_temporal_centered.py tests/test_m3w_source_pretrained_temporal.py
```

200-update atomic checkpoints contain optimizer,Torch RNG,sampler RNG and draw
counts. Reuse the same entry after interruption. Complete trials hash-verify and
skip training. Heartbeat records PID and step. No outcome-dependent restart.
The full legacy integration/training suite is not rerun for this scoped repair.
Past-only input audit entry:scripts/audit_m3w_source_temporal_information.py.
No alternative input or target columns are inferred from a filename.
""")
    cache=ROOT/reg['output']/'plot_cache';cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache));os.environ.setdefault('XDG_CACHE_HOME',str(cache))
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']=file_digest(args.registration)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    for i,(arm,color) in enumerate(zip(arms,['#666666','#217884','#A64E39','#357B3C','#C39312'])):
        x=summary[arm];point=x['equal_site_gain_percent'];lo,hi=x['conditional_four_site_ci95']
        ax.errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o',capsize=5,color=color)
    ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(arms)),arms);ax.invert_yaxis()
    ax.set_xlabel('Equal-site ADE gain vs stationary CV (%)');ax.grid(alpha=.2)
    ax.set_title('Fixed input repair: conditional four-site intervals, n=2,000')
    svg=out/'comparison.svg';fig.savefig(svg,metadata={'Date':None})
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    fig.savefig(ROOT/reg['output']/'comparison.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(verdict=verdict,fit_seconds=seconds,new_deployment=False)))


if __name__=='__main__':main()
