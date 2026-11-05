"""Completed ego-frame comparison; preserve all failures and original rules."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_ego_frame as run


def pct(v): return 'NA' if v is None else f'{100*v:+.6f}%'


def main():
    a=run.source.read(run.PUBLIC/'analysis.json'); v=run.source.read(run.PUBLIC/'verification.json')
    run.rel.base.exact(v['analysis_sha256'],run.source.sha(run.PUBLIC/'analysis.json'))
    table=['| Context | Primary MSE reduction | Primary cost reduction | Original harmful cost | New harmful cost | Screen |',
        '|---|---:|---:|---:|---:|---|']
    for i,c in enumerate(a['contexts']):
        s=c['summary']; old,new=s['source_macro']['original120'],s['source_macro']['history121']
        table.append(f"| {i} | {pct(s['primary_MSE_reduction'])} | {pct(s['primary_cost_reduction'])} | "
            f"{old['false_safe_cost']:.9g} | {new['false_safe_cost']:.9g} | "
            f"{'pass' if s['passes_information_screen'] else 'fail'} |")
    text='''# Does An Ego-History Frame Repair Cross-Source Risk Learning?

Sixteen new forest models are freshly trained against eight hash-verified
original-frame controls. All24models reproduce by full refit, and all target
RMS values match exactly. Forecasts and original signed targets are unchanged.
This is a source-held fitting experiment, not independent confirmation or
new trajectory dynamics. No safety rule, threshold or data role changed.

## Fixed Primary

history121 uses ego-history-scaled coordinates plus the causal log scale ratio.
original120 uses the existing neighbor-inclusive scale. Both use the same
64-tree/depth8/minleaf32 learner. The121st scalar is an explicit difference;
history120 is a secondary ablation, not a posthoc replacement primary.
Positive reductions mean better. Costs are the unchanged signed-target
decision diagnostic, not ADE improvement or full deployment risk ratios.

'''+ '\n'.join(table)+f'''

Internal follow-up screen: **{a['passing_contexts']}/3**. The fixed rule requires
>=1% original decision-cost reduction and nonworse false-safe cost in>=2contexts.
It does not certify the original complete safety contract. The complete
source/arm breakdown, including secondary and negative results, follows.

## Source Breakdown

| Context/source | Arm | Known rows | Normalized MSE | Joint cost | Harmful admissions | All admissions |
|---|---|---:|---:|---:|---:|---:|
'''
    for i,c in enumerate(a['contexts']):
        for f in c['folds']:
            for arm,r in f['arms'].items():
                m=r['metrics']
                text+=f"| {i}/{f['source']} | {arm} | {r['rows']} | {m['normalized_MSE']:.7g} | "
                text+=f"{m['total_joint_cost']:.7g} | {m['false_safe_count']} | {m['accepted_known_proposals']} |\n"
    text+='''
## Causal Scale Audit

Ratios are ego-history scale / existing context scale on all proposed causal
rows, before label-availability filtering. A ratio below1 indicates the outer
neighbor context controls the old scale. This diagnoses representation, not
pixel-to-metre or temporal calibration.

| Context/source | Proposed rows | Ratio min/Q25/median/Q75/max | Fraction below1 |
|---|---:|---|---:|
'''
    for i,c in enumerate(a['contexts']):
        for r in c['scale_audit']:
            values=', '.join(f'{n:.6g}' for n in r['ego_to_context_scale_quantiles'])
            text+=f"| {i}/{r['source']} | {r['proposed_rows']} | {values} | {100*r['history_scale_smaller_fraction']:.3f}% |\n"
    text+='''
## Runtime And Reproduction

Use native arm64 `.venv-pytorch/bin/python`, CPU4/inter-op1/workers0.
The registered forest uses n_jobs=4 despite a legacy startup banner field.

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_ego_frame.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_ego_frame.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_ego_frame.py run
.venv-pytorch/bin/python scripts/run_m3w_moving_ego_frame.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_ego_frame.py
.venv-pytorch/bin/python scripts/run_m3w_moving_ego_frame.py replay
```

Completed phases are immutable. An interrupted run resumes with `run --resume`
after verifying the old process ended. Per-cell atomic checkpoints are reused;
an unfinished forest is refitted, not resumed tree-by-tree. The private history
assets and frozen models are required; reports alone are not a standalone
dataset or complete anonymous reproduction package. No new CREATE job is needed.

| Phase | PID | Seconds | Peak RSS bytes |
|---|---:|---:|---:|
'''
    runtime={}
    for phase in ('pilot','run','verify'):
        r=run.source.read(run.PUBLIC/(phase+'_execution.json')); runtime[phase]=r
        text+=f"| {phase} | {r['pid']} | {r['seconds']:.3f} | {r['peak_RSS_bytes']} |\n"
    text+='''
## Claim Boundary

Three dependent contexts from one producer seed17,48,061known and1,274unknown
proposals. Held-source scores are still exposed-development evidence. No new
C/H, reserved readout or bootstrap claim from dependent rows. Unknown futures
are not safe outcomes. Native observation8/prediction12 image coordinates are
not metric or seconds-level results. No3D/foundation/JEPA/dynamics claim.

Original2% all/easy harm, easy-net, support and empty-control NA stay unchanged.
There is no identity-fallback exemption. Original24-group primary remains NA,
not passed. No deployment promotion, Stage5C, SMC or submission-ready claim.
Exact repeats show engineering reproducibility, not independent confirmation.

See [failure analysis](failure_analysis.md), [protocol](protocol.md), and
[complete metrics](analysis.json). The compiled main paper is unchanged.
'''
    run.PUBLIC.joinpath('README.md').write_text(text)
    summary=dict(analysis_sha256=v['analysis_sha256'],fresh_fits=16,cached_controls=8,all24_refits_exact=True,
        passing_contexts=a['passing_contexts'],neural_expansion_supported=a['neural_expansion_supported'],
        summaries=[c['summary'] for c in a['contexts']],execution=runtime,
        original_rules_unchanged=True,new_V_or_H=False,identity_certification=False,independent_confirmation=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
    run.PUBLIC.joinpath('result_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
