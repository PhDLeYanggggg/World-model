"""Aggregate frozen source readout without publishing query-level cache records."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_dimensionless_intervention as run
from src.evaluation.m3w_agent_track_refit import paired_localities


def fmt(v):
    if isinstance(v, dict):
        if v.get('point') is None: return 'undefined (support)'
        ci = v['ci95']
        return f"{v['point']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
    return 'undefined' if v is None else f'{v:.4f}'


def write(name, text):
    path = run.PUBLIC/name
    path.write_text(text.rstrip()+'\n')


def main():
    d = json.loads((run.PUBLIC/'evaluation.json').read_text())
    reg = json.loads((run.PUBLIC/'registration.json').read_text())
    assert d['identity'] == reg
    frozen = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    training = []
    for ref in frozen['heads']:
        assert run.artifact(ROOT/ref['path']) == ref
        r = json.loads((ROOT/ref['path']).read_text())
        training.append(dict(group=r['identity']['group'], task=r['identity']['task'],
            seconds=r['fit']['seconds'], steps=r['fit']['step'], parameters=r['fit']['parameters'],
            unknown_rows_sampled=r['fit']['unknown_rows_sampled'], trace=r['fit']['trace']))
    sites = sorted(sum(reg['rosters'], []))
    seed_summary = []
    for seed in (17,29,43):
        table = [r for r in d['paired_rows'] if f'_seed{seed}' in r['trial'] and
                 r['scope']=='full' and r['policy']=='point' and r['endpoint']=='ADE' and r['subset']=='all']
        seed_summary.append(dict(seed=seed, neural_vs_protected_damping=paired_localities(table, sites, 'gain_vs_legacy_percent')))
    query_summary = []
    for candidate in ('dimensionless', 'damped'):
        q = [r for r in d['query_diagnostics'] if r['candidate']==candidate]
        query_summary.append(dict(candidate=candidate, dependent_query_views=len(q),
            matched=sum(r['matched'] for r in q), matched_nonzero=sum(r['matched_nonzero'] for r in q),
            active_nonadditive=sum(r['nonadditive_supported_edges']>0 for r in q),
            joint_changed_queries=sum(r['joint_changes']>0 for r in q),
            solver_floor_queries=sum(any('failed' in s or 'not_optimal' in s for s in r['solver_reasons']) for r in q),
            mean_proxy={a:float(np.mean([r['arms'][a]['mean_pair_proxy'] for r in q]))
                        for a in run.JOINT_POLICIES}))
    zero = [r for r in d['rows'] if r['scope']=='full' and r['policy']=='point' and r['endpoint']=='ADE'
            and r['subset']=='zero_CV' and r['metric']['rows']>0]
    light = dict(evaluation_sha256=run.digest(run.PUBLIC/'evaluation.json'), result_source=d['result_source'],
        summaries=d['summaries'], neural_vs_protected_damping=d['neural_vs_protected_damping'],
        joint_contrasts=d['joint_contrasts'], gates=d['gates'], seed_summary=seed_summary,
        query_summary=query_summary, zero_reference_views=zero, head_count=len(training),
        total_fit_seconds=sum(r['seconds'] for r in training), updates=sum(r['steps'] for r in training),
        independent_roles_read=False, deployment_changed=False)
    run.immutable_json(run.PUBLIC/'summary_metrics.json', light)
    run.immutable_json(run.PUBLIC/'training.json', dict(heads=training, training_source='fresh_run'))
    def get(candidate, policy, endpoint, subset, scope='full'):
        return next(r['metrics'] for r in d['summaries'] if all(r[k]==v for k,v in
            dict(candidate=candidate, policy=policy, endpoint=endpoint, subset=subset, scope=scope).items()))
    primary = next(r['metrics']['gain_vs_legacy_percent'] for r in d['neural_vs_protected_damping'] if
                   r['scope']=='full' and r['policy']=='point' and r['endpoint']=='ADE' and r['subset']=='all')
    text = ['# Matched Frozen-Predictor Intervention Results', '',
        'Result provenance: 108 fresh Torch cost heads and causal decisions; cached_verified frozen forecasts.',
        'Exploratory opened European source-development only. Obs8/pred12 at raw stride 12.',
        'Detector-derived silver image-local trajectories, not human gold, metric, calibrated seconds,',
        'independent confirmation, physical safety, true3D or foundation evidence. No deployment change.', '',
        '## Full-Bank Pointwise and Unprotected Comparisons', '',
        'Positive values mean lower error. Equal-locality percent gains; 3,000 paired locality-bootstrap draws.',
        'Average seed/producer contexts inside each of 12 localities before resampling; overlapping windows',
        'are not independent samples. CV is the fixed fallback; training-selected causal references are',
        'reported separately, not silently called CV the strongest.', '',
        '| Candidate | Policy | ADE vs CV | Easy ADE vs CV | Hard ADE vs CV | FDE vs CV |',
        '|---|---|---:|---:|---:|---:|']
    for candidate in ('dimensionless','damped'):
        for policy in run.FULL_POLICIES:
            vals = [get(candidate,policy,ep,sub)['gain_vs_CV_percent'] for ep,sub in
                    [('ADE','all'),('ADE','positive_easy'),('ADE','hard'),('FDE','all')]]
            text.append('| '+candidate+' | '+policy+' | '+' | '.join(map(fmt,vals))+' |')
    text += ['', '**Primary: pointwise neural vs equally protected damping ADE:** '+fmt(primary)+'%.', '',
        '## Three Seeds', '', '| Seed | Primary ADE gain, % |', '|---|---:|']
    text += [f"| {r['seed']} | {fmt(r['neural_vs_protected_damping'])} |" for r in seed_summary]
    text += ['', '## Locality Safety', '', '| Locality | Pointwise neural ADE vs CV, % | Easy gain, % |', '|---|---:|---:|']
    all_site=get('dimensionless','point','ADE','all')['gain_vs_CV_percent']['by_site']
    easy_site=get('dimensionless','point','ADE','positive_easy')['gain_vs_CV_percent']['by_site']
    text += [f'| {s} | {fmt(all_site[s])} | {fmt(easy_site[s])} |' for s in sites]
    text += ['', '## Joint Query Subset', '',
        '96 hash-selected recording/frame queries per locality; all indexed query agents retained.',
        'This is not full-bank joint evaluation. Exact-count diagnostics retain solver failures and',
        'zero matches. A zero intervention rate or lower proximity proxy alone is not neural gain.', '',
        '| Candidate | Dependent query views | Nonzero matched | Active nonadditive | Joint changes | Solver floors |',
        '|---|---:|---:|---:|---:|---:|']
    text += [f"| {r['candidate']} | {r['dependent_query_views']} | {r['matched_nonzero']} | {r['active_nonadditive']} | {r['joint_changed_queries']} | {r['solver_floor_queries']} |" for r in query_summary]
    text += ['', '| Candidate | Joint vs independent ADE, all query subset | Joint vs unary ADE |', '|---|---:|---:|']
    for c in ('dimensionless','damped'):
        vals=[next(r['metrics']['gain_vs_legacy_percent'] for r in d['joint_contrasts'] if r['candidate']==c and
              r['control']==control and r['endpoint']=='ADE' and r['subset']=='all' and r['population']=='all_queries')
              for control in ('half_independent','half_unary')]
        text.append('| '+c+' | '+' | '.join(map(fmt,vals))+' |')
    text += ['', 'The preceding table retains solver-floor outcomes. It is not necessarily matched coverage.',
        'Restricting to queries with verified equal nonfailed counts gives:', '',
        '| Candidate | Matched joint vs independent ADE | Matched joint vs unary ADE |', '|---|---:|---:|']
    for c in ('dimensionless','damped'):
        vals=[next(r['metrics']['gain_vs_legacy_percent'] for r in d['joint_contrasts'] if r['candidate']==c and
              r['control']==control and r['endpoint']=='ADE' and r['subset']=='all' and r['population']=='matched')
              for control in ('half_independent','half_unary')]
        text.append('| '+c+' | '+' | '.join(map(fmt,vals))+' |')
    text += ['', 'Matched zero-action queries remain present. The matched-nonadditive population is',
        'separate in summary_metrics.json; missing locality support is not silently dropped.']
    text += ['', '## Zero-Reference Costs', '',
        'Reference-exact rows have undefined percentage gains. These are repeated producer/seed views,',
        'not additional independent queries. The complete supported absolute-cost table follows.', '',
        '| Candidate | Trial/controller | Site | Rows | Harmed rows | Mean selected ADE |', '|---|---|---|---:|---:|---:|']
    text += [f"| {r['candidate']} | {r['trial']}/{r['controller']} | {r['site']} | {r['metric']['rows']} | {r['metric']['absolute_zero_harmed']} | {fmt(r['metric']['new_mean'])} |" for r in zero]
    text += ['', '## Gates', '']+[f"- {k}: {str(v).lower()}" for k,v in d['gates'].items()]
    text += ['', f"Fresh training: {len(training)} heads, {sum(r['steps'] for r in training):,} updates; cumulative fit time {light['total_fit_seconds']:.2f}s.",
             'Training loss logs are optimization evidence, not a validation or generalization score.',
             'Detailed query-level caches remain local; summary_metrics.json contains aggregate evidence.']
    write('results.md', '\n'.join(text))
    write('model_data_card.md', '''# Model and Data Card

Frozen dimensionless forecast bank, plus separate three-moment envelope heads
for it and fixed damping. Heads have 22,914 parameters each, width 64, GELU, and 2,000 updates.
Source-only controller preprocessing; twelve opened source localities, three
producer groups, six ordered role assignments, seeds 17/29/43. No readout fitting.
All 108 heads are fresh training; forecasters are hash-verified cached results.

The 355 causal features contain target/neighbor histories, full predicted
candidate/CV rollouts and observed scale. No future endpoint, mask, target latent,
central velocity, test endpoint goal or test-fitted statistics. Label-derived
hard/easy events are evaluation/training labels, not inference features.

Fixed 2% predicted all/easy moment screens, no readout threshold tuning. Their
predicted ratios are not realized guarantees. Joint controls optimize an image
proximity proxy at a matched count; it is not a physical collision measure.
Nonadditive geometry support and numerical solver failures must accompany claims.

318,969 source-training target histories; recordings and localities are not new
confirmation datasets. Silver detector tracks, image-local coordinates,
obs8/pred12 raw stride 12. Independent roles remain closed. No deployment,
Stage5C, SMC, metric/seconds, true3D, human-gold or foundation claim.
''')
    write('reproducibility.md', '''# Reproduction

Use the existing native arm64 .venv-pytorch; CPU 4/interop 1, workers 0. No resource
probing, multiprocessing or NumPy replacement for training. Private data and
checkpoint caches are not distributed. Registration binds source and parent
seals; original fitting states, forecasts and split lineage must be available.

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_intervention.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_intervention.py --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_intervention.py --phase decide --resume
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_intervention.py --phase replay_heads --resume
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_intervention.py --phase replay_decisions --resume
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_intervention.py --phase evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_intervention.py --phase verify_eval
.venv-pytorch/bin/python scripts/report_m3w_european_dimensionless_intervention.py
```

Register before fresh training; commit decision_freeze.json before readout.
On completed artifacts use replay phases; do not delete checkpoints to restart.
The first pilot resumes inside its 2,000-update budget. Checkpoint every 200 updates,
heartbeat and PIDs in the private event log. At least 10 GiB free before each fit.
Resume skips hash-verified completed heads and completed decision groups.
Shared simulation jobs/environment must never be modified. Local fit is measured
to fit the machine. CREATE queue readout is not M3W remote training evidence.

Report complete source replay separately from independent confirmation, raw-data
rebuild, formal calibration, deployment and full historical test-suite coverage.
None of the latter follow from a successful cache replay.
''')
    print(json.dumps(dict(primary=primary,gates=d['gates'],query_summary=query_summary)))


if __name__ == '__main__': main()
