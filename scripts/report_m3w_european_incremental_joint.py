"""Aggregate every frozen joint-control view, retaining undefined and adverse results."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_incremental_joint as run
from scripts.report_m3w_european_floor_relative import spread, dump
from scripts.report_m3w_european_protected_motion import compact_metric

SUBSETS = ('all', 'easy', 'hard', 'complete', 'matched', 'nonadditive')


def gates(s):
    def supported(v): return v['defined'] == 36 and v['positive_CI'] > 0 and v['negative_CI'] == 0
    comparisons = s['joint_vs_controls']; safety = s['policies']['half_joint']['safety']
    return dict(parent_assets_verified=True, frozen_before_readout=True, original_actions_preserved=True,
        same_forecasts=True, actual_counts_and_caps_verified=True,
        all_solvers_certified=s['queries']['unmatched'] == 0,
        nontrivial_pair_support=s['queries']['nonadditive'] > 0,
        joint_changes_unary=s['queries']['joint_unary_changed_queries'] > 0,
        joint_beats_independent=supported(comparisons['half_independent']['all']),
        joint_beats_unary=supported(comparisons['half_unary']['all']),
        hard_comparisons_not_negative=all(comparisons[k]['hard']['defined'] == 36
            and comparisons[k]['hard']['negative_CI'] == 0 for k in ('half_independent', 'half_unary')),
        easy_preservation=safety['defined_views'] == 36 and safety['worst_positive_degradation_percent'] <= 2,
        independent_calibration=False, independent_confirmation=False, deployment_promoted=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)


def query_summary(queries):
    fields = dict(views=len(queries), nonzero=sum(q['count'] > 0 for q in queries),
        nonadditive=sum(q['nonadditive_supported_edges_at_count'] > 0 for q in queries),
        unmatched=sum(not q['matched'] for q in queries),
        joint_unary_changed_queries=sum(q['joint_unary_changed_agents'] > 0 for q in queries),
        joint_independent_changed_queries=sum(q['joint_independent_changed_agents'] > 0 for q in queries),
        joint_unary_changed_agents=sum(q['joint_unary_changed_agents'] for q in queries),
        joint_independent_changed_agents=sum(q['joint_independent_changed_agents'] for q in queries),
        unique_queries=len({(q['site'], q['recording'], q['frame']) for q in queries}),
        unique_nonadditive_queries=len({(q['site'], q['recording'], q['frame']) for q in queries
            if q['nonadditive_supported_edges_at_count'] > 0}),
        unique_joint_unary_changed_queries=len({(q['site'], q['recording'], q['frame']) for q in queries
            if q['joint_unary_changed_agents'] > 0}))
    fields['predicted_proxy_locality_means'] = {}
    for name in queries[0]['arms']:
        fields['predicted_proxy_locality_means'][name] = {
            site: float(np.mean([q['arms'][name]['mean_pair_proxy'] for q in queries if q['site'] == site]))
            for site in sorted({q['site'] for q in queries})}
    fields['solver_failure_reasons'] = [dict(site=q['site'], recording=q['recording'], frame=q['frame'],
        hash_reason=q['hash_reason'], reasons={k: r['reason'] for k, r in q['base_comparison']['controls'].items()})
        for q in queries if not q['matched']]
    return fields


def summarize(rows, queries):
    policies = {}
    for p in rows[0]['views']:
        vv = [r['views'][p] for r in rows]
        easy = [v['easy_vs_CV']['worst_scene_gain_percent'] for v in vv]
        known = [v for v in easy if v is not None]
        policies[p] = {field: {s: spread([v[field][s] for v in vv]) for s in SUBSETS}
            for field in ('ADE_vs_incumbent', 'ADE_vs_full_add')}
        policies[p].update(FDE_vs_incumbent=spread([v['FDE_vs_incumbent'] for v in vv]),
            safety=dict(defined_views=len(known), worst_positive_degradation_percent=max(0., -min(known)) if known else None,
                zero_CV_harm_views=sum(v['zero_CV']['harmed_rows'] > 0 for v in vv)),
            switch_rate_range=[min(v['switch_rate'] for v in vv), max(v['switch_rate'] for v in vv)],
            unknown_switches_range=[min(v['unknown_ADE_switches'] for v in vv), max(v['unknown_ADE_switches'] for v in vv)])
    return dict(result_source='fresh_run_joint_control_inference_readout_cached_verified_models', new_training=False,
        groups=len(rows), views=len(rows)*len(policies), policies=policies,
        joint_vs_controls={p: {s: spread([r['contrasts'][p][s] for r in rows]) for s in SUBSETS}
            for p in ('half_independent', 'half_hash', 'half_unary')}, queries=query_summary(queries),
        bootstrap_resamples=3000, independent_localities=12, localities_per_view=4,
        uncertainty='paired_source_bootstrap_conditional_overlapping_development_views',
        reserved_roles_opened=False, deployment_changed=False, calibrated_safety=False)


def fmt(values): return 'undefined' if values is None else f'{values[0]:+.6f}% to {values[1]:+.6f}%'


def main():
    frozen = run.ensure_frozen(); done = json.loads((run.PRIVATE/'evaluation_complete.json').read_text())
    assert done['all_passed'] and done['identity'] == frozen['identity']
    rows, queries, pergroup = [], [], {}; counts = dict(coordinate_arrays=0, metric_reductions=0)
    for ref in done['groups']:
        assert run.artifact(ROOT/ref['path']) == ref
        r = json.loads((ROOT/ref['path']).read_text()); assert r['verified'] and r['identity'] == done['identity']
        assert len({r[k] for k in ('producer', 'controller', 'readout')}) == 3
        rows.append(r)
        for a, b in [('coordinate_arrays', 'coordinate_arrays_verified'), ('metric_reductions', 'metric_reductions_verified')]: counts[a] += r[b]
        qq = json.loads((run.PRIVATE/'decisions'/(r['group']+'_queries.json')).read_text())
        for q in qq:
            for p in ('half_independent', 'half_hash', 'half_unary', 'half_joint'):
                assert q['arms'][p]['predicted_constraints_satisfied']
        queries.extend(qq); pergroup[r['group']] = query_summary(qq)
        public = {k: v for k, v in r.items() if k != 'identity'}
        for v in public['views'].values():
            for f in ('ADE_vs_incumbent', 'ADE_vs_full_add'):
                v[f] = {s: compact_metric(m, scenes=True) for s, m in v[f].items()}
            for f in ('FDE_vs_incumbent', 'easy_vs_CV'): v[f] = compact_metric(v[f], scenes=True)
        # Keep full private metrics for the summary; the public form is compact.
        rows[-1] = json.loads((ROOT/ref['path']).read_text())
        public['contrasts'] = {p: {s: compact_metric(m, scenes=True) for s, m in v.items()} for p, v in public['contrasts'].items()}
        public['query_summary'] = pergroup[r['group']]
        dump(run.PUBLIC/'groups'/(r['group']+'.json'), public)
    assert len(rows) == 36 and len(queries) == 36*384
    s = summarize(rows, queries); dump(run.PUBLIC/'summary_metrics.json', s); gg = gates(s); dump(run.PUBLIC/'gates.json', gg)
    events = [json.loads(l) for l in (run.PRIVATE/'events.jsonl').read_text().splitlines()]
    frozen_at = min(e['utc'] for e in events if e['state'] == 'phase_complete' and e.get('phase') == 'decide')
    readout_at = min(e['utc'] for e in events if e['state'] == 'phase_started' and e.get('phase') == 'evaluate')
    assert frozen_at < readout_at
    durations = {p: float((np.datetime64(max(e['utc'].removesuffix('Z') for e in events if e['state'] == 'phase_complete' and e.get('phase') == p))-
        np.datetime64(min(e['utc'].removesuffix('Z') for e in events if e['state'] == 'phase_started' and e.get('phase') == p)))/np.timedelta64(1, 's')) for p in ('decide', 'evaluate')}
    dump(run.PUBLIC/'compute_receipt.json', dict(counts=counts, seconds=durations, frozen_at=frozen_at, readout_at=readout_at,
        decisions=run.artifact(run.PRIVATE/'decisions_complete.json'), evaluation=run.artifact(run.PRIVATE/'evaluation_complete.json'),
        new_training=False, runtime='native_arm64_CPU4_interop1_workers0', independent_queries=s['queries']['unique_queries'],
        create_queue_receipt_sha256=run.digest(run.PRIVATE/'create_queue.json'), remote_jobs_submitted=0))
    lines = ['# Incremental Joint Control: All Registered Results', '',
        'Frozen forecasters and cost heads; no new training. Restricted past-hash query population, not full parent rows.',
        '36 dependent role/seed/event views; four source localities per view,12 unique localities. No independent confirmation.', '',
        '| Policy | All ADE gain vs incumbent | Positive / negative CI | Hard ADE gain | Worst easy degradation vs CV |', '|---|---:|---:|---:|---:|']
    for p, v in s['policies'].items():
        m = v['ADE_vs_incumbent']['all']; safe = v['safety']['worst_positive_degradation_percent']
        lines.append(f"|{p}|{fmt(m['range'])}|{m['positive_CI']} / {m['negative_CI']}|{fmt(v['ADE_vs_incumbent']['hard']['range'])}|{safe}|")
    lines += ['', '| Joint vs matched control | All ADE gain | Positive / negative CI | Hard gain | Hard positive / negative CI |', '|---|---:|---:|---:|---:|']
    for p, v in s['joint_vs_controls'].items():
        m, h = v['all'], v['hard']; lines.append(f"|{p}|{fmt(m['range'])}|{m['positive_CI']} / {m['negative_CI']}|{fmt(h['range'])}|{h['positive_CI']} / {h['negative_CI']}|")
    lines += ['', '## Identification And Safety Limits', '',
        f"There are{s['queries']['views']} repeated query/views, but only{s['queries']['unique_queries']} unique current queries.",
        f"{s['queries']['nonadditive']} query/views permit non-additive interactions; joint and unary differ in{s['queries']['joint_unary_changed_queries']} query/views ({s['queries']['unique_joint_unary_changed_queries']} unique queries).",
        f"{s['queries']['unmatched']} query/views failed a solver certificate; all matched variants then retain the independent reference, never an unmatched-rate advantage.",
        'All future-unknown indexed rows are retained for inference; label-unknown costs remain undefined. Partial ADE and complete-window ADE are separate; FDE requires the actual endpoint.',
        'Proxy reduction follows an optimized proximity objective and is not evidence of collision avoidance. Predicted harm budgets are not calibrated risk bounds.',
        'Hash priorities are outcome-independent, but feasibility still uses the same predicted harm cap. Uniform control is not rate matched.',
        'Unknown/non-indexed agents are omitted from the pair graph, not assumed collision-free. Scene selection only coordinates indexed eight-history agents.',
        'Source-bootstrap intervals use3000 resamples and four localities; no IID-window or multiplicity-corrected claims. All adverse branches remain in groups/*.json.',
        'Image-pixel raw-frame obs8/pred12, released detector tracks; not metric/seconds/human-gold/true3D/foundation/physical-safety evidence. No Stage5C, SMC or deployment.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    (run.PUBLIC/'gates.md').write_text('# Evidence Gates\n\n'+''.join(f'- {k}: {v}\n' for k, v in gg.items())+
        '\nExecution flags are prohibitions, not passed research gates. No promotion from opened-source development.\n')
    (run.PUBLIC/'reproducibility.md').write_text('''# Reproduction

Registration commit a0d63f52 preceded new decisions. Parent199 public artifacts,
source bindings,144 trained neural heads,72 ridge fits and parent frozen decisions
are hash-verified. No new model fitting. No reserved roles opened.

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_incremental_joint.py --phase decide --resume
.venv-pytorch/bin/python scripts/run_m3w_european_incremental_joint.py --phase evaluate --resume
.venv-pytorch/bin/python scripts/report_m3w_european_incremental_joint.py
```

Use the reporter on completed artifacts; resume retains immutable group receipts.
Only one process can own the run lock. Heartbeat/events and per-group decisions
are private and resumable. Native arm64 CPU4/inter-op1, no loader multiprocessing.
The authorized CREATE queue observation was read-only; this small inference task
required no remote job or upload. No remote artifact directory was assumed.

Public files are aggregate reports/config/code, not raw tracks, forecasts,
per-agent outcomes or checkpoints. Reproduction needs the verified local asset
chain. Full legacy suite is not_run; scoped checks are recorded separately.
''')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    controls = ('half_independent', 'half_hash', 'half_unary')
    for ax, subset in zip(axes, ('all', 'hard')):
        for i, p in enumerate(controls):
            for fold, color in enumerate(('#087e8b', '#c74b36', '#525e70')):
                values = [r['contrasts'][p][subset]['equal_scene_gain_percent'] for r in rows if r['producer'] == fold]
                ax.scatter(np.full(len(values), i+(fold-1)*.15), values, s=16, alpha=.65, color=color,
                           label=f'Producer fold{fold}' if i == 0 else None)
        ax.axhline(0, color='black', linewidth=.8); ax.set_xticks(range(3), ['Independent', 'Hash priority', 'Unary geometry'])
        ax.set_ylabel(f'Joint {subset} ADE gain over control (%)'); ax.legend()
    fig.suptitle('Same predictions, counts and predicted-risk caps:36 dependent development views')
    fig.savefig(run.PUBLIC/'joint_contrasts.svg'); fig.savefig(run.PRIVATE/'joint_contrasts.png', dpi=160); plt.close(fig)
    print(json.dumps(dict(groups=len(rows), counts=counts, gates=gg, queries=s['queries']), allow_nan=False))


if __name__ == '__main__': main()
