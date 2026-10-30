"""Separate a common fixed-set training probe from a moving-policy loss."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_current_policy_critic as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import pack, verify

np = run.np


def comparable_curve(trace, role):
    if role not in ('fixed', 'current'): raise ValueError('Explicit fixed/current supervision required')
    steps = [r['step'] for r in trace]
    if not steps or steps != sorted(set(steps)): raise ValueError('Unique ordered trace steps required')
    key = 'frozen_selected_query_score_original_units' if role == 'current' else 'selected_query_score_original_units'
    if any(key not in r['monitor'] for r in trace): raise ValueError('Comparable fixed-original-set probe required')
    return dict(steps=steps, frozen_probe=[r['monitor'][key] for r in trace],
        pointwise=[r['monitor']['pointwise_risk'] for r in trace],
        moving_query_term=[r['monitor']['selected_query_score_original_units'] for r in trace] if role == 'current' else None,
        complete_selected_queries=[r['monitor']['complete_selected_queries'] for r in trace] if role == 'current' else None,
        selected_rows=[r['monitor']['current_selected_rows'] for r in trace] if role == 'current' else None)


def training():
    p = run.PUBLIC; reg = json.loads(run.REG.read_text())
    for rel, digest in reg['bindings'].items():
        if run.sha(ROOT/rel) != digest: raise ValueError('Registered current-query design changed')
    freeze = json.loads((p/'training_freeze.json').read_text()); fits = []; pairs = []
    for ref in freeze['fits']:
        if run.sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('Frozen current-query fit receipt changed')
        row = json.loads((ROOT/ref['path']).read_text()); states = {}
        for role, key in (('fixed', 'cached_fixed_control'), ('current', 'checkpoint')):
            cp = row[key]
            if run.sha(ROOT/cp['path']) != cp['sha256']: raise ValueError('Paired critic checkpoint changed')
            states[role] = run.parent.api.read_checkpoint(ROOT/cp['path'])
            if states[role]['step'] != 2000 or states[role]['parameters'] != 30726:
                raise ValueError('Complete original-budget paired fits required')
        run.parent.api.assert_matched(states['fixed'], states['current'])
        if (states['fixed']['identity']['query_unit_normalizer'] != states['current']['identity']['query_unit_normalizer']
                or row['query_normalizer'] != states['current']['identity']['query_unit_normalizer']):
            raise ValueError('Only selected supervision may change,not query normalization')
        pairs.append(dict(context=row['context'], seed=row['seed'], arm=row['arm'],
            fixed=states['fixed']['trace'], current=states['current']['trace'],
            unknown_rows_drawn=states['current']['unknown_rows_drawn'], unknown_rows_with_loss=states['current']['unknown_rows_with_loss']))
        fits.append(row)
    expected = {(c, seed, arm) for c in reg['contexts'] for seed in (17, 29, 43) for arm in run.parent.api.ARMS}
    if len(fits) != 36 or {(r['context'], r['seed'], r['arm']) for r in fits} != expected:
        raise ValueError('All36 fixed matched fitting contexts required')
    for arm in run.parent.api.ARMS:
        if not json.loads((p/('first_full_refit_'+arm+'.json')).read_text())['scientific_state_exact']:
            raise ValueError('Both separate full2,000-step refits required')
    return freeze, fits, pairs


def training_stats(pairs):
    arms = {}
    for arm in run.parent.api.ARMS:
        subset = [r for r in pairs if r['arm'] == arm]; roles = {}
        for role in ('fixed', 'current'):
            curves = [comparable_curve(r[role], role) for r in subset]
            if any(c['steps'] != curves[0]['steps'] for c in curves): raise ValueError('Same complete trace schedule required')
            snapshots = [t for r in subset for t in r[role] if 'pointwise_gradient_norm' in t]
            ratios = [t['query_over_pointwise_gradient'] for t in snapshots if t['query_over_pointwise_gradient'] is not None]
            roles[role] = dict(heads=len(subset),
                frozen_probe_decreased_heads=sum(c['frozen_probe'][-1] < c['frozen_probe'][0] for c in curves),
                median_initial_frozen_probe=float(np.median([c['frozen_probe'][0] for c in curves])),
                median_final_frozen_probe=float(np.median([c['frozen_probe'][-1] for c in curves])),
                median_final_pointwise=float(np.median([c['pointwise'][-1] for c in curves])),
                gradient_snapshots=len(snapshots), zero_query_gradient_snapshots=sum(t['selected_query_gradient_norm'] == 0 for t in snapshots),
                query_pointwise_gradient_ratio_quantiles=np.quantile(ratios, [0, .25, .5, .75, 1]).tolist() if ratios else None,
                median_final_moving_masked_query_term=float(np.median([c['moving_query_term'][-1] for c in curves])) if role == 'current' else None,
                median_initial_monitor_selected_rows=float(np.median([c['selected_rows'][0] for c in curves])) if role == 'current' else None,
                median_final_monitor_selected_rows=float(np.median([c['selected_rows'][-1] for c in curves])) if role == 'current' else None,
                median_final_monitor_complete_selected_queries=float(np.median([c['complete_selected_queries'][-1] for c in curves])) if role == 'current' else None)
        arms[arm] = dict(roles=roles, unknown_rows_drawn=sum(r['unknown_rows_drawn'] for r in subset),
            unknown_rows_with_loss=sum(r['unknown_rows_with_loss'] for r in subset))
    return dict(arms=arms, fresh_primary_updates=72000, exact_refit_updates=4000, cached_verified_fixed_heads=36,
        original_set_probe_comparable=True, moving_loss_not_common_population_error=True,
        fitting_monitor_not_validation=True, new_trajectory_forecaster_fits=0, independent_confirmation=False, deployment_changed=False)


def training_text(freeze, stats):
    lines = ['# Current-query supervision: real training telemetry', '', '## Material Passport', '',
        'fresh_run:36 native Torch risk fits,72,000 primary updates,4,000 separate full-refit updates.',
        'cached_verified:36 fixed-query critic controls,unchanged source inputs,forecasts,initial states and draws.',
        f"Training PID{freeze['pid']}: {freeze['seconds']:.3f}s,peak RSS{freeze['peak_RSS_bytes']}B.",
        'Only query supervision changes to current decoded choices,detached from gradients. Context remains original.',
        'Both full first-head2,000-step refits agree exactly. All unknown draws have zero supervised loss.', '',
        '## Comparable fixed-original-set probe', '',
        'These16 fixed fitting-monitor queries per head are not validation/test or full-dataset means.',
        'The ORIGINAL selected set and original error units are common in this probe. New moving-set',
        'loss values are not substituted for that probe or called the same-population error.', '',
        '|Arm|Training target|Initial common probe|Final common probe|Heads decreased|Final point loss|Query/point gradient median|',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm, pair in stats['arms'].items():
        for role, r in pair['roles'].items():
            q = r['query_pointwise_gradient_ratio_quantiles']
            lines.append(f"|{arm}|{role}|{r['median_initial_frozen_probe']:.9f}|{r['median_final_frozen_probe']:.9f}|"
                f"{r['frozen_probe_decreased_heads']}/{r['heads']}|{r['median_final_pointwise']:.9f}|{q[2] if q else None}|")
    lines += ['', '## Moving target and supervision coverage', '',
        '|Arm|Final moving masked query term,original units|Initial selected rows|Final selected rows|Final complete queries(out of16)|',
        '|---|---:|---:|---:|---:|']
    for arm, pair in stats['arms'].items():
        r = pair['roles']['current']
        lines.append(f"|{arm}|{r['median_final_moving_masked_query_term']:.9f}|{r['median_initial_monitor_selected_rows']}|"
            f"{r['median_final_monitor_selected_rows']}|{r['median_final_monitor_complete_selected_queries']}|")
    lines += ['', 'The moving registered loss averages over the sampled queries and gives no supervised contribution',
        'to empty or selected-unknown sets. Zero contribution is NOT zero risk. Changing loss can reflect',
        'changing choices/support,not only better prediction. Fixed-control moving-set curves were not',
        'logged and are not invented. Full final-policy utility and original recording-risk tests remain decisive.',
        'Gradient ratios are actual same-batch component norms;the supervised populations differ by target.', '',
        '## Scope', '', 'Exposed SOURCE fitting/resubstitution,not independent or whole-pipeline OOF confirmation.',
        'Final step2,000 is fixed;no loss-based checkpoint or threshold selection.',
        'No new trajectory forecaster,physical guarantee,metric/seconds,human-gold,true3D,foundation or deployment claim.',
        'Task:obs8/pred12/rawstride12,detector-silver/local-image coordinates. Stage5C execution/SMC stay off.', '']
    return '\n'.join(lines)


def training_plot(pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for arm, color in zip(run.parent.api.ARMS, ('#666666', '#007C91')):
        subset = [r for r in pairs if r['arm'] == arm]
        for role, style in (('fixed', '--'), ('current', '-')):
            curves = [comparable_curve(r[role], role) for r in subset]; steps = curves[0]['steps']
            for ax, key in ((axes[0, 0], 'frozen_probe'), (axes[0, 1], 'pointwise')):
                ax.plot(steps, np.median([c[key] for c in curves], 0), style, color=color, label=arm+' / '+role)
            if role == 'current':
                axes[1, 0].plot(steps, np.median([c['moving_query_term'] for c in curves], 0), color=color, label=arm)
                axes[1, 1].plot(steps, np.median([c['complete_selected_queries'] for c in curves], 0), color=color, label=arm)
    axes[0, 0].set_title('Common frozen-set score probe'); axes[0, 1].set_title('Same all-action point objective')
    axes[1, 0].set_title('Current-only moving masked query term'); axes[1, 1].set_title('Complete selected-query supervision')
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]): ax.set_ylabel('Median fixed fitting-monitor value')
    axes[1, 1].set_ylabel('Median complete queries / 16'); axes[1, 1].set_ylim(0, 16.5)
    for ax in axes.flat: ax.set_xlabel('Actual update'); ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Fixed-set probe versus moving-policy loss: fitting only,not validation', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96)); folder = run.PUBLIC/'figures'; folder.mkdir(exist_ok=True)
    path = folder/'current_query_training.svg'; fig.savefig(path, metadata={'Date': None}, bbox_inches='tight')
    path.write_text('\n'.join(v.rstrip() for v in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'current_query_training.png', dpi=160, bbox_inches='tight'); plt.close(fig)


def export_training():
    freeze, _, pairs = training(); p = run.PUBLIC; stats = training_stats(pairs)
    run.once(p/'training_trace_metrics.json', dict(training_freeze_sha256=run.sha(p/'training_freeze.json'),
        pairs=pairs, raw_data_or_weights_included=False, original_set_probe_comparable=True))
    run.write(p/'training_diagnostics.json', stats); (p/'training_report.md').write_text(training_text(freeze, stats)); training_plot(pairs)
    print(json.dumps(dict(full_real_fits=36, current_policy_values_read=False, diagnostics=stats)))


def build():
    p = run.PUBLIC; c, s, v = [json.loads((p/n).read_text()) for n in ('complete.json', 'summary.json', 'verification.json')]
    if (not v['all72groups_exact'] or not v['all144_fixed_query_controls_exact']
            or v['summary_sha256'] != run.sha(p/'summary.json') or v['complete_sha256'] != run.sha(p/'complete.json')
            or c['training_freeze_sha256'] != run.sha(p/'training_freeze.json')):
        raise ValueError('Separate complete current-query replay and unchanged training required')
    for rel, digest in json.loads(run.REG.read_text())['bindings'].items():
        if run.sha(ROOT/rel) != digest: raise ValueError('Registered current-query scientific definition changed')
    groups = aggregate_groups(run.NAME); rows = [groups[Path(r['path']).name] for r in c['groups']]
    if run.summarize(rows, json.loads(run.CONFIG.read_text())) != s: raise ValueError('Canonical current-query reduction differs')
    nonoptimal = Counter()
    for row in rows: nonoptimal.update(row['result']['allocation']['nonoptimal_queries'])
    rp = p/'replay_runtime.json'
    return dict(summary=s, complete=c, nonoptimal_queries=dict(nonoptimal),
        replay_runtime=json.loads((rp if rp.exists() else run.PRIVATE/'verification_runtime.json').read_text()))


def fmt(v):
    return f"{v['point']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]" if v['CI95'] else 'not estimable'


def render(value):
    s = value['summary']; lines = ['# Current-query training: complete matched policy readout', '', '## Material Passport', '',
        'fresh_run:36 current-query Torch fits,72,000 updates plus4,000 exact refits,full72-group inference/allocation/scoring.',
        'cached_verified:36 fixed-original-query controls,unchanged features,normalizers,draws,forecasts and original context.',
        'Old control predictions are freshly re-inferred and hash-checked;their immutable choices are rescored,not reoptimized.',
        f"Readout PID{value['complete']['pid']}: {value['complete']['seconds']:.3f}s;full replay PID{value['replay_runtime']['pid']}: {value['replay_runtime']['seconds']:.3f}s.",
        'All72 groups,144 fixed full-policy controls and the canonical summary reproduce exactly.',
        'This is exposed SOURCE fitting/resubstitution,not independent evidence or new trajectory dynamics.', '',
        '## Prespecified query-target effect', '',
        'Conditional95% paired locality bootstrap,3,000 draws over12 exposed localities after context/head-seed averaging.',
        'Primary:selected-set current-query joint minus fixed-original-query joint full completion-lower utility.', '',
        '|Arm,current minus fixed query supervision|Full known increment pp [95% CI]|Full wholly-unknown completion-lower pp [95% CI]|',
        '|---|---:|---:|']
    for arm in ('row', 'set'):
        v = s['current_query_contrasts'][arm]; lines.append(f"|{arm}|{fmt(v['full_known_pp'])}|{fmt(v['full_lower_pp'])}|")
    lines += ['', '## Complete policy utility and unchanged risk', '',
        '|Policy|Known ADE gain over floor [95% CI]|Supported recording-risk views|Known easy violations|Undefined easy|Selected views|',
        '|---|---:|---:|---:|---:|---:|']
    for name in (*run.parent.evaluation.NAMES, 'fixed_query_row_joint', 'fixed_query_set_joint'):
        fixed = name.startswith('fixed_query_'); v = (s['fixed_query_accuracy'] if fixed else s['accuracy'])[name]
        risk = (s['fixed_query_safety'] if fixed else s['safety'])[name]
        lines.append(f"|{name}|{fmt(v)}|{risk['supported_record_views']}/{risk['required_record_views']}|"
            f"{risk['known_easy_risk_violations']}|{risk['undefined_easy_record_views']}|{risk['selected']}|")
    lines += ['', 'All original2% recording all/easy positive-harm/reference and easy-degradation requirements remain.',
        'Common-count and generic trimmed comparisons are utility-only,not safety repairs. Empty comparisons are not estimable.',
        'The moving loss can change its selected population;lower masked training loss alone does not establish calibration.',
        '[Actual original-set probe,point loss,moving term and coverage](training_report.md).',
        f"Nonoptimal query views: {value['nonoptimal_queries']}.", '',
        '## Context and cost-fit diagnostics', '',
        '|Current-target contrast|Full known pp [95% CI]|Full completion-lower pp [95% CI]|', '|---|---:|---:|']
    for name in ('set_joint_minus_row_joint', 'set_matched_minus_row_matched'):
        v = s['comparisons'][name]; lines.append(f"|{name}|{fmt(v['full_known_pp'])}|{fmt(v['full_lower_pp'])}|")
    lines += ['', '|Estimator|All-known-legal risk MSE [95% CI]|', '|---|---:|']
    for name, v in s['calibration_risk_MSE'].items(): lines.append(f'|{name}|{fmt(v)}|')
    lines += ['', 'This fitting MSE is neither probability ECE nor calibrated final-policy risk.',
        'The machine-readable summary also retains FDE availability,all subsets and locality/seed contrasts;',
        'per-record aggregates retain tails,unknown completion and predicted-path proxies. They are not relabelled',
        'as complete-label,physical or independent evidence.', '', '## Fitting checks', '']
    lines += [f"- {k}: {'pass' if v else 'fail'}" for k, v in s['fitting_gates'].items()]
    lines += ['', f"{sum(s['fitting_gates'].values())}/{len(s['fitting_gates'])} fitting checks;advancement {s['fitting_advancement_gate']}.", '',
        '## Interpretation limits', '',
        'Moving-policy score regression uses stopped discrete choices;it is not an exact policy gradient,SPO control or calibration theorem.',
        'Known-label ADE is not full-roster FDE;missing required endpoints remain not estimable.',
        'Wholly-unknown completion bounds do not validate missing interior points. No unknown row is dropped or imputed.',
        'The same149,510 target rows,18,766 queries,74 recordings and12 localities are reused. Three critic seeds share',
        'one forecaster seed43. Overlapping windows/context views are not independent observations;intervals are nominal development summaries.',
        'All seven forecasts remain fixed;no new physical-bank reoptimization with these heads or independent/SDD transfer is claimed.',
        'Predicted-path geometry is a proxy,not physical safety or learned pair dynamics.',
        'Task:obs8/pred12/rawstride12,European detector-silver/local-image coordinates,not historical raw-frame t+50.',
        'No metric,seconds,human-gold,true3D,foundation,independent-success,deployment or submission-ready claim.',
        'Independent roles,Stage5C execution and SMC stay closed.', '']
    return '\n'.join(lines)


def complete_report():
    value = build(); p = run.PUBLIC; pack(ROOT, p); verify(ROOT, p)
    run.once(p/'replay_runtime.json', value['replay_runtime']); (p/'readout.md').write_text(render(value))
    run.write(p/'evidence.json', dict(summary_sha256=run.sha(p/'summary.json'), complete_sha256=run.sha(p/'complete.json'),
        verification_sha256=run.sha(p/'verification.json'), report_code_sha256=run.sha(Path(__file__)),
        report_sha256=run.sha(p/'readout.md'), independent_confirmation=False, deployment_changed=False, submission_ready=False))
    print(json.dumps(dict(report_generated=True, gates=value['summary']['fitting_gates'], deployment_changed=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('training', 'complete'))
    a = p.parse_args(); export_training() if a.phase == 'training' else complete_report()
