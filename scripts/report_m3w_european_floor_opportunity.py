"""Publish every frozen-floor diagnostic, without selecting a deployable winner."""
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_floor_opportunity as run
from scripts.report_m3w_european_protected_motion import compact_metric
from src.evaluation.m3w_floor_opportunity import COMPONENTS, SUBSETS


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')
    if path.stat().st_size >= 1024**2:
        raise ValueError('Aggregate exceeds 1MiB public limit: '+str(path))


def distribution(rows):
    valid = [r for r in rows if r['equal_locality'] is not None]
    return dict(views=len(rows), defined=len(valid),
        range=[min(r['equal_locality'] for r in valid), max(r['equal_locality'] for r in valid)] if valid else None,
        positive_points=sum(r['equal_locality'] > 0 for r in valid),
        positive_CI=sum(r['ci95'] is not None and r['ci95'][0] > 0 for r in valid),
        negative_CI=sum(r['ci95'] is not None and r['ci95'][1] < 0 for r in valid))


def dist_metrics(rows):
    return distribution([dict(equal_locality=r['equal_scene_gain_percent'], ci95=r['scene_bootstrap_ci95']) for r in rows])


def verify(mode):
    receipt = json.loads((run.PRIVATE/mode/'complete.json').read_text())
    identity = receipt['identity']
    for p, sha in identity['bindings'].items():
        if run.digest(ROOT/p) != sha:
            raise ValueError('Registered analysis code changed: '+p)
    run.parent.configure(mode)
    parent = run.verified_parent()
    if (identity['prior_analysis_sha256'] != run.digest(run.parent.PUBLIC/'analysis.json')
            or identity['prior_identity_sha256'] != run.digest(run.parent.PRIVATE/'identity.json')
            or identity['prior_decisions_sha256'] != run.digest(run.parent.PRIVATE/'decisions_complete.json')
            or not receipt['all_passed'] or len(receipt['groups']) != 18):
        raise ValueError('Completed fixed predecessor and all groups required')
    rows = {}
    for group, ref in receipt['groups'].items():
        p = ROOT/ref['path']
        if run.digest(p) != ref['sha256']:
            raise ValueError('Group receipt changed')
        r = json.loads(p.read_text())
        if (r['identity'] != identity or r['group'] != group or not r['checks']['all_passed']
                or r['checks']['exact_parent_metrics'] != 11 or len(r['locality_roster']) != 8):
            raise ValueError('Incomplete verified group')
        rows[group] = r
    return rows, receipt


def publish(mode, rows):
    pub = run.PUBLIC/mode; pub.mkdir(parents=True, exist_ok=True)
    views, ledgers, safety = {}, {}, {}
    for group, r in rows.items():
        ledgers[group] = {s: r['ledgers'][s]['summary'] for s in SUBSETS}
        views[group] = {a: {name: ({s: compact_metric(v) for s, v in item.items()}
                                 if name != 'zero_CV' else item)
                           for name, item in m.items()} for a, m in r['metrics'].items()}
        views[group]['rebase_vs_original'] = {s: compact_metric(v) for s, v in r['rebase_vs_original'].items()}
        safety[group] = dict(neural_switch_rate=r['neural_switch_rate'], floor_switch_rate=r['floor_switch_rate'],
            rebase_changes_rows=r['rebase_changes_rows'], indexed_rows=r['indexed_rows'])
        for a in ('original_neural', 'rebased_neural'):
            m = r['metrics'][a]
            easy = m['ADE_vs_CV']['easy']['worst_scene_gain_percent']
            zero = m['zero_CV']
            safety[group][a] = dict(worst_easy_degradation_percent=-easy if easy is not None else None,
                zero_CV=zero, observed_preservation=bool(easy is not None and easy >= -2 and not zero['harmed_rows']))
    for file, value in [('view_metrics.json', views), ('ledger_summary.json', ledgers), ('safety.json', safety)]:
        dump(pub/file, value)
    for fold in range(3):
        with (pub/f'fold{fold}_locality_ledger.csv').open('w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(['group', 'subset', 'locality', 'indexed_rows', 'supported_rows', 'unknown_rows',
                        'floor_error_sum', *[k+'_cost_sum' for k in COMPONENTS], *[k+'_pp' for k in COMPONENTS]])
            for group, r in rows.items():
                if not group.startswith(f'fold{fold}_'):
                    continue
                for subset, ledger in r['ledgers'].items():
                    for site, v in ledger['by_scene'].items():
                        w.writerow([group, subset, site, v['indexed_rows'], v['supported_rows'], v['unknown_rows'],
                                    v['floor_error_sum'], *[v['sums'][k] for k in COMPONENTS],
                                    *[v['contributions_percent'][k] for k in COMPONENTS]])
        with (pub/f'fold{fold}_locality_metrics.csv').open('w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            fields = ('rows', 'model_error', 'reference_error', 'gain_percent', 'model_p95', 'model_p99', 'reference_p95')
            w.writerow(['group', 'action', 'metric', 'subset', 'locality', *fields])
            for group, r in rows.items():
                if not group.startswith(f'fold{fold}_'):
                    continue
                for action, m in r['metrics'].items():
                    for family in ('ADE_vs_floor', 'FDE_vs_floor', 'ADE_vs_CV'):
                        for subset, value in m[family].items():
                            for site, v in value['by_scene'].items():
                                w.writerow([group, action, family, subset, site, *[v.get(k) for k in fields]])
    summary = {}
    for event in ('all', 'easy'):
        rr = [r for group, r in rows.items() if group.endswith('_'+event)]
        assert len(rr) == 9
        summary[event] = dict(ledger={s: {k: distribution([r['ledgers'][s]['summary'][k] for r in rr])
            for k in COMPONENTS} for s in SUBSETS}, actions={a: {family: {s: dist_metrics([
                r['metrics'][a][family][s] for r in rr]) for s in SUBSETS}
                for family in ('ADE_vs_floor', 'FDE_vs_floor')} for a in rr[0]['metrics']},
            safety={a: dict(views=9, observed_preservation=sum(safety[r['group']][a]['observed_preservation'] for r in rr),
                worst_easy_degradation_percent=max(safety[r['group']][a]['worst_easy_degradation_percent'] for r in rr),
                zero_harm_views=sum(safety[r['group']][a]['zero_CV']['harmed_rows'] > 0 for r in rr))
                for a in ('original_neural', 'rebased_neural')})
    return summary


def figures(all_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for col, (mode, rows) in enumerate(all_rows.items()):
        for row, event in enumerate(('all', 'easy')):
            ax = axes[row, col]
            rr = [r for group, r in rows.items() if group.endswith('_'+event)]
            for offset, action, color, label in ((-.12, 'original_neural', '#b33e44', 'Original vs damping floor'),
                    (.12, 'rebased_neural', '#13766c', 'Rebased diagnostic vs floor')):
                x, lo, hi = [], [], []
                for r in rr:
                    v = r['metrics'][action]['ADE_vs_floor']['all']
                    x.append(v['equal_scene_gain_percent']); lo.append(v['scene_bootstrap_ci95'][0]); hi.append(v['scene_bootstrap_ci95'][1])
                ax.errorbar(x, np.arange(len(rr))+offset, xerr=[np.array(x)-lo, np.array(hi)-x],
                            fmt='o', markersize=4, color=color, label=label)
            ax.axvline(0, color='#777777', linewidth=.8)
            ax.set_yticks(np.arange(len(rr)), [r['group'].replace('_'+event, '') for r in rr], fontsize=8)
            ax.set_title(f'{mode} normalizer / {event} event')
            ax.set_xlabel('All-ADE gain over protected damping (%)')
            ax.grid(axis='x', alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc='upper center', bbox_to_anchor=(.5, .92),
               ncol=2, frameon=False)
    fig.suptitle('Same neural decisions; different default action\nOpened-development diagnostic, not calibrated deployment', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .88])
    svg = run.PUBLIC/'floor_rebase.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'floor_rebase.png', dpi=140)
    plt.close(fig)


def main():
    modes = {}; refs = {}; summary = {}
    for mode in ('batch', 'fitting'):
        rows, receipt = verify(mode)
        modes[mode] = rows; refs[mode] = receipt['groups']; summary[mode] = publish(mode, rows)
    checks = {k: sum(r['checks'][k] for rows in modes.values() for r in rows.values())
              for k in ('coordinate_arrays', 'decision_arrays', 'exact_parent_metrics', 'metric_reductions', 'independent_ledgers')}
    assert checks == dict(coordinate_arrays=144, decision_arrays=72, exact_parent_metrics=396,
                          metric_reductions=3312, independent_ledgers=288)
    support = {}
    for rows in modes.values():
        for r in rows.values():
            for site, value in r['support'].items():
                if site in support:
                    assert support[site] == value
                support[site] = value
    assert len(support) == 12
    dump(run.PUBLIC/'summary_metrics.json', dict(result_source='fresh_run_on_cached_verified_forecasts',
        modes=summary, checks=checks, locality_support=support, unique_localities=12,
        primary_population_rows=sum(v['indexed_rows'] for v in support.values()),
        new_training=False, new_predictions=False, independent_confirmation=False,
        reserved_roles_opened=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    lines = ['# Protected-Floor Opportunity: All Registered Groups', '',
        'All values below use all-row ADE and equally weighted localities. Benefit/harm entries are',
        'percentage points with locality sum(floor ADE) as denominator. Conditional intervals appear',
        'in view_metrics.json and ledger_summary.json. No threshold or winner selected.', '',
        '| Mode | Group | Reachable neural benefit | Captured | Selected harm | Missed | CV regression | CV relief | Original gain | Rebased gain |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode, rows in modes.items():
        for group, r in rows.items():
            keys = ('oracle_gain', 'captured_gain', 'selected_harm', 'missed_gain', 'fallback_regression',
                    'fallback_relief', 'original_gain', 'rebased_gain')
            vals = [r['ledgers']['all']['summary'][k]['equal_locality'] for k in keys]
            lines.append('| '+mode+' | '+group+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines += ['', 'Eight annotation-support slices are fully reported; unknown rows are never zero-error',
        'successes. ADE-chosen oracles are unavailable future-label diagnostics; FDE uses their same',
        'chosen forecasts. Rebase keeps the neural mask but is not recalibrated against its new floor.',
        'Three seeds, 3,000 locality resamples per view; dependent unadjusted development intervals.',
        'Image pixels, obs8/pred12 rawstride12. No historical Stage37 recertification, metric/seconds,',
        'human gold, physical safety, true3D or foundation claim. No deployment, Stage5C or SMC.', '']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines))
    figures(modes)
    paths = sorted(p for p in run.PUBLIC.rglob('*') if p.is_file() and p.name not in ('completion_checks.json',))
    for p in paths:
        if p.stat().st_size >= 1024**2:
            raise ValueError('Public output exceeds1MiB: '+str(p))
    test_path = run.PRIVATE/'test_receipt.json'
    tests = json.loads(test_path.read_text())
    test_log = run.PRIVATE/'tests.txt'
    if tests['returncode'] != 0 or tests['log_sha256'] != run.digest(test_log):
        raise ValueError('Passing scoped test log required')
    passed = re.search(r'(\d+) passed', test_log.read_text())
    if passed is None:
        raise ValueError('Missing test completion evidence')
    dump(run.PUBLIC/'completion_checks.json', dict(all_passed=True, checks=checks, group_receipts=refs,
        summary_sha256=run.digest(run.PUBLIC/'summary_metrics.json'),
        aggregate_sha256={str(p.relative_to(run.PUBLIC)): run.digest(p) for p in paths},
        reporter_sha256=run.digest(Path(__file__)), tests_passed=int(passed.group(1)),
        scoped_test_files=tests['files'], test_log_sha256=tests['log_sha256'], full_legacy_suite_run=False,
        new_training=False, deployment_changed=False))
    print(json.dumps(dict(groups=36, checks=checks, summary=str(run.PUBLIC/'summary_metrics.json'))))


if __name__ == '__main__':
    main()
