"""Descriptive fixed-cohort risk and event/magnitude decomposition."""
import argparse
import json
from pathlib import Path
import statistics

from scripts import run_m3w_selection_conditioned_analogue as run
from scripts.verify_m3w_selection_conditioned_analogue import require


def aggregate(rows):
    result = {}
    for arm in run.api.ARMS:
        values = [r['scores'][arm] for r in rows]
        out = {k: sum(v[k] for v in values) for k in
               ('rows', 'known', 'unknown', 'supported', 'unsupported', 'benefit_harm_envelope_exceedances')}
        for target in ('all', 'easy'):
            part = {k: sum(v[target][k] for v in values) for k in
                    ('events', 'observed_harm', 'predicted_harm', 'false_safe_rows')}
            for key in ('full_cohort_MSE', 'AUROC', 'AP'):
                nums = [v[target][key] for v in values]
                part[key] = statistics.fmean(nums) if None not in nums else None
                part['undefined_'+key+'_views'] = nums.count(None)
            out[target] = part
        result[arm] = out
    return result


def decomposition(rows):
    out = {}
    for prefix in ('pooled', 'selected'):
        out[prefix] = {}
        for t in ('all', 'easy'):
            data = [r['decomposition'][prefix][t] for r in rows]
            sums = {k: sum(d[k] for d in data) for k in data[0] if k != 'note'}
            sums['event_prevalence'] = sums['events']/sums['known'] if sums['known'] else None
            sums['mean_predicted_probability'] = sums['predicted_event_mass']/sums['known'] if sums['known'] else None
            sums['brier'] = sums['brier_sum']/sums['known'] if sums['known'] else None
            sums['non_event_SSE_fraction'] = sums['non_event_harm_SSE']/sums['harm_SSE'] if sums['harm_SSE'] else None
            sums['non_event_predicted_harm_fraction'] = sums['predicted_harm_on_non_events']/(sums['predicted_harm_on_events']+sums['predicted_harm_on_non_events']) if sums['predicted_harm_on_events']+sums['predicted_harm_on_non_events'] else None
            n = sums['positive_queries_with_positive_donors']
            sums['positive_severity_MSE_supported_only'] = sums['positive_severity_SSE']/n if n else None
            out[prefix][t] = sums
    return out


def fmt(x):
    return 'undefined' if x is None else f'{x:.9g}'


def describe(rows):
    out = dict(all=aggregate(rows), decomposition=decomposition(rows), posthoc_descriptive_only=True,
        whole_pipeline_OOF=False, held_inference=False, independent_roles_read=False,
        screen_changed=False, deployment_changed=False)
    lines = ['# Frozen-Policy Donor Conditioning: Descriptive Readout', '',
        'All72 views retained; repeated occurrences, not independent samples.',
        'Signed-risk MSE is source-normalized and equal-view averaged. Masses and event diagnostics sum occurrences.', '',
        '| Arm | All risk MSE | Easy risk MSE | Predicted easy harm | Observed easy harm | Unsupported |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, a in out['all'].items():
        lines.append(f"| {arm} | {fmt(a['all']['full_cohort_MSE'])} | {fmt(a['easy']['full_cohort_MSE'])} | {fmt(a['easy']['predicted_harm'])} | {fmt(a['easy']['observed_harm'])} | {a['unsupported']} |")
    lines += ['', '## Event and Magnitude Diagnostics', '',
        'Unknown labels excluded only from observed scoring. Severity MSE conditions on positive query outcomes',
        'and positive donor-event support; it is a diagnostic, not an inference input or comparable unconditionally.', '',
        '| Bank / target | Known | Events | Mean event probability | Observed prevalence | Brier | Non-event harm SSE fraction | Positive queries without positive donor |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for prefix, targets in out['decomposition'].items():
        for t, a in targets.items():
            lines.append(f"| {prefix}/{t} | {a['known']} | {a['events']} | {fmt(a['mean_predicted_probability'])} | {fmt(a['event_prevalence'])} | {fmt(a['brier'])} | {fmt(a['non_event_SSE_fraction'])} | {a['positive_queries_without_positive_donors']} |")
    lines += ['', '## Fixed Slices', '', '| Slice | Views | Original easy MSE | Full-bank nearest | Selected nearest | Unsupported |', '|---|---:|---:|---:|---:|---:|']
    for key in ('context', 'seed', 'source'):
        out[key] = {}
        for value in sorted({r[key] for r in rows}):
            group = [r for r in rows if r[key] == value]; a = aggregate(group)
            out[key][str(value)] = dict(views=len(group), arms=a, decomposition=decomposition(group))
            lines.append(f"| {key}:{value} | {len(group)} | {fmt(a['origin']['easy']['full_cohort_MSE'])} | {fmt(a['pooled_nearest32']['easy']['full_cohort_MSE'])} | {fmt(a['selected_nearest32']['easy']['full_cohort_MSE'])} | {a['selected_nearest32']['unsupported']} |")
    lines += ['', 'No selection, threshold fit, held/independent inference, policy execution or confirmatory CI.',
        'Obs8/pred12 rawstride12, detector-silver pixel/local only;2% positive selected harm/reference unchanged.',
        'No metric/seconds/gold/physical-safety/true3D/foundation claim. Stage5C/SMC disabled.']
    return out, '\n'.join(lines)+'\n'


def execute(check):
    done = json.loads((run.PUBLIC/'complete.json').read_text()); rows = []
    for ref in done['groups']:
        p = run.ROOT/ref['path']; require(run.sha(p) == ref['sha256'], 'Group changed')
        rows.append(json.loads(p.read_text()))
    data, md = describe(rows)
    data.update(complete_sha256=run.sha(run.PUBLIC/'complete.json'), reporter_sha256=run.sha(Path(__file__)),
        result_source='cached_verified_receipt_decomposition_no_new_inference')
    p, table = run.PUBLIC/'slice_diagnostics.json', run.PUBLIC/'slice_diagnostics.md'
    if check:
        require(data == json.loads(p.read_text()) and md == table.read_text(), 'Readout replay differs')
        r = json.loads((run.PUBLIC/'figure_receipt.json').read_text())
        require(run.sha(run.ROOT/r['path']) == r['sha256'] and run.sha(p) == r['slice_sha256'], 'Figure changed')
        print('All21 slices, decomposition and figure hash verified'); return
    run.once(p, data)
    if table.exists(): require(table.read_text() == md, 'Existing table changed')
    else: table.write_text(md)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout='constrained')
    arms = ('origin', 'pooled_mean', 'pooled_nearest32', 'selected_mean', 'selected_nearest32', 'selected_shuffled32')
    names = ('Original', 'Full mean', 'Full nearest', 'Selected mean', 'Selected nearest', 'Selected shuffle')
    colors = ('#687580', '#b47942', '#a95166', '#7c6b53', '#007d78', '#5485aa')
    for ax, target, metric, title in zip(axes, ('all', 'easy', 'easy'),
            ('full_cohort_MSE', 'full_cohort_MSE', 'predicted_harm'),
            ('All signed-risk MSE', 'Easy signed-risk MSE', 'Predicted easy-harm mass')):
        vals = [data['all'][a][target][metric] for a in arms]
        for i, val in enumerate(vals):
            if val is None: ax.text(i, 0, 'undefined', rotation=90)
            else: ax.bar(i, val, color=colors[i]); ax.text(i, val, f'{val:.3g}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(arms)), names, rotation=30, ha='right'); ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max([v for v in vals if v is not None]+[1e-12])*1.3)
        if metric == 'predicted_harm':
            observed = data['all']['origin']['easy']['observed_harm']
            ax.axhline(observed, color='#222222', ls='--', label=f'observed {observed:.4g}'); ax.legend(fontsize=8)
            ax.set_ylim(0, max(ax.get_ylim()[1], observed*1.2))
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('Same frozen queries, different supervised donor population', fontsize=14)
    fig.supxlabel('72 exposed-TRAIN fitting views; no new policy or independent test', fontsize=10)
    path = run.PRIVATE/'selection_conditioning.png'; fig.savefig(path, dpi=160); plt.close(fig)
    run.once(run.PUBLIC/'figure_receipt.json', dict(path=str(path.relative_to(run.ROOT)), sha256=run.sha(path),
        slice_sha256=run.sha(p), raw_data_in_figure=False))
    print(json.dumps(dict(slices=21, figure=str(path))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--check', action='store_true'); execute(p.parse_args().check)
