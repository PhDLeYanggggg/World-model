"""Reduce completed signed-utility controls without opening independent roles."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_signed_utility_allocation as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import pack, verify


def build():
    p = run.PUBLIC
    complete, summary, replay = [json.loads((p/n).read_text()) for n in ('complete.json', 'summary.json', 'verification.json')]
    if (not replay['all72groups_exact'] or not replay['all72_original_controls_exact'] or not replay['all148_oracles_exact']
            or replay['summary_sha256'] != run.sha(p/'summary.json') or replay['complete_sha256'] != run.sha(p/'complete.json')):
        raise ValueError('Complete separate inference/allocation replay required')
    for rel, h in json.loads(run.REG.read_text())['bindings'].items():
        if run.sha(ROOT/rel) != h: raise ValueError('Scientific registration changed')
    groups = aggregate_groups(run.NAME); rows = [groups[Path(r['path']).name] for r in complete['groups']]
    for r in complete['groups']:
        if run.sha(ROOT/r['path']) != r['sha256']: raise ValueError('Aggregate changed')
    if run.summarize(rows, json.loads(run.CONFIG.read_text())) != summary: raise ValueError('Canonical reduction differs')
    reasons = {a: Counter() for a in run.api.NAMES[1:]}
    for r in rows:
        for a, counts in r['result']['allocation']['reasons'].items(): reasons[a].update(counts)
    gaps = [o['full']['optimal_gain_lower_mass']-o['positive_action_only']['optimal_gain_lower_mass']
        for r in rows if r['seed'] == 17 for o in r['result']['oracle_admissibility'].values()
        if o['full']['optimal_gain_lower_mass'] is not None and o['positive_action_only']['optimal_gain_lower_mass'] is not None]
    return dict(summary=summary, complete=complete, reasons={a: dict(v) for a, v in reasons.items()},
        oracle_gain_equality=dict(comparable_record_contexts=len(gaps), exact_equal=sum(g == 0 for g in gaps),
            max_abs_gap=max(map(abs, gaps)) if gaps else None),
        replay_runtime=json.loads((p/'verification_runtime.json').read_text()))


def fmt(v):
    if v['point'] is None or v['CI95'] is None: return 'not_estimable'
    return f"{v['point']:+.6f} [{v['CI95'][0]:+.6f},{v['CI95'][1]:+.6f}]"


def render(value):
    s = value['summary']; lines = ['# Joint utility admissibility on frozen risks', '', '## Material Passport', '',
        'fresh_run:72 complete causal allocation/scoring groups and a separate full replay.',
        'cached_verified:18 fixed selected-set critics,all source inputs/forecasts,and original full-policy choices.',
        'not_run:new training,trajectory dynamics,independent calibration/confirmation or deployment.',
        f"Readout PID{value['complete']['pid']}:{value['complete']['seconds']:.3f}s,peakRSS{value['complete']['peak_RSS_bytes']}B.",
        f"Replay PID{value['replay_runtime']['pid']}:{value['replay_runtime']['seconds']:.3f}s,peakRSS{value['replay_runtime']['peak_RSS_bytes']}B.",
        'All72 groups,72 original full-policy controls,148 unique record/context oracle controls and the summary agree exactly.', '',
        '## Controlled factor', '',
        'Only per-action positive utility is removed from the signed admissible set. Positive reference/easy-reference,',
        'at most one action per agent,nonnegative total predicted utility,and both ORIGINAL2% positive-harm/reference',
        'inequalities remain. Benefit never cancels positive harm in either numerator. All predictions are frozen.',
        'The matched pair retains exactly the same intervention count in every current query;free counts are secondary.', '',
        '## Prespecified paired effects', '',
        '|Signed minus positive-action control|Known utility pp [95% CI]|Wholly-unknown completion-lower pp [95% CI]|',
        '|---|---:|---:|']
    for mode in ('matched', 'free'):
        v = s['effects'][mode]; lines.append(f"|{mode}|{fmt(v['known_pp'])}|{fmt(v['lower_pp'])}|")
    lines += ['', 'Primary:matched full completion-lower effect. Nominal3,000-draw paired locality bootstrap over12',
        'EXPOSED SOURCE localities after context/head-seed averaging;not adaptive-search-adjusted confirmation.', '',
        '|Policy|Known ADE gain over floor [95% CI]|FDE gain [95% CI]|Supported original risk views|Known easy violations|Undefined easy|Selected views|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a in run.api.NAMES:
        r = s['safety'][a]
        lines.append(f"|{a}|{fmt(s['accuracy'][a])}|{fmt(s['FDE'][a])}|{r['supported_record_views']}/{r['required_record_views']}|"
            f"{r['known_easy_risk_violations']}|{r['undefined_easy_record_views']}|{r['selected']}|")
    o = s['oracle_admissibility']
    lines += ['', '## Label-side structural diagnostic', '',
        f"All{ o['required_record_contexts']} unique record/context views are retained. Unrestricted known-label oracle:",
        f"{o['full_supported']} supported,{o['full_positive_feasible']} positive feasible. True-positive-action-only oracle:",
        f"{o['positive_action_only_supported']} supported,{o['positive_action_only_positive_feasible']} positive feasible.",
        'The unrestricted control is freshly recomputed and verified against the existing result;it is not newly discovered',
        'opportunity. The restricted oracle uses future costs only for diagnosis,not model inference or selection.',
        'It differs from a conditional expected-moment policy. Neither feasible oracle proves causal learnability.', '',
        '## Fitting checks', '']
    lines += [f"- {k}: {'pass' if v else 'fail'}" for k, v in s['fitting_gates'].items()]
    lines += ['', f"Advancement:{s['fitting_advancement_gate']}. Deployment unchanged.", '',
        f"Oracle utility comparison,head seeds deduplicated:{value['oracle_gain_equality']}", '',
        '## Numerical and statistical limits', '', f"Solver reason counts:{value['reasons']}",
        'A fallback retains an actual feasible incumbent,not a global-optimality claim. Predicted nesting is not observed safety.',
        'Empty matched risk is not_estimable;count controls do not replace original complete recording risk.',
        'No independent rows are created by repeated contexts,seeds or overlapping windows. Three critic seeds share',
        'forecast producer43. All149,510 target rows,74 recordings and12 localities remain;no failing source is dropped.',
        'Unknown selected outcomes and missing endpoints remain unobserved. Bounds do not complete interior labels.',
        'The summary retains supplementary hard/easy slices;they do not replace the original risk requirement.',
        'This decoder-only experiment does not newly score interaction-path/tail proxies or SDD safety;those are not_run',
        'here,not borrowed from older policies. No world-dynamics contribution or complete benchmark is claimed.',
        'Obs8/pred12/rawstride12,oblique detector-silver/local-image coordinates,not SDDt50,metric/seconds,human gold,',
        'true3D,foundation,independent success or submission-ready evidence. Independent roles,Stage5C and SMC stay closed.', '',
        '[Full-policy figure](figures/allocation_result.svg),[mechanism counterexample](mechanism_note.md),',
        '[registered protocol](protocol.md),[Chinese reproduction guide](reproduction_zh.md).', '']
    return '\n'.join(lines)


def plot(value):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    s = value['summary']; fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for i, a in enumerate(run.api.NAMES):
        v = s['accuracy'][a]
        axes[0].errorbar(v['point'], i, xerr=[[v['point']-v['CI95'][0]], [v['CI95'][1]-v['point']]],
            fmt='o', color='#007C91' if a.startswith('signed') else '#777777', capsize=3)
    axes[0].set_yticks(range(4), ('Positive / matched', 'Signed / matched', 'Positive / free', 'Signed / free'))
    axes[0].invert_yaxis(); axes[0].set_title('Known-label fitting utility'); axes[0].set_xlabel('ADE gain over floor (%)')
    for i, (mode, key) in enumerate((('matched', 'known_pp'), ('matched', 'lower_pp'), ('free', 'known_pp'), ('free', 'lower_pp'))):
        v = s['effects'][mode][key]
        axes[1].errorbar(v['point'], i, xerr=[[v['point']-v['CI95'][0]], [v['CI95'][1]-v['point']]],
            fmt='o', color='#007C91' if key == 'known_pp' else '#B43A32', capsize=3)
    axes[1].set_yticks(range(4), ('Matched / known', 'Matched / lower', 'Free / known', 'Free / lower'))
    axes[1].invert_yaxis(); axes[1].set_title('Signed minus positive control'); axes[1].set_xlabel('Paired utility difference (pp)')
    axes[1].axvline(0, color='#777777', linewidth=.7)
    o = s['oracle_admissibility']; counts = [o['full_positive_feasible'], o['positive_action_only_positive_feasible']]
    axes[2].barh([0, 1], counts, color=['#777777', '#007C91'])
    for i, n in enumerate(counts): axes[2].text(n-4, i, str(n), ha='right', va='center', color='white')
    axes[2].set_yticks([0, 1], ('Full oracle', 'Positive-only oracle')); axes[2].invert_yaxis()
    axes[2].set_xlim(0, 155); axes[2].set_title('Future-label diagnostic only'); axes[2].set_xlabel('Positive feasible record/context views')
    for ax in axes: ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Exposed SOURCE development: nominal intervals, 3,000 resamples of 12 localities', fontsize=11)
    fig.tight_layout(rect=(0, .1, 1, .91))
    fig.text(.5, .015, 'All four policies: original recording risk only 1/444 supported. No deployment or independent-evidence claim.',
        ha='center', fontsize=9)
    folder = run.PUBLIC/'figures'; folder.mkdir(exist_ok=True); path = folder/'allocation_result.svg'
    fig.savefig(path, bbox_inches='tight', metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'allocation_result.png', bbox_inches='tight', dpi=160); plt.close(fig)


def complete_report():
    value = build(); p = run.PUBLIC; pack(ROOT, p); verify(ROOT, p)
    (p/'readout.md').write_text(render(value)); plot(value)
    run.write(p/'evidence.json', dict(summary_sha256=run.sha(p/'summary.json'), complete_sha256=run.sha(p/'complete.json'),
        verification_sha256=run.sha(p/'verification.json'), report_code_sha256=run.sha(Path(__file__)),
        report_sha256=run.sha(p/'readout.md'), new_training_updates=0,
        independent_confirmation=False, deployment_changed=False, submission_ready=False))
    print(json.dumps(dict(report_generated=True, gates=value['summary']['fitting_gates'], deployment_changed=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.parse_args(); complete_report()
