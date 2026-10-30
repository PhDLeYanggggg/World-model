"""Report exact frozen-choice neural-use effects without promoting a policy."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_frame_allocation_attribution as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups


def build():
    public = run.PUBLIC
    complete, summary, verification = [json.loads((public/n).read_text()) for n in
        ('complete.json', 'summary.json', 'verification.json')]
    if (not verification['all72groups_exact'] or not verification['all288action_hashes_exact']
            or verification['summary_sha256'] != run.sha(public/'summary.json')
            or verification['complete_sha256'] != run.sha(public/'complete.json')
            or complete['summary_sha256'] != run.sha(public/'summary.json')):
        raise ValueError('Complete separately verified diagnostic required')
    reg = json.loads(run.REG.read_text())
    for rel, digest in reg['bindings'].items():
        if run.sha(ROOT/rel) != digest:
            raise ValueError('Diagnostic scientific dependency changed: '+rel)
    if (reg['parent_complete_sha256'] != run.sha(run.parent.PUBLIC/'complete.json')
            or reg['parent_summary_sha256'] != run.sha(run.parent.PUBLIC/'summary.json')):
        raise ValueError('Main frozen experiment changed')
    values = aggregate_groups(run.NAME)
    rows = [values[Path(ref['path']).name] for ref in complete['groups']]
    cfg = json.loads(run.parent.CONFIG.read_text())
    if run.summarize(rows, cfg) != summary:
        raise ValueError('Diagnostic summary reconstruction differs')
    return dict(summary=summary, complete=complete, rows=rows,
        main=json.loads((run.parent.PUBLIC/'summary.json').read_text()),
        verification_runtime=json.loads((run.PRIVATE/'verification_runtime.json').read_text()))


def fmt(v):
    return f"{v['point']:+.6f} [{v['CI95'][0]:+.6f},{v['CI95'][1]:+.6f}]" if v['CI95'] else 'not estimable'


def render(value):
    s, complete = value['summary'], value['complete']
    lines = ['# Frozen-query neural use and harm attribution', '', '## Material Passport', '',
        'Registered after the main readout,before these new attribution values. Post-readout diagnostic,not a primary efficacy test.',
        'Fresh inference and arithmetic on cached_verified models,inputs and immutable action choices.',
        'All72 original independent attributions and288 action hashes match. Separate full diagnostic replay is exact.',
        f"PID{complete['pid']} takes{complete['seconds']:.3f}s;replay PID{value['verification_runtime']['pid']} takes{value['verification_runtime']['seconds']:.3f}s.",
        'SOURCE fitting/resubstitution only. No training,policy search,independent confirmation or deployment.', '',
        '## Actual neural use within each frozen allocation', '',
        'Delete ONLY selected neural actions to the unchanged floor;keep all other actions fixed.',
        'This is NOT a reoptimized physical-only controller. The ablation itself need not satisfy',
        'the original budget,because removing a reference-contributing action can change the ratio.', '',
        '|Frozen decoder|Known neural-use increment pp [95% CI]|Wholly-unknown completion-lower pp [95% CI]|',
        '|---|---:|---:|']
    for a, effect in s['actual_neural_use'].items():
        lines.append(f"|{a}|{fmt(effect['full_known_pp'])}|{fmt(effect['full_lower_pp'])}|")
    inc = s['joint_minus_independent_incremental_neural_use']
    remaining = s['neural_deleted_joint_minus_independent']
    lines += ['', '## Where the joint-minus-independent gain comes from', '',
        f"Incremental neural-use reallocation:known{fmt(inc['known_pp'])};completion-lower{fmt(inc['lower_pp'])}.",
        f"Remaining action reallocation:known{fmt(remaining['known_pp'])};completion-lower{fmt(remaining['lower_pp'])}.",
        'The known point estimates sum to the main+0.648388pp increment. Interval endpoints do not sum.',
        'Shared unknown neural uses cancel before bounding the difference. The incremental lower',
        'bound is not a subtraction of two lower bounds. These are forecast-use arithmetic effects,',
        'not causal effects on real agent behavior,new dynamics training or independent neural superiority.', '',
        '## Why easy protection still fails', '',
        '|Decoder|Observed easy harm bookkeeping|Predicted easy harm bookkeeping|Median within-record predicted/observed easy harm|Fully known selected queries with easy excess|',
        '|---|---:|---:|---:|---:|']
    for a, v in s['attribution'].items():
        c = v['channels']['easy']; q = c['within_record_predicted_over_observed_harm_quantiles']
        lines.append(f"|{a}|{c['observed_harm_mass_bookkeeping']:.6f}|{c['predicted_harm_mass_bookkeeping']:.6f}|"
            f"{q[2] if q else None}|{c['fully_known_queries_with_observed_excess']}|")
    primary = s['attribution']['joint_matched']; ch = primary['channels']['easy']
    lines += ['', f"Joint-matched fully known query witnesses:{ch['fully_known_query_witness_counts']}.",
        f"Selected-unknown query views:{primary['selected_unknown_query_views']};empty query views:{primary['empty_query_views']}.",
        'Pooling deliberately permits individual predicted row excess. Such a row is not necessarily',
        'a false query admission; only complete-query arithmetic on known selected outcomes is used',
        'for that diagnosis. Mixed/unknown queries remain in full policy scores and are not imputed.',
        'The easy-harm estimate is much too small on the selected population,even while total harm',
        'can be conservative in aggregate. Fewer positive-excess rows do not imply smaller harm mass.',
        'Witnesses describe numerator/reference errors algebraically,not a uniquely identified causal root cause.', '',
        '## Interpretation and boundaries', '',
        'Raw masses sum reused local-image-coordinate views and are bookkeeping,not cross-domain physical metrics.',
        'Within-record ratios exclude records with zero observed harm;their quantiles are not a calibration guarantee.',
        'Intervals are3,000 locality bootstrap draws after context/head-seed averaging on12 exposed localities.',
        'Three head seeds are not three forecaster seeds or independent scenes. The main complete-policy',
        'risk gate still fails:324known easy violations,43/444 supported recording views. No positive',
        'secondary effect can replace it. Independent confirmation and submission readiness remain false.',
        'The next strong control must reoptimize physical-only allocation at a feasible common query',
        'count,not rename neural deletion as that comparator. Selected-query easy-risk estimation then',
        'needs a distinct,evidence-driven repair and exposure-aware calibration protocol.',
        'No metric/seconds/human-gold/true3D/foundation claim,Stage5C execution or SMC.', '']
    return '\n'.join(lines)


def main():
    value = build(); s = value['summary']; public = run.PUBLIC
    (public/'readout.md').write_text(render(value))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    effects = [s['joint_minus_independent_incremental_neural_use']['known_pp'],
        s['neural_deleted_joint_minus_independent']['known_pp'],
        value['main']['comparisons']['joint_matched_minus_independent']['full_known_pp']]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, (v, color) in enumerate(zip(effects, ('#007C91', '#888888', '#333333'))):
        low, high = v['CI95']
        ax.errorbar(v['point'], i, xerr=[[v['point']-low], [high-v['point']]],
            fmt='o', capsize=4, color=color)
    ax.set_yticks(range(3), ('Neural-use reallocation', 'Remaining action reallocation', 'Total joint allocation'))
    ax.invert_yaxis(); ax.axvline(0, color='#999999', linewidth=.8)
    ax.set_xlabel('Known ADE increment (percentage points), locality 95% CI')
    ax.grid(axis='x', alpha=.2)
    ax.set_title('Fitting attribution only; complete-policy risk still fails', fontsize=11)
    fig.tight_layout(); folder = public/'figures'; folder.mkdir(exist_ok=True)
    path = folder/'neural_use_attribution.svg'
    fig.savefig(path, bbox_inches='tight', metadata={'Date': None})
    path.write_text('\n'.join(v.rstrip() for v in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'neural_use_attribution.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    run.write(public/'evidence.json', dict(summary_sha256=run.sha(public/'summary.json'),
        complete_sha256=run.sha(public/'complete.json'), verification_sha256=run.sha(public/'verification.json'),
        report_code_sha256=run.sha(Path(__file__)), report_sha256=run.sha(public/'readout.md'),
        figure_sha256=run.sha(path), original_choices_unchanged=True, scientific_rescoring=False,
        optimized_physical_only_comparison=False, causal_identification=False,
        independent_confirmation=False, deployment_changed=False, submission_ready=False))
    print(json.dumps(dict(report_generated=True, all72groups_exact=True, deployment_changed=False)))


if __name__ == '__main__':
    main()
