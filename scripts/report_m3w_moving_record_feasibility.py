"""Render the verified fixed-proposal feasibility result without dropping failures."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_record_feasibility_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_record_feasibility_v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    analysis=read(PUBLIC/'analysis.json')
    if (not read(PUBLIC/'verify_execution.json')['all_scalars_exact']
            or not analysis['exact_witness_gate_agreement_all_views']):
        raise ValueError('Complete independent scalar replay and unchanged gate agreement required')
    for relative,expected in analysis['bindings'].items():
        if sha(ROOT/relative) != expected:
            raise ValueError('Frozen diagnostic code changed')
    cfg=read(ROOT/'configs/m3w_moving_source_forecast_supported_v1.json')
    admission_path=ROOT/cfg['admission']
    if sha(admission_path) != cfg['admission_sha256']:
        raise ValueError('Original admission mapping required')
    admission=read(admission_path)
    nodes=sorted(admission['recordings'])
    grouped=defaultdict(list)
    for row in analysis['rows']:
        grouped[row['recording']].append(row)
    if len(grouped) != 58 or any(len(v) != 6 for v in grouped.values()):
        raise ValueError('All58recordings and6producer/seed views required')
    records=[]
    for key,rows in sorted(grouped.items(),key=lambda p:int(p[0])):
        node=nodes[int(key)]
        metadata=admission['recordings'][node]
        counts=Counter(r['reason']for r in rows)
        records.append(dict(recording_id=int(key),recording_node=node,clip=metadata['clip'],
            source_group=metadata['exclusion_group'],required_views=len(rows),
            feasible_views=counts['feasible_label_aware_witness'],
            no_positive_easy_reference_views=counts['no_known_positive_easy_reference'],
            no_legal_easy_reference_views=counts['no_legal_positive_easy_reference'],
            all_legal_easy_actions_above_budget_views=counts['every_legal_easy_harm_ratio_exceeds_original_budget']))
    counts=Counter(r['reason']for r in analysis['rows'])
    blocked=[r for r in records if r['feasible_views'] != r['required_views']]
    evaluation,replay=(read(PUBLIC/(phase+'_execution.json'))for phase in ('evaluate','verify'))
    lines=['# The Fixed Binary Proposal Cannot Pass The Whole Recording Roster','',
        '## Material Passport','',
        'Fresh_run: all 348 recording/producer/seed views and their label-aware',
        'single-row witnesses, checked against the unchanged original evaluator.',
        'Cached_verified: original data and saved EqMotion forecasts, followed',
        'by a separate exact scalar replay. This is post-readout development',
        'diagnosis, not learned performance or independent final confirmation.','',
        '## Main Result','',
        f"Only **{analysis['feasible_record_views']}/{analysis['required_record_views']}** repeated views admit any finite-support binary neural/floor policy",
        'that meets the original all/easy 2% positive-harm and signed easy-net',
        'requirements. Thus the whole fixed roster cannot pass, even with labels.',
        'This is not a failed optimization budget or a conclusion from a poor',
        'classifier. The complete unchanged-gate witness check is exact.','',
        '| Fixed-view condition | Repeated views |','|---|---:|',
        f"| Label-aware feasible witness | {counts['feasible_label_aware_witness']} |",
        f"| No known positive easy reference | {counts['no_known_positive_easy_reference']} |",
        f"| All legal easy proposal harm ratios exceed 2% | {counts['every_legal_easy_harm_ratio_exceeds_original_budget']} |",
        f"| Positive easy reference exists but no legal proposal | {counts['no_legal_positive_easy_reference']} |",'',
        f'{len(blocked)} of 58 unique actual-query recordings have at least one infeasible',
        'view. The 348 views repeat excluded producers and three actual seeds;',
        'they are not 348 independent recordings. All 58 recordings remain.','',
        'The 15 reference-limited views have no observed easy row at all under',
        'the unchanged producer-fitting CV cutoff. This is an absent event in',
        'the observed evaluation support, not proof that hidden easy outcomes',
        'exist or that more annotations would necessarily repair it.','',
        '## Why The Criterion Is Exact','',
        'For known legal easy rows with positive floor error R, define',
        'H=max(proposal_error-floor_error,0). A selected easy risk is a',
        'reference-weighted average of H/R. If every legal ratio exceeds 0.02,',
        'no subset passes. Non-easy rows cannot subsidize the easy denominator;',
        'unknown selected rows add nonnegative bound mass but no known reference.',
        'With no positive easy reference, the ratio is undefined, not safe zero.','',
        'Conversely, selecting one legal easy row with H/R<=0.02 and keeping',
        'all others at the floor meets both harm budgets. Its signed easy-net',
        'increase is at most H divided by the full easy reference, also <=0.02.',
        'Every such witness was scored through the unchanged policy evaluator.',
        'The labels are used only for diagnosis. No oracle row choice is exported',
        'as a model input, policy, training target store or deployable result.','',
        '## Recordings With Any Infeasible View','',
        '| Clip | Source group | Feasible / 6 | No easy reference | All easy actions above 2% |',
        '|---|---|---:|---:|---:|']
    for r in blocked:
        lines.append(f"| {r['clip']} | {r['source_group']} | {r['feasible_views']}/6 | "
            f"{r['no_positive_easy_reference_views']} | {r['all_legal_easy_actions_above_budget_views']} |")
    lines += ['','The complete 58-record summary is in `record_summary.json`; per-view',
        'support, minimum ratios and unchanged evaluator checks are in',
        '`analysis.json`. No failing recording or seed is removed.','',
        '## What This Changes','',
        'Stop blind loss, width or duration sweeps of this same binary selector.',
        'Even perfect selection cannot repair 49 of the required fixed views.',
        'The 34 candidate-limited views motivate a different, better-controlled',
        'proposal class. Their failure is not a theorem about every possible',
        'new neural forecaster or bounded blend. The 15 empty-event views cannot',
        'be repaired by changing only the proposal while retaining the same',
        'floor, event and observed support. Data/event support must be addressed',
        'explicitly; no evaluation requirement is silently relaxed.','',
        'Do not filter to the 299 feasible views and call that the original',
        'primary. A future cohort/primary-role change requires an explicit',
        'prospective scientific decision. New-source role allocation remains',
        'pending; DUT and DroneCrowd reservations stay unchanged.','',
        'The previous loss repair remains negative and undeployed. This audit',
        'does not establish a learned method benefit, safe world model, true 3D,',
        'foundation status or submission readiness. Original 24-group primary',
        'remains not estimable; 23-group results are exposed-source secondary.',
        'Pixels/native 8/12 steps are not verified metres or seconds.','',
        '## Measured Execution','',
        f"Evaluation PID {evaluation['pid']}: {evaluation['seconds']:.6f} s, observed exit 0.",
        f"Separate replay PID {replay['pid']}: {replay['seconds']:.6f} s, observed exit 0.",
        'All scalars and 348 original-gate witness checks agree. No optimizer',
        'updates, new thresholds, raw-array export or independent outcome use.',
        'Stage5C execution and SMC remain disabled.']
    (PUBLIC/'readout.md').write_text('\n'.join(lines)+'\n')
    (PUBLIC/'record_summary.json').write_text(json.dumps(dict(records=records,
        repeated_view_counts=dict(counts),result_source='derived_hash_verified_feasibility_not_new_training'),
        indent=2,sort_keys=True,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(7.5,3.4))
    vals=[counts[k]for k in ('feasible_label_aware_witness','every_legal_easy_harm_ratio_exceeds_original_budget','no_known_positive_easy_reference')]
    ax.barh(['Label-aware feasible','Candidate limited','No observed easy event'],vals,color=['#2e7770','#b85b39','#6c6d76'])
    ax.invert_yaxis()
    for y,value in enumerate(vals):
        ax.text(value+2,y,str(value),va='center')
    ax.set_xlim(0,348)
    ax.set_xlabel('Repeated recording/producer/seed views, not independent samples')
    ax.set_title('Fixed binary proposal: 299 / 348 views feasible\nExact diagnosis, not learned or deployable performance')
    fig.tight_layout()
    (PUBLIC/'figures').mkdir(exist_ok=True)
    (PRIVATE/'figure_previews').mkdir(exist_ok=True)
    fig.savefig(PUBLIC/'figures/feasibility.svg')
    fig.savefig(PRIVATE/'figure_previews/feasibility.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
