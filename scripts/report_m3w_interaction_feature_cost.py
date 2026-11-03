"""Reconstruct the matched feature study from complete replayed scalar evidence."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_interaction_feature_cost_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_interaction_feature_cost_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    read = lambda p: json.loads(p.read_text())
    summary = read(PUBLIC/'summary.json')
    replay = read(PUBLIC/'verification.json')
    inference = read(PUBLIC/'inference_verification.json')
    train = read(PUBLIC/'training_freeze.json')
    pilot = read(PUBLIC/'pilot.json')
    if (not replay['all72_metric_groups_exact'] or not inference['all36_cost_amplitude_arrays_exact']
            or replay['summary_sha256'] != sha(PUBLIC/'summary.json')):
        raise ValueError('Complete exact replay is required for this report')
    from scripts import run_m3w_interaction_feature_cost as runner
    cfg = read(runner.CONFIG)
    reg = read(runner.REG)
    paths = [PUBLIC/'groups'/(j['key']+'_'+site+'.json') for j in reg['jobs'] for site in j['evaluation_sites']]
    groups = [read(p) for p in paths]
    if runner.summarize(groups, cfg, summary['source_localities']) != summary:
        raise ValueError('Every registered public group must reproduce the canonical summary')
    fits = []
    for ref in train['fits']:
        path = ROOT/ref['path']
        if sha(path) != ref['sha256']:
            raise ValueError('Fitting receipt changed')
        fits.append(read(path))
    fmt = lambda r: 'not estimable' if r['point'] is None else f'{r["point"]:+.4f} [{r["CI95"][0]:+.4f}, {r["CI95"][1]:+.4f}]'
    distance = lambda r: 'not estimable' if r['point'] is None else f'{r["point"]:.4f} [{r["CI95"][0]:.4f}, {r["CI95"][1]:.4f}]'
    primary = summary['contrasts']['primary']
    names = ('floor','constant_velocity_causal_fd','full_neural','direct_bound_bounded','zero64_bounded','interaction64_bounded','zero64_binary','interaction64_binary')
    lines = ['# Frozen Interaction Features For Intervention Costs', '', '## Material Passport', '',
             'Fresh_run36real Torch cost-head fits and complete held SOURCE inference,',
             'evaluation and replay;cached_verified frozen forecasters and h64cache.',
             'This report reconstructs public scalar evidence,not new inference.', '',
             '## Registered Primary', '',
             f'Interaction64minuszero64direct paired lower: **{fmt(primary["completion_lower_pp"])} pp**.',
             f'Known-label effect: {fmt(primary["known_pp"])} pp.',
             f'Upper effect: {fmt(primary["completion_upper_pp"])} pp.',
             f'Primary supported under the registered SOURCE bootstrap: {summary["primary_supported"]}.',
             f'Original full-risk pass: {summary["original_risk_pass"]}.', '',
             'The two new508-input heads have identical33349parameter capacity,initial',
             'weights,fit-only normalizer,full query draws,objective and2%policy rule.',
             'Zero64removes the final standardized feature block exactly;the other',
             'uses ego h64from a frozen producer-excluded EqMotion interaction layer.',
             'The old444-input direct control has29253parameters and is secondary.', '',
             'h64uses the original forecaster neighbor packet;the444descriptors use',
             'the later partial-neighbor packet. This contrast tests the full added',
             'feature bundle,not learned encoding isolated from raw neighbor coverage.',
             'A positive result still needs a raw-original-context control before',
             'that stronger attribution. This caveat was fixed before head training.', '',
             '|Policy|Known gain % [95% CI]|Individual completion lower % [95% CI]|Supported/444|Worst signed easy net %|',
             '|---|---:|---:|---:|---:|']
    for name in names:
        a,r = summary['accuracy'][name],summary['safety'][name]
        worst = r['worst_signed_easy_net_percent']
        worst_text = f'{worst:.4f}' if worst is not None else 'not estimable'
        lines.append(f'|{name}|{fmt(a["known"])}|{fmt(a["completion_lower"])}|{r["supported_record_views"]}/444|{worst_text}|')
    lines += ['', '## Accuracy And Tail Diagnostics', '',
              '|Policy|Known ADE pixel [95% CI]|Available endpoint FDE diagnostic pixel [95% CI]|Known gain over CV % [95% CI]|Mean role-specific p95 ADE pixel [95% CI]|',
              '|---|---:|---:|---:|---:|']
    for name in names:
        a = summary['accuracy'][name]
        lines.append(f'|{name}|{distance(a["known_ADE_pixel"])}|{distance(a["available_endpoint_FDE_diagnostic_pixel"])}|{fmt(a["known_gain_CV_percent"])}|{distance(a["mean_role_specific_p95_ADE_pixel"])}|')
    lines += ['', 'The tail column is an equal-locality average of role-specific95th percentiles,',
              'not a pooled-window percentile. Endpoint FDE uses only released valid',
              'endpoints,not full-roster endpoint accuracy. The floor is producer-fit-',
              'selected;CVremains a separate fixed causal comparator,not held-selected.', '',
              'CV-relative percentages are averaged within localities before the final',
              'mean;they are not the ratio of the two macro pixel-ADEaverages.', '',
              'A paired lower comparison is not the subtraction of two individual lower',
              'bounds. An empty selected reference is not estimable;the unchanged floor',
              'is not thereby operationally unsafe or a learned-intervention success.', '',
              '## Full Risk And Subsets', '',
              '|Policy|Known all/easy violations|Completion all/easy violations|Undefined easy|Active %|Hard gain % [95% CI]|Easy gain % [95% CI]|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('direct_bound_bounded','zero64_bounded','interaction64_bounded'):
        a,r = summary['accuracy'][name],summary['safety'][name]
        lines.append(f'|{name}|{r["known_all_violations"]}/{r["known_easy_violations"]}|{r["completion_all_violations"]}/{r["completion_easy_violations"]}|{r["undefined_easy_views"]}|{100*a["switch_rate"]["point"]:.4f}|{fmt(a["hard"])}|{fmt(a["easy"])}|')
    lines += ['', '## Secondary Contrasts', '']
    for name in ('interaction_minus_direct','zero_minus_direct'):
        x = summary['contrasts'][name]
        lines.append(f'{name}: known {fmt(x["known_pp"])} pp;direct paired lower {fmt(x["completion_lower_pp"])} pp.')
    lines += ['', 'Secondary results cannot replace a failed registered primary.', '',
              '## Fitting And Locality Evidence', '',
              '|Feature arm|Mean full-bound fitting objective|Mean known-only fitting objective|',
              '|---|---:|---:|']
    fitting = {}
    for arm in cfg['arms']:
        rows = [r for r in fits if r['arm']==arm]
        if len(rows)!=18:
            raise ValueError('All18fits per arm required')
        fitting[arm] = {k:sum(r['fitting_objectives'][k]['total'] for r in rows)/18 for k in ('bound_aware','known_only')}
        lines.append(f'|{arm}|{fitting[arm]["bound_aware"]:.6f}|{fitting[arm]["known_only"]:.6f}|')
    lines += ['', 'These common decoded fitting objectives use the actual cyclic query visit',
              'weights;they are not held evaluation or raw-FDE validation losses.', '',
              '|Locality|Primary known pp|Primary paired lower pp|', '|---|---:|---:|']
    for site in summary['source_localities']:
        k = primary['known_pp']['by_locality'][site]
        lo = primary['completion_lower_pp']['by_locality'][site]
        lines.append(f'|{site}|{k:+.4f}|{lo:+.4f}|')
    lines += ['', '## Endpoint And Cohort Support', '',
              'Posthoc descriptive label-support counts,not a new primary cohort.',
              'All six producer/seed views must agree on the same target-row support.', '',
              '|Locality|Unique windows|Wholly unknown future|Complete12|Valid final endpoint|',
              '|---|---:|---:|---:|---:|']
    support = {}
    for site in summary['source_localities']:
        values = [r['policies']['floor'] for r in groups if r['site']==site]
        keys = ('rows','wholly_unknown_rows','complete12_rows','endpoint_rows')
        if len(values)!=6 or any(len({r[k] for r in values})!=1 for k in keys):
            raise ValueError('Repeated views cannot change unique cohort support')
        row = {k:values[0][k] for k in keys}
        support[site] = row
        lines.append(f'|{site}|{row["rows"]}|{row["wholly_unknown_rows"]}|{row["complete12_rows"]}|{row["endpoint_rows"]}|')
    if sum(r['rows'] for r in support.values())!=149510:
        raise ValueError('Unique SOURCE whitelist must be preserved')
    lines += ['', '082has zero released final endpoints in the unchanged whitelist;the',
              'all-required-locality endpoint interval is therefore not estimable.',
              'Do not fill endpoints,drop082or call available-labelADEa full12score.',
              '008and074contain most windows,but the registered bootstrap gives each',
              'locality equal weight. Window count alone is not independent support.']
    lines += ['', '## Reproduction And Scope', '',
              f'Training PID{train["pid"]}:{train["seconds"]:.3f}s phase time including preparation/scoring,peakRSS{train["peak_RSS_bytes"]:,}bytes.',
              f'Primary updates:{train["primary_optimizer_updates"]};first full refits:{train["first_full_refit_updates"]};pilot:{pilot["actual_optimizer_updates"]}.',
              f'Complete replay PID{replay["pid"]}:{replay["seconds"]:.3f}s,including{inference["seconds"]:.3f}s held prediction replay.',
              'All36cost/amplitude pairs and72groups reproduce exactly;the old direct',
              'policy matches every original metric view. Atomic checkpoint/resume,',
              'arm64CPU4/inter-op1/workers0. See protocol and frozen registrations.', '',
              '149510overlapping SOURCE windows,74recordings and12exposed localities.',
              '444recording/pipeline views repeat those recordings,not444independent',
              'scenes. Three actual seeds17/29/43;3000paired locality bootstraps after',
              'role/seed averaging. Scoring sites are excluded from both producer and',
              'critic fits,but SOURCE has been used for development,not final confirmation.', '',
              'The source-cache v1pilot failed from neighbor-packet mismatch;v2restores',
              'the original packet and separately replays all coordinates and h64rows.',
              'The initial head pilot also failed before any optimizer update:the',
              'wrapper requested Hrather than G,and the site assertion stopped it.',
              'Temporary Hlabel preprocessing had been computed and discarded;',
              'it was never used to train a head or select a policy. The explicit',
              'registration_v2correction forwards the role and checks it earlier.',
              'No private calibration/confirmation/DroneCrowd use for this repair.',
              'Future targets and masks are loss/evaluation only. Available-label ADE',
              'is not full12ADE;unknown-row bounds do not complete partial futures.',
              'Pixels/rawstride12/silver,not metric,seconds,human gold or physical safety.',
              'No new trajectory/JEPA/Transformer training,no deployment,Stage5C orSMC.',
              'Licensed private data and checkpoints are needed for full raw reruns;',
              'the public scalar reconstruction is not a standalone cold-start package.']
    (PUBLIC/'readout.md').write_text('\n'.join(lines)+'\n')
    evidence = dict(result_source='verified_public_scalar_reconstruction_not_new_training',
                    summary_sha256=sha(PUBLIC/'summary.json'),verification_sha256=sha(PUBLIC/'verification.json'),
                    reporter_sha256=sha(Path(__file__)),groups={str(p.relative_to(ROOT)):sha(p) for p in paths},
                    fitting=fitting,posthoc_unique_label_support=support,
                    public_groups_reproduce_summary_exactly=True,submission_ready=False,deployment_changed=False)
    (PUBLIC/'report_sources.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    gate_lines = ['# Matched Interaction Feature Gates', '',
                  '|Gate|Pass|', '|---|---|',
                  f'|Complete36real fits|{train["all36complete"]}|',
                  f'|Matched common fields|{train["paired_common_fields_exact"]}|',
                  f'|Every36prediction pair exact|{inference["all36_cost_amplitude_arrays_exact"]}|',
                  f'|All72groups exact|{replay["all72_metric_groups_exact"]}|',
                  f'|Registered primary supported|{summary["primary_supported"]}|',
                  f'|Original complete risk|{summary["original_risk_pass"]}|',
                  '|Independent confirmation|False|', '|Submission ready|False|',
                  '|Stage5Cexecution|False|', '|SMCenabled|False|', '',
                  'Engineering gates and positive SOURCE mean effects cannot certify',
                  'complete risk or independent science. No promotion in this report.']
    (PUBLIC/'gates.md').write_text('\n'.join(gate_lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    old = PUBLIC.parent/'european_bound_aware_cost_v1'
    direct = []
    direct_refs = []
    for ref in read(old/'training_freeze.json')['fits']:
        p = ROOT/ref['path']
        if sha(p)!=ref['sha256']:
            raise ValueError('Frozen direct fitting reference changed')
        r = read(p)
        if r['arm']=='bound_aware':
            direct.append(r['fitting_objectives']['bound_aware']['total'])
            direct_refs.append(ref)
    if len(direct)!=18:
        raise ValueError('Complete old direct fitting control required')
    plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'m3w-interaction-feature-cost-v1','font.size':9})
    fig,axes = plt.subplots(1,3,figsize=(15,4),layout='constrained')
    labels = ('Direct444','Zero64','Interaction64')
    colors = ('#75818a','#a66b35','#087f89')
    axes[0].bar(labels,[float(np.mean(direct)),fitting['zero64']['bound_aware'],fitting['interaction64']['bound_aware']],color=colors)
    axes[0].set(title='Fitting costs improve',ylabel='Mean decoded full-bound fitting objective',ylim=(0,1.7))
    for j,(field,label) in enumerate((('known_pp','Known labels'),('completion_lower_pp','Registered lower'))):
        value = primary[field]
        p = value['point']
        lo,hi = value['CI95']
        axes[1].errorbar(p,j,xerr=[[p-lo],[hi-p]],fmt='o',color=('#087f89','#b24952')[j],capsize=4)
    axes[1].axvline(0,color='#777777',linestyle=':')
    axes[1].set(title='Held benefit is not established',yticks=[0,1],yticklabels=['Known labels','Registered lower'],
                xlabel='Interaction64 minus Zero64 (pp), 95% locality CI')
    names = ('direct_bound_bounded','zero64_bounded','interaction64_bounded')
    axes[2].bar(labels,[summary['safety'][k]['supported_record_views'] for k in names],color=colors)
    axes[2].axhline(444,color='#777777',linestyle='--')
    axes[2].text(2.4,449,'Required: 444',ha='right',va='bottom',color='#555555',fontsize=8)
    axes[2].set(title='No complete risk pass',ylabel='Supported repeated recording/pipeline views',ylim=(0,470))
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Frozen interaction features: better fitting is not a safe intervention result')
    home = PUBLIC/'figures'
    home.mkdir(exist_ok=True)
    fig.savefig(home/'interaction_feature_cost.svg',metadata={'Date':None})
    plt.close(fig)
    (home/'sources.json').write_text(json.dumps(dict(result_source='verified_scalar_figure_not_new_inference',
        summary_sha256=sha(PUBLIC/'summary.json'),verification_sha256=sha(PUBLIC/'verification.json'),
        old_direct_fitting_receipts=direct_refs,reporter_sha256=sha(Path(__file__)),
        svg_sha256=sha(home/'interaction_feature_cost.svg'),deployment_changed=False),indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(readout=str(PUBLIC/'readout.md'),primary=primary,risk=summary['original_risk_pass'])))


if __name__=='__main__':
    main()
