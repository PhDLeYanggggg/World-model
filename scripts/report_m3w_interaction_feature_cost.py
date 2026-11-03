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
    reg = read(PUBLIC/'registration.json')
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
    primary = summary['contrasts']['primary']
    names = ('floor','full_neural','direct_bound_bounded','zero64_bounded','interaction64_bounded','zero64_binary','interaction64_binary')
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
        lines.append(f'|{name}|{fmt(a["known"])}|{fmt(a["completion_lower"])}|{r["supported_record_views"]}/444|{worst if worst is not None else "not estimable"}|')
    lines += ['', 'A paired lower comparison is not the subtraction of two individual lower',
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
                    fitting=fitting,public_groups_reproduce_summary_exactly=True,submission_ready=False,deployment_changed=False)
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
    print(json.dumps(dict(readout=str(PUBLIC/'readout.md'),primary=primary,risk=summary['original_risk_pass'])))


if __name__=='__main__':
    main()
