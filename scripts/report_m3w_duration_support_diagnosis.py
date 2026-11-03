"""Report label-support contributions without redefining the full risk cohort."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_duration_support_diagnosis_v1'


def median_text(stat, percent=False):
    values = stat['quantiles_0_25_50_75_100']
    if values is None:
        return 'not estimable'
    if percent:
        return f'{100*values[2]:.2f}%'
    return f'{values[2]:.6g}' if 0 < abs(values[2]) < .0001 else f'{values[2]:.6f}'


def main():
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    summary, replay, complete = [json.loads((PUBLIC/name).read_text())
        for name in ('summary.json','verification.json','complete.json')]
    if (summary['models'] != 18 or not replay['all18_partitions_replayed_exact']
            or replay['summary_sha256'] != sha(PUBLIC/'summary.json')):
        raise ValueError('Require complete separate18-job partition replay')
    lines = ['# Selected Harm By Released-Label Support', '',
        'Fresh_run: all18 original G-role model pairs are inferred and all three',
        'selected sets are decomposed. A second full execution reproduces every',
        'partition. Cached_verified: original/extended models and unpartitioned',
        'costs. No training update, new H score or threshold selection occurs.', '',
        '## Numerator Contributions', '',
        'All entries summarize the same444 repeated recording/pipeline views,',
        'not444 independent scenes. Unknown EH is a conservative missing-label',
        'contribution, not measured forecast error. Partial means available-label',
        'evaluation only, not a complete twelve-step trajectory.', '',
        '| Selected set | Median unknown share of EH | Whole easy violations | Observed-only easy violations / defined views | Whole violations also observed |',
        '|---|---:|---:|---:|---:|']
    for name, value in summary['results'].items():
        lines.append(f'| {name} | {median_text(value["unknown_share_of_target_EH"], True)} | '
            f'{value["whole_easy_violations"]} | {value["observed_easy_violations"]}/{value["observed_easy_defined"]} | '
            f'{value["whole_and_observed_easy_violations"]} |')
    lines += ['', '## Predicted / Target EH Ratios Within Each Partition', '',
        'These are evaluation-only label partitions of the unchanged decisions.',
        'No future availability mask is an inference feature. Undefined target',
        'mass stays not estimable, never zero risk.', '',
        '| Selected set | Complete labels | Partial labels | Wholly unknown |',
        '|---|---:|---:|---:|']
    for name, value in summary['results'].items():
        lines.append(f'| {name} | '+' | '.join(median_text(value['parts'][part]['EH_prediction_target_ratio'])
                                             for part in ('complete','partial','unknown'))+' |')
    lines += ['', '## Execution And Limits', '',
        f'Original PID{complete["pid"]}: {complete["seconds"]:.3f}s, peak RSS{complete["peak_RSS_bytes"]:,}bytes.',
        f'Separate replay PID{replay["pid"]}: {replay["seconds"]:.3f}s, peak RSS{replay["peak_RSS_bytes"]:,}bytes.',
        'Every original selected row count and unpartitioned predicted/target',
        'mass matches exactly; the complete/partial/unknown contributions sum',
        'to that same mass. This is descriptive fitting evidence, not a new',
        'primary, independent confirmation or uniquely identified cause.', '',
        'Observed-only ratios cannot replace the original complete risk gate.',
        'A view that fails only after conservative bounds is not thereby safe;',
        'a view that also fails on available labels cannot be repaired by merely',
        'dropping unknown rows. Neither subset permits future-validity filtering.', '',
        '```sh', '.venv-pytorch/bin/python scripts/run_m3w_duration_support_diagnosis.py register',
        '.venv-pytorch/bin/python scripts/run_m3w_duration_support_diagnosis.py run',
        '.venv-pytorch/bin/python scripts/run_m3w_duration_support_diagnosis.py verify',
        '.venv-pytorch/bin/python scripts/report_m3w_duration_support_diagnosis.py', '```', '',
        'Unchanged interrupted execution supports run --resume. Completed phases',
        'are immutable. Private inputs/checkpoints are required. SOURCE remains',
        'exposed development, image pixels/raw frames and silver labels, not',
        'metric/seconds, human gold or new dynamics evidence. Original failures',
        'and deployment stand; Stage5C/SMC remain off and the goal is unmet.']
    (PUBLIC/'readout.md').write_text('\n'.join(lines)+'\n')
    receipt = dict(result_source='verified_scalar_readout_not_new_inference',
        summary_sha256=sha(PUBLIC/'summary.json'),verification_sha256=sha(PUBLIC/'verification.json'),
        reporter_sha256=sha(Path(__file__)),deployment_changed=False)
    (PUBLIC/'report_sources.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(readout=str(PUBLIC/'readout.md'))))


if __name__=='__main__':
    main()
