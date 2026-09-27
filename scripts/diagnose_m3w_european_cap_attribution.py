"""Absolute-cost companion for every fixed arm; no selection or new contrasts."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_cap_attribution as run
from scripts.report_m3w_european_cap_attribution import distribution


def main():
    rows = json.loads((run.PRIVATE/'readout.json').read_text())['rows']; out = {}; counts = {}
    for pair in ('full', 'motion_only'):
        rr = [r for r in rows if r['input']['pair'] == pair]; out[pair] = {}; counts[pair] = {}
        for arm in run.parent.method.ARMS:
            ds = [r['diagnostics'][arm] for r in rr]
            counts[pair][arm] = dict(dependent_views=len(ds),
                lower_easy_MSE=sum(d['cap_relaxation_MSE_benefit'] > 0 for d in ds),
                higher_easy_MSE=sum(d['cap_relaxation_MSE_benefit'] < 0 for d in ds),
                unchanged_easy_MSE=sum(d['cap_relaxation_MSE_benefit'] == 0 for d in ds),
                event_benefit_positive=sum(d['event_cap_relaxation_contribution'] > 0 for d in ds),
                event_benefit_but_total_worse=sum(d['event_cap_relaxation_contribution'] > 0
                    and d['cap_relaxation_MSE_benefit'] < 0 for d in ds),
                zero_event_contribution_nonpositive=all(d['zero_event_cap_relaxation_contribution'] <= 0 for d in ds),
                coupled_all_harm_MSE_worse=sum(r['metrics'][arm+'_envelope_coupled']['H_all_MSE']
                    > r['metrics'][arm+'_frozen_cap']['H_all_MSE'] for r in rr))
            for mode in run.method.PROJECTIONS:
                name = arm+'_'+mode; ms = [r['metrics'][name] for r in rr]
                assert len(ms) == 216 and all(m['status'] == 'measured' for m in ms)
                out[pair][name] = {key:distribution([m[key] for m in ms])
                    for key in ('H_easy_MSE', 'H_all_MSE', 'actual_harm_mean', 'predicted_harm_mean')}
    run.immutable_json(run.PUBLIC/'absolute_costs.json', out)
    run.immutable_json(run.PUBLIC/'mechanism_counts.json', counts)
    lines = ['# Absolute Cost Companion', '',
        'Post-screen descriptions of every fixed arm, not a new comparison or selection criterion.',
        'Each min/median/max summarizes216 dependent views; medians are not paired effects or pooled error.',
        'MSE is squared native pixel-derived expected-cost error, not trajectory ADE/FDE.', '',
        '| Family / arm | Easy-harm MSE min / median / max | All-harm MSE min / median / max |', '|---|---|---|']
    for pair, arms in out.items():
        for name, d in arms.items(): lines.append(f"| {pair} / {name} | {d['H_easy_MSE']} | {d['H_all_MSE']} |")
    lines += ['', 'Large relative percentages may reflect tiny denominators. No unfavorable arm is removed.',
        'A pointwise label-derived ceiling is not evidence of conditional-mean bias.',
        'The registered paired intervals, not these descriptive medians, determine the information screen.',
        'No independent outcomes, refitting, trajectory deployment, metric/seconds, Stage5C or SMC.']
    lines += ['', '## Post-Hoc View Counts', '',
        'Counts describe overlapping views, not independent votes or a new gate.',
        'A zero easy-harm label includes non-easy samples as well as non-harm events; it is not synonymous with an easy case.', '',
        '| Family / arm | Dependent view counts |', '|---|---|']
    for pair, arms in counts.items():
        for arm, value in arms.items(): lines.append(f'| {pair} / {arm} | {value} |')
    (run.PUBLIC/'absolute_costs.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(rows=len(rows), families=len(out), arms_per_family=16)))


if __name__ == '__main__': main()
