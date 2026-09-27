"""Post-screen support and ceiling diagnostics; no policy/model selection."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_temporal_support as run


def distribution(values):
    a = np.array([v for v in values if v is not None], float)
    return dict(n=len(a), min=float(a.min()), median=float(np.median(a)), max=float(a.max())) if len(a) else dict(n=0)


def main():
    doc = json.loads((run.PRIVATE/'readout.json').read_text()); output = {}
    for pair in ('full', 'motion_only'):
        rows = [r for r in doc['rows'] if r['input']['pair'] == pair]
        supports = [s for r in rows for s in r['scoring_support']]
        locality = {}
        for site in sorted({s['site'] for s in supports}):
            ss = [s for s in supports if s['site'] == site]
            locality[site] = {key:distribution([s[key] for s in ss]) for key in
                ('easy_harm_rows', 'event_tracks', 'event_recordings', 'event_nonoverlap_recording_queries',
                 'event_track_mass_effective_count', 'largest_event_track_mass_share')}
        transfer = {}
        for control in ('score_only', 'old_summary', 'ordered_history'):
            counts = dict(fit_and_inner_improve=0, fit_only=0, inner_only=0, neither=0)
            for row in rows:
                fit = [row['fitting_metrics'][k]['training_metrics']['H_easy_MSE'] for k in ('history_neighbors', control)]
                held = [row['metrics'][k]['H_easy_MSE'] for k in ('history_neighbors', control)]
                i, j = fit[0] < fit[1], held[0] < held[1]
                counts['fit_and_inner_improve' if i and j else 'fit_only' if i else 'inner_only' if j else 'neither'] += 1
            transfer[control] = counts
        cap = [r['metrics']['raw']['frozen_cap_unavoidable_MSE']/r['metrics']['raw']['H_easy_MSE']
            for r in rows if r['metrics']['raw']['H_easy_MSE'] > 0]
        output[pair] = dict(dependent_views=len(rows), locality_support=locality,
            absolute_costs={arm:{key:distribution([r['metrics'][arm][key] for r in rows])
                for key in ('H_easy_MSE', 'H_all_MSE', 'actual_harm_mean', 'predicted_harm_mean', 'coverage')}
                for arm in ('raw', *run.method.ARMS)},
            fewer_than_10_event_tracks=sum(s['event_tracks'] < 10 for s in supports),
            fewer_than_10_effective_event_tracks=sum(s['event_track_mass_effective_count'] < 10 for s in supports),
            at_most_one_event_recording=sum(s['event_recordings'] <= 1 for s in supports),
            cap_unavoidable_fraction_of_raw_MSE=distribution(cap),
            cap_fraction_above_half=sum(v > .5 for v in cap), fitting_to_inner_descriptive=transfer)
    run.immutable_json(run.PUBLIC/'support_diagnostics.json', output)
    lines = ['# Event Support and Residual Ceiling', '', '## Scope',
        'Fresh diagnostic computations over frozen inner fitting-locality predictions.',
        'No refitting, threshold change or favorable subgroup selection. Counts are dependent views.', '',
        '| Family | Views | <10 event tracks | <10 harm-mass effective tracks | <=1 event recording | Cap floor / raw MSE, min / median / max |',
        '|---|---:|---:|---:|---:|---|']
    for pair, d in output.items():
        lines.append(f"| {pair} | {d['dependent_views']} | {d['fewer_than_10_event_tracks']} | {d['fewer_than_10_effective_event_tracks']} | {d['at_most_one_event_recording']} | {d['cap_unavoidable_fraction_of_raw_MSE']} |")
    lines += ['', 'The cap floor is label-derived mean(max(true easy harm - frozen predicted all harm,0)^2).',
        'It is a lower bound for this easy-only clipped probe, not all possible risk models or dynamics models.',
        'Harm-mass effective count describes concentration, not an independent sample-size or power calculation.', '',
        '## Absolute Cost Companion', '',
        'Native pixel-cost squared MSE across dependent views, not pooled trajectory ADE/FDE or an independent sample.',
        'A percentage contrast can be large when its control MSE is tiny; these values retain that denominator context.', '',
        '| Family / arm | Easy-harm MSE, min / median / max | Coverage, min / median / max |', '|---|---|---|']
    for pair, d in output.items():
        for arm, m in d['absolute_costs'].items():
            lines.append(f"| {pair} / {arm} | {m['H_easy_MSE']} | {m['coverage']} |")
    lines += ['',
        '## Fitting to Inner Transfer', '',
        'These row-weighted within-view summaries are descriptive. Training optimizes equal-locality loss,',
        'so a pooled fitting MSE change is not itself an optimization certificate.', '',
        '| Family / control | Better on both | Fitting only | Inner only | Neither |', '|---|---:|---:|---:|---:|']
    for pair, d in output.items():
        for control, counts in d['fitting_to_inner_descriptive'].items():
            lines.append(f"| {pair} / {control} | "+' | '.join(str(v) for v in counts.values())+' |')
    lines += ['', '## Locality Support', '',
        'Min/median/max include seeds, overlapping producer assignments, changing fitting cuts and outer contexts.',
        'A support count is never summed across these replicas.', '',
        '| Family / locality | Event tracks | Effective event tracks | Largest track share |', '|---|---|---|---|']
    for pair, d in output.items():
        for site, s in d['locality_support'].items():
            lines.append(f"| {pair} / {site} | {s['event_tracks']} | {s['event_track_mass_effective_count']} | {s['largest_event_track_mass_share']} |")
    lines += ['', 'Source-development only; obs8/pred12 sampled annotation steps, detector pixels.',
        'No seconds/metric/physical-safety/true3D/foundation claim. Stage5C/SMC remain off.']
    (run.PUBLIC/'support_diagnostics.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:{a:v for a,v in d.items() if a != 'locality_support'} for k,d in output.items()}, indent=2))


if __name__ == '__main__': main()
