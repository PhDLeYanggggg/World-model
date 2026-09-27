"""Retain every dependent inner screen; uncertainty clusters by source locality."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))


def difference(a, b):
    if a['status'] != 'measured' or b['status'] != 'measured':
        return dict(primary=None, top10_pp=None, coverage_log=None, all_harm=None)
    gain = lambda x, y: 100*(y-x)/y if y > 0 else None
    x, y = a['coverage'], b['coverage']
    ta, tb = a['top10_mass_share'], b['top10_mass_share']
    return dict(primary=gain(a['H_easy_MSE'], b['H_easy_MSE']),
        top10_pp=100*(ta-tb) if ta is not None and tb is not None else None,
        coverage_log=float(abs(np.log(y))-abs(np.log(x))) if x is not None and y is not None and min(x, y) > 0 else None,
        all_harm=gain(a['H_all_MSE'], b['H_all_MSE']))


def contrasts(rows, cfg):
    result = {}
    for pair in ('full', 'motion_only'):
        result[pair] = {}
        for new, old in cfg['comparisons']:
            key = new+'_vs_'+old; result[pair][key] = {}
            for producer, controller in sorted({(r['input']['producer'], r['input']['controller']) for r in rows}):
                group = [r for r in rows if r['input']['pair'] == pair and
                    (r['input']['producer'], r['input']['controller']) == (producer, controller)]
                sites = sorted({r['input']['inner'] for r in group})
                if len(sites) != 4: raise ValueError('All four scoring localities required')
                output = {}
                for metric in ('primary', 'top10_pp', 'coverage_log', 'all_harm'):
                    values = []
                    for site in sites:
                        seed_values = []
                        for seed in cfg['seeds']:
                            views = [r for r in group if r['input']['inner'] == site and r['input']['seed'] == seed]
                            if len(views) != 3 or len({r['input']['outer'] for r in views}) != 3:
                                raise ValueError('Exactly three excluded-outer contexts per locality and seed required')
                            ds = [difference(r['metrics'][new], r['metrics'][old])[metric] for r in views]
                            seed_values.append(float(np.mean(ds)) if all(v is not None and np.isfinite(v) for v in ds) else None)
                        values.append(float(np.mean(seed_values)) if all(v is not None for v in seed_values) else None)
                    if any(v is None for v in values):
                        output[metric] = dict(status='not_estimable', reason='missing_support_not_dropped')
                    else:
                        rng = np.random.default_rng(cfg['bootstrap_seed']); a = np.asarray(values)
                        bootstrap = a[rng.integers(0, 4, (cfg['bootstrap_draws'], 4))].mean(1)
                        ci = np.quantile(bootstrap, [.025, .975]).tolist()
                        output[metric] = dict(status='measured', point=float(a.mean()), CI=ci,
                            locality_points=dict(zip(sites, values)), localities=4, seeds=3,
                            outer_contexts_per_site_seed=3,
                            sign='positive' if ci[0] > 0 else 'negative' if ci[1] < 0 else 'overlap')
                result[pair][key][f'producer{producer}_controller{controller}'] = output
    return result


def summarize(cs):
    output = {}
    for pair, comparisons in cs.items():
        output[pair] = {}
        for name, groups in comparisons.items():
            output[pair][name] = {}
            for metric in ('primary', 'top10_pp', 'coverage_log', 'all_harm'):
                values = [g[metric] for g in groups.values()]; good = [v for v in values if v['status'] == 'measured']
                output[pair][name][metric] = dict(
                    positive=sum(v['sign'] == 'positive' for v in good), negative=sum(v['sign'] == 'negative' for v in good),
                    overlap=sum(v['sign'] == 'overlap' for v in good), not_estimable=len(values)-len(good),
                    point_range=[min(v['point'] for v in good), max(v['point'] for v in good)] if good else None)
    return output


def main():
    from scripts import run_m3w_european_temporal_support as run
    cfg, _ = run.registration(); readout = json.loads((run.PRIVATE/'readout.json').read_text())
    rows = readout['rows']; assert len(rows) == 432
    cs = contrasts(rows, cfg); summary = summarize(cs)
    planned = [summary['full']['history_neighbors_vs_'+s] for s in ('score_only', 'old_summary')]
    primary = all(c['primary']['positive'] >= 2 and c['primary']['negative'] == c['primary']['not_estimable'] == 0 for c in planned)
    guards = all(c[k]['negative'] == c[k]['not_estimable'] == 0 for c in planned for k in ('top10_pp', 'coverage_log', 'all_harm'))
    gates = dict(fitting_only_screen_complete=True, exclusion_checked=True, primary_screen=primary,
        protection_screen=guards, outer_context_study_justified=primary and guards,
        independent_confirmation=False, deployment_changed=False, new_neural_training=False,
        stage5c_executed=False, smc_enabled=False)
    support = [s for r in rows for s in r['scoring_support']]
    fields = ('supported_rows', 'easy_harm_rows', 'recordings', 'event_recordings', 'tracks', 'event_tracks',
        'agent_queries', 'event_agent_queries', 'event_scene_queries', 'event_nonoverlap_recording_queries',
        'event_track_mass_effective_count', 'largest_event_track_mass_share')
    support_summary = {f:dict(min=min(s[f] for s in support if s[f] is not None),
        median=float(np.median([s[f] for s in support if s[f] is not None])),
        max=max(s[f] for s in support if s[f] is not None)) for f in fields}
    cap = []
    for row in rows:
        m = row['metrics']['raw']
        cap.append(dict(tag=row['input']['tag'], inner=row['input']['inner'],
            unavoidable_fraction_of_raw_MSE=m['frozen_cap_unavoidable_MSE']/m['H_easy_MSE'] if m['H_easy_MSE'] else None,
            harm_above_cap_rows=m['harm_above_cap_rows']))
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json', dict(contrasts=cs, summary=summary,
        scoring_support_dependent_view_ranges=support_summary, cap_diagnostics=cap,
        cohort=readout['cohort'], detailed_metrics=run.artifact(run.PRIVATE/'readout.json')))
    run.immutable_json(run.PUBLIC/'gates.json', gates)
    lines = ['# Temporal Context Information Screen', '', '## Material Passport',
        'Fresh fitting-only ridge probes; cached-verified neural nuisance models. No neural retraining.',
        'No outer-source scoring or independent evaluation. Detector pixels, obs8/pred12 annotation steps.', '',
        '## Results', '', '| Pair / contrast | Metric | Positive / negative / overlap / missing intervals | Point range |',
        '|---|---|---|---|']
    for pair, comparisons in summary.items():
        for name, metrics in comparisons.items():
            for metric, v in metrics.items():
                lines.append(f"| {pair} / {name} | {metric} | {v['positive']} / {v['negative']} / {v['overlap']} / {v['not_estimable']} | {v['point_range']} |")
    lines += ['', 'Primary is expected easy-harm MSE gain, not trajectory improvement or easy degradation.',
        'Three outer-context differences are averaged within locality and seed, then three seeds within locality.',
        '3000 paired resamples of four localities. Six assignments overlap; intervals are exploratory, not multiplicity adjusted.',
        'Unknown or zero-denominator comparisons are retained as not_estimable, not dropped.', '', '## Support', '',
        '| Quantity | Minimum / median / maximum across dependent scoring views |', '|---|---|']
    for field, value in support_summary.items(): lines.append(f"| {field} | {value} |")
    lines += ['', 'Counts above are not summed across overlapping cuts, seeds, producer assignments or outer contexts.',
        'Non-overlap intervals and harm-mass effective counts are descriptive, not statistical sample-size estimates.',
        'Distinct physical cohort: '+json.dumps(readout['cohort'], sort_keys=True), '', '## Gates', '',
        *['- '+k+': '+str(v).lower() for k, v in gates.items()], '',
        'Cap diagnostics quantify the loss no easy-only correction can avoid while retaining the frozen all-harm cap.',
        'They use evaluation labels only for analysis; never for inputs or intervention.',
        'No metric/seconds, physical safety, human-gold, true3D or foundation claims. Stage5C/SMC remain off.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=summary, gates=gates), indent=2))


if __name__ == '__main__': main()
