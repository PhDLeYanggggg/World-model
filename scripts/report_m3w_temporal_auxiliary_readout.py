"""Render the complete frozen readout, including failed and undefined comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT/'outputs/publication_readiness_2026_09/european_temporal_auxiliary_v1/readout'
COMPARATORS = ('original', 'additive', 'poisson', 'cost', 'none', 'rowmean')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(row):
    if row['CI95'] is None:
        return 'undefined ('+row['status']+')'
    lo, hi = row['CI95']
    return f"{row['mean']:+.6f} [{lo:+.6f}, {hi:+.6f}]"


def diagnose(rows):
    keys = [(r['group'], r['source'], r['head_seed']) for r in rows]
    if len(rows) != 72 or len(set(keys)) != 72:
        raise ValueError('Full unique development grid required')
    risk, decomposition = {}, {}
    for arm in ('original', 'additive', 'poisson', 'cost', 'none', 'rowmean', 'temporal'):
        policies = [r['result']['policies'][arm] for r in rows]
        known = [p['known_easy_positive_risk'] for p in policies if p['known_easy_positive_risk'] is not None]
        risk[arm] = dict(known_defined=len(known), known_violations=sum(v > .02+1e-12 for v in known),
            known_worst=max(known) if known else None, known_median=statistics.median(known) if known else None,
            upper_violations_with_known_safe=sum(p['easy_selected_risk_upper'] is not None
                and p['easy_selected_risk_upper'] > .02+1e-12 and p['known_easy_positive_risk'] is not None
                and p['known_easy_positive_risk'] <= .02+1e-12 for p in policies))
    for arm in ('original', 'none', 'rowmean'):
        decomposition[arm] = {}
        for mode in ('full', 'matched'):
            localities = {}
            for row in rows:
                result = row['result']; pair = result['pairs'][arm+'_'+mode]
                den = result['full_known_reference_mass']
                if den <= 0: raise ValueError('Undefined known-reference decomposition; do not drop')
                localities.setdefault(row['source'], []).append((100*pair['known_difference_mass']/den,
                    100*pair['unknown_exchanged_envelope_mass']/den))
            known = statistics.mean(statistics.mean(v[0] for v in pp) for pp in localities.values())
            penalty = statistics.mean(statistics.mean(v[1] for v in pp) for pp in localities.values())
            decomposition[arm][mode] = dict(known_utility_difference_mean=known,
                unknown_disagreement_penalty_mean=penalty, paired_lower_mean=known-penalty)
    missing = [{k: r[k] for k in ('group', 'source', 'head_seed')} for r in rows
               if r['result']['scores']['temporal']['original_selected']['conditional_MSE'] is None]
    return dict(result_source='cached_verified_aggregate_decomposition', risk=risk,
                paired_utility_decomposition=decomposition, undefined_common_cohort=missing,
                risk_budget=.02, counts_are_repeated_views=True, new_training=False,
                new_forecast_evaluation=False, policy_changed=False, independent_confirmation=False)


def render(summary, complete):
    if (summary['groups'] != 72 or summary['localities'] != 12 or summary['trained_cost_heads'] != 216
            or summary['independent_confirmation'] or summary['deployment_changed']
            or summary['transfer_evaluated'] or summary['stage5c_executed'] or summary['smc_enabled']):
        raise ValueError('Expected exposed-development experiment and claim boundary changed')
    rows = [
        '# Temporal Auxiliary: Complete Development Readout', '',
        'Result source: `fresh_run` fixed-final development predictions, scalar checks',
        'and 3000 paired locality bootstrap draws. Models and original controls are',
        '`cached_verified`. All 216 heads completed 2000 updates before this readout.',
        'There are 72 source/head-seed views over 12 exposed localities, not 72',
        'independent scenes. Head seeds are not independently retrained forecasters.', '',
        '## Registered Decision', '',
        'Advance to transfer design: **'+str(summary['advance_to_transfer_design']).lower()+'**.',
        'Deployment remains unchanged. Independent calibration and confirmation remain closed.',
        'No validation threshold, checkpoint, architecture or comparator selection was performed.', '',
        '## Primary Cost and Paired Utility', '',
        'Each entry is temporal minus comparator, mean [nominal 95% CI]. Negative',
        'signed-score MSE is better; positive paired-completion utility is better.',
        'Utility is percent of full known reference cost, not raw FDE improvement.',
        'These are exposed-development, non-multiplicity-adjusted intervals. Missing',
        'support stays undefined; no head is dropped to manufacture a CI.', '',
        '| Comparator | All MSE | Original-selected MSE | Full paired lower utility | Matched paired lower utility |',
        '|---|---|---|---|---|']
    for arm in COMPARATORS:
        suffixes = ('all_MSE', 'original_selected_MSE', 'full_paired_lower_percent', 'matched_paired_lower_percent')
        rows.append('| '+arm+' | '+' | '.join(interval(summary['contrasts'][arm+'_'+s]) for s in suffixes)+' |')
    rows += ['', '## Lower-Proxy Differences', '',
        'These are differences of lower bounds, not bounds on paired differences.',
        'They remain separate from the same-outcome paired completion intervals above.', '',
        '| Comparator | Full lower-proxy difference | Matched lower-proxy difference |', '|---|---|---|']
    for arm in COMPARATORS:
        rows.append('| '+arm+' | '+' | '.join(interval(summary['contrasts'][arm+'_'+s+'_lower_proxy_delta_percent'])
                                              for s in ('full', 'matched'))+' |')
    rows += ['', '## Safety and Support', '',
        'Counts below are repeated source/seed occurrences, not distinct agents or',
        'independent windows. The registered budget is 2% selected positive easy',
        'harm/reference. Undefined reference support is not a passing safety result.', '',
        '| Policy | Selected occurrences | Unknown selected | Undefined easy risk /72 | Easy-upper violations /72 | Worst easy upper | Complete finite support /72 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm, row in summary['safety'].items():
        worst = 'undefined' if row['worst_easy_upper'] is None else f"{100*row['worst_easy_upper']:.6f}%"
        rows.append(f"| {arm} | {row['selected_occurrences']} | {row['unknown_selected_occurrences']} | "
                    f"{row['undefined_easy_risk']} | {row['easy_upper_violations']} | {worst} | {row['finite_completion_supported']} |")
    rows += ['', '## Unmet Conditions', '']
    rows += ['- '+reason for reason in summary['failure_reasons']] or ['None in this development screen; not independent confirmation.']
    rows += ['', '## Verification and Limits', '',
        f"- {complete['independent_scalar_checks']} scalar cross-checks; original-control replay and repeated neural inference match exactly.",
        f"- Readout process elapsed {complete['seconds']:.2f} seconds; peak RSS {complete['peak_RSS_bytes']/2**30:.3f} GiB.",
        '- No checkpoint disk cache, new training, transfer evaluation or deployment change during readout.',
        '- Image-local detector-silver obs8/pred12 at raw stride12; no meter/seconds/human-gold/physical-safety claims.',
        '- Cost-head learning around frozen predictors is not new world-dynamics pretraining, true 3D, or foundation-model evidence.',
        '- Stage5C and SMC remain disabled.', '']
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, default=HOME)
    args = parser.parse_args(); home = args.home
    summary = json.loads((home/'summary.json').read_text())
    complete = json.loads((home/'complete.json').read_text())
    reader = json.loads((home/'create_reader_complete.json').read_text())
    if (sha(home/'summary.json') != complete['summary_sha256']
            or sha(home.parent/'training_freeze.json') != complete['training_freeze_sha256']
            or not reader['no_checkpoint_disk_cache'] or not reader['original_evaluator_unchanged']):
        raise ValueError('Complete checksum-verified run required')
    rows = []
    for ref in complete['groups']:
        if sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('Changed group output')
        rows.append(json.loads((ROOT/ref['path']).read_text()))
    diagnostic = diagnose(rows)
    dpath = home/'failure_diagnostic.json'
    raw = json.dumps(diagnostic, indent=2)+'\n'
    if dpath.exists() and dpath.read_text() != raw: raise ValueError('Preserve different prior diagnosis')
    dpath.write_text(raw)
    text = render(summary, complete)
    path = home/'report.md'
    if path.exists() and path.read_text() != text: raise ValueError('Preserve existing different report')
    path.write_text(text)
    print(json.dumps(dict(report=str(path), registered_screen_pass=summary['advance_to_transfer_design'],
                          new_evaluation=False, numerical_source='cached_verified')))


if __name__ == '__main__': main()
