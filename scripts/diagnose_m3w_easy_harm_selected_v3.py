"""Describe selected-harm errors from the completed readout without new fitting."""
import argparse
import json
import math
import statistics

from scripts import report_m3w_easy_harm_readout_v3 as report

HOME = report.HOME
COMPONENTS = ('predicted_slack_pp', 'harm_underprediction_pp',
              'reference_overprediction_pp', 'realized_excess_pp')


def decompose(d, budget):
    p, y = d['predicted_moments'], d['observed_moments']
    if (budget != .02 or d['unknown_outcomes_imputed'] is not False
            or len(p) != 5 or len(y) != 5 or d['known_selected'] < 0
            or any(not math.isfinite(v) or v < 0 for v in p + y)):
        raise ValueError('Fixed nonnegative known-selected moments and 2% budget required')
    terms = (p[4] - budget*p[3], y[4] - p[4], budget*(p[3] - y[3]))
    excess = y[4] - budget*y[3]
    if not math.isclose(math.fsum(terms), excess, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('Selected-risk accounting identity failed')
    return dict(**{k: 100*v/y[3] if y[3] > 0 else None
                   for k, v in zip(COMPONENTS, (*terms, excess))},
                predicted_actual_harm_ratio=p[4]/y[4] if y[4] > 0 else None,
                known_risk_percent=100*y[4]/y[3] if y[3] > 0 else None,
                harm_underpredicted=p[4] < y[4] - 1e-12,
                known_selected=d['known_selected'])


def equal_locality_mean(values, expected=12):
    sites = {}
    for site, value in values:
        if value is None:
            return None
        sites.setdefault(site, []).append(value)
    if len(sites) != expected:
        return None
    return statistics.mean(statistics.mean(v) for v in sites.values())


def build(cfg, summary, rows):
    arms = {}
    for arm in report.run.original.api.ARMS:
        views = [dict(group=r['group'], source=r['source'], head_seed=r['head_seed'],
                      **decompose(r['result']['diagnostics'][arm], cfg['risk_budget']))
                 for r in rows]
        ratios = [v['predicted_actual_harm_ratio'] for v in views]
        risks = [v['known_risk_percent'] for v in views]
        arms[arm] = dict(views=len(views),
            harm_underpredicted_views=sum(v['harm_underpredicted'] for v in views),
            undefined_harm_ratios=sum(v is None for v in ratios),
            median_defined_predicted_actual_harm_ratio=statistics.median(
                v for v in ratios if v is not None) if any(v is not None for v in ratios) else None,
            undefined_risk_views=sum(v is None for v in risks),
            median_defined_known_risk_percent=statistics.median(
                v for v in risks if v is not None) if any(v is not None for v in risks) else None,
            components={k: equal_locality_mean([(v['source'], v[k]) for v in views])
                        for k in COMPONENTS},
            selected_occurrences=summary['safety'][arm]['selected_occurrences'],
            unknown_selected_occurrences=summary['safety'][arm]['unknown_selected_occurrences'],
            known_violations=summary['safety'][arm]['known_easy_violations'],
            completion_upper_violations=summary['safety'][arm]['easy_upper_violations'],
            per_view=views)
    return dict(result_source='fresh_run_descriptive_accounting_of_cached_verified_readout',
        evidence_role='post_readout_exposed_development_diagnostic',
        summary_sha256=report.run.sha(HOME/'summary.json'),
        complete_sha256=report.run.sha(HOME/'complete.json'),
        budget=.02, identity_checks=len(rows)*len(arms), arms=arms,
        hypothesis_test=False, calibration_claim=False, causal_attribution=False,
        new_training=False, new_inference=False, threshold_search=False,
        independent_roles_read=False, deployment_changed=False)


def render(e):
    fmt = lambda v: 'undefined' if v is None else f'{v:.6f}'
    lines = ['# Selected-Harm Failure Diagnosis', '',
        'Post-readout descriptive accounting, not a new hypothesis test or independent confirmation.',
        'Inputs are the hash-verified completed six-arm development readout; no model or threshold changed.', '',
        '| Arm | Harm underpredicted /72 | Undefined harm ratios | Median defined predicted/actual harm | Median defined known risk (%) | Known violations | Completion-upper violations |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm, r in e['arms'].items():
        lines.append(f"| {arm} | {r['harm_underpredicted_views']} | {r['undefined_harm_ratios']} | "
                     f"{fmt(r['median_defined_predicted_actual_harm_ratio'])} | "
                     f"{fmt(r['median_defined_known_risk_percent'])} | {r['known_violations']} | "
                     f"{r['completion_upper_violations']} |")
    lines += ['', 'Medians use defined per-view ratios only; undefined counts remain explicit.',
        'Different policies select different populations. A better ratio is not itself a causal effect or policy gain.', '',
        '| Arm | Predicted slack (pp) | Harm underprediction (pp) | Reference overprediction (pp) | Realized excess (pp) |',
        '|---|---:|---:|---:|---:|']
    for arm, r in e['arms'].items():
        lines.append('| '+arm+' | '+' | '.join(fmt(r['components'][k]) for k in COMPONENTS)+' |')
    lines += ['', 'Each term is divided by actual easy-reference mass within the view, then averaged',
        'within locality and across all 12 localities. Any undefined view makes that aggregate undefined.',
        'The signed sum is an accounting identity, not a new theorem or attribution of causal mechanisms.', '',
        'The deviance repair fails both the matched-quadratic advancement requirement and selected-risk checks.',
        'Known-label risk failures cannot be explained solely by pessimistic missing-outcome bounds.',
        'The complete comparison also shows that lower intervention count does not by itself establish safety.',
        'This diagnosis does not prove that all decision-focused objectives fail, or that features are sufficient.', '',
        '## Next Controlled Repair', '',
        'Test a decision-aligned loss on the same frozen forecast candidates, TRAIN roles and sampling budget.',
        'Keep quadratic/deviance controls, the existing 2% positive-easy-harm definition and fixed readout.',
        'Separate optimization of expected intervention utility from independent risk calibration.',
        'First verify gradients and a TRAIN-only pilot, then freeze the full comparison before development readout.',
        'Unknown outcomes must not become zero-cost negatives; independent selection/calibration/confirmation stay closed.',
        'Matched decision-learning training is not_run, not established by this diagnosis.', '',
        'No deployment change; no metric/seconds, true-3D or foundation-model claim. Stage5C and SMC remain off.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    cfg, _, summary, rows = report.validate()
    e = build(cfg, summary, rows)
    for name, text in {'selected_harm_diagnosis.json':json.dumps(e, indent=2, allow_nan=False)+'\n',
                       'selected_harm_diagnosis.md':render(e)}.items():
        path = HOME/name
        if path.exists():
            if path.read_bytes() != text.encode():
                raise ValueError('Preserve changed diagnosis: '+name)
        elif args.check:
            raise ValueError('Missing diagnosis: '+name)
        else:
            path.write_bytes(text.encode())
    print(json.dumps(dict(status='checked' if args.check else 'reported',
                          identity_checks=e['identity_checks'], new_training=False)))


if __name__ == '__main__':
    main()
