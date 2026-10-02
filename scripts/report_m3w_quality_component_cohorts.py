"""Posthoc locality-level decomposition, with no component/model selection."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_quality_components_v1'


def cohort_summary(rows, cohort):
    sites = {}; n = unknown = defined_groups = 0
    fields = ('actual_known_easy_risk_percent', 'predicted_excess_pp',
              'harm_underestimate_pp', 'reference_inflation_pp')
    for r in rows:
        c = r['result']['arms']['quality']['cohorts'][cohort]
        n += c['selected']; unknown += c['unknown']
        den = c['known_truth_moments'][3]
        if den > 0:
            defined_groups += 1
            e = c['easy']
            values = [c['known_truth_moments'][4], e['predicted_excess_mass'],
                      e['harm_underestimate_mass'], e['reference_inflation_budget_mass']]
            sites.setdefault(r['source'], []).append([100*x/den for x in values])
    localities = {s: {k: math.fsum(row[j] for row in v)/len(v) for j, k in enumerate(fields)}
                  for s, v in sorted(sites.items())}
    means = {k: math.fsum(v[k] for v in localities.values())/len(localities)
             if localities else None for k in fields}
    if localities:
        assert math.isclose(means[fields[0]], 2+math.fsum(means[k] for k in fields[1:]),
                            abs_tol=1e-8, rel_tol=1e-10)
    return dict(selected_occurrences=n, unknown_selected=unknown, defined_groups=defined_groups,
                defined_localities=len(localities), locality_means=localities,
                equal_locality_mean=means, unknown_harm_not_observed=True)


def main():
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    v = json.loads((PUBLIC/'verification.json').read_text())
    assert sha(PUBLIC/'complete.json') == v['complete_sha256']
    assert sha(PUBLIC/'summary.json') == v['summary_sha256']
    c = json.loads((PUBLIC/'complete.json').read_text()); rows = []
    for ref in c['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        rows.append(json.loads((ROOT/ref['path']).read_text()))
    assert len(rows) == 72
    fields = ('nonpositive_utility', 'all_risk', 'easy_risk')
    result = dict(status='fresh_run_posthoc_descriptive_reduction_no_selection',
                  source_verification_sha256=sha(PUBLIC/'verification.json'),
                  cohorts={k: cohort_summary(rows, k) for k in ('added', 'removed', 'retained')},
                  original_failed_constraints_on_quality_added={k: sum(
                      r['result']['arms']['quality']['original_failed_constraints_on_added'][k]
                      for r in rows) for k in fields},
                  constraint_counts_overlap=True, independent_confirmation=False)
    text = json.dumps(result, indent=2)+'\n'; path = PUBLIC/'cohort_decomposition.json'
    if path.exists(): assert path.read_text() == text
    else:
        with path.open('x') as f: f.write(text)
    print(json.dumps({k: {f: value for f, value in d.items() if f != 'locality_means'}
                      for k, d in result['cohorts'].items()}, indent=2))


if __name__ == '__main__': main()
