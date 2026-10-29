"""Decompose frozen decisions without training, policy changes or new data reads."""
import argparse
from collections import defaultdict
import json

import numpy as np


def components(result, other, mode):
    denominator = result['full_known_reference_mass']
    policies = result['policies'] if mode == 'full' else result['matched'][other]
    left, right = policies[other], policies['peer']
    pair = result['pairs'][other+'_'+mode]
    if denominator <= 0:
        raise ValueError('A positive full known reference mass is required')
    benefit = 100*(right['selected_known_benefit_mass']-left['selected_known_benefit_mass'])/denominator
    harm = 100*(right['selected_known_harm_mass']-left['selected_known_harm_mass'])/denominator
    penalty = 100*pair['unknown_exchanged_envelope_mass']/denominator
    total = benefit-harm-penalty
    np.testing.assert_allclose(total, result['contrasts'][other+'_'+mode], atol=1e-10)
    return dict(known_benefit_delta_pp=benefit, known_harm_delta_pp=harm,
                unknown_penalty_pp=penalty, paired_lower_utility_pp=total)


def locality_mean(entries):
    if not entries:
        raise ValueError('No observations')
    grouped = defaultdict(list)
    for source, entry in entries:
        grouped[source].append(entry)
    fields = tuple(entries[0][1])
    per_source = {s: {k: float(np.mean([r[k] for r in rows])) for k in fields}
                  for s, rows in sorted(grouped.items())}
    return dict(locality_mean={k: float(np.mean([r[k] for r in per_source.values()])) for k in fields},
                per_locality=per_source)


def calculate():
    from scripts import run_m3w_query_context_risk as run
    reg, _, _ = run.admission()
    complete = json.loads((run.PUBLIC/'complete.json').read_text())
    if run.sha(run.PUBLIC/'summary.json') != complete['summary_sha256']:
        raise ValueError('Summary changed')
    rows = []
    for ref in complete['readouts']:
        path = run.ROOT/ref['path']
        if run.sha(path) != ref['sha256']:
            raise ValueError('Readout changed')
        rows.append(json.loads(path.read_text()))
    if len(rows) != 72 or {(r['context'], r['source'], r['seed']) for r in rows} != run.expected(reg):
        raise ValueError('Incomplete frozen roster')
    decomposition = {other+'_'+mode: locality_mean([
        (r['source'], components(r['result'], other, mode)) for r in rows])
        for other in run.evaluation.CONTROLS for mode in ('full', 'matched')}
    diagnostics = {}
    for arm in run.evaluation.ARMS:
        entries = []
        for r in rows:
            result = r['result']; d = result['selected_diagnostics'][arm]
            den = result['full_known_reference_mass']
            values = {'conditional_MSE': result['scores'][arm]}
            for prefix, moments in [('predicted', d['predicted_moments']), ('observed', d['observed_moments'])]:
                for field, value in zip(('benefit', 'harm', 'reference', 'easy_reference', 'easy_harm'), moments):
                    values[prefix+'_'+field+'_full_reference_pp'] = 100*value/den
            entries.append((r['source'], values))
        diagnostics[arm] = locality_mean(entries)
    return dict(result_source='fresh_run_decomposition_of_hash_verified_frozen_readouts',
        summary_sha256=complete['summary_sha256'], views=72, localities=12,
        decomposition=decomposition, selected_diagnostics=diagnostics,
        denominator='Full known reference mass within view, then equal locality mean; not pooled raw units',
        unknown_outcomes_imputed=False, new_training=False, policies_changed=False,
        independent_roles_read=False, deployment_changed=False,
        limitation='Post-readout descriptive decomposition, not a newly independent test or causal attribution')


if __name__ == '__main__':
    from scripts import run_m3w_query_context_risk as run
    p = argparse.ArgumentParser(); p.add_argument('--check', action='store_true'); a = p.parse_args()
    doc = calculate(); path = run.PUBLIC/'diagnostics.json'
    if a.check:
        if json.loads(path.read_text()) != doc:
            raise ValueError('Diagnostic replay changed')
    else:
        run.once(path, doc)
    print(json.dumps({k: v['locality_mean'] for k, v in doc['decomposition'].items()}))
