"""Rebuild a frozen-guard diagnostic report from public scalar evidence only."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/european_current_bank_deferral_v1'
FIELDS = (
    'known_easy_rows', 'legal_known_easy_action_occurrences',
    'guard_kept_known_easy_action_occurrences',
    'true_signed_risk_safe_known_easy_action_occurrences',
    'true_signed_risk_safe_known_easy_rejected',
    'true_beneficial_safe_known_easy_rejected',
    'known_signed_risk_unsafe_admitted_action_occurrences',
)


def summarize(result):
    rows = result['recordings']
    keys = [(r['context'], r['source'], r['seed'], r['recording']) for r in rows]
    if result['groups'] != 72 or len(rows) != 444 or len(set(keys)) != 444:
        raise ValueError('Complete unique original 72-group/444-view diagnostic required')
    blocked = [r for r in rows if r['frozen_pointwise_guard_cannot_supply_known_easy_reference']]
    singleton = [r for r in blocked if r['frozen_native_joint_cannot_supply_known_easy_reference_singleton_proof']]
    if (blocked != result['blocked_views']
            or len(blocked) != result['pointwise_support_blocked_views']
            or len(singleton) != result['native_joint_singleton_proof_views']):
        raise ValueError('Stored obstruction count differs from original recording cells')
    for row in blocked:
        if not (row['full_known_easy_reference_mass'] > 0
                and row['guard_kept_known_easy_action_occurrences'] == 0
                and row['original_legal_bank_has_known_easy_reference']):
            raise ValueError('Obstruction must retain original legal easy support')
    for row in singleton:
        if not row['known_easy_rows'] == row['known_easy_rows_in_singleton_target_queries']:
            raise ValueError('No joint impossibility inference for multi-target queries')
    totals = {k: sum(r[k] for r in rows) for k in FIELDS}
    return dict(
        result_source='public_scalar_reconstruction_of_fresh_label_aware_SOURCE_diagnostic',
        groups=72, repeated_record_views=444,
        pointwise_blocked_views=len(blocked),
        unique_blocked_source_record_pairs=len({(r['source'], r['recording']) for r in blocked}),
        native_joint_singleton_proof_views=len(singleton),
        action_occurrences_repeat_context_seed_and_candidates=True,
        all_views_totals=totals,
        blocked_views_totals={k: sum(r[k] for r in blocked) for k in FIELDS},
        overlapping_rejection_reason_occurrences={
            k: sum(r['false_rejection_reason_occurrences'][k] for r in rows)
            for k in rows[0]['false_rejection_reason_occurrences']},
        rejected_safe_easy_fraction=(totals['true_signed_risk_safe_known_easy_rejected']
                                    / totals['true_signed_risk_safe_known_easy_action_occurrences']),
        no_new_model_or_threshold=True, inference_used_future_labels=False,
        diagnostic_used_future_labels=True, independent_confirmation=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False,
    )


def main():
    path = PUBLIC / 'guard_support_diagnostic.json'
    result = json.loads(path.read_text())
    reg = PUBLIC / 'guard_diagnostic_registration.json'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if result['registration_sha256'] != sha(reg):
        raise ValueError('Diagnostic registration changed')
    registration = json.loads(reg.read_text())
    for rel, digest in dict(registration['bindings'], **registration['artifacts']).items():
        if sha(ROOT / rel) != digest:
            raise ValueError('Registered diagnostic or parent result changed')
    s = summarize(result)
    s['diagnostic_sha256'] = sha(path)
    s['registration_sha256'] = sha(reg)
    (PUBLIC / 'guard_support_statistics.json').write_text(json.dumps(s, indent=2) + '\n')
    a, b = s['all_views_totals'], s['blocked_views_totals']
    report = f"""# Frozen Guard: Two-Sided Failure Diagnosis

**fresh_run diagnostic; public-scalar report reconstruction.** All 72 original
groups and 444 repeated recording/context/seed views are retained. Actual run:
PID {result['pid']}, {result['seconds']:.6f} s, peak {result['peak_RSS_bytes']:,} B.
The diagnostic was registered after the primary readout and before these counts.
It is label-aware SOURCE failure analysis, not a new causal policy or a primary
test. Future labels are used only to identify errors; they never define eligibility.

## Observed Errors

|Quantity|Repeated action occurrences|
|---|---:|
|Legal known easy actions|{a['legal_known_easy_action_occurrences']:,}|
|True signed-risk-safe known easy actions|{a['true_signed_risk_safe_known_easy_action_occurrences']:,}|
|True safe easy actions rejected by the fixed guard|{a['true_signed_risk_safe_known_easy_rejected']:,}|
|Rejected safe easy actions with strictly positive true benefit|{a['true_beneficial_safe_known_easy_rejected']:,}|
|Known signed-risk-unsafe actions admitted, all known rows|{a['known_signed_risk_unsafe_admitted_action_occurrences']:,}|

These repeat actions across candidates, contexts and cost seeds, not independent
examples. The last row is not an easy-only false-positive rate and is not the
number actually chosen by deferral. Safety here means only the observed two
signed-risk inequalities, not certified full-outcome or physical safety. The guard
rejects {s['rejected_safe_easy_fraction']:.2%} of the available true safe easy action
occurrences. Rejection reasons overlap;their counts must not be added.

## Missing Reference Support

{s['pointwise_blocked_views']}/444 views, covering
{s['unique_blocked_source_record_pairs']} unique source/record pairs, have positive
full known easy reference but no known easy action surviving the pointwise guard.
Their original legal banks retain {b['true_signed_risk_safe_known_easy_action_occurrences']}
true signed-risk-safe easy occurrences, including
{b['true_beneficial_safe_known_easy_rejected']} with positive benefit; all are rejected.
This is not evidence that the whole candidate bank has no useful forecast.

For {s['native_joint_singleton_proof_views']} repeated views of locality082/record51,
all three known easy rows belong to singleton ORIGINAL TARGET queries. The
native query allocator cannot pool those anchors with a contemporaneous second
target to recover known easy support. A wider real-world neighborhood is not
being asserted empty. For the remaining blocked views, multi-target pooling may
change support; no joint impossibility is inferred from pointwise rejection alone.
Unknown easy outcomes cannot supply a known denominator for the unchanged
conservative certificate. This does not prove their realized reference is zero.

## Consequence For The Next Repair

Better published-loss ranking alone cannot undo an unchanged eligibility mask.
Uniformly tightening thresholds can further remove safe denominator anchors;
uniform relaxation can admit additional harm. The critic therefore needs a
two-sided repair: reduce false low-risk admissions AND recover accurately modeled
beneficial reference support. Any such fit must be separately registered, retain
the original 2% contract and be tested against the complete causal policy, not an
oracle-filtered subset. A label-aware feasible witness is a diagnosis, not a model
result. Partial-label uncertainty and observed cost errors remain separate.

The current comparison remains 5/9 fitting gates, not independently confirmed or
deployable. No new training, threshold, calibration, data role or deployment change
occurred in this diagnostic. No SDD t+50, metric, seconds-level, true 3D, foundation,
human-gold or new dynamics claim follows. Stage5C/SMC remain off.
"""
    (PUBLIC / 'guard_support_readout.md').write_text(report)
    print(json.dumps(s, indent=2))


if __name__ == '__main__':
    main()
