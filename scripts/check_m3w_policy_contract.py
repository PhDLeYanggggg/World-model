"""Hash-bound preflight of the prospective contract, without model selection."""
import argparse
import hashlib
import json
from pathlib import Path

from src.evaluation.m3w_policy_contract_preflight import assess_count_contract

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT/'outputs/publication_readiness_2026_09/european_gate_contract_v1'
SOURCE = ROOT/'outputs/publication_readiness_2026_09/european_scaled_risk_v1/readout'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence():
    reg = json.loads((SOURCE/'registration.json').read_text())
    complete = json.loads((SOURCE/'complete.json').read_text())
    if (sha(SOURCE/'summary.json') != complete['summary_sha256']
            or reg['risk_budget'] != .02 or reg['independent_roles_read']):
        raise ValueError('Changed readout or role boundary')
    for rel, digest in reg['bindings'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Changed frozen source; do not rewrite the old gate')
    expected = {(r['group'],r['source'],r['head_seed']) for r in reg['expected']}
    if len(expected) != 72:
        raise ValueError('All 72 frozen views required')
    seen = set()
    counts = []
    matched_empty = {k:0 for k in reg['primary_comparators']}
    full = []
    for ref in complete['groups']:
        path = ROOT/ref['path']
        if (path.is_symlink() or not path.resolve().is_relative_to((SOURCE/'groups').resolve())
                or sha(path) != ref['sha256']):
            raise ValueError('Changed readout aggregate')
        row = json.loads(path.read_text())
        key = row['group'],row['source'],row['head_seed']
        if key in seen or key not in expected:
            raise ValueError('Unexpected or duplicate view')
        seen.add(key)
        if row['readout_registration_sha256'] != sha(SOURCE/'registration.json'):
            raise ValueError('Changed readout registration')
        policies = row['result']['policies']
        full.append(policies['budget_scaled'])
        for arm in reg['primary_comparators']:
            count = policies[arm]['selected_count']
            counts.append(dict(group=key[0],source=key[1],seed=key[2],anchor=arm,
                               selected_count=count))
            candidate = policies['budget_scaled_matched_'+arm]
            comparator = policies[arm+'_matched_budget_scaled']
            if candidate['selected_count'] != comparator['selected_count']:
                raise ValueError('Count matching no longer holds')
            if candidate['selected_count'] == 0:
                matched_empty[arm] += 1
            if count == 0 and (candidate['selected_count'] != 0
                    or candidate['easy_selected_risk_upper'] is not None
                    or candidate['finite_completion_supported']):
                raise ValueError('Empty-anchor proof not reproduced')
    if seen != expected:
        raise ValueError('Incomplete fixed roster')
    before = assess_count_contract(counts,require_matched_finite_risk=True)
    after = assess_count_contract(counts,require_matched_finite_risk=False)
    files = [Path(__file__).resolve(),ROOT/'src/evaluation/m3w_policy_contract_preflight.py',
             ROOT/'tests/test_m3w_policy_contract_preflight.py',HOME/'prospective_protocol.md',
             SOURCE/'registration.json',SOURCE/'complete.json',SOURCE/'summary.json']
    return dict(result_source='fresh_run_structural_preflight_of_cached_verified_aggregates',
        bindings={str(path.relative_to(ROOT)):sha(path) for path in files},
        views=72,anchor_views=len(counts),anchor_roster=counts,
        old_contract=before,prospective_preflight=after,
        actual_empty_matched_views=matched_empty,
        current_candidate_known_easy_violations=sum(p['known_easy_positive_risk'] is not None
            and p['known_easy_positive_risk'] > .02+1e-12 for p in full),
        current_candidate_full_completion_support=sum(p['finite_completion_supported'] for p in full),
        prospective_candidate_evaluation='not_run_old_failed_candidate_not_promoted',
        registered_legacy_result_unchanged=True,new_training=False,new_predictions=False,
        independent_roles_read=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)


def content(value):
    counts = {arm:sum(r['anchor']==arm for r in value['old_contract']['forced_empty_comparisons'])
              for arm in value['actual_empty_matched_views']}
    table = ['| Anchor | Empty full-anchor views /72 | Empty matched views /72 |',
             '|---|---:|---:|']
    for arm,count in counts.items():
        table.append(f"| {arm} | {count} | {value['actual_empty_matched_views'][arm]} |")
    return f'''# Prospective Gate Preflight

Fresh_run structural check of 72 cached_verified frozen development aggregates.
No new predictions, fit, threshold search or independent-role access occurred.
The full 288-anchor roster and source hashes are recorded in evidence.json.

{chr(10).join(table)}

The original forest is empty in {counts['original']}/72 views. Exact same-count
matching must then be empty on both sides: zero reference makes selected risk
undefined, not zero. The old requirement that every matched candidate have
finite risk is structurally impossible. This was an evaluation-design error,
not a neural optimization target. The admission helper rejects that contract.

The user-approved prospective contract separates comparison utility from actual
deployment safety. It keeps the same 2% full-policy requirements, all original
full/matched utility comparisons and all zero-count views. The preflight now has
no zero-anchor contradiction; it does not establish that a model can pass the
remaining gates. Empty comparison risk remains not estimable. No old gate code,
registration, score or failed verdict was changed.

The scaled candidate still has {value['current_candidate_known_easy_violations']}/72
known easy-risk violations and {value['current_candidate_full_completion_support']}/72
full finite-completion supports in the frozen development result. These are real
model/support failures, independent of the protocol error. The candidate is not
re-evaluated or promoted under the new rule. Deployment remains unchanged.

```bash
.venv-pytorch/bin/python -m scripts.check_m3w_policy_contract --check
```

The executable future gate rejects incomplete rosters, unknown/empty full-policy
support, nonfinite risk, any unchanged-budget violation or a nonpositive primary
utility interval. A passed development gate still does not imply independent
confirmation, population safety or submission readiness. Main observation8 /
prediction12, raw-frame and dataset-local conventions remain unchanged. Not
true3D or foundation-scale; no metric/seconds claim. Stage5C and SMC stay disabled.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    value = evidence()
    for name, text in {'evidence.json':json.dumps(value,indent=2,allow_nan=False)+'\n',
                       'conclusions.md':content(value)}.items():
        path = HOME/name
        if args.check or path.exists():
            if path.read_text() != text:
                raise ValueError('Preserve conflicting preflight evidence')
        else:
            with path.open('x') as stream:
                stream.write(text)
    print(json.dumps(dict(status='checked' if args.check else 'reported',
                         legacy_structural_obstruction=value['old_contract']['structural_obstruction_detected'],
                         scientific_success=False,deployment_changed=False)))


if __name__ == '__main__':
    main()
