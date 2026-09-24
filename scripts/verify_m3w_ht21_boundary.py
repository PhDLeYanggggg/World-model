"""Real-annotation prefix-invariance and source-audit admission checks."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_m3w_ht21_annotations import ARCHIVE, PUBLIC, sha
from src.evaluation.m3w_ht21_admission import admission, require_role
from src.evaluation.m3w_ht21_intake import history, human_observation_mask, read_ground_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    audit = json.loads((PUBLIC/'analysis.json').read_text())
    if sha(ARCHIVE.read_bytes()) != audit['archive_sha256']:
        raise ValueError('Source archive changed')
    checks = []
    with zipfile.ZipFile(ARCHIVE) as z:
        for recording in audit['recordings']:
            if not recording['ground_truth_present']:
                continue
            payload = z.read(f"HT21Labels/train/{recording['sequence']}/gt/gt.txt")
            if sha(payload) != recording['ground_truth_sha256']:
                raise ValueError('Annotation bytes changed')
            rows = read_ground_truth(payload, recording['metadata'])
            visible = rows[human_observation_mask(rows)]
            for fraction in (.25, .5, .75):
                query = int(recording['metadata']['frames']*fraction)
                agents = visible[visible[:,0] == query,1]
                if not len(agents):
                    raise ValueError('Fixed audit query has no visible humans')
                for agent in (int(agents.min()), int(agents.max())):
                    prefix = rows[rows[:,0] <= query]
                    poisoned = rows.copy()
                    poisoned[poisoned[:,0] > query, 2:6] = 1e6
                    static_flip = rows.copy()
                    static_flip[np.isin(static_flip[:,7], [1,2]),7] = 1
                    for length in (8,16,32,64):
                        expected = history(rows, agent, query, length=length)
                        for altered in (prefix, poisoned, static_flip):
                            actual = history(altered, agent, query, length=length)
                            for key in expected:
                                np.testing.assert_array_equal(expected[key], actual[key])
                        checks.append(dict(sequence=recording['sequence'], query_frame=query,
                                           agent_id=agent, length=length,
                                           valid_observations=int(expected['valid_mask'].sum())))
    decisions = []
    for recording in audit['recordings']:
        require_role(recording, 'source_audit')
        for role in ('supervised_training','representation_pretraining','risk_calibration','official_eval','confirmation'):
            try:
                require_role(recording, role)
            except ValueError:
                pass
            else:
                raise AssertionError('Unregistered scientific role admitted')
        decisions.append(admission(recording, 'source_audit'))
    result = dict(result_source='fresh_run', all_checks_passed=True,
        analysis_sha256=sha((PUBLIC/'analysis.json').read_bytes()),
        checked_histories=len(checks), history_checks=checks, admission=decisions,
        producer_level_online_causality_proven=False,
        scope='past-indexed interface invariance only; not annotation-producer causality',
        blockers_are_provenance_not_a_request_for_routine_user_audit=True,
        deployment_changed=False, training=False, forecasting_metrics_computed=False,
        implementation_sha256={p:sha((ROOT/p).read_bytes()) for p in (
            'scripts/verify_m3w_ht21_boundary.py','src/evaluation/m3w_ht21_admission.py',
            'src/evaluation/m3w_ht21_intake.py','tests/test_m3w_ht21_admission.py')})
    path = PUBLIC/'boundary_verification.json'
    if args.verify:
        if result != json.loads(path.read_text()):
            raise ValueError('Boundary replay differs')
        print('cached_verified: boundary replay exact')
    else:
        with path.open('x') as f:
            f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(checked_histories=len(checks), role_admissions=0, all_checks_passed=True)))


if __name__ == '__main__':
    main()
