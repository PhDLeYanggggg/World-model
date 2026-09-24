"""Post-readout numerical impact accounting; no policy or threshold modification."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.run_m3w_risk_subsidy import load
from scripts.run_m3w_native_forecast import file_digest, immutable_json, assert_current


def main():
    cfg, data, _, old, _, identity = load()
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']
    analysis = json.loads((public/'analysis.json').read_text())
    dm = json.loads((root/'decisions_complete.json').read_text())
    assert analysis['decision_manifest_sha256'] == file_digest(root/'decisions_complete.json')
    assert analysis['experiment_sha256'] == file_digest(root/'identity.json')
    arms = cfg['policies']; changes = {}; failures = []; outcomes = {}
    for ref in old['outcome_archives']:
        assert file_digest(ROOT/ref['path']) == ref['sha256']
        with np.load(ROOT/ref['path'], allow_pickle=False) as z:
            outcomes[Path(ref['path']).stem] = {k: z[k].copy() for k in ('cv', 'candidate_ade')}
    for ref in dm['receipts']:
        assert file_digest(ROOT/ref['path']) == ref['sha256']
        r = json.loads((ROOT/ref['path']).read_text())
        assert file_digest(ROOT/r['path']) == r['sha256']
        with np.load(ROOT/r['path'], allow_pickle=False) as z:
            ids, choices = z['ids'].copy(), z['choices'].copy()
        v = changes.setdefault(r['action'], dict(changed_row_seed_instances=0, changed_query_seed_instances=0))
        changed = choices[:, arms.index('net_population')] != choices[:, arms.index('parent_net_population')]
        v['changed_row_seed_instances'] += int(changed.sum())
        seed = int(r['view'].rsplit('_seed', 1)[1])
        o = outcomes[f"{r['action']}_seed{seed}"]
        for query in r['queries']:
            mask = (data['recordings'][ids] == query['recording']) & (data['frames'][ids] == query['frame'])
            v['changed_query_seed_instances'] += int(changed[mask].any())
            for mode, info in query['arms'].items():
                if not info['failed_closed']:
                    continue
                qids = ids[mask]; site = query['recording'].split('/')[0]
                supported = np.isfinite(o['cv'][qids])
                assert np.array_equal(supported, np.isfinite(o['candidate_ade'][qids]))
                denominator = np.nansum(o['cv'][data['sites'] == site])
                assert denominator > 0
                # A superset bound: any subset of all query agents, ignoring
                # causal eligibility and predicted-risk feasibility restrictions.
                absolute = np.abs(o['candidate_ade'][qids][supported]-o['cv'][qids][supported]).sum()
                bound = 100*absolute/denominator/len(cfg['sites'])/len(cfg['seeds'])
                failures.append(dict(view=r['view'], action=r['action'], recording=query['recording'],
                    frame=query['frame'], policy=mode, status=info['status'], query_rows=len(qids),
                    supported_rows=int(supported.sum()), selected=info['selected'],
                    absolute_primary_ADE_gain_impact_bound_pp=float(bound),
                    bound_scope='observed_supported_ADE_not_unknown_future_safety',
                    bound_ignores_eligibility_and_budget=True))
    assert len(failures) == sum(v['failed_closed'] for aa in analysis['solver'].values() for v in aa.values())
    assert_current(identity)
    result = dict(result_source='fresh_post_readout_diagnostic_cached_verified_decisions_and_labels',
        analysis_sha256=file_digest(public/'analysis.json'), decision_manifest_sha256=file_digest(root/'decisions_complete.json'),
        script_sha256=file_digest(Path(__file__)), same_rule_parent_changes=changes,
        solver_failure_cases=failures, policy_changed=False, independent_confirmation=False)
    immutable_json(public/'numerical_impact.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
