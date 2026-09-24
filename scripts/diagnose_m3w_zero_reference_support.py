"""Frozen-policy failure diagnosis; no target is supplied to motion features."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_net_easy_moment_guarded import load
from scripts.run_m3w_native_forecast import file_digest, immutable_json, assert_current, array_hash
from src.world_model.m3w_causal_motion_support import motion_profile
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/zero_reference_support_v2'


def native_history(pack):
    data, context = pack[1][1], pack[1][2]
    n = len(data['sites']); history = np.zeros((n, 8, 2)); offsets = np.zeros((n, 8), np.int64)
    seen = np.zeros(n, bool)
    for rec in context['records']:
        assert file_digest(ROOT/rec['cache']['path']) == rec['cache']['sha256']
        with np.load(ROOT/rec['cache']['path'], allow_pickle=False) as z:
            valid = z['context_target_rows'] >= 0
            ids = z['context_target_rows'][valid]
            assert not seen[ids].any() and z['context_history_mask'][valid].all()
            np.testing.assert_array_equal(z['context_frame_ids'][valid], data['frames'][ids])
            np.testing.assert_array_equal(z['context_agent_ids'][valid],
                [int(v.rsplit(':', 1)[1]) for v in data['tracks'][ids]])
            assert np.all(data['recordings'][ids] == rec['recording'])
            history[ids], offsets[ids] = z['context_history'][valid], z['context_history_offsets'][valid]
            seen[ids] = True
    assert seen.all() and np.all(offsets == np.arange(-7, 1)*12)
    np.testing.assert_array_equal(history[:, -1], data['origin'])
    return history, offsets


def main():
    start = time.monotonic(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    print(json.dumps(dict(pid=os.getpid(), state='verify_frozen_inputs')), flush=True)
    pack = load(); cfg, ap, _, _, identity = pack
    data = ap[1]; n = len(data['sites'])
    source = ROOT/cfg['reports']/'analysis.json'
    assert file_digest(source) == '6780efc6c4b4a3759229ad157866a5fb1ed894ade1e0b30af99c7e9ebd004a31'
    analysis = json.loads(source.read_text())
    history, offsets = native_history(pack); profile = motion_profile(history, offsets)
    manifest_path = ROOT/cfg['output']/'decisions_complete.json'
    assert file_digest(manifest_path) == analysis['decision_manifest_sha256']
    assert file_digest(ROOT/cfg['output']/'identity.json') == analysis['experiment_sha256']
    decision = json.loads(manifest_path.read_text())
    assert decision['experiment_sha256'] == analysis['experiment_sha256']
    score = {(s, a):dict(seen=np.zeros(n, bool), selected=np.zeros(n, bool),
        **{k:np.zeros(n) for k in ('distance', 'gain', 'positive_risk', 'net_risk', 'denominator',
                                  'predicted_zero_proxy', 'query_budget', 'query_signed_sum', 'query_negative_sum')})
        for s in cfg['seeds'] for a in cfg['actions']}
    col = cfg['policies'].index('net_population')
    for ref in decision['receipts']:
        assert file_digest(ROOT/ref['path']) == ref['sha256']
        r = json.loads((ROOT/ref['path']).read_text()); assert file_digest(ROOT/r['path']) == r['sha256']
        g = score[int(r['view'].rsplit('_seed', 1)[1]), r['action']]
        with np.load(ROOT/r['path'], allow_pickle=False) as z:
            ids, bits = z['ids'], z['choices'][:, col]
            assert not g['seen'][ids].any(); g['seen'][ids] = True; g['selected'][ids] = bits
            for k in ('distance', 'gain', 'positive_risk', 'net_risk', 'denominator'): g[k][ids] = z[k]
            # Easy-event probability is not probability of zero reference error.
            g['predicted_zero_proxy'][ids] = z['fractions'][:, 3]
            for q in r['queries']:
                mask = (data['recordings'][ids] == q['recording']) & (data['frames'][ids] == q['frame'])
                selected = mask & bits
                g['query_budget'][ids[mask]] = .02*z['denominator'][mask].sum()
                g['query_signed_sum'][ids[mask]] = z['net_risk'][selected].sum()
                g['query_negative_sum'][ids[mask]] = np.minimum(z['net_risk'][selected], 0).sum()
    assert all(v['seen'].all() for v in score.values())
    records, cases, supports = [], [], []
    for action in cfg['actions']:
        for seed in cfg['seeds']:
            ref = next(v for v in ap[3]['outcome_archives'] if v['path'].endswith(f'{action}_seed{seed}.npz'))
            assert file_digest(ROOT/ref['path']) == ref['sha256']
            with np.load(ROOT/ref['path'], allow_pickle=False) as z:
                cv, e, full, hard, easy = [z[k].copy() for k in ('cv', 'candidate_ade', 'complete', 'hard', 'positive_easy')]
            g = score[seed, action]; zero = full & (cv == 0); harmed = zero & g['selected'] & (e > 0)
            expected = analysis['summary'][action+'__net_population']['seeds'][str(seed)]['zero_CV_harmed']
            assert int(harmed.sum()) == expected
            for site in cfg['sites']:
                for name, mask in [('all', np.ones(n, bool)), ('zero_CV', zero), ('positive_easy', easy),
                                   ('hard', hard), ('unknown_or_partial', ~full)]:
                    use = mask & (data['sites'] == site)
                    records.append(dict(action=action, seed=seed, site=site, subset=name,
                        rows=int(use.sum()), selected=int((use & g['selected']).sum()),
                        last_stop=int((use & profile['last_stop']).sum()),
                        exact_past_cv=int((use & profile['exact_past_cv']).sum()),
                        exact_past_cv_selected=int((use & profile['exact_past_cv'] & g['selected']).sum()),
                        zero_harmed=int((use & harmed).sum())))
                # Counts only, not independent training trials or OOF calibration.
                outside = data['sites'] != site
                for name, mask in [('zero_CV_moving', zero & ~profile['last_stop']),
                                   ('zero_CV_moving_exact_past_cv', zero & ~profile['last_stop'] & profile['exact_past_cv'])]:
                    use = outside & mask
                    supports.append(dict(outer=site, action=action, seed=seed, subset=name,
                        exposed_source_outcome_support_rows=int(use.sum()), exposed_source_outcome_support_tracks=len(set(data['tracks'][use])),
                        role='source_population_support_not_rebuilt_nested_cost_labels'))
            for i in np.flatnonzero(harmed):
                cases.append(dict(action=action, seed=seed, row=int(i), site=str(data['sites'][i]),
                    recording=str(data['recordings'][i]), frame=int(data['frames'][i]), track=str(data['tracks'][i]),
                    candidate_ADE=float(e[i]),
                    **{k:bool(v[i]) if v.dtype == bool else float(v[i]) for k,v in profile.items()},
                    **{k:float(g[k][i]) for k in ('distance', 'gain', 'positive_risk', 'net_risk', 'denominator',
                                                 'query_budget', 'query_signed_sum', 'query_negative_sum')},
                    predicted_easy_probability=float(g['predicted_zero_proxy'][i]),
                    individual_signed_rule_pass=bool(g['net_risk'][i] <= .02*g['denominator'][i])))
    assert_current(identity)
    result = dict(result_source='fresh_postdecision_diagnosis_cached_verified_inputs_no_policy_change',
        parent_analysis_sha256=file_digest(source), history_sha256=array_hash(history, offsets),
        source_bindings={p:file_digest(ROOT/p) for p in ('scripts/diagnose_m3w_zero_reference_support.py',
            'src/world_model/m3w_causal_motion_support.py', 'tests/test_m3w_causal_motion_support.py')},
        rows=n, records=records, zero_harmed_cases=cases, source_support=supports,
        unique_harmed_rows=len({v['row'] for v in cases}), unique_harmed_tracks=len({v['track'] for v in cases}),
        causal_feature_uses_future=False, new_policy=False, new_training=False, threshold_search=False,
        external_readout=False, independent_confirmation=False, deployment=False,
        stage5c_executed=False, smc_enabled=False)
    immutable_json(PUBLIC/'analysis.json', result)
    print(json.dumps(dict(state='complete', seconds=time.monotonic()-start,
        repeated_harms=len(cases), unique_rows=result['unique_harmed_rows'], unique_tracks=result['unique_harmed_tracks'],
        exact_history_cv_harms=sum(v['exact_past_cv'] for v in cases),
        individually_net_feasible_harms=sum(v['individual_signed_rule_pass'] for v in cases),
        cases=cases)), flush=True)


if __name__ == '__main__': main()
