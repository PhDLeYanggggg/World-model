import copy

import numpy as np
import pytest

from src.data_unification.m3w_producer_crossfit import plan, eligibility, selected_records
from src.world_model.m3w_european_source_forecast import fit_design


def registry():
    groups = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]
    jobs = []
    for producer, critic, evaluation in ((1, 0, 2), (0, 1, 2), (0, 2, 1)):
        jobs.append(dict(key=f'p{producer}c{critic}', producer_fold=producer,
            producer_sites=groups[producer], critic_sites=groups[critic],
            evaluation_sites=groups[evaluation], forecast_key=f'p{producer}', easy_cut=producer+1))
    parts = {j['key']: [dict(part=i, names=[s]) for i, s in enumerate(j['critic_sites'])] for j in jobs}
    return dict(jobs=jobs, partitions=parts)


def test_six_source_disjoint_models_inside_parent_A():
    reg = registry()
    jobs = plan(reg)
    assert len(jobs) == 6
    for j in jobs:
        assert not set(j['fit_sites']) & set(j['donor_sites'])
        assert set(j['fit_sites']) | set(j['donor_sites']) == set(j['producer_sites'])
        for p in reg['jobs']:
            if p['key'] in j['parent_contexts']:
                assert not set(j['producer_sites']) & (set(p['critic_sites']) | set(p['evaluation_sites']))


def test_reject_overlap_and_missing_partition():
    reg = registry()
    reg['partitions']['p1c0'][0]['names'].append('b')
    with pytest.raises(ValueError, match='exactly'):
        plan(reg)
    reg = registry()
    reg['jobs'][0]['evaluation_sites'].append('d')
    with pytest.raises(ValueError, match='overlaps'):
        plan(reg)


def test_same_teacher_is_not_permission_for_cross_context_pooling():
    rows = eligibility(registry())
    same = next(r for r in rows if r['target'] == 'p0c1' and r['donor'] == 'p0c2')
    assert 'different_teacher' not in same['reasons']
    assert 'donor_is_target_outer_H' in same['reasons']
    assert not same['blind_pooling_allowed']


def test_subset_offsets_skip_unread_source_and_preserve_empty_queries():
    admission = {'recordings': {k: dict(clip=k, exclusion_group=g) for k, g in [('r0', 'x'), ('r1', 'y'), ('r2', 'z')]}}
    structure = {'recordings': [dict(clip=k, image_frames=n,
        window_support={'obs8_pred12_native_stride1': {'history_complete_agent_instances': count}})
        for k, n, count in [('r0', 23, 8), ('r1', 18, 0), ('r2', 25, 12)]]}
    rows = selected_records(admission, structure, {'z'})
    assert len(rows) == 1
    assert (rows[0]['row_offset'], rows[0]['query_offset'], rows[0]['index']) == (8, 4, 2)


def test_training_design_invariant_to_held_labels():
    data = {'sites': np.array(['a', 'a', 'b', 'b', 'c', 'c'])}
    ade = np.arange(1, 37, dtype=float).reshape(6, 6)
    a = fit_design(data, ['a', 'b'], ade)
    changed = ade.copy()
    changed[4:] = 1e8
    b = fit_design(data, ['a', 'b'], changed)
    for k in ('train_ids', 'factors', 'selection_relative_scores'):
        np.testing.assert_array_equal(a[k], b[k])
    assert a['baseline_index'] == b['baseline_index']
    assert a['normalizers'] == b['normalizers']
