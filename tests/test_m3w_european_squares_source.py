import numpy as np
import pytest
from src.evaluation.m3w_european_squares_intake import DTYPE, past_scene
from src.data_unification.m3w_european_squares_source import (
    build_recording, past_queries, choose_query_frames, INPUT_FIELDS, LABEL_FIELDS)


def rows():
    out = np.zeros(50,dtype=DTYPE)
    out['agent'][:30] = 1
    out['frame'][:30] = np.arange(30)*12
    out['agent'][30:] = 2
    out['frame'][30:] = np.arange(6,26)*12
    out['x_min'] = out['frame']*0.1
    out['x_max'] = out['x_min']+3
    out['y_min'] = out['agent']*4
    out['y_max'] = out['y_min']+5
    out['confidence'] = 1
    return out


def test_queries_only_depend_on_available_past():
    f = np.array([0,12,24,36,48,60,72,84,96,132,144,156,168,180,192,204,216])
    assert np.array_equal(past_queries(f),[84,96,216])
    assert np.array_equal(past_queries(f)[past_queries(f)<=144],past_queries(f[f<=144]))
    with pytest.raises(ValueError):
        past_queries([1,1])


def test_future_mutation_and_removal_do_not_change_any_input():
    source = rows()
    before, labels = build_recording(source,[84])
    changed = source.copy()
    for field in ['x_min','y_min','x_max','y_max']:
        changed[field][changed['frame']>84] += 1234
    after, changed_labels = build_recording(changed,[84])
    truncated, _ = build_recording(source[source['frame']<=84],[84])
    for key in INPUT_FIELDS:
        np.testing.assert_array_equal(before[key],after[key])
        np.testing.assert_array_equal(before[key],truncated[key])
    assert not np.array_equal(labels['future_xy'],changed_labels['future_xy'])
    assert not set(INPUT_FIELDS)&set(LABEL_FIELDS)


def test_visible_short_history_neighbors_retained_and_no_future_filter():
    source = rows()
    inputs, labels = build_recording(source,[84,348])
    np.testing.assert_array_equal(inputs['query_offsets'],[0,2,3])
    np.testing.assert_array_equal(inputs['target_eligible'],[True,False,True])
    assert not labels['future_valid'][-1].any()
    assert inputs['history_valid'][1].sum() == 2
    np.testing.assert_allclose(inputs['velocity_causal_fd'][0,1:,0],0.1)
    np.testing.assert_allclose(inputs['baseline_cv'][0],labels['future_xy'][0])


def test_input_history_matches_existing_raw_prefix_reader():
    source = rows()
    inputs,_ = build_recording(source,[84])
    ref = past_scene(source,84)
    for i,agent in enumerate(inputs['agent_id']):
        np.testing.assert_array_equal(inputs['history_xy'][i],ref[int(agent)]['box_center'])
        np.testing.assert_array_equal(inputs['history_valid'][i],ref[int(agent)]['valid'])
        np.testing.assert_array_equal(inputs['velocity_causal_fd'][i],ref[int(agent)]['velocity_causal_fd'])


def test_selected_query_budget_and_eligibility_reproduce():
    source = rows()
    q, stats = choose_query_frames(source,4)
    assert len(q) == 4 and len(np.unique(q)) == 4
    assert stats['all_past_eligible_targets'] == 23+13
    assert np.array_equal(q,choose_query_frames(source,4)[0])
    with pytest.raises(ValueError):
        choose_query_frames(source,0)
    with pytest.raises(ValueError):
        build_recording(source,[84,84])
