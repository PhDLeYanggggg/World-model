import pytest

from scripts.report_m3w_movingdronecrowd_background import candidate_support


def recording(clip,targets,visible,incomplete):
    return dict(clip=clip,publisher_scene_label=clip.split('/')[0],rows=200,agents=7,
        window_support=dict(obs8_pred12_native_stride1=dict(target_agent_windows=targets,
            past_visible_agent_instances=visible,history_complete_agent_instances=targets+incomplete,
            history_complete_but_future_incomplete_instances=incomplete,queries_with_targets=9,
            nonoverlapping_raw_intervals_with_targets=2)),
        endpoint_support={h:dict(endpoint_label_windows=targets,complete_interval_windows=targets-1)
                          for h in ('10','25','50','100')})


def test_candidate_support_keeps_missing_future_and_visible_counts_separate():
    rows=[recording('scene_1/1',15,40,20),recording('scene_2/1',100,150,30)]
    value=candidate_support(rows,['moving_scene_1__1'],[['moving_scene_1__1'],['moving_scene_2__1']])
    assert value['rows']==200 and value['clips']==1
    assert value['native_obs8_pred12']['target_agent_windows']==15
    assert value['native_obs8_pred12']['past_visible_agent_instances']==40
    assert value['native_obs8_pred12']['history_complete_but_future_incomplete_instances']==20
    assert value['native_endpoint_support']['50']['complete_interval_windows']==14
    assert not value['candidate_counts_are_admission'] and not value['windows_are_independent_samples']


def test_empty_candidate_support_does_not_invent_data_or_sites():
    value=candidate_support([recording('scene_1/1',10,20,5)],[],[['moving_scene_1__1']])
    assert value['rows']==value['clips']==value['conservative_components_touching_candidates']==0
    assert value['native_obs8_pred12']['target_agent_windows']==0


def test_candidate_support_rejects_unknown_or_duplicate_source_and_missing_groups():
    row=recording('scene_1/1',10,20,5)
    with pytest.raises(ValueError,match='source-traceable'):
        candidate_support([row],['moving_scene_2__1'],[])
    with pytest.raises(ValueError,match='source-traceable'):
        candidate_support([row,row],[],[])
    with pytest.raises(ValueError,match='partition'):
        candidate_support([row],['moving_scene_1__1'],[])
