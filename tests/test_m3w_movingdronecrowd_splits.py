import pytest

from src.evaluation.m3w_movingdronecrowd_splits import expand_publisher_list,compare_publisher_splits


def test_scene_and_clip_identifiers_follow_the_publisher_loader_semantics():
    value=expand_publisher_list('scene_1\nscene_2/1\n',{'scene_1/1','scene_1/2','scene_2/1','scene_2/2'})
    assert value['clips']==['scene_1/1','scene_1/2','scene_2/1']
    assert [v['kind'] for v in value['entries']]==['scene','clip']


@pytest.mark.parametrize('text',['scene_1\nscene_1/1','scene_9','scene_1/3','../scene_1','scene_1,0'])
def test_missing_ambiguous_or_unsafe_entries_are_not_silently_repaired(text):
    with pytest.raises(ValueError):expand_publisher_list(text,{'scene_1/1','scene_1/2'})


def test_disjoint_clips_do_not_imply_disjoint_scene_labels_or_physical_sites():
    clips={'scene_1/1','scene_2/1','scene_2/2'}
    parts={role:expand_publisher_list(text,clips) for role,text in
           dict(train='scene_1',val='scene_2/1',test='scene_2/2').items()}
    result=compare_publisher_splits(parts)
    assert result['clip_overlaps']['val_test']==[]
    assert result['scene_label_overlaps']['val_test']==['scene_2']
    assert not result['independent_physical_sites_verified']
    assert not result['roles_assigned']
