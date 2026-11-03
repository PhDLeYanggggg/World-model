import numpy as np
import pytest

from src.evaluation import m3w_dut_background_bridge as bridge


def test_background_inventory_rejects_wrong_tree_partial_set_and_unbounded_blob():
    row=dict(path='data/backgrounds/intersection_01_background_GSOC.png',size=5,sha='a'*40,type='blob',mode='100644')
    tree=dict(sha='tree',tree=[row],truncated=False)
    assert bridge.select_backgrounds(tree,['intersection_01'],'tree',5)[0]['size']==5
    with pytest.raises(ValueError):bridge.select_backgrounds(tree,['intersection_01'],'other',5)
    with pytest.raises(ValueError):bridge.select_backgrounds(tree,['intersection_01','intersection_02'],'tree',5)
    with pytest.raises(ValueError):bridge.select_backgrounds(dict(tree,tree=[dict(row,size=2097153)]),['intersection_01'],'tree',2097153)


def test_full_roster_is_complete_balanced_unique_and_not_old_old():
    m=[f'moving_{i}' for i in range(120)];d=[f'drone_{i}' for i in range(112)];u=[f'dut_{i}' for i in range(28)]
    rows=bridge.pair_roster(m,d,u,20261103)
    assert len(rows)==len({(r['left'],r['right']) for r in rows})==6874
    assert {k:sum(r['kind']==k for r in rows) for k in ('moving_to_dut','drone_to_dut','within_dut')}==dict(
        moving_to_dut=3360,drone_to_dut=3136,within_dut=378)
    assert all(sum(r['kind']==k for r in rows[:60])==20 for k in ('moving_to_dut','drone_to_dut','within_dut'))
    assert rows==bridge.pair_roster(m,d,u,20261103)


def test_one_background_cannot_be_multiview_even_with_three_strong_left_views(monkeypatch):
    monkeypatch.setattr(bridge,'pair_evidence',lambda *args:dict(status='strong',inliers=50))
    value=bridge.compare_available_views({1:None,2:None,3:None},{1:None},{})
    assert value['status']=='ambiguous' and value['frame_pairs_attempted']==3
    assert len(value['support'])==3 and not value['multiview_support_claimed']
    with pytest.raises(ValueError):bridge.compare_available_views({1:None},{1:None,2:None},{})


def test_empty_descriptors_are_unsupported_not_physical_independence():
    empty=dict(xy=np.empty((0,2)),descriptors=None,background_fraction=1.)
    value=bridge.compare_available_views({1:empty},{1:empty},{})
    assert value['status']=='unsupported' and value['frame_pairs_attempted']==1


def test_weak_bridge_preserves_whole_DUT_site_and_parent_publisher_component():
    parent=[['moving_a','moving_b'],['drone_1']]
    dut=['dut_intersection_01','dut_intersection_02','dut_roundabout_01']
    value=bridge.conservative_bridge(parent,dut,[dict(left='moving_a',right='dut_intersection_01',status='ambiguous')])
    assert value['moving_nodes_connected_to_reserved_sources']==['moving_a','moving_b']
    assert not value['scientific_roles_assigned'] and not value['negative_matching_certifies_geographic_independence']
    conflict=bridge.conservative_bridge(parent,dut,[dict(left='drone_1',right='dut_roundabout_01',status='ambiguous')])
    assert len(conflict['DUT_DroneCrowd_cross_reservation_components'])==1
