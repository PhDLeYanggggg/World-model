import pytest

from src.evaluation.m3w_european_squares_grouping import camera_identity, locality_groups


def site(sid, city, lat=None, lon=None, url=''):
    return dict(square_id=sid,city=city,country='A',latitude=lat,longitude=lon,camera_url=url)


def test_same_city_and_transitive_locality_cannot_be_split():
    sites=[site(1,'City'),site(2,' city ',0,0),site(3,'Different',0,.001)]
    assert locality_groups(sites)['groups']==[[1,2,3]]


def test_missing_location_is_not_guessed_zero():
    assert locality_groups([site(1,'A'),site(2,'B',0,0)])['groups']==[[1],[2]]


def test_shared_camera_and_order_stability():
    a=site(9,'A',url='https://youtu.be/abc123?t=1')
    b=site(3,'B',url='https://www.youtube.com/watch?v=abc123&other=x')
    assert locality_groups([a,b])==locality_groups([b,a])
    assert locality_groups([a,b])['groups']==[[3,9]]


def test_outcomes_do_not_change_grouping():
    a=[site(1,'A'),site(2,'B')]
    old=locality_groups(a)
    a[0]['future_error']=10000
    a[1]['future_error']=-10000
    assert locality_groups(a)==old


def test_distinct_camera_ids_remain_distinct():
    assert camera_identity('https://youtube.com/live/abc')!=camera_identity('https://youtu.be/def')
    assert camera_identity('') is None


def test_invalid_source_identity_and_location_refused():
    with pytest.raises(ValueError):
        locality_groups([site(1,'A'),site(1,'B')])
    with pytest.raises(ValueError):
        locality_groups([site(1,'A',91,0),site(2,'B',0,0)])
