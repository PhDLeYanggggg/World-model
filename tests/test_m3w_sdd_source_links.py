import pytest

from src.world_model.m3w_sdd_source_links import lexical_reindex_links


def test_lexical_compression_order_is_not_numeric_source_identity():
    links = lexical_reindex_links(['video'+str(i) for i in range(12)])
    assert links['video2'] == 'video4'
    assert links['video6'] == 'video8'
    assert links['video10'] == 'video2'
    assert len(set(links.values())) == 12


def test_single_digit_series_unchanged():
    names = ['video'+str(i) for i in range(8)]
    assert lexical_reindex_links(names) == dict(zip(names, names))


@pytest.mark.parametrize('names', [[], ['video0', 'video0'], ['video0', 'video2'], ['arbitrary']])
def test_incomplete_or_ambiguous_inventory_fails(names):
    with pytest.raises(ValueError):
        lexical_reindex_links(names)
