import numpy as np
import pytest

from reproducibility.opendd_vehicle_direction_supplement.verify import replay
from scripts import build_m3w_vehicle_direction_supplement as pack


def test_both_anonymous_panels_reproduce_and_require_all_seed_directions():
    rows, _ = pack.paper.inputs()
    panels, expected = {}, {}
    for label, r in zip(('forced', 'one_rotation'), rows, strict=True):
        panels[label], expected[label] = pack.export(r, label == 'forced')
    data = dict(panels=panels, draws=np.random.default_rng(2917).integers(7, size=(3000, 7)).tolist())
    assert pack.prior.previous.prior_builder.check(replay(data), expected)
    panels['forced'][0][0].pop()
    with pytest.raises(ValueError, match='six directions'):
        replay(data)


def test_vehicle_paragraph_only_adds_registered_results():
    rows, _ = pack.paper.inputs()
    original = 'Before\n'+pack.paper.ANCHOR+'\nAfter'
    revised, markers = pack.paper.revise(original, rows)
    assert revised.startswith('Before\n'+pack.paper.ANCHOR)
    assert revised.endswith('\nAfter')
    assert set(markers) == {'1,431', '1,393', '0.0446', '364 chosen actors'}
