import copy

import numpy as np
import pytest

from scripts.build_m3w_decision_supplement_v12 import (
    COMPARISONS,
    contrast_influence,
    source_influence,
)
from src.evaluation.m3w_joint_path_audit import bootstrap


def test_influence_holds_seeds_together_and_keeps_zero_source():
    result = source_influence([[1, 2, 3], [-3, -2, -1], [-1, 0, 1]])
    assert result['mean'] == 0
    assert result['sources'] == 3
    assert result['positive_sources'] == 1
    assert result['zero_sources'] == 1
    assert result['leave_one_source_out_range'] == [-1, 1]
    assert result['seed_mean_range'] == [-1, 1]


def test_missing_required_cell_is_not_dropped():
    assert source_influence([[1, None, 2], [1, 0, -1]]) is None
    with pytest.raises(ValueError, match='multiple sources'):
        source_influence([[1, 0, 2]])


def test_frozen_bootstrap_must_match_before_influence_is_used():
    roster = {f'R{i}': f'S{i % 14:02d}' for i in range(34)}
    metrics = ('observed_ade_reduction', 'grid_ade_lower', 'grid_ade_upper')
    packet = dict(schema_version=1, roster=roster, seeds=[1, 2, 3],
                  joint_cells={g: {str(s): {k: dict.fromkeys(metrics, float(s))
                                            for k in COMPARISONS}
                                   for s in (1, 2, 3)} for g in set(roster.values())})
    draw = np.random.default_rng(73107).integers(14, size=(3000, 14))
    expected = {k: dict.fromkeys(metrics, bootstrap([[1, 2, 3]]*14, draw))
                for k in COMPARISONS}
    assert contrast_influence(packet, expected)['joint_vs_cv']['grid_ade_lower']['mean'] == 2
    changed = copy.deepcopy(packet)
    changed['joint_cells']['S00']['1']['joint_vs_cv']['grid_ade_lower'] += 1
    with pytest.raises(ValueError, match='Paired replay does not match'):
        contrast_influence(changed, expected)
