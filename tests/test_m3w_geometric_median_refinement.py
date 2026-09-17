import numpy as np
import pytest

from src.evaluation.m3w_geometric_median_refinement import refine_geometric_median


def test_exact_atom_after_initial_iteration_limit():
    point,info=refine_geometric_median([[0,0],[2,0]],[.4,.6],[.8,0])
    np.testing.assert_array_equal(point,[2,0])
    assert info['converged'] and info['method']=='certified_support_atom'


def test_smooth_triangle_certificate_and_native_scale():
    points=np.array([[0,0],[2,0],[1,np.sqrt(3)]])*37
    answer,info=refine_geometric_median(points,[1,1,1],[30,20])
    np.testing.assert_allclose(answer,points.mean(0),atol=1e-6)
    assert info['converged'] and info['gap_bound']<1e-6


def test_invalid_initial_or_weights_rejected():
    with pytest.raises(ValueError):
        refine_geometric_median([[0,0]],[1],[np.nan,0])
