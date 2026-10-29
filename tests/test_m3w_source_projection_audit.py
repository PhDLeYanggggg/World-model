import numpy as np
import pytest

from scripts.audit_m3w_source_projection import resolution,points,homography,project


def test_resolution_tokens_do_not_treat_qhd_plus_as_plain_hd():
    np.testing.assert_array_equal(resolution('126_QHD+.points'),[2688,1520])
    np.testing.assert_array_equal(resolution('126_FHD_polygon.csv'),[1920,1080])
    with pytest.raises(ValueError):resolution('unknown.points')


def test_qgis_comments_latin1_enabled_rows_and_y_sign():
    blob=b'# CRS degree\xb0\nmapX,mapY,sourceX,sourceY,enable\n'
    blob+=b'0,0,0,0,1\n1,0,1,0,1\n0,1,0,-1,1\n1,1,1,-1,1\n9,9,2,-2,0\n'
    image,mapped,info=points(blob)
    assert len(image)==4 and info['total']==5
    np.testing.assert_array_equal(image,mapped)


def test_centered_projective_fit_large_map_ordinates_roundtrips():
    image=np.array([[0.,0.],[100.,0.],[0.,100.],[100.,100.],[40.,60.],[60.,40.]])
    h=np.array([[.3,.02,2e6],[.01,.4,6e6],[.0001,.0002,1.]])
    mapped=project(image,h);fit,origin=homography(image,mapped)
    np.testing.assert_allclose(project(mapped-origin,np.linalg.inv(fit)),image,atol=.02)
    np.testing.assert_allclose(project(project(image,fit),np.linalg.inv(fit)),image,atol=1e-6)
