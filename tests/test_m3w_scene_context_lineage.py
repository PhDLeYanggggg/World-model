import zipfile
import numpy as np
from src.world_model.m3w_european_source_forecast import pack_scene
from src.world_model.m3w_observation_quality import masked_neighbors, repair_geometry
from src.world_model.m3w_static_scene_context import source_polygons


def test_current_partial_context_packing_also_discards_global_translation():
    h = np.arange(48).reshape(3, 8, 2).astype(float)
    valid = np.ones((3, 8), bool); valid[2, :4] = False
    inputs = dict(history_xy=h, history_valid=valid, agent_id=np.arange(3),
        target_eligible=valid.all(1), baseline_cv=np.repeat(h[:, -1, None], 12, axis=1))
    translated = {**inputs, 'history_xy':h+100, 'baseline_cv':inputs['baseline_cv']+100}
    def pack(value):
        g, _ = pack_scene(value)
        return repair_geometry(g, masked_neighbors(value))
    np.testing.assert_array_equal(pack(inputs), pack(translated))


def test_resource_mapping_uses_raw_camera_not_merged_locality(tmp_path):
    archive = tmp_path/'resources.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('02.Resources/02.Place_polygons/07_image-test_place_polygon.csv', 'x,y\n0,0\n10,0\n10,10\n')
        z.writestr('02.Resources/02.Place_polygons/45_image-test_place_polygon.csv', 'x,y\n0,0\n20,0\n20,20\n')
    a, b = 'raw/07_2024-record.csv', 'raw/45_2024-record.csv'
    result = source_polygons(archive, [a, b])
    assert result[a]['vertices'].max() == 10
    assert result[b]['vertices'].max() == 20
