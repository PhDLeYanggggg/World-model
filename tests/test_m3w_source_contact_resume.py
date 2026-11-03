import numpy as np
import pytest
from PIL import Image

from scripts.render_m3w_movingdronecrowd_background import save_contact


def test_resume_requires_identical_pixels_and_preserves_original_bytes(tmp_path):
    path=tmp_path/'contact.png'
    image=Image.fromarray(np.full((10,12,3),73,np.uint8))
    save_contact(image,path)
    original=path.read_bytes()
    with pytest.raises(ValueError,match='Preserve'):
        save_contact(image,path)
    save_contact(image,path,resume=True)
    assert path.read_bytes()==original
    with pytest.raises(ValueError,match='Changed'):
        save_contact(Image.new('RGB',(12,10),'black'),path,resume=True)
    assert path.read_bytes()==original
