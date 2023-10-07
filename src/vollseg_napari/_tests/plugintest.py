import napari
import pytest
from vollseg import test_image_carcinoma_3dt

from .. import make_dock_widget


@pytest.fixture(scope="function")
def plugin():
    return make_dock_widget()





@pytest.fixture(scope="session")
def test_3dt():
    img = test_image_carcinoma_3dt()
    img = img[0:2,0:10, 0:10, 0:10] 
    return napari.layers.Image(img, name="test_3dt")


