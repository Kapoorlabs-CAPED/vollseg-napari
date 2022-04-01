import numpy as np
from tifffile import imwrite
import pytest
from vollseg_napari import _dock_widget
import functools
from vollseg import UNET, StarDist3D, StarDist2D
from magicgui.events import Signal
from napari.qt.threading import thread_worker
from vollseg.pretrained import get_registered_models, get_model_folder, get_model_details, get_model_instance
from csbdeep.utils import load_json
from csbdeep.utils import _raise, normalize, axes_check_and_normalize, axes_dict
import time
from typing import List, Union

@pytest.fixture
def test_unet_checkbox_toggle(qtbot, make_napari_viewer):

    viewer = make_napari_viewer()
    widg = _dock_widget.plugin_wrapper_vollseg()
    qtbot.addWidget(widg)
  