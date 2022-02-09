from vollseg_napari import napari_get_reader
import numpy as np
from tifffile import imwrite
import pytest
from vollseg_napari import _test_dock_widget
import functools
from vollseg import UNET, StarDist3D, StarDist2D
from magicgui.events import Signal
from napari.qt.threading import thread_worker
from vollseg.pretrained import get_registered_models, get_model_folder, get_model_details, get_model_instance
from csbdeep.utils import load_json
from csbdeep.utils import _raise, normalize, axes_check_and_normalize, axes_dict
import time
from typing import List, Union


def test_unet_checkbox_toggle(qtbot, make_napari_viewer):

    viewer = make_napari_viewer()
    widg = _test_dock_widget.plugin_wrapper_vollseg()
    plugin = widg[0]
    plugin_star_parameters = widg[1]
    plugin_extra_parameters = widg[2]
    
    def change_handler(*widgets, init=True):
        def decorator_change_handler(handler):
            @functools.wraps(handler)
            def wrapper(*args):
                source = Signal.sender()
                emitter = Signal.current_emitter()
               
                    
                return handler(*args)

            for widget in widgets:
                widget.changed.connect(wrapper)
                if init:
                    widget.changed(widget.value)
            return wrapper

        return decorator_change_handler
   
    model_unet_configs = dict()
    model_selected_unet = None
    
    widget_for_modeltype = {
        
        UNET: plugin.model_unet,
       
        'NOUNET': plugin.model_unet_none
        
    }
    
       


    @change_handler( plugin.model_unet, plugin.model_unet_none, init=False) 
    def _model_change_unet():
        
        
                model_class_unet = ( UNET if Signal.sender() is plugin.model_unet else UNET if plugin.model_unet.value is not None and Signal.sender() is None else None ) 
                
                if model_class_unet is not None:
                        plugin_extra_parameters.dounet.value = True  
                        model_name = plugin.model_unet.value
                        key_unet = model_class_unet, model_name
                      
                else:
                    plugin_extra_parameters.dounet.value = False
                    plugin_extra_parameters.unet_model_axes.value = ''
                    plugin.model_folder_unet.line_edit.tooltip = (
                            'Invalid model directory'
                        )

    @change_handler(plugin.unet_seg_model_type, init=False)
    def _seg_model_type_change_unet(seg_model_type: Union[str, type]):
        selected = widget_for_modeltype[seg_model_type]
        for w in set(
            (plugin.model_unet, plugin.model_unet_none, plugin.model_folder_unet)
        ) - {selected}:
            w.hide()
        selected.show()
        # trigger _model_change_unet
        selected.changed(selected.value)    
    plugin.model_unet_none.value = 'NOUNET'            
    _seg_model_type_change_unet(plugin.model_unet_none.value)
    assert plugin_extra_parameters.dounet.value == False
    _seg_model_type_change_unet(UNET)
    assert plugin_extra_parameters.dounet.value == True
  


@pytest.fixture
def write_results(tmp_path):
    

    def write_func(filename):

       test_file, original_data, reader = test_get_reader_returns_callable(tmp_path, filename)
       
       return test_file, original_data, reader


    return write_func
#tmp_path is a pytest fixture    
def test_get_reader_returns_callable(tmp_path, filename = 'file.tif'):
     
       test_file = str(tmp_path/filename)
       original_data = np.random.rand(20,20,20)
       reader = napari_get_reader(test_file)
       imwrite(test_file, original_data)
       assert callable(reader), f'{reader} is not a valid file to be read by this function' 

       return test_file, original_data, reader


def test_reader_round_trip(write_results):


           test_file, original_data, reader = write_results("file.tif")

           layer_data_list = reader(test_file)

           assert isinstance(layer_data_list, list) and len(layer_data_list) > 0

           layer_data_tuple = layer_data_list[0]

           layer_data = layer_data_tuple[0]

           np.testing.assert_allclose(layer_data, original_data)
   

