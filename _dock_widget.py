# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:15:34 2021

@author: vkapoor
"""
from napari_plugin_engine import napari_hook_implementation
from magicgui import magicgui
from magicgui import widgets as mw
from magicgui.events import Event, Signal
from magicgui.application import use_app
import napari
from napari.qt.threading import thread_worker
from napari.utils.colormaps import label_colormap
from typing import List, Union
import functools
import time
from enum import Enum
import numpy as np
from pathlib import Path
from warnings import warn
def plugin_wrapper():
    
    from csbdeep.utils import _raise, normalize, axes_check_and_normalize, axes_dict
    from csbdeep.models.pretrained import get_registered_models, get_model_folder
    from csbdeep.utils import load_json
    DEBUG = False
    from stardist.models import StarDist2D, StarDist3D
    from vollseg import SmartSeedPrediction3D, SmartSeedPrediction2D
    from csbdeep.models import Config, CARE
    from n2v.models import N2V
    
    
    # get available models
    _models2d, _aliases2d = get_registered_models(StarDist2D)
    _models3d, _aliases3d = get_registered_models(StarDist3D)
    # use first alias for model selection (if alias exists)
    models2d = [((_aliases2d[m][0] if len(_aliases2d[m]) > 0 else m),m) for m in _models2d]
    models3d = [((_aliases3d[m][0] if len(_aliases3d[m]) > 0 else m),m) for m in _models3d]
    
    _models2d_unet, _aliases2d_unet = get_registered_models(CARE)
    _models3d_unet, _aliases3d_unet = get_registered_models(CARE)
    # use first alias for model selection (if alias exists)
    models2d_unet = [((_aliases2d_unet[m][0] if len(_aliases2d_unet[m]) > 0 else m),m) for m in _models2d_unet]
    models3d_unet = [((_aliases3d_unet[m][0] if len(_aliases3d_unet[m]) > 0 else m),m) for m in _models3d_unet]
    
    model_configs = dict()
    model_threshs = dict()
    model_selected = None

    CUSTOM_MODEL = 'CUSTOM_MODEL'
    model_type_choices = [('2D', StarDist2D), ('3D', StarDist3D), ('Custom 2D/3D', CUSTOM_MODEL)]

    @functools.lru_cache(maxsize=None)
    def get_model(model_type, model_star, model_unet):
        if model_type == CUSTOM_MODEL:
            path_star = Path(model_star)
            path_star.is_dir() or _raise(FileNotFoundError(f"{path_star} is not a directory"))
            
            path_unet = Path(model_unet)
            path_unet.is_dir() or _raise(FileNotFoundError(f"{path_unet} is not a directory"))
            
            config = model_configs[(model_type,model_star)]
            model_class = StarDist2D if config['n_dim'] == 2 else StarDist3D
            return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent))
        else:
            return model_type.from_pretrained(model_star), model_type.from_pretrained(model_unet)

    class Output(Enum):
        Labels = 'Label Image'
        Prob  = 'Probability Map'
        Markers = 'Markers'
        All   = 'All'
    output_choices = [Output.Labels.value, Output.Prob.value, Output.Markers.value, Output.All.value]  
    
    DEFAULTS = dict (
    model_type     = StarDist2D,
    model2d        = models2d[0][1],
    model2d_unet   = models2d_unet[0][1],
    model3d        = models3d[0][1],
    model3d_unet   = models3d_unet[0][1],
    norm_image     = True,
    perc_low       =  1.0,
    perc_high      = 99.8,
    norm_axes      = 'ZYX',
    prob_thresh    = 0.5,
    nms_thresh     = 0.4,
    output_type    = Output.All.value,
    n_tiles        = 'None'
)                  
     

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return plugin_wrapper, dict(name='VollSeg', add_vertical_stretch=False)



@napari_hook_implementation
def napari_provide_sample_data():
    from vollseg import data
    return {
        'test_image_cell_2d': {
            'data': lambda: [(data.test_image_cell_2d(), {'name': 'cell2d'})],
            'display_name': 'Cell (2D)',
        },
        'test_image_cell_3d': {
            'data': lambda: [(data.test_image_cell_3d(), {'name': 'cell3d'})],
            'display_name': 'Cell (3D)',
        },
    }