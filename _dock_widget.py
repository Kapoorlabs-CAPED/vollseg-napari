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
from tifffile import imread, imwrite
from vollseg import inrimage, klb, h5, spatial_image
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
    
    _models_denoise_care, _aliases_denoise_care = get_registered_models(CARE)
    # use first alias for model selection (if alias exists)
    models_denoise_care = [((_aliases_denoise_care[m][0] if len(_aliases_denoise_care[m]) > 0 else m),m) for m in _models_denoise_care]
    
    _models_denoise_n2v, _aliases_denoise_n2v = get_registered_models(N2V)
    # use first alias for model selection (if alias exists)
    models_denoise_n2v = [((_aliases_denoise_n2v[m][0] if len(_aliases_denoise_n2v[m]) > 0 else m),m) for m in _models_denoise_n2v]
    
    
    model_configs = dict()
    model_threshs = dict()
    model_selected = None

    CUSTOM_MODEL = 'CUSTOM_MODEL'
    seg_model_type_choices = [('2D', StarDist2D), ('3D', StarDist3D), ('Custom 2D/3D', CUSTOM_MODEL)]
    den_model_type_choices = [ ('DenoiseCARE', CARE) , ('DenoiseN2V', N2V) ('Custom N2V/CARE', CUSTOM_MODEL)]
    @functools.lru_cache(maxsize=None)
    def get_model(model_type, model_star, model_unet, model_care, model_n2v):
        if model_type == CUSTOM_MODEL:
            path_star = Path(model_star)
            path_star.is_dir() or _raise(FileNotFoundError(f"{path_star} is not a directory"))
            
            path_unet = Path(model_unet)
            path_unet.is_dir() or _raise(FileNotFoundError(f"{path_unet} is not a directory"))
            
            config = model_configs[(model_type,model_star)]
            model_class = StarDist2D if config['n_dim'] == 2 else StarDist3D
            
            if model_care is not None:
                path_care = Path(model_care)
                path_care.is_dir() or _raise(FileNotFoundError(f"{path_care} is not a directory"))
                return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent)), CARE(None, name=path_care.name, basedir=str(path_care.parent))
            if model_n2v is not None:
                path_n2v = Path(model_n2v)
                path_n2v.is_dir() or _raise(FileNotFoundError(f"{path_n2v} is not a directory"))
                return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent)), N2V(None, name=path_n2v.name, basedir=str(path_n2v.parent))

              
            elif model_care == None and model_n2v == None:
                
                 return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent))
        
        else:
            
            if model_care is not None:
                
                 return model_type.from_pretrained(model_star), model_type.from_pretrained(model_unet), model_type.from_pretrained(model_care) 
            if model_n2v is not None:
                
                 return model_type.from_pretrained(model_star), model_type.from_pretrained(model_unet), model_type.from_pretrained(model_n2v)
                
            elif model_care == None and model_n2v == None:
                
                 return model_type.from_pretrained(model_star), model_type.from_pretrained(model_unet)

    class Output(Enum):
        Labels = 'Label Image'
        Binary_mask = 'Binary Image'
        Denoised_image = 'Denoised Image'
        Prob  = 'Probability Map'
        Markers = 'Markers'
        All   = 'All'
    output_choices = [Output.Labels.value, Output.Binary_mask.value, Output.Denoised_image.value, Output.Prob.value, Output.Markers.value, Output.All.value]  
    
    DEFAULTS = dict (
            seg_model_type = SmartSeedPrediction3D,
            den_model_type = N2V,
            model2d_star        = models2d[0][1],
            model2d_unet   = models2d_unet[0][1],
            model3d_star        = models3d[0][1],
            model3d_unet   = models3d_unet[0][1],
            model_denoise_n2v  = models_denoise_n2v[0][1],
            model_denoise_care  = models_denoise_care[0][1],
            norm_image     = True,
            perc_low       =  1.0,
            perc_high      = 99.8,
            norm_axes      = 'ZYX',
            prob_thresh    = 0.5,
            nms_thresh     = 0.4,
            output_type    = Output.All.value,
            n_tiles        = 'None'
    )                  
     
    logo = abspath(__file__, 'resources/vollseg_logo_napari.png')

    @magicgui (
        label_head      = dict(widget_type='Label', label=f'<h1><img src="{logo}">VollSeg</h1>'),
        image           = dict(label='Input Image'),
        axes            = dict(widget_type='LineEdit', label='Image Axes'),
        label_nn        = dict(widget_type='Label', label='<br><b>Neural Network Prediction:</b>'),
        seg_model_type  = dict(widget_type='RadioButtons', label='Seg Model Type', orientation='horizontal', choices=seg_model_type_choices, value=DEFAULTS['seg_model_type']),
        den_model_type  = dict(widget_type='RadioButtons', label='Den Model Type', orientation='horizontal', choices=den_model_type_choices, value=DEFAULTS['den_model_type']),
        model2d_star    = dict(widget_type='ComboBox', visible=False, label='Pre-trained StarDist Model', choices=models2d, value=DEFAULTS['model2d_star']),
        model3d_star    = dict(widget_type='ComboBox', visible=False, label='Pre-trained StarDist Model', choices=models3d, value=DEFAULTS['model3d_star']),
        
        model2d_unet    = dict(widget_type='ComboBox', visible=False, label='Pre-trained UNET Model', choices=models2d_unet, value=DEFAULTS['model2d_unet']),
        model3d_unet    = dict(widget_type='ComboBox', visible=False, label='Pre-trained UNET Model', choices=models3d_unet, value=DEFAULTS['model3d_unet']),
        
        model_denoise_n2v   = dict(widget_type='ComboBox', visible=False, label='Pre-trained N2V Model', choices=models_denoise_n2v, value=DEFAULTS['model_denoise_n2v']), 
        model_denoise_care   = dict(widget_type='ComboBox', visible=False, label='Pre-trained CARE Model', choices=models_denoise_care, value=DEFAULTS['model_denoise_care']),
        
        model_folder    = dict(widget_type='FileEdit', visible=False, label='Custom Model', mode='d'),
        model_axes      = dict(widget_type='LineEdit', label='Model Axes', value=''),
        norm_image      = dict(widget_type='CheckBox', text='Normalize Image', value=DEFAULTS['norm_image']),
        label_nms       = dict(widget_type='Label', label='<br><b>NMS Postprocessing:</b>'),
        perc_low        = dict(widget_type='FloatSpinBox', label='Percentile low',              min=0.0, max=100.0, step=0.1,  value=DEFAULTS['perc_low']),
        perc_high       = dict(widget_type='FloatSpinBox', label='Percentile high',             min=0.0, max=100.0, step=0.1,  value=DEFAULTS['perc_high']),
        norm_axes       = dict(widget_type='LineEdit',     label='Normalization Axes',                                         value=DEFAULTS['norm_axes']),
        prob_thresh     = dict(widget_type='FloatSpinBox', label='Probability/Score Threshold', min=0.0, max=  1.0, step=0.05, value=DEFAULTS['prob_thresh']),
        nms_thresh      = dict(widget_type='FloatSpinBox', label='Overlap Threshold',           min=0.0, max=  1.0, step=0.05, value=DEFAULTS['nms_thresh']),
        output_type     = dict(widget_type='ComboBox', label='Output Type', choices=output_choices, value=DEFAULTS['output_type']),
        label_adv       = dict(widget_type='Label', label='<br><b>Advanced Options:</b>'),
        n_tiles         = dict(widget_type='LiteralEvalLineEdit', label='Number of Tiles', value=DEFAULTS['n_tiles']),
        timelapse_opts  = dict(widget_type='ComboBox', label='Time-lapse Labels ', choices=timelapse_opts, value=DEFAULTS['timelapse_opts']),
        set_thresholds  = dict(widget_type='PushButton', text='Set optimized postprocessing thresholds (for selected model)'),
        defaults_button = dict(widget_type='PushButton', text='Restore Defaults'),
        progress_bar    = dict(label=' ', min=0, max=0, visible=False),
        layout          = 'vertical',
        persist         = True,
        call_button     = True,
    )

def inrimage_file_reader(path):
   array = inrimage.read_inrimage(path)
   # return it as a list of LayerData tuples,
   # here with no optional metadata
   return [(array,)]

def klbimage_file_reader(path):
   array = klb.read_klb(path)
   # return it as a list of LayerData tuples,
   # here with no optional metadata
   return [(array,)]

def tifimage_file_reader(path):
   array = imread(path)
   # return it as a list of LayerData tuples,
   # here with no optional metadata
   return [(array,)]

def h5image_file_reader(path):
   array = h5.read_h5(path)
   # return it as a list of LayerData tuples,
   # here with no optional metadata
   return [(array,)]


@napari_hook_implementation
def napari_get_reader(path):
   # If we recognize the format, we return the actual reader function
   if isinstance(path, str) and path.endswith(".inr"):
      return inrimage_file_reader
   if isinstance(path, str) and path.endswith(".klb"):
      return klbimage_file_reader
   if isinstance(path, str) and path.endswith(".tif"):
      return tifimage_file_reader
   if isinstance(path, str) and path.endswith(".h5"):
      return h5image_file_reader
    
   else:
      # otherwise we return None.
      return None


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