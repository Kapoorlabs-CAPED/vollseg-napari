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
from napari.layers import Image, Shapes
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
    
    from stardist.models import StarDist2D, StarDist3D
    from vollseg import VollSeg3D, VollSeg2D
    from csbdeep.models import Config, CARE
    
    
    from n2v.models import N2V
    from stardist.utils import abspath
    
    
    DEBUG = False
    def get_data(image):
       image = image.data[0] if image.multiscale else image.data
       # enforce dense numpy array in case we are given a dask array etc
       return np.asarray(image)

    def change_handler(*widgets, init=True, debug=DEBUG):
       def decorator_change_handler(handler):
           @functools.wraps(handler)
           def wrapper(*args):
               source = Signal.sender()
               emitter = Signal.current_emitter()
               if debug:
                   # print(f"{emitter}: {source} = {args!r}")
                   print(f"{str(emitter.name).upper()}: {source.name} = {args!r}")
               return handler(*args)
           for widget in widgets:
               widget.changed.connect(wrapper)
               if init:
                   widget.changed(widget.value)
           return wrapper
       return decorator_change_handler
    
    # get available models
    _models2d_star, _aliases2d_star = get_registered_models(StarDist2D)
    _models3d_star, _aliases3d_star = get_registered_models(StarDist3D)
    
    
    
    # use first alias for model selection (if alias exists)
    models2d_star = [((_aliases2d_star[m][0] if len(_aliases2d_star[m]) > 0 else m),m) for m in _models2d_star]
    models3d_star = [((_aliases3d_star[m][0] if len(_aliases3d_star[m]) > 0 else m),m) for m in _models3d_star]
    
    _models2d_unet, _aliases2d_unet = get_registered_models(StarDist3D)
    _models3d_unet, _aliases3d_unet = get_registered_models(StarDist3D)
    # use first alias for model selection (if alias exists)
    models2d_unet = [((_aliases2d_unet[m][0] if len(_aliases2d_unet[m]) > 0 else m),m) for m in _models2d_unet]
    models3d_unet = [((_aliases3d_unet[m][0] if len(_aliases3d_unet[m]) > 0 else m),m) for m in _models3d_unet]
    
    _models_den_care, _aliases_den_care = get_registered_models(StarDist3D)
    # use first alias for model selection (if alias exists)
    models_den_care = [((_aliases_den_care[m][0] if len(_aliases_den_care[m]) > 0 else m),m) for m in _models_den_care]
    
    _models_den_n2v, _aliases_den_n2v = get_registered_models(StarDist3D)
    # use first alias for model selection (if alias exists)
    models_den_n2v = [((_aliases_den_n2v[m][0] if len(_aliases_den_n2v[m]) > 0 else m),m) for m in _models_den_n2v]
    
    
    model_star_configs = dict()
    model_unet_configs = dict()
    model_den_care_configs = dict()
    model_den_n2v_configs = dict()
    model_star_threshs = dict()
    model_selected_star = None
    model_selected_unet = None
    model_selected_care = None
    model_selected_n2v = None
    
    CUSTOM_SEG_MODEL = 'CUSTOM_SEG_MODEL'
    CUSTOM_DEN_MODEL = 'CUSTOM_DEN_MODEL'
    seg_model_type_choices = [('2D', VollSeg2D), ('3D', VollSeg3D), ('Custom 2D/3D', CUSTOM_SEG_MODEL)]
    den_model_type_choices = [ ('DenoiseCARE', CARE) , ('DenoiseN2V', N2V), ('NoDenoising', None), ('Custom N2V/CARE', CUSTOM_DEN_MODEL)]
    @functools.lru_cache(maxsize=None)
    def get_model(seg_model_type, den_model_type, model_star, model_unet, model_den_care, model_den_n2v):
        if seg_model_type == CUSTOM_SEG_MODEL:
            path_star = Path(model_star)
            path_star.is_dir() or _raise(FileNotFoundError(f"{path_star} is not a directory"))
            
            path_unet = Path(model_unet)
            path_unet.is_dir() or _raise(FileNotFoundError(f"{path_unet} is not a directory"))
            
            config = model_star_configs[(seg_model_type,model_star)]
            model_class = VollSeg2D if config['n_dim'] == 2 else VollSeg3D
            if den_model_type == CUSTOM_DEN_MODEL:    
                if model_den_care is not None:
                    path_care = Path(model_den_care)
                    path_care.is_dir() or _raise(FileNotFoundError(f"{path_care} is not a directory"))
                    return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent)), CARE(None, name=path_care.name, basedir=str(path_care.parent))
                if model_den_n2v is not None:
                    path_n2v = Path(model_den_n2v)
                    path_n2v.is_dir() or _raise(FileNotFoundError(f"{path_n2v} is not a directory"))
                    return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent)), N2V(None, name=path_n2v.name, basedir=str(path_n2v.parent))
    
            elif den_model_type != CUSTOM_DEN_MODEL and model_den_care is not None:
                
                 return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent)), den_model_type.from_pretrained(model_den_care)
             
            elif den_model_type != CUSTOM_DEN_MODEL and model_den_n2v is not None:
                
                 return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent)), den_model_type.from_pretrained(model_den_n2v)
                 
                
            elif den_model_type != CUSTOM_DEN_MODEL and model_den_care == None and model_den_n2v == None:
                    
                     return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent))
        
        else:
            
              if den_model_type == CUSTOM_DEN_MODEL:    
                  if model_den_care is not None:
                      path_care = Path(model_den_care)
                      path_care.is_dir() or _raise(FileNotFoundError(f"{path_care} is not a directory"))
                      return seg_model_type.from_pretrained(model_star), seg_model_type.from_pretrained(model_unet), CARE(None, name=path_care.name, basedir=str(path_care.parent))
                  if model_den_n2v is not None:
                      path_n2v = Path(model_den_n2v)
                      path_n2v.is_dir() or _raise(FileNotFoundError(f"{path_n2v} is not a directory"))
                      return seg_model_type.from_pretrained(model_star), seg_model_type.from_pretrained(model_unet), N2V(None, name=path_n2v.name, basedir=str(path_n2v.parent))
      
              elif den_model_type != CUSTOM_DEN_MODEL and model_den_n2v is not None:
                  
                   return seg_model_type.from_pretrained(model_star), seg_model_type.from_pretrained(model_unet), den_model_type.from_pretrained(model_den_n2v)
                   
              elif den_model_type != CUSTOM_DEN_MODEL and model_den_care is not None:
                  
                   return seg_model_type.from_pretrained(model_star), seg_model_type.from_pretrained(model_unet), den_model_type.from_pretrained(model_den_care)    
               
                
              elif den_model_type != CUSTOM_DEN_MODEL and model_den_care == None and model_den_n2v == None:
                      
                       return seg_model_type.from_pretrained(model_star), seg_model_type.from_pretrained(model_unet)
        


    class Output(Enum):
        Labels = 'Label Image'
        Binary_mask = 'Binary Image'
        Denoised_image = 'Denoised Image'
        Prob  = 'Probability Map'
        Markers = 'Markers'
        All   = 'All'
    output_choices = [Output.Labels.value, Output.Binary_mask.value, Output.Denoised_image.value, Output.Prob.value, Output.Markers.value, Output.All.value]  
    
    DEFAULTS = dict (
            seg_model_type = VollSeg3D,
            den_model_type = N2V,
            model2d_star        = models2d_star[0][1],
            model2d_unet   = models2d_unet[0][1],
            model3d_star        = models3d_star[0][1],
            model3d_unet   = models3d_unet[0][1],
            model_den_n2v  = models_den_n2v[0][1],
            model_den_care  = models_den_care[0][1],
            norm_image     = True,
            perc_low       =  1.0,
            perc_high      = 99.8,
            min_size_mask = 100,
            min_size = 100,
            max_size = 10000,
            norm_axes      = 'ZYX',
            prob_thresh    = 0.5,
            nms_thresh     = 0.4,
            output_type    = Output.All.value,
            n_tiles        = 'None',
            dounet         = True,
            prob_map_watershed = True,
            
    )                  
     
    logo = abspath(__file__, 'resources/vollseg_logo_napari.png')

    @magicgui (
        label_head      = dict(widget_type='Label', label=f'<h1><img src="{logo}">VollSeg</h1>'),
        image           = dict(label='Input Image'),
        axes            = dict(widget_type='LineEdit', label='Image Axes'),
        label_nn        = dict(widget_type='Label', label='<br><b>Neural Network Prediction:</b>'),
        seg_model_type  = dict(widget_type='RadioButtons', label='Seg Model Type', orientation='horizontal', choices=seg_model_type_choices, value=DEFAULTS['seg_model_type']),
        den_model_type  = dict(widget_type='RadioButtons', label='Den Model Type', orientation='horizontal', choices=den_model_type_choices, value=DEFAULTS['den_model_type']),
        model2d_star    = dict(widget_type='ComboBox', visible=False, label='Pre-trained StarDist Model', choices=models2d_star, value=DEFAULTS['model2d_star']),
        model3d_star    = dict(widget_type='ComboBox', visible=False, label='Pre-trained StarDist Model', choices=models3d_star, value=DEFAULTS['model3d_star']),
        
        model2d_unet    = dict(widget_type='ComboBox', visible=False, label='Pre-trained UNET Model', choices=models2d_unet, value=DEFAULTS['model2d_unet']),
        model3d_unet    = dict(widget_type='ComboBox', visible=False, label='Pre-trained UNET Model', choices=models3d_unet, value=DEFAULTS['model3d_unet']),
        
        model_den_care   = dict(widget_type='ComboBox', visible=False, label='Pre-trained CARE Denoising Model', choices=models_den_care, value=DEFAULTS['model_den_care']),
        model_den_n2v   = dict(widget_type='ComboBox', visible=False, label='Pre-trained N2V Denoising Model', choices=models_den_n2v, value=DEFAULTS['model_den_n2v']),
        
        model_folder_star    = dict(widget_type='FileEdit', visible=False, label='Custom StarDist Model', mode='d'),
        model_folder_unet    = dict(widget_type='FileEdit', visible=False, label='Custom UNET Model', mode='d'),
        model_folder_den    = dict(widget_type='FileEdit', visible=False, label='Custom Denoising Model', mode='d'),
        
        
        result_folder    = dict(widget_type='FileEdit', visible=False, label='Result Folder', mode='d'),
        model_axes      = dict(widget_type='LineEdit', label='Model Axes', value=''),
        norm_image      = dict(widget_type='CheckBox', text='Normalize Image', value=DEFAULTS['norm_image']),
        label_nms       = dict(widget_type='Label', label='<br><b>NMS Postprocessing:</b>'),
        min_size_mask        = dict(widget_type='FloatSpinBox', label='Min Size UNET',     min=0.0, max=1000.0, step=1,  value=DEFAULTS['min_size_mask']),
        min_size       = dict(widget_type='FloatSpinBox', label='Min Size Star',             min=0.0, max=1000.0, step=1,  value=DEFAULTS['min_size']),
        max_size       = dict(widget_type='FloatSpinBox', label='Max Size Star',             min=1000, max=100000.0, step=100,  value=DEFAULTS['max_size']),
        
        perc_low        = dict(widget_type='FloatSpinBox', label='Percentile low',              min=0.0, max=100.0, step=0.1,  value=DEFAULTS['perc_low']),
        perc_high       = dict(widget_type='FloatSpinBox', label='Percentile high',             min=0.0, max=100.0, step=0.1,  value=DEFAULTS['perc_high']),
        norm_axes       = dict(widget_type='LineEdit',     label='Normalization Axes',                                         value=DEFAULTS['norm_axes']),
        prob_thresh     = dict(widget_type='FloatSpinBox', label='Probability/Score Threshold', min=0.0, max=  1.0, step=0.05, value=DEFAULTS['prob_thresh']),
        nms_thresh      = dict(widget_type='FloatSpinBox', label='Overlap Threshold',           min=0.0, max=  1.0, step=0.05, value=DEFAULTS['nms_thresh']),
        output_type     = dict(widget_type='ComboBox', label='Output Type', choices=output_choices, value=DEFAULTS['output_type']),
        label_adv       = dict(widget_type='Label', label='<br><b>Advanced Options:</b>'),
        n_tiles         = dict(widget_type='LiteralEvalLineEdit', label='Number of Tiles', value=DEFAULTS['n_tiles']),
        prob_map_watershed      = dict(widget_type='CheckBox', text='Use Probability Map (watershed)', value=DEFAULTS['prob_map_watershed']),
        dounet      = dict(widget_type='CheckBox', text='Use UNET for binary mask (else denoised)', value=DEFAULTS['dounet']),
        set_thresholds  = dict(widget_type='PushButton', text='Set optimized postprocessing thresholds (for selected model)'),
        defaults_button = dict(widget_type='PushButton', text='Restore Defaults'),
        progress_bar    = dict(label=' ', min=0, max=0, visible=False),
        layout          = 'vertical',
        persist         = True,
        call_button     = True,
    )
    
    def plugin (
       viewer: napari.Viewer,
       label_head,
       image: Image,
       axes,
       label_nn,
       seg_model_type,
       den_model_type,
       model2d_star,
       model3d_star,
       model2d_unet,
       model3d_unet,
       model_den_care,
       model_den_n2v,
       model_folder_star,
       model_folder_unet,
       model_folder_den_care,
       model_folder_den_n2v,
       result_folder,
       model_axes,
       norm_image,
       perc_low,
       perc_high,
       min_size_mask,
       min_size,
       max_size,
       label_nms,
       prob_thresh,
       nms_thresh,
       prob_map_watershed,
       dounet,
       output_type,
       label_adv,
       n_tiles,
       norm_axes,
       set_thresholds,
       defaults_button,
       progress_bar: mw.ProgressBar,
   ) -> List[napari.types.LayerDataTuple]:

       model_star = get_model(*model_selected_star)
       model_unet = get_model(*model_selected_unet)
       model_den_care = get_model(*model_selected_care)
       model_den_n2v = get_model(*model_selected_n2v)
       lkwargs = {}
       x = get_data(image)
       axes = axes_check_and_normalize(axes, length=x.ndim)

       if not axes.replace('T','').startswith(model_star._axes_out.replace('C','')):
           warn(f"output images have different axes ({model_star._axes_out.replace('C','')}) than input image ({axes})")
           # TODO: adjust image.scale according to shuffled axes

       if norm_image:
           axes_norm = axes_check_and_normalize(norm_axes)
           axes_norm = ''.join(set(axes_norm).intersection(set(axes))) # relevant axes present in input image
           assert len(axes_norm) > 0
           # always jointly normalize channels for RGB images
           if ('C' in axes and image.rgb == True) and ('C' not in axes_norm):
               axes_norm = axes_norm + 'C'
               warn("jointly normalizing channels of RGB input image")
           ax = axes_dict(axes)
           _axis = tuple(sorted(ax[a] for a in axes_norm))
           # # TODO: address joint vs. channel/time-separate normalization properly (let user choose)
           # #       also needs to be documented somewhere
           # if 'T' in axes:
           #     if 'C' not in axes or image.rgb == True:
           #          # normalize channels jointly, frames independently
           #          _axis = tuple(i for i in range(x.ndim) if i not in (ax['T'],))
           #     else:
           #         # normalize channels independently, frames independently
           #         _axis = tuple(i for i in range(x.ndim) if i not in (ax['T'],ax['C']))
           # else:
           #     if 'C' not in axes or image.rgb == True:
           #          # normalize channels jointly
           #         _axis = None
           #     else:
           #         # normalize channels independently
           #         _axis = tuple(i for i in range(x.ndim) if i not in (ax['C'],))
           x = normalize(x, perc_low,perc_high, axis=_axis)

       # TODO: progress bar (labels) often don't show up. events not processed?
       if 'T' in axes:
           app = use_app()
           t = axes_dict(axes)['T']
           n_frames = x.shape[t]
           if n_tiles is not None:
               # remove tiling value for time axis
               n_tiles = tuple(v for i,v in enumerate(n_tiles) if i != t)
           def progress(it, **kwargs):
               progress_bar.label = 'VollSeg Prediction (frames)'
               progress_bar.range = (0, n_frames)
               progress_bar.value = 0
               progress_bar.show()
               app.process_events()
               for item in it:
                   yield item
                   progress_bar.increment()
                   app.process_events()
               app.process_events()
       elif n_tiles is not None and np.prod(n_tiles) > 1:
           n_tiles = tuple(n_tiles)
           app = use_app()
           def progress(it, **kwargs):
               progress_bar.label = 'CNN Prediction (tiles)'
               progress_bar.range = (0, kwargs.get('total',0))
               progress_bar.value = 0
               progress_bar.show()
               app.process_events()
               for item in it:
                   yield item
                   progress_bar.increment()
                   app.process_events()
               #
               progress_bar.label = 'NMS Postprocessing'
               progress_bar.range = (0, 0)
               app.process_events()
       else:
           progress = False
           progress_bar.label = 'VollSeg Prediction'
           progress_bar.range = (0, 0)
           progress_bar.show()
           use_app().process_events()

       if 'T' in axes:
           x_reorder = np.moveaxis(x, t, 0)
           axes_reorder = axes.replace('T','')
           
           if isinstance(model_star, VollSeg3D):
          
                  if model_den_care is not None:
                      noise_model = model_den_care
                  if model_den_n2v is not None:
                      noise_model = model_den_n2v 
                  if model_den_care == None and model_den_n2v == None:
                      noise_model = None
                  res = tuple(zip(*tuple( VollSeg3D( _x,  model_unet, model_star, axes=axes_reorder, noise_model = noise_model, prob_thresh=prob_thresh, nms_thresh=nms_thresh, min_size_mask = min_size_mask, min_size = min_size, max_size = max_size,
                  n_tiles = n_tiles, UseProbability = prob_map_watershed, dounet = dounet)
                                                   for _x in progress(x_reorder))))
            
           elif isinstance(model_star, VollSeg2D):
          
                  if model_den_care is not None:
                      noise_model = model_den_care
                  if model_den_n2v is not None:
                      noise_model = model_den_n2v 
                  if model_den_care == None and model_den_n2v == None:
                      noise_model = None
                  res = tuple(zip(*tuple( VollSeg2D( _x,  model_unet, model_star, axes=axes_reorder, noise_model = noise_model, prob_thresh=prob_thresh, nms_thresh=nms_thresh, min_size_mask = min_size_mask, min_size = min_size, max_size = max_size,
                  n_tiles = n_tiles, UseProbability = prob_map_watershed, dounet = dounet)
                                                   for _x in progress(x_reorder))))
               
               
               
          
           if noise_model is not None:
               
               labels, SizedMask, StarImage, ProbabilityMap, Markers, denimage = res
           else:
               labels, SizedMask, StarImage, ProbabilityMap, Markers = res

           labels = np.asarray(labels)

           labels = np.moveaxis(labels, 0, t)

           

       else:
           # TODO: possible to run this in a way that it can be canceled?
           if isinstance(model_star, VollSeg3D):
          
                  if model_den_care is not None:
                      noise_model = model_den_care
                  if model_den_n2v is not None:
                      noise_model = model_den_n2v 
                  if model_den_care == None and model_den_n2v == None:
                      noise_model = None
                  pred =  VollSeg3D( x,  model_unet, model_star, axes=axes_reorder, noise_model = noise_model, prob_thresh=prob_thresh, nms_thresh=nms_thresh, min_size_mask = min_size_mask, min_size = min_size, max_size = max_size,
                  n_tiles = n_tiles, UseProbability = prob_map_watershed, dounet = dounet)
            
           elif isinstance(model_star, VollSeg2D):
          
                  if model_den_care is not None:
                      noise_model = model_den_care
                  if model_den_n2v is not None:
                      noise_model = model_den_n2v 
                  if model_den_care == None and model_den_n2v == None:
                      noise_model = None
                  pred = VollSeg2D( x,  model_unet, model_star, axes=axes_reorder, noise_model = noise_model, prob_thresh=prob_thresh, nms_thresh=nms_thresh, min_size_mask = min_size_mask, min_size = min_size, max_size = max_size,
                  n_tiles = n_tiles, UseProbability = prob_map_watershed, dounet = dounet)
                  
                  
       progress_bar.hide()


       layers = []
     
       if noise_model is not None:
            
            labels, SizedMask, StarImage, ProbabilityMap, Markers, denimage = pred
       else:
            labels, SizedMask, StarImage, ProbabilityMap, Markers = pred
            
       
       # if timeseries, only scale spatial part...
       im_scale = tuple(s for i,s in enumerate(image.scale) if not axes[i] in ('T','C'))
       scale = list(s1*s2 for s1, s2 in zip(im_scale, model_star.config.grid))
       # small correction as napari centers object
       translate = list(0.5*(s-1) for s in model_star.config.grid)
       if 'T' in axes:
            scale.insert(t,1)
            translate.insert(t,0)
        
       layers.append((ProbabilityMap, dict(name='Base Watershed Image',
                                  scale=scale, translate=translate,
                                  **lkwargs), 'image'))

       if output_type in (Output.Labels.value,Output.All.value):
           layers.append((labels, dict(name='VollSeg labels', scale=image.scale, opacity=.5, **lkwargs), 'labels'))
       if output_type in (Output.Binary_mask.value,Output.All.value):
           layers.append((SizedMask, dict(name='VollSeg Binary', scale=image.scale, opacity=.5, **lkwargs), 'binary mask'))
       if output_type in (Output.Denoised_image.value,Output.All.value) and noise_model is not None:
           layers.append((denimage, dict(name='Denoised Image', scale=image.scale, opacity=.5, **lkwargs), 'denoised image'))    
       if output_type in (Output.Markers.value,Output.All.value):
           layers.append((Markers, dict(name='Markers', scale=image.scale, opacity=.5, **lkwargs), 'markers'))    
       return layers
    # -------------------------------------------------------------------------

    # don't want to load persisted values for these widgets
    plugin.axes.value = ''
    plugin.n_tiles.value = DEFAULTS['n_tiles']
    plugin.label_head.value = '<small>VollSeg segmentation for 2D and 3D images.<br>If you are using this in your research please <a href="https://github.com/kapoorlab/vollseg#how-to-cite" style="color:gray;">cite us</a>.</small><br><br><tt><a href="http://conference.scipy.org/proceedings/scipy2021/varun_kapoor.html" style="color:gray;">http://conference.scipy.org/proceedings/scipy2021/varun_kapoor.html</a></tt>'

    # make labels prettier (https://doc.qt.io/qt-5/qsizepolicy.html#Policy-enum)
    for w in (plugin.label_head, plugin.label_nn, plugin.label_nms, plugin.label_adv):
       w.native.setSizePolicy(1|2, 0)

    # -------------------------------------------------------------------------

    widget_for_modeltype = {
       VollSeg2D:   (plugin.model2d_star, plugin.model2d_unet),
       VollSeg3D:   (plugin.model3d_star, plugin.model3d_unet),
       CARE:         plugin.model_den_care,
       N2V:          plugin.model_den_n2v,
       CUSTOM_SEG_MODEL: (plugin.model_folder_star, plugin.model_folder_unet),
       CUSTOM_DEN_MODEL: (plugin.model_folder_den_care, plugin.model_folder_den_n2v),
    }

    def widgets_inactive(*widgets, active):
       for widget in widgets:
           widget.visible = active
           # widget.native.setStyleSheet("" if active else "text-decoration: line-through")

    def widgets_valid(*widgets, valid):
       for widget in widgets:
           widget.native.setStyleSheet("" if valid else "background-color: lightcoral")
   
    # allow some widgets to shrink because their size depends on user input
    plugin.image.native.setMinimumWidth(240)
    plugin.model2d_star.native.setMinimumWidth(240)
    plugin.model3d_star.native.setMinimumWidth(240)
   

    plugin.label_head.native.setOpenExternalLinks(True)
    # make reset button smaller
    # plugin.defaults_button.native.setMaximumWidth(150)

    # plugin.model_axes.native.setReadOnly(True)
    plugin.model_axes.enabled = False

    # push 'call_button' and 'progress_bar' to bottom
    layout = plugin.native.layout()
    layout.insertStretch(layout.count()-2)
   
    class Updater:
        def __init__(self, debug=DEBUG):
            from types import SimpleNamespace
            self.debug = debug
            self.valid = SimpleNamespace(**{k:False for k in ('image_axes', 'model_star', 'model_unet', 'model_den', 'n_tiles', 'norm_axes', 'dounet', 'prob_map_watershed' , 'min_size', 'min_size_mask', 'max_size')})
            self.args  = SimpleNamespace()
            self.viewer = None

        def __call__(self, k, valid, args=None):
            assert k in vars(self.valid)
            setattr(self.valid, k, bool(valid))
            setattr(self.args,  k, args)
            self._update()

        def help(self, msg):
            if self.viewer is not None:
                self.viewer.help = msg
            elif len(str(msg)) > 0:
                print(f"HELP: {msg}")

        def _update(self):

            # try to get a hold of the viewer (can be None when plugin starts)
            if self.viewer is None:
                # TODO: when is this not safe to do and will hang forever?
                # while plugin.viewer.value is None:
                #     time.sleep(0.01)
                if plugin.viewer.value is not None:
                    self.viewer = plugin.viewer.value
                    if DEBUG:
                        print("GOT viewer")

                    @self.viewer.layers.events.removed.connect
                    def _layer_removed(event):
                        layers_remaining = event.source
                        if len(layers_remaining) == 0:
                            plugin.image.tooltip = ''
                            plugin.axes.value = ''
                            plugin.n_tiles.value = 'None'


            def _model(valid):
                widgets_valid(plugin.model2d_star, plugin.model2d_unet, plugin.model3d_star, plugin.model3d_unet, plugin.model_den_n2v, plugin.model_den_care, plugin.model_folder_star.line_edit, plugin.model_folder_unet.line_edit, plugin.model_folder_den_care.line_edit, plugin.model_folder_den_n2v.line_edit,  valid=valid)
                if valid:
                    config = self.args.model
                    axes = config.get('axes', 'ZYXC'[-len(config['net_input_shape']):])
                    if 'T' in axes:
                        raise RuntimeError("model with axis 'T' not supported")
                    plugin.model_axes.value = axes.replace("C", f"C[{config['n_channel_in']}]")
                    plugin.model_folder_star.line_edit.tooltip = ''
                    plugin.model_folder_unet.line_edit.tooltip = ''
                    plugin.model_folder_den_care.line_edit.tooltip = ''
                    plugin.model_folder_den_n2v.line_edit.tooltip = ''
                    
                    
                    return axes, config
                else:
                    plugin.model_axes.value = ''
                    plugin.model_folder_star.line_edit.tooltip = 'Invalid model directory'
                    plugin.model_folder_unet.line_edit.tooltip = 'Invalid model directory'
                    plugin.model_folder_den_care.line_edit.tooltip = 'Invalid model directory'
                    plugin.model_folder_den_n2v.line_edit.tooltip = 'Invalid model directory'

            def _image_axes(valid):
                axes, image, err = getattr(self.args, 'image_axes', (None,None,None))
                widgets_valid(plugin.axes, valid=(valid or (image is None and (axes is None or len(axes) == 0))))
                if valid and 'T' in axes and plugin.output_type.value in (Output.Binary_mask.value,Output.Labels.value,Output.Markers.value, Output.Denoised_image.value, Output.Prob.value , Output.All.value):
                    plugin.output_type.native.setStyleSheet("background-color: orange")
                    plugin.output_type.tooltip = 'Displaying many labels can be very slow.'
                else:
                    plugin.output_type.native.setStyleSheet("")
                    plugin.output_type.tooltip = ''
                if valid:
                    plugin.axes.tooltip = '\n'.join([f'{a} = {s}' for a,s in zip(axes,get_data(image).shape)])
                    return axes, image
                else:
                    if err is not None:
                        err = str(err)
                        err = err[:-1] if err.endswith('.') else err
                        plugin.axes.tooltip = err
                        # warn(err) # alternative to tooltip (gui doesn't show up in ipython)
                    else:
                        plugin.axes.tooltip = ''

            def _norm_axes(valid):
                norm_axes, err = getattr(self.args, 'norm_axes', (None,None))
                widgets_valid(plugin.norm_axes, valid=valid)
                if valid:
                    plugin.norm_axes.tooltip = f"Axes to jointly normalize (if present in selected input image). Note: channels of RGB images are always normalized together."
                    return norm_axes
                else:
                    if err is not None:
                        err = str(err)
                        err = err[:-1] if err.endswith('.') else err
                        plugin.norm_axes.tooltip = err
                        # warn(err) # alternative to tooltip (gui doesn't show up in ipython)
                    else:
                        plugin.norm_axes.tooltip = ''

            def _n_tiles(valid):
                n_tiles, image, err = getattr(self.args, 'n_tiles', (None,None,None))
                widgets_valid(plugin.n_tiles, valid=(valid or image is None))
                if valid:
                    plugin.n_tiles.tooltip = 'no tiling' if n_tiles is None else '\n'.join([f'{t}: {s}' for t,s in zip(n_tiles,get_data(image).shape)])
                    return n_tiles
                else:
                    msg = str(err) if err is not None else ''
                    plugin.n_tiles.tooltip = msg

            def _no_tiling_for_axis(axes_image, n_tiles, axis):
                if n_tiles is not None and axis in axes_image:
                    return n_tiles[axes_dict(axes_image)[axis]] == 1
                return True

            def _restore():
                widgets_valid(plugin.image, valid=plugin.image.value is not None)


            all_valid = False
            help_msg = ''

            if self.valid.image_axes and self.valid.n_tiles and self.valid.model and self.valid.norm_axes:
                axes_image, image  = _image_axes(True)
                axes_model, config = _model(True)
                axes_norm          = _norm_axes(True)
                n_tiles = _n_tiles(True)
                if not _no_tiling_for_axis(axes_image, n_tiles, 'C'):
                    # check if image axes and n_tiles are compatible
                    widgets_valid(plugin.n_tiles, valid=False)
                    err = 'number of tiles must be 1 for C axis'
                    plugin.n_tiles.tooltip = err
                    _restore()
                elif not _no_tiling_for_axis(axes_image, n_tiles, 'T'):
                    # check if image axes and n_tiles are compatible
                    widgets_valid(plugin.n_tiles, valid=False)
                    err = 'number of tiles must be 1 for T axis'
                    plugin.n_tiles.tooltip = err
                    _restore()
                elif set(axes_norm).isdisjoint(set(axes_image)):
                    # check if image axes and normalization axes are compatible
                    widgets_valid(plugin.norm_axes, valid=False)
                    err = f"Image axes ({axes_image}) must contain at least one of the normalization axes ({', '.join(axes_norm)})"
                    plugin.norm_axes.tooltip = err
                    _restore()
                elif 'T' in axes_image and config.get('n_dim') == 3 and plugin.output_type.value in  (Output.Binary_mask.value,Output.Labels.value,Output.Markers.value, Output.Denoised_image.value, Output.Prob.value , Output.All.value):
                    # not supported
                    widgets_valid(plugin.output_type, valid=False)
                    plugin.output_type.tooltip = '3D timelapse data'
                    _restore()
                else:
                    # check if image and model are compatible
                    ch_model = config['n_channel_in']
                    ch_image = get_data(image).shape[axes_dict(axes_image)['C']] if 'C' in axes_image else 1
                    all_valid = set(axes_model.replace('C','')) == set(axes_image.replace('C','').replace('T','')) and ch_model == ch_image

                    widgets_valid(plugin.image, plugin.model2d_star, plugin.model2d_unet, plugin.model3d_star, plugin.model3d_unet, plugin.model_den_care, plugin.model_den_n2v, plugin.model_folder_star.line_edit, plugin.model_folder_unet.line_edit, plugin.model_folder_den_care.line_edit, plugin.model_folder_den_n2v.line_edit, valid=all_valid)
                    if all_valid:
                        help_msg = ''
                    else:
                        help_msg = f'Model with axes {axes_model.replace("C", f"C[{ch_model}]")} and image with axes {axes_image.replace("C", f"C[{ch_image}]")} not compatible'
            else:
                _image_axes(self.valid.image_axes)
                _norm_axes(self.valid.norm_axes)
                _n_tiles(self.valid.n_tiles)
                _model(self.valid.model)
                _restore()

            self.help(help_msg)
            plugin.call_button.enabled = all_valid
            # widgets_valid(plugin.call_button, valid=all_valid)
            if self.debug:
                print(f"valid ({all_valid}):", ', '.join([f'{k}={v}' for k,v in vars(self.valid).items()]))

    update = Updater()


    def select_model(key_star, key_unet, key_den_care, key_den_n2v):
        nonlocal model_selected_star, model_selected_unet, model_selected_den_care, model_selected_den_n2v 
        model_selected_star = key_star
        config_star = model_star_configs.get(key_star)
        update('model_star', config_star is not None, config_star)
        
        model_selected_unet = key_unet
        config_unet = model_unet_configs.get(key_unet)
        update('model_unet', config_unet is not None, config_unet)
        
        
        model_selected_den_care = key_den_care
        config_den_care = model_den_care_configs.get(key_den_care)
        update('model_den_care', config_den_care is not None, config_den_care)
        
        model_selected_den_n2v = key_den_n2v
        config_den_n2v = model_den_n2v_configs.get(key_den_n2v)
        update('model_den_n2v', config_den_n2v is not None, config_den_n2v)
        

    # -------------------------------------------------------------------------

    # hide percentile selection if normalization turned off
    @change_handler(plugin.norm_image)
    def _norm_image_change(active: bool):
        widgets_inactive(plugin.perc_low, plugin.perc_high, plugin.norm_axes, active=active)
    
    @change_handler(plugin.dounet)
    def _dounet_change(active: bool):
        widgets_inactive(plugin.dounet, active=active)
        
    @change_handler(plugin.dounet)
    def _prob_map_watershed_change(active: bool):
        widgets_inactive(plugin.prob_map_watershed, active=active)
                         
                         
    # ensure that percentile low < percentile high
    @change_handler(plugin.perc_low)
    def _perc_low_change():
        plugin.perc_high.value = max(plugin.perc_low.value+0.01, plugin.perc_high.value)

    @change_handler(plugin.perc_high)
    def _perc_high_change():
        plugin.perc_low.value  = min(plugin.perc_low.value, plugin.perc_high.value-0.01)

    @change_handler(plugin.norm_axes)
    def _norm_axes_change(value: str):
        if value != value.upper():
            with plugin.axes.changed.blocked():
                plugin.norm_axes.value = value.upper()
        try:
            axes = axes_check_and_normalize(value, disallowed='S')
            if len(axes) >= 1:
                update('norm_axes', True, (axes, None))
            else:
                update('norm_axes', False, (axes, 'Cannot be empty'))
        except ValueError as err:
            update('norm_axes', False, (value, err))

    # -------------------------------------------------------------------------

    # RadioButtons widget triggers a change event initially (either when 'value' is set in constructor, or via 'persist')
    # TODO: seems to be triggered too when a layer is added or removed (why?)
    @change_handler(plugin.model_type, init=False)
    def _model_type_change(model_type: Union[str, type]):
        selected = widget_for_modeltype[model_type]
        for w in set((plugin.model2d_star, plugin.model2d_unet, plugin.model3d_star, plugin.model3d_unet, plugin.model_den_care, plugin.model_den_n2v, plugin.model_folder_star, plugin.model_folder_unet, plugin.model_folder_den_care, plugin.model_folder_den_n2v )) - {selected}:
            w.hide()
        selected.show()
        # trigger _model_change
        selected.changed(selected.value)


    # show/hide model folder picker
    # load config/thresholds for selected pretrained model
    # -> triggered by _model_type_change
    @change_handler(plugin.model2d_star, plugin.model2d_unet, plugin.model3d_star, plugin.model3d_unet, plugin.model_den_care, plugin.model_den_n2v, init=False)
    def _model_change(model_name_star: str, model_name_unet: str, model_name_den: str):
        model_class_star, model_class_unet = VollSeg2D if Signal.sender() is plugin.model2d_star else VollSeg3D
        model_class_den = CARE if Signal.sender() is plugin.model_den_care else N2V
        
        key_star = model_class_star, model_name_star
        key_unet =  model_class_unet, model_name_unet
        key_den = model_class_den, model_name_den
        if key_star not in model_star_configs:
            @thread_worker
            def _get_model_folder():
                return get_model_folder(*key_star), get_model_folder(*key_unet), get_model_folder(*key_den)

            def _process_model_folder(path):
                try:
                    model_star_configs[key_star] = load_json(str(path/'config.json'))
                    try:
                        # not all models have associated thresholds
                        model_star_threshs[key_star] = load_json(str(path/'thresholds.json'))
                    except FileNotFoundError:
                        pass
                finally:
                    select_model(key_star)
                    plugin.progress_bar.hide()

            worker = _get_model_folder()
            worker.returned.connect(_process_model_folder)
            worker.start()

            # delay showing progress bar -> won't show up if model already downloaded
            # TODO: hacky -> better way to do this?
            time.sleep(0.1)
            plugin.call_button.enabled = False
            plugin.progress_bar.label = 'Downloading model'
            plugin.progress_bar.show()

        else:
            select_model(key_star)


    # load config/thresholds from custom model path
    # -> triggered by _model_type_change
    # note: will be triggered at every keystroke (when typing the path)
    @change_handler(plugin.model_folder, init=False)
    def _model_folder_change(_path: str):
        path = Path(_path)
        key = CUSTOM_SEG_MODEL, path
        try:
            if not path.is_dir(): return
            model_star_configs[key] = load_json(str(path/'config.json'))
            model_star_threshs[key] = load_json(str(path/'thresholds.json'))
        except FileNotFoundError:
            pass
        finally:
            select_model(key)

    # -------------------------------------------------------------------------

    # -> triggered by napari (if there are any open images on plugin launch)
    @change_handler(plugin.image, init=False)
    def _image_change(image: napari.layers.Image):
        ndim = get_data(image).ndim
        plugin.image.tooltip = f"Shape: {get_data(image).shape}"

        # dimensionality of selected model: 2, 3, or None (unknown)
        ndim_model = None
        if plugin.model_type.value == VollSeg2D:
            ndim_model = 2
        elif plugin.model_type.value == VollSeg3D:
            ndim_model = 3
        else:
            if model_selected_star in model_star_configs:
                config = model_star_configs[model_selected_star]
                ndim_model = config.get('n_dim')

        # TODO: guess images axes better...
        axes = None
        if ndim == 2:
            axes = 'YX'
        elif ndim == 3:
            axes = 'YXC' if image.rgb else ('ZYX' if ndim_model == 3 else 'TYX')
        elif ndim == 4:
            axes = ('ZYXC' if ndim_model == 3 else 'TYXC') if image.rgb else 'TZYX'
        else:
            raise NotImplementedError()

        if (axes == plugin.axes.value):
            # make sure to trigger a changed event, even if value didn't actually change
            plugin.axes.changed(axes)
        else:
            plugin.axes.value = axes
        plugin.n_tiles.changed(plugin.n_tiles.value)
        plugin.norm_axes.changed(plugin.norm_axes.value)


    # -> triggered by _image_change
    @change_handler(plugin.axes, init=False)
    def _axes_change(value: str):
        if value != value.upper():
            with plugin.axes.changed.blocked():
                plugin.axes.value = value.upper()
        image = plugin.image.value
        axes = ''
        try:
            image is not None or _raise(ValueError("no image selected"))
            axes = axes_check_and_normalize(value, length=get_data(image).ndim, disallowed='S')
            update('image_axes', True, (axes, image, None))
        except ValueError as err:
            update('image_axes', False, (value, image, err))
        finally:
            widgets_inactive(plugin.timelapse_opts, active=('T' in axes))


    @change_handler(plugin.min_size, init=False)
    def _min_size_change():
        value = plugin.min_size.get_value()
        update(plugin.min_size, value)
    
    @change_handler(plugin.min_size_mask, init=False)
    def _min_size_mask_change():
        value = plugin.min_size_mask.get_value()
        update(plugin.min_size_mask, value)
        
    @change_handler(plugin.max_size, init=False)
    def _max_size_change():
        value = plugin.max_size.get_value()
        update(plugin.max_size, value)    
    
    # -> triggered by _image_change
    @change_handler(plugin.n_tiles, init=False)
    def _n_tiles_change():
        image = plugin.image.value
        try:
            image is not None or _raise(ValueError("no image selected"))
            value = plugin.n_tiles.get_value()
            if value is None:
                update('n_tiles', True, (None, image, None))
                return
            shape = get_data(image).shape
            try:
                value = tuple(value)
                len(value) == len(shape) or _raise(TypeError())
            except TypeError:
                raise ValueError(f'must be a tuple/list of length {len(shape)}')
            if not all(isinstance(t,int) and t >= 1 for t in value):
                raise ValueError(f'each value must be an integer >= 1')
            update('n_tiles', True, (value, image, None))
        except (ValueError, SyntaxError) as err:
            update('n_tiles', False, (None, image, err))


    # -------------------------------------------------------------------------

    # set thresholds to optimized values for chosen model
    @change_handler(plugin.set_thresholds, init=False)
    def _set_thresholds():
        if model_selected_star in model_star_threshs:
            thresholds = model_star_threshs[model_selected_star]
            plugin.nms_thresh.value = thresholds['nms']
            plugin.prob_thresh.value = thresholds['prob']

    # output type changed
    @change_handler(plugin.output_type, init=False)
    def _output_type_change():
        update._update()

    # restore defaults
    @change_handler(plugin.defaults_button, init=False)
    def restore_defaults():
        for k,v in DEFAULTS.items():
            getattr(plugin,k).value = v

    # -------------------------------------------------------------------------

    # allow some widgets to shrink because their size depends on user input
    plugin.image.native.setMinimumWidth(240)
    plugin.model2d.native.setMinimumWidth(240)
    plugin.model3d.native.setMinimumWidth(240)
    plugin.timelapse_opts.native.setMinimumWidth(240)

    plugin.label_head.native.setOpenExternalLinks(True)
    # make reset button smaller
    # plugin.defaults_button.native.setMaximumWidth(150)

    # plugin.model_axes.native.setReadOnly(True)
    plugin.model_axes.enabled = False

    # push 'call_button' and 'progress_bar' to bottom
    layout = plugin.native.layout()
    layout.insertStretch(layout.count()-2)
  
   
    return plugin
   
    
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
