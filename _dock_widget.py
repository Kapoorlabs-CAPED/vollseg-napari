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
    
    
    model_configs = dict()
    model_threshs = dict()
    model_selected_star = None
    model_selected_unet = None
    model_selected_care = None
    model_selected_n2v = None
    
    CUSTOM_SEG_MODEL = 'CUSTOM_SEG_MODEL'
    CUSTOM_DEN_MODEL = 'CUSTOM_DEN_MODEL'
    seg_model_type_choices = [('2D', VollSeg2D), ('3D', VollSeg3D), ('Custom 2D/3D', CUSTOM_SEG_MODEL)]
    den_model_type_choices = [ ('DenoiseCARE', CARE) , ('DenoiseN2V', N2V), ('Custom N2V/CARE', CUSTOM_DEN_MODEL)]
    @functools.lru_cache(maxsize=None)
    def get_model(seg_model_type, den_model_type, model_star, model_unet, model_care, model_n2v):
        if seg_model_type == CUSTOM_SEG_MODEL:
            path_star = Path(model_star)
            path_star.is_dir() or _raise(FileNotFoundError(f"{path_star} is not a directory"))
            
            path_unet = Path(model_unet)
            path_unet.is_dir() or _raise(FileNotFoundError(f"{path_unet} is not a directory"))
            
            config = model_configs[(seg_model_type,model_star)]
            model_class = StarDist2D if config['n_dim'] == 2 else StarDist3D
            if den_model_type == CUSTOM_DEN_MODEL:    
                if model_care is not None:
                    path_care = Path(model_care)
                    path_care.is_dir() or _raise(FileNotFoundError(f"{path_care} is not a directory"))
                    return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent)), CARE(None, name=path_care.name, basedir=str(path_care.parent))
                if model_n2v is not None:
                    path_n2v = Path(model_n2v)
                    path_n2v.is_dir() or _raise(FileNotFoundError(f"{path_n2v} is not a directory"))
                    return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent)), N2V(None, name=path_n2v.name, basedir=str(path_n2v.parent))
    
            elif den_model_type != CUSTOM_DEN_MODEL and model_care is not None:
                
                 return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent)), den_model_type.from_pretrained(model_care)
             
            elif den_model_type != CUSTOM_DEN_MODEL and model_n2v is not None:
                
                 return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent)), den_model_type.from_pretrained(model_n2v)
                 
                
            elif den_model_type != CUSTOM_DEN_MODEL and model_care == None and model_n2v == None:
                    
                     return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent))
        
        else:
            
              if den_model_type == CUSTOM_DEN_MODEL:    
                  if model_care is not None:
                      path_care = Path(model_care)
                      path_care.is_dir() or _raise(FileNotFoundError(f"{path_care} is not a directory"))
                      return seg_model_type.from_pretrained(model_star), seg_model_type.from_pretrained(model_unet), CARE(None, name=path_care.name, basedir=str(path_care.parent))
                  if model_n2v is not None:
                      path_n2v = Path(model_n2v)
                      path_n2v.is_dir() or _raise(FileNotFoundError(f"{path_n2v} is not a directory"))
                      return seg_model_type.from_pretrained(model_star), seg_model_type.from_pretrained(model_unet), N2V(None, name=path_n2v.name, basedir=str(path_n2v.parent))
      
              elif den_model_type != CUSTOM_DEN_MODEL and model_n2v is not None:
                  
                   return seg_model_type.from_pretrained(model_star), seg_model_type.from_pretrained(model_unet), den_model_type.from_pretrained(model_n2v)
                   
              elif den_model_type != CUSTOM_DEN_MODEL and model_care is not None:
                  
                   return seg_model_type.from_pretrained(model_star), seg_model_type.from_pretrained(model_unet), den_model_type.from_pretrained(model_care)    
               
                
              elif den_model_type != CUSTOM_DEN_MODEL and model_care == None and model_n2v == None:
                      
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
        
        model_den_care   = dict(widget_type='ComboBox', visible=False, label='Pre-trained CARE Model', choices=models_den_care, value=DEFAULTS['model_den_care']),
        model_den_n2v   = dict(widget_type='ComboBox', visible=False, label='Pre-trained N2V Model', choices=models_den_n2v, value=DEFAULTS['model_den_n2v']), 
       
        
        model_folder    = dict(widget_type='FileEdit', visible=False, label='Custom Model', mode='d'),
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
       model_den_n2v,
       model_den_care,
       model_folder,
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
       model_care = get_model(*model_selected_care)
       model_n2v = get_model(*model_selected_n2v)
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
          
                  if model_care is not None:
                      noise_model = model_care
                  if model_n2v is not None:
                      noise_model = model_n2v 
                  if model_care == None and model_n2v == None:
                      noise_model = None
                  res = tuple(zip(*tuple( VollSeg3D( _x,  model_unet, model_star, axes=axes_reorder, noise_model = noise_model, prob_thresh=prob_thresh, nms_thresh=nms_thresh, min_size_mask = min_size_mask, min_size = min_size, max_size = max_size,
                  n_tiles = n_tiles, UseProbability = prob_map_watershed, dounet = dounet)
                                                   for _x in progress(x_reorder))))
            
           elif isinstance(model_star, VollSeg2D):
          
                  if model_care is not None:
                      noise_model = model_care
                  if model_n2v is not None:
                      noise_model = model_n2v 
                  if model_care == None and model_n2v == None:
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
          
                  if model_care is not None:
                      noise_model = model_care
                  if model_n2v is not None:
                      noise_model = model_n2v 
                  if model_care == None and model_n2v == None:
                      noise_model = None
                  pred =  VollSeg3D( x,  model_unet, model_star, axes=axes_reorder, noise_model = noise_model, prob_thresh=prob_thresh, nms_thresh=nms_thresh, min_size_mask = min_size_mask, min_size = min_size, max_size = max_size,
                  n_tiles = n_tiles, UseProbability = prob_map_watershed, dounet = dounet)
            
           elif isinstance(model_star, VollSeg2D):
          
                  if model_care is not None:
                      noise_model = model_care
                  if model_n2v is not None:
                      noise_model = model_n2v 
                  if model_care == None and model_n2v == None:
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
       CUSTOM_SEG_MODEL: plugin.model_folder,
       CUSTOM_DEN_MODEL: plugin.model_folder,
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
