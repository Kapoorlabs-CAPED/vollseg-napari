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
    _models2d_star, _aliases2d_star = get_registered_models(StarDist2D)
    _models3d_star, _aliases3d_star = get_registered_models(StarDist3D)
    # use first alias for model selection (if alias exists)
    models2d_star = [((_aliases2d_star[m][0] if len(_aliases2d_star[m]) > 0 else m),m) for m in _models2d_star]
    models3d_star = [((_aliases3d_star[m][0] if len(_aliases3d_star[m]) > 0 else m),m) for m in _models3d_star]
    
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
    model_selected_star = None
    model_selected_unet = None
    model_selected_care = None
    model_selected_n2v = None
    
    CUSTOM_MODEL = 'CUSTOM_MODEL'
    seg_model_type_choices = [('2D', StarDist2D), ('3D', StarDist3D), ('Custom 2D/3D', CUSTOM_MODEL)]
    den_model_type_choices = [ ('DenoiseCARE', CARE) , ('DenoiseN2V', N2V) ('Custom N2V/CARE', CUSTOM_MODEL)]
    @functools.lru_cache(maxsize=None)
    def get_model(seg_model_type, den_model_type, model_star, model_unet, model_care, model_n2v):
        if seg_model_type == CUSTOM_MODEL:
            path_star = Path(model_star)
            path_star.is_dir() or _raise(FileNotFoundError(f"{path_star} is not a directory"))
            
            path_unet = Path(model_unet)
            path_unet.is_dir() or _raise(FileNotFoundError(f"{path_unet} is not a directory"))
            
            config = model_configs[(model_type,model_star)]
            model_class = StarDist2D if config['n_dim'] == 2 else StarDist3D
            if den_model_type == CUSTOM_MODEL:    
                if model_care is not None:
                    path_care = Path(model_care)
                    path_care.is_dir() or _raise(FileNotFoundError(f"{path_care} is not a directory"))
                    return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent)), CARE(None, name=path_care.name, basedir=str(path_care.parent))
                if model_n2v is not None:
                    path_n2v = Path(model_n2v)
                    path_n2v.is_dir() or _raise(FileNotFoundError(f"{path_n2v} is not a directory"))
                    return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent)), N2V(None, name=path_n2v.name, basedir=str(path_n2v.parent))
    
            elif den_model_type != CUSTOM_MODEL and model_care is not None:
                
                 return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent)), model_type.from_pretrained(model_care)
             
            elif den_model_type != CUSTOM_MODEL and model_n2v is not None:
                
                 return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent)), model_type.from_pretrained(model_n2v)
                 
                
            elif den_model_type != CUSTOM_MODEL and model_care == None and model_n2v == None:
                    
                     return model_class(None, name=path_star.name, basedir=str(path_star.parent)), CARE(None, name=path_unet.name, basedir=str(path_unet.parent))
        
        else:
            
              if den_model_type == CUSTOM_MODEL:    
                  if model_care is not None:
                      path_care = Path(model_care)
                      path_care.is_dir() or _raise(FileNotFoundError(f"{path_care} is not a directory"))
                      return model_type.from_pretrained(model_star), model_type.from_pretrained(model_unet), CARE(None, name=path_care.name, basedir=str(path_care.parent))
                  if model_n2v is not None:
                      path_n2v = Path(model_n2v)
                      path_n2v.is_dir() or _raise(FileNotFoundError(f"{path_n2v} is not a directory"))
                      return model_type.from_pretrained(model_star), model_type.from_pretrained(model_unet), N2V(None, name=path_n2v.name, basedir=str(path_n2v.parent))
      
              elif den_model_type != CUSTOM_MODEL and model_n2v is not None:
                  
                   return model_type.from_pretrained(model_star), model_type.from_pretrained(model_unet), model_type.from_pretrained(model_n2v)
                   
              elif den_model_type != CUSTOM_MODEL and model_care is not None:
                  
                   return model_type.from_pretrained(model_star), model_type.from_pretrained(model_unet), model_type.from_pretrained(model_care)    
               
                
              elif den_model_type != CUSTOM_MODEL and model_care == None and model_n2v == None:
                      
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
            model2d_star        = models2d_star[0][1],
            model2d_unet   = models2d_unet[0][1],
            model3d_star        = models3d_star[0][1],
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
        model2d_star    = dict(widget_type='ComboBox', visible=False, label='Pre-trained StarDist Model', choices=models2d_star, value=DEFAULTS['model2d_star']),
        model3d_star    = dict(widget_type='ComboBox', visible=False, label='Pre-trained StarDist Model', choices=models3d_star, value=DEFAULTS['model3d_star']),
        
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
    
    def plugin (
       viewer: napari.Viewer,
       label_head,
       image: napari.layers.Image,
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
       model_axes,
       norm_image,
       perc_low,
       perc_high,
       label_nms,
       prob_thresh,
       nms_thresh,
       output_type,
       label_adv,
       n_tiles,
       norm_axes,
       timelapse_opts,
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

       if not axes.replace('T','').startswith(model._axes_out.replace('C','')):
           warn(f"output images have different axes ({model._axes_out.replace('C','')}) than input image ({axes})")
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
           
           
           
           res = tuple(zip(*tuple(model.predict_instances(_x, axes=axes_reorder,
                                           prob_thresh=prob_thresh, nms_thresh=nms_thresh,
                                           n_tiles=n_tiles,
                                           sparse=(not cnn_output), return_predict=cnn_output)
                                            for _x in progress(x_reorder))))

           if cnn_output:
               labels, polys = tuple(zip(*res[0]))
               cnn_output = tuple(np.stack(c, t) for c in tuple(zip(*res[1])))
           else:
               labels, polys = res

           labels = np.asarray(labels)

           if len(polys) > 1:
               if timelapse_opts == TimelapseLabels.Match.value:
                   # match labels in consecutive frames (-> simple IoU tracking)
                   labels = match_labels(labels, iou_threshold=0)
               elif timelapse_opts == TimelapseLabels.Unique.value:
                   # make label ids unique (shift by offset)
                   offsets = np.cumsum([len(p['points']) for p in polys])
                   for y,off in zip(labels[1:],offsets):
                       y[y>0] += off
               elif timelapse_opts == TimelapseLabels.Separate.value:
                   # each frame processed separately (nothing to do)
                   pass
               else:
                   raise NotImplementedError(f"unknown option '{timelapse_opts}' for time-lapse labels")

           labels = np.moveaxis(labels, 0, t)

           if isinstance(model, StarDist3D):
               # TODO poly output support for 3D timelapse
               polys = None
           else:
               polys = dict(
                   coord =  np.concatenate(tuple(np.insert(p['coord'],  t, _t, axis=-2) for _t,p in enumerate(polys)),axis=0),
                   points = np.concatenate(tuple(np.insert(p['points'], t, _t, axis=-1) for _t,p in enumerate(polys)),axis=0)
               )

           if cnn_output:
               pred = (labels, polys), cnn_output
           else:
               pred = labels, polys

       else:
           # TODO: possible to run this in a way that it can be canceled?
           pred = model.predict_instances(x, axes=axes, prob_thresh=prob_thresh, nms_thresh=nms_thresh,
                                          n_tiles=n_tiles, show_tile_progress=progress,
                                          sparse=(not cnn_output), return_predict=cnn_output)
       progress_bar.hide()


       layers = []
       if cnn_output:
           (labels,polys), cnn_out = pred
           prob, dist = cnn_out[:2]
           # if timeseries, only scale spatial part...
           im_scale = tuple(s for i,s in enumerate(image.scale) if not axes[i] in ('T','C'))
           scale = list(s1*s2 for s1, s2 in zip(im_scale, model.config.grid))
           # small correction as napari centers object
           translate = list(0.5*(s-1) for s in model.config.grid)
           if 'T' in axes:
               scale.insert(t,1)
               translate.insert(t,0)
           dist = np.moveaxis(dist, -1,0)
           layers.append((dist, dict(name='StarDist distances',
                                     scale=[1]+scale, translate=[0]+translate,
                                     **lkwargs), 'image'))
           layers.append((prob, dict(name='StarDist probability',
                                     scale=scale, translate=translate,
                                     **lkwargs), 'image'))
       else:
           labels, polys = pred

       if output_type in (Output.Labels.value,Output.Both.value):
           layers.append((labels, dict(name='StarDist labels', scale=image.scale, opacity=.5, **lkwargs), 'labels'))
       if output_type in (Output.Polys.value,Output.Both.value):
           n_objects = len(polys['points'])
           if isinstance(model, StarDist3D):
               surface = surface_from_polys(polys)
               layers.append((surface, dict(name='StarDist polyhedra',
                                            contrast_limits=(0,surface[-1].max()),
                                            scale=image.scale,
                                            colormap=label_colormap(n_objects), **lkwargs), 'surface'))
           else:
               # TODO: sometimes hangs for long time (indefinitely?) when returning many polygons (?)
               #       seems to be a known issue: https://github.com/napari/napari/issues/2015
               # TODO: coordinates correct or need offset (0.5 or so)?
               shapes = np.moveaxis(polys['coord'], -1,-2)
               layers.append((shapes, dict(name='StarDist polygons', shape_type='polygon',
                                           scale=image.scale,
                                           edge_width=0.75, edge_color='yellow', face_color=[0,0,0,0], **lkwargs), 'shapes'))
       return layers

   # -------------------------------------------------------------------------

   # don't want to load persisted values for these widgets
   plugin.axes.value = ''
   plugin.n_tiles.value = DEFAULTS['n_tiles']
   plugin.label_head.value = '<small>Star-convex object detection for 2D and 3D images.<br>If you are using this in your research please <a href="https://github.com/stardist/stardist#how-to-cite" style="color:gray;">cite us</a>.</small><br><br><tt><a href="https://stardist.net" style="color:gray;">https://stardist.net</a></tt>'

   # make labels prettier (https://doc.qt.io/qt-5/qsizepolicy.html#Policy-enum)
   for w in (plugin.label_head, plugin.label_nn, plugin.label_nms, plugin.label_adv):
       w.native.setSizePolicy(1|2, 0)

   # -------------------------------------------------------------------------

   widget_for_modeltype = {
       StarDist2D:   plugin.model2d,
       StarDist3D:   plugin.model3d,
       CUSTOM_MODEL: plugin.model_folder,
   }

   def widgets_inactive(*widgets, active):
       for widget in widgets:
           widget.visible = active
           # widget.native.setStyleSheet("" if active else "text-decoration: line-through")

   def widgets_valid(*widgets, valid):
       for widget in widgets:
           widget.native.setStyleSheet("" if valid else "background-color: lightcoral")

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