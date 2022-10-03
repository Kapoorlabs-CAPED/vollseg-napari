# -*- coding: utf-8 -*-
'''
Created on Wed Dec  8 16:15:34 2021
@author: vkapoor
'''

from napari_plugin_engine import napari_hook_implementation
from magicgui import magicgui
from magicgui import widgets as mw
from psygnal import Signal
from magicgui.application import use_app
import napari
from napari.qt.threading import thread_worker
from typing import List, Union
import functools
import time
import numpy as np
from pathlib import Path
from warnings import warn
from tifffile import imread
from vollseg import inrimage, h5
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QSizePolicy
)


def plugin_wrapper_vollseg():

    from csbdeep.utils import _raise, axes_check_and_normalize, axes_dict
    from vollseg.pretrained import get_registered_models, get_model_folder
    from csbdeep.utils import load_json
    from vollseg import VollSeg, CARE, UNET, StarDist2D, StarDist3D, MASKUNET
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
                   
                    print(f'{str(emitter.name).upper()}: {source.name} = {args!r}')
                return handler(*args)

            for widget in widgets:
                widget.changed.connect(wrapper)
                if init:
                    widget.changed(widget.value)
            return wrapper

        return decorator_change_handler

   
   

    model_star_configs = dict()
    model_unet_configs = dict()
    model_den_configs = dict()
    model_roi_configs = dict()
    model_star_threshs = dict()
    model_selected_star = None
    model_selected_unet = None
    model_selected_den = None
    model_selected_roi = None
    worker = None
    DEFAULTS_MODEL = dict(
        star_seg_model_type=StarDist3D,
        unet_seg_model_type=UNET,
        den_model_type=CARE,
        roi_model_type = MASKUNET,
        model_den_none='NODEN',
        model_star_none='NOSTAR',
        model_unet_none='NOUNET',
        model_roi_none='NOROI',
        norm_axes='ZYX',
        axes = 'ZYX'
    )

    DEFAULTS_STAR_PARAMETERS = dict(
        norm_image=True,
        perc_low=1.0,
        perc_high=99.8,
        prob_thresh=0.5,
        nms_thresh=0.4,
        n_tiles=(1,1,1),
    )

    DEFAULTS_VOLL_PARAMETERS = dict(
        min_size_mask=1.0,
        min_size=1.0,
        max_size=10000000.0,
        isRGB = False,
        dounet=True,
        slicemerge = False,
        seedpool = True,
        iouthresh = 0.5,
        prob_map_watershed=True,
    )
    
    DEFAULTS_DISPLAY_PARAMETERS = dict(
        
        display_prob=True,
        display_vollseg = True,
        display_skeleton = True,
        display_stardist = True,
        display_unet = True,
        display_denoised = True,
        display_markers = True,
        display_roi = True,
        
    ) 

    CUSTOM_SEG_MODEL_STAR = 'CUSTOM_SEG_MODEL_STAR'
    CUSTOM_SEG_MODEL_UNET = 'CUSTOM_SEG_MODEL_UNET'
    CUSTOM_DEN_MODEL = 'CUSTOM_DEN_MODEL'
    CUSTOM_ROI_MODEL = 'CUSTOM_ROI_MODEL'
    star_seg_model_type_choices = [
        ('PreTrained2D', StarDist2D),
        ('PreTrained3D', StarDist3D),
        ('NOSTAR', 'NOSTAR'),
        ('Custom STAR', CUSTOM_SEG_MODEL_STAR),
    ]
    unet_seg_model_type_choices = [
        ('PreTrained', UNET),
        ('NOUNET', 'NOUNET'),
        ('Custom UNET', CUSTOM_SEG_MODEL_UNET),
    ]
    den_model_type_choices = [
        ('PreTrained', CARE),
        ('NODEN', 'NODEN'),
        ('Custom CARE', CUSTOM_DEN_MODEL),
    ]

    roi_model_type_choices = [
        ('PreTrained', MASKUNET),
        ('NOROI', 'NOROI'),
        ('Custom ROI', CUSTOM_ROI_MODEL)
    ]

    @functools.lru_cache(maxsize=None)
    def get_model_star(star_seg_model_type, model_star):
        if star_seg_model_type == CUSTOM_SEG_MODEL_STAR:
            path_star = Path(model_star)
            path_star.is_dir() or _raise(
                FileNotFoundError(f'{path_star} is not a directory')
            )
            config_star = model_star_configs[(star_seg_model_type, model_star)]
            model_class_star = StarDist2D if config_star['n_dim'] == 2 else StarDist3D
            return model_class_star(
                None, name=path_star.name, basedir=str(path_star.parent)
            )
       
        elif star_seg_model_type !=DEFAULTS_MODEL['model_star_none']:
            return star_seg_model_type.local_from_pretrained(model_star)
        else:
            return None
    
    @functools.lru_cache(maxsize=None)
    def get_model_roi(roi_model_type, model_roi):

        if roi_model_type == CUSTOM_ROI_MODEL:
            path_roi = Path(model_roi)
            path_roi.is_dir() or _raise(
                FileNotFoundError(f'{path_roi} is not a directory')
            )
            model_class_roi = MASKUNET 
            return model_class_roi(
                None, name=path_roi.name, basedir=str(path_roi.parent)
            )
        
        elif roi_model_type !=DEFAULTS_MODEL['model_roi_none']:
            return roi_model_type.local_from_pretrained(model_roi)

        else:
            return None


    @functools.lru_cache(maxsize=None)
    def get_model_unet(unet_seg_model_type, model_unet):
        if unet_seg_model_type == CUSTOM_SEG_MODEL_UNET:
            path_unet = Path(model_unet)
            path_unet.is_dir() or _raise(
                FileNotFoundError(f'{path_unet} is not a directory')
            )
            model_class_unet = UNET
            return model_class_unet(
                None, name=path_unet.name, basedir=str(path_unet.parent)
            )
        

        elif unet_seg_model_type !=DEFAULTS_MODEL['model_unet_none']:
            return unet_seg_model_type.local_from_pretrained(model_unet)
        else:
            
            return None

    @functools.lru_cache(maxsize=None)
    def get_model_den(den_model_type, model_den):
        if den_model_type == CUSTOM_DEN_MODEL:
            path_den = Path(model_den)
            path_den.is_dir() or _raise(
                FileNotFoundError(f'{path_den} is not a directory')
            )
            model_class_den = CARE
            return model_class_den(
                None, name=path_den.name, basedir=str(path_den.parent)
            )
        elif den_model_type != DEFAULTS_MODEL['model_den_none']:
            return den_model_type.local_from_pretrained(model_den)
        else:
           return None


    

    @magicgui(
        norm_image=dict(
            widget_type='CheckBox',
            text='Normalize Image',
            value=DEFAULTS_STAR_PARAMETERS['norm_image'],
        ),
        perc_low=dict(
            widget_type='FloatSpinBox',
            label='Percentile low',
            min=0.0,
            max=100.0,
            step=0.1,
            value=DEFAULTS_STAR_PARAMETERS['perc_low'],
        ),
        perc_high=dict(
            widget_type='FloatSpinBox',
            label='Percentile high',
            min=0.0,
            max=100.0,
            step=0.1,
            value=DEFAULTS_STAR_PARAMETERS['perc_high'],
        ),
        prob_thresh=dict(
            widget_type='FloatSpinBox',
            label='Probability/Score Threshold',
            min=0.0,
            max=1.0,
            step=0.05,
            value=DEFAULTS_STAR_PARAMETERS['prob_thresh'],
        ),
        nms_thresh=dict(
            widget_type='FloatSpinBox',
            label='Overlap Threshold',
            min=0.0,
            max=1.0,
            step=0.05,
            value=DEFAULTS_STAR_PARAMETERS['nms_thresh'],
        ),
        set_thresholds=dict(
            widget_type='PushButton',
            text='Set optimized postprocessing thresholds (for selected model)',
        ),
        n_tiles=dict(
            widget_type='LiteralEvalLineEdit',
            label='Number of Tiles',
            value=DEFAULTS_STAR_PARAMETERS['n_tiles'],
        ),
        star_model_axes=dict(widget_type='LineEdit', label='Star Model Axes', value=''),
        
        defaults_star_parameters_button=dict(
            widget_type='PushButton', text='Restore StarDist Parameter Defaults'
        ),
        call_button=False,
    )
    def plugin_star_parameters(
        norm_image,
        perc_low,
        perc_high,
        prob_thresh,
        nms_thresh,
        set_thresholds,
        n_tiles,
        star_model_axes,
       
        defaults_star_parameters_button
        
    ):

        return plugin_star_parameters

    @magicgui(
        min_size_mask=dict(
            widget_type='FloatSpinBox',
            label='Min Size Mask (px)',
            min=0.0,
            max=1000000.0,
            step=10,
            value=DEFAULTS_VOLL_PARAMETERS['min_size_mask'],
        ),
        min_size=dict(
            widget_type='FloatSpinBox',
            label='Min Size Cells (px)',
            min=0.0,
            max=10000000.0,
            step=10,
            value=DEFAULTS_VOLL_PARAMETERS['min_size'],
        ),
        max_size=dict(
            widget_type='FloatSpinBox',
            label='Max Size Cells (px)',
            min=0.0,
            max=1.0E100,
            step=100,
            value=DEFAULTS_VOLL_PARAMETERS['max_size'],
        ),
        
        prob_map_watershed=dict(
            widget_type='CheckBox',
            text='Use Probability Map (watershed)',
            value=DEFAULTS_VOLL_PARAMETERS['prob_map_watershed'],
        ),
        dounet=dict(
            widget_type='CheckBox',
            text='Use UNET for binary mask (else denoised)',
            value=DEFAULTS_VOLL_PARAMETERS['dounet'],
        ),
        isRGB=dict(
            widget_type='CheckBox',
            text='RGB image',
            value=DEFAULTS_VOLL_PARAMETERS['isRGB'],
        ),
        seedpool=dict(
            widget_type='CheckBox',
            text='Do seed pooling',
            value=DEFAULTS_VOLL_PARAMETERS['seedpool'],
        ),
        slicemerge=dict(
            widget_type='CheckBox',
            text='Merge slices (UNET)',
            value=DEFAULTS_VOLL_PARAMETERS['slicemerge'],
        ),
        iouthresh=dict(
            widget_type='FloatSpinBox',
            label='Threshold linkining',
            min=0.0,
            max=1.0,
            step=0.1,
            value=DEFAULTS_VOLL_PARAMETERS['iouthresh'],
        ),
        unet_model_axes=dict(widget_type='LineEdit', label='Unet Model Axes', value=''),
        den_model_axes=dict(widget_type='LineEdit', label='Denoising Model Axes', value=''),
        defaults_vollseg_parameters_button=dict(
            widget_type='PushButton', text='Restore VollSeg Parameter Defaults'
        ),
        
        call_button=False,
    )
    def plugin_extra_parameters(
        min_size_mask,
        min_size,
        max_size,
        prob_map_watershed,
        dounet,
        isRGB,
        seedpool,
        slicemerge,
        iouthresh,
        unet_model_axes,
        den_model_axes,
        defaults_vollseg_parameters_button,
        
    ):

        return plugin_extra_parameters
    
    @magicgui(
       
        display_prob=dict(
            widget_type='CheckBox',
            text='Display Probability Map',
            value=DEFAULTS_DISPLAY_PARAMETERS['display_prob'],
        ),
        display_unet=dict(
            widget_type='CheckBox',
            text='Display UNET Results',
            value=DEFAULTS_DISPLAY_PARAMETERS['display_unet'],
        ),
        
        display_stardist =dict(
            widget_type='CheckBox',
            text='Display StarDist Results',
            value=DEFAULTS_DISPLAY_PARAMETERS['display_stardist'],
        ),
        display_markers=dict(
            widget_type='CheckBox',
            text='Display Markers Results',
            value=DEFAULTS_DISPLAY_PARAMETERS['display_markers'],
        ),
        display_vollseg=dict(
            widget_type='CheckBox',
            text='Display VollSeg Results',
            value=DEFAULTS_DISPLAY_PARAMETERS['display_vollseg'],
        ),
        display_denoised=dict(
            widget_type='CheckBox',
            text='Display Denoised Results',
            value=DEFAULTS_DISPLAY_PARAMETERS['display_denoised'],
        ),
        display_skeleton=dict(
            widget_type='CheckBox',
            text='Display VollSeg Skeleton Results',
            value=DEFAULTS_DISPLAY_PARAMETERS['display_skeleton'],
        ),
        defaults_display_parameters_button=dict(
            widget_type='PushButton', text='Restore Display Defaults'
        ),
        display_roi=dict(
            widget_type='CheckBox',
            text='Display Roi Results',
            value=DEFAULTS_DISPLAY_PARAMETERS['display_roi'],
        ),
        call_button=False,
    )
    def plugin_display_parameters(
        display_prob,
        display_unet,
        display_stardist,
        display_markers,
        display_roi,
        display_vollseg,
        display_denoised,
        display_skeleton,

        defaults_display_parameters_button,
        
    ):

        return plugin_display_parameters


    @magicgui(
       
       
        stop_button=dict(
            widget_type='PushButton', text='Stop TimeLapse Calculation', value = True,
        ),
        call_button=False,
    )
    def plugin_stop_parameters(
        stop_button
        
    ):

        return plugin_stop_parameters    

    logo = abspath(__file__, 'resources/vollseg_logo_napari.png')

    @magicgui(
        label_head=dict(
            widget_type='Label', label=f'<h1><img src="{logo}">VollSeg</h1>'
        ),
        image=dict(label='Input Image'),
        axes=dict(widget_type='LineEdit', label='Image Axes',value=DEFAULTS_MODEL['axes']),
        star_seg_model_type=dict(
            widget_type='RadioButtons',
            label='StarDist Model Type',
            orientation='horizontal',
            choices=star_seg_model_type_choices,
            value=DEFAULTS_MODEL['star_seg_model_type'],
        ),
        unet_seg_model_type=dict(
            widget_type='RadioButtons',
            label='Unet Model Type',
            orientation='horizontal',
            choices=unet_seg_model_type_choices,
            value=DEFAULTS_MODEL['unet_seg_model_type'],
        ),
        den_model_type=dict(
            widget_type='RadioButtons',
            label='Denoising Model Type',
            orientation='horizontal',
            choices=den_model_type_choices,
            value=DEFAULTS_MODEL['den_model_type'],
        ),
        roi_model_type=dict(
            widget_type='RadioButtons',
            label='Region of Interest Model Type',
            orientation='horizontal',
            choices=roi_model_type_choices,
            value=DEFAULTS_MODEL['roi_model_type'],
        ),
       
       
        model_den_none=dict(widget_type='Label', visible=False, label='No Denoising'),
        model_unet_none=dict(widget_type='Label', visible=False, label='NOUNET'),
        model_star_none=dict(widget_type='Label', visible=False, label='NOSTAR'),
        model_roi_none=dict(widget_type='Label', visible=False, label='NOROI'),
        model_folder_star=dict(
            widget_type='FileEdit',
            visible=False,
            label='Custom StarDist Model',
            mode='d',
        ),
        model_folder_unet=dict(
            widget_type='FileEdit', visible=False, label='Custom UNET Model', mode='d'
        ),
        model_folder_den=dict(
            widget_type='FileEdit',
            visible=False,
            label='Custom Denoising Model',
            mode='d',
        ),
         model_folder_roi=dict(
            widget_type='FileEdit', visible=False, label='Custom ROI Model', mode='d'
        ),
        norm_axes=dict(
            widget_type='LineEdit',
            label='Normalization Axes',
            value=DEFAULTS_MODEL['norm_axes'],
        ),
        defaults_model_button=dict(
            widget_type='PushButton', text='Restore Model Defaults'
        ),
       
        progress_bar=dict(label=' ', min=0, max=0, visible=False),
        layout='vertical',
        persist=True,
        call_button=True,
    )
    def plugin(
        viewer: napari.Viewer,
        label_head,
        image: napari.layers.Image,
        axes,
        star_seg_model_type,
        unet_seg_model_type,
        den_model_type,
        roi_model_type,
        model2d_star,
        model3d_star,
        model_unet,
        model_den,
        model_roi,
        model_den_none,
        model_unet_none,
        model_star_none,
        model_roi_none,
        model_folder_star,
        model_folder_unet,
        model_folder_den,
        model_folder_roi,
        norm_axes,
        defaults_model_button,
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:
        
        
        x = get_data(image)
        
        nonlocal worker 
        axes = axes_check_and_normalize(axes, length=x.ndim)
        progress_bar.label = 'Starting VollSeg'
        if model_selected_star is not None:    
           model_star = get_model_star(*model_selected_star)
        else: model_star = None   
        if model_selected_unet is not None:
           model_unet = get_model_unet(*model_selected_unet)
        else: model_unet = None   
        if model_selected_roi is not None:
           model_roi = get_model_roi(*model_selected_roi)
        else: model_roi = None   
        if model_selected_den is not None:
           model_den = get_model_den(*model_selected_den)
        else: model_den = None   
         
        

        if model_star is not None:
                if not axes.replace('T', '').startswith(model_star._axes_out.replace('C', '')):
                    warn(
                        f'output images have different axes ({model_star._axes_out.replace("C","")}) than input image ({axes})'
                    )
                    
        if model_unet is not None:
                if not axes.replace('T', '').startswith(model_unet._axes_out.replace('C', '')):
                    warn(
                        f'output images have different axes ({model_unet._axes_out.replace("C","")}) than input image ({axes})'
                    )      
        if model_den is not None:
                if not axes.replace('T', '').startswith(model_den._axes_out.replace('C', '')):
                    warn(
                        f'output images have different axes ({model_den._axes_out.replace("C","")}) than input image ({axes})'
                    )            

        if 'T' in axes:
            app = use_app()
            t = axes_dict(axes)['T']
            n_frames = x.shape[t]
           

            def progress_thread(current_time):
                
                progress_bar.label = 'VollSeg Prediction (frames)'
                progress_bar.range = (0, n_frames)
                progress_bar.value = current_time + 1
                progress_bar.show()
                plugin_stop_parameters.stop_button.native.setStyleSheet('')
                
           

        elif plugin_star_parameters.n_tiles.value is not None and np.prod(plugin_star_parameters.n_tiles.value) > 1:
            plugin_star_parameters.n_tiles.value = tuple(plugin_star_parameters.n_tiles.value)
        axes_out = None
        if model_star is not None:
                # semantic output axes of predictions
                assert model_star._axes_out[-1] == 'C'
                axes_out = list(model_star._axes_out[:-1])
        if model_unet is not None:
                # semantic output axes of predictions
                assert model_unet._axes_out[-1] == 'C'
                axes_out = list(model_unet._axes_out[:-1])  
        
        if model_den is not None:
                # semantic output axes of predictions
                assert model_den._axes_out[-1] == 'C'
                axes_out = list(model_den._axes_out[:-1])

        if model_roi is not None:
                # semantic output axes of predictions
                assert model_roi._axes_out[-1] == 'C'
                axes_out = list(model_roi._axes_out[:-1])        
        scale_in_dict = dict(zip(axes, image.scale))
        scale_out = [scale_in_dict.get(a, 1.0) for a in axes_out]       
        if 'T' in axes and axes_out is not None:
            x_reorder = np.moveaxis(x, t, 0)
            
            axes_reorder = axes.replace('T', '')
            axes_out.insert(t, 'T')
            # determine scale for output axes
            scale_in_dict = dict(zip(axes, image.scale))
            scale_out = [scale_in_dict.get(a, 1.0) for a in axes_out]     
            

            

            if model_star is not None:   
                worker = _VollSeg_time(model_star, model_unet, model_roi, x_reorder, axes_reorder, model_den, scale_out, t, x)
                worker.returned.connect(return_segment_time)
                worker.yielded.connect(progress_thread)
               
            if model_star is None:
                   
                        
                    worker = _Unet_time( model_unet, model_roi, x_reorder, axes_reorder, model_den, scale_out, t, x)
                    worker.returned.connect(return_segment_unet_time)
                    worker.yielded.connect(progress_thread)
            
        else:
            
           

                    if model_star is not None: 
                        worker = _Segment(model_star, model_unet, model_roi, x, axes, model_den,scale_out)
                        worker.returned.connect(return_segment)
                    if model_star is None:
                        worker = _Unet(model_unet, model_roi, x, axes, model_den,scale_out)
                        worker.returned.connect(return_segment_unet)
                        
           



        # add a button to the viewew that, when clicked, stops the worker
        
        progress_bar.hide()
        plugin_stop_parameters.stop_button.native.setStyleSheet('')
            
    plugin.axes.value = ''
    plugin_star_parameters.n_tiles.value = DEFAULTS_STAR_PARAMETERS['n_tiles']
    plugin.label_head.value = '<small>VollSeg segmentation for 2D and 3D images.<br>If you are using this in your research please <a href="https://github.com/kapoorlab/vollseg#how-to-cite" style="color:gray;">cite us</a>.</small><br><br><tt><a href="http://conference.scipy.org/proceedings/scipy2021/varun_kapoor.html" style="color:gray;">VollSeg Scipy</a></tt>'
    plugin.label_head.native.setSizePolicy(
        QSizePolicy.MinimumExpanding, QSizePolicy.Fixed
    )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    #
    widget_for_modeltype = {
        StarDist2D: plugin.model2d_star,
        StarDist3D: plugin.model3d_star,
        UNET: plugin.model_unet,
        CARE: plugin.model_den,
        MASKUNET: plugin.model_roi,
        'NODEN': plugin.model_den_none,
        'NOUNET': plugin.model_unet_none,
        'NOSTAR': plugin.model_star_none,
        'NOROI' : plugin.model_roi_none,
        CUSTOM_SEG_MODEL_STAR: plugin.model_folder_star,
        CUSTOM_SEG_MODEL_UNET: plugin.model_folder_unet,
        CUSTOM_DEN_MODEL: plugin.model_folder_den,
        CUSTOM_ROI_MODEL: plugin.model_folder_roi,
    }

    tabs = QTabWidget()

    parameter_star_tab = QWidget()
    _parameter_star_tab_layout = QVBoxLayout()
    parameter_star_tab.setLayout(_parameter_star_tab_layout)
    _parameter_star_tab_layout.addWidget(plugin_star_parameters.native)
    tabs.addTab(parameter_star_tab, 'StarDist Parameter Selection')

    parameter_extra_tab = QWidget()
    _parameter_extra_tab_layout = QVBoxLayout()
    parameter_extra_tab.setLayout(_parameter_extra_tab_layout)
    _parameter_extra_tab_layout.addWidget(plugin_extra_parameters.native)
    tabs.addTab(parameter_extra_tab, 'VollSeg Parameter Selection')

    parameter_display_tab = QWidget()
    _parameter_display_tab_layout = QVBoxLayout()
    parameter_display_tab.setLayout(_parameter_display_tab_layout)
    _parameter_display_tab_layout.addWidget(plugin_display_parameters.native)
    tabs.addTab(parameter_display_tab, 'Display Output Selection')

    parameter_kill_tab = QWidget()
    _parameter_kill_tab_layout = QVBoxLayout()
    parameter_kill_tab.setLayout(_parameter_kill_tab_layout)
    _parameter_kill_tab_layout.addWidget(plugin_stop_parameters.native)
    tabs.addTab(parameter_kill_tab, 'Interrupt Running computation')

    
   
    plugin.native.layout().addWidget(tabs)

    def widgets_inactive(*widgets, active):
        for widget in widgets:
            widget.visible = active
            # widget.native.setStyleSheet('' if active else 'text-decoration: line-through')

    def widgets_valid(*widgets, valid):
        for widget in widgets:
            widget.native.setStyleSheet('' if valid else 'background-color: red')



    class MaskUnet_roi_updater:
                def __init__(self, debug=DEBUG):
                    from types import SimpleNamespace

                    self.debug = debug
                    self.valid = SimpleNamespace(
                        **{
                            k: False
                            for k in ('model_roi',)
                        }
                    )
                    self.args = SimpleNamespace()
                    self.viewer = None

                def __call__(self, k, valid, args=None):
                    assert k in vars(self.valid)
                    setattr(self.valid, k, bool(valid))
                    setattr(self.args, k, args)
                    self._update()

                def help(self, msg):
                    if self.viewer is not None:
                        self.viewer.help = msg
                    elif len(str(msg)) > 0:
                        print(f'HELP: {msg}')
                
                def _update(self):

                    # try to get a hold of the viewer (can be None when plugin starts)
                    if self.viewer is None:
                        # TODO: when is this not safe to do and will hang forever?
                        # while plugin.viewer.value is None:
                        #     time.sleep(0.01)
                        if plugin.viewer.value is not None:
                            self.viewer = plugin.viewer.value
                            if DEBUG:
                                print('GOT viewer')


                    def _model(valid):
                        widgets_valid(
                            plugin.model_roi, plugin.model_folder_roi.line_edit, valid=valid,
                        )
                        if valid:
                            config_roi = self.args.model_roi
                            axes_roi = config_roi.get(
                                'axes', 'ZYXC'[-len(config_roi['unet_input_shape']) :]
                            )
                            if 'T' in axes_roi:
                                raise RuntimeError('model with axis "T" not supported')
                            plugin.model_folder_roi.line_edit.tooltip = ''
                            return axes_roi, config_roi
                        else:
                            
                            plugin.model_folder_roi.line_edit.tooltip = (
                                'Invalid model directory'
                            )

                    


                    def _restore():
                        widgets_valid(plugin.image, valid=plugin.image.value is not None)
                    
                    all_valid = False
                    for layer in list(plugin.viewer.value.layers):
                            if isinstance(layer, napari.layers.Image):
                                all_valid = True
                                break;   

                    help_msg = ''

                    
                    if (
                        self.valid.model_roi
                    ):
                       
                       

                            widgets_valid(
                              
                                plugin.model_roi,
                                plugin.model_folder_roi.line_edit,
                                valid=all_valid,
                            )
                            
                    else:
                        
                      
                        _model(self.valid.model_roi)

                        _restore()

                    self.help(help_msg)
                    plugin.call_button.enabled = all_valid
                    if self.debug:
                        print(
                            f'valid ({all_valid}):',
                            ', '.join([f'{k}={v}' for k, v in vars(self.valid).items()]),
                        ) 

    class Unet_den_updater:
        def __init__(self, debug=DEBUG):
            from types import SimpleNamespace

            self.debug = debug
            self.valid = SimpleNamespace(
                **{
                    k: False
                    for k in ('image_axes', 'model_den', 'n_tiles')
                }
            )
            self.args = SimpleNamespace()
            self.viewer = None

        def __call__(self, k, valid, args=None):
            assert k in vars(self.valid)
            setattr(self.valid, k, bool(valid))
            setattr(self.args, k, args)
            self._update()

        def help(self, msg):
            if self.viewer is not None:
                self.viewer.help = msg
            elif len(str(msg)) > 0:
                print(f'HELP: {msg}')
        
        def _update(self):

            # try to get a hold of the viewer (can be None when plugin starts)
            if self.viewer is None:
                # TODO: when is this not safe to do and will hang forever?
                # while plugin.viewer.value is None:
                #     time.sleep(0.01)
                if plugin.viewer.value is not None:
                    self.viewer = plugin.viewer.value
                    if DEBUG:
                        print('GOT viewer')


            def _model(valid):
                widgets_valid(
                    plugin.model_den, plugin.model_folder_den.line_edit, valid=valid,
                )
                if valid:
                    config_den = self.args.model_den
                    axes_den = config_den.get(
                        'axes', 'ZYXC'[-len(config_den['unet_input_shape']) :]
                    )
                    if 'T' in axes_den:
                        raise RuntimeError('model with axis "T" not supported')
                    plugin_extra_parameters.den_model_axes.value = axes_den.replace('C', f'C[{config_den["n_channel_in"]}]')    
                    plugin.model_folder_den.line_edit.tooltip = ''
                    return axes_den, config_den
                else:
                    
                    plugin_extra_parameters.den_model_axes.value = ''
                    plugin.model_folder_den.line_edit.tooltip = (
                        'Invalid model directory'
                    )

            def _image_axes(valid):
                axes, image, err = getattr(self.args, 'image_axes', (None, None, None))
                
                widgets_valid(
                    plugin.axes,
                    valid=(
                        valid or (image is None and (axes is None or len(axes) == 0))
                    ),
                )
                

                if valid:
                    plugin.axes.tooltip = '\n'.join(
                        [f'{a} = {s}' for a, s in zip(axes, get_data(image).shape)]
                    )
                    return axes, image
                else:
                    if err is not None:
                        err = str(err)
                        err = err[:-1] if err.endswith('.') else err
                        plugin.axes.tooltip = err
                        # warn(err) # alternative to tooltip (gui doesn't show up in ipython)
                    else:
                        plugin.axes.tooltip = ''

           

            def _n_tiles(valid):
                n_tiles, image, err = getattr(self.args, 'n_tiles', (1, 1, 1))

                widgets_valid(
                    plugin_star_parameters.n_tiles, valid=(valid or image is None)
                )
                if valid:
                    plugin_star_parameters.n_tiles.tooltip = (
                         '\n'.join(
                            [
                                f'{t}: {s}'
                                for t, s in zip(n_tiles, get_data(image).shape)
                            ]
                        )
                    )
                    return n_tiles
                else:
                    msg = str(err) if err is not None else ''
                    plugin_star_parameters.n_tiles.tooltip = msg

            def _no_tiling_for_axis(axes_image, n_tiles, axis):
                if n_tiles is not None and axis in axes_image:
                    return n_tiles[axes_dict(axes_image)[axis]] == 1
                return True

            def _restore():
                widgets_valid(plugin.image, valid=plugin.image.value is not None)

            all_valid = False
            help_msg = ''

            
            if (
                self.valid.image_axes
                and self.valid.n_tiles
                and self.valid.model_den
            ):
                axes_image, image = _image_axes(True)
                (axes_model_den, config_den) = _model(True)
                n_tiles = _n_tiles(True)
                if not _no_tiling_for_axis(axes_image, n_tiles, 'C'):
                    # check if image axes and n_tiles are compatible
                    widgets_valid(plugin_star_parameters.n_tiles, valid=False)
                    err = 'number of tiles must be 1 for C axis'
                    plugin_star_parameters.n_tiles.tooltip = err
                    _restore()
                elif not _no_tiling_for_axis(axes_image, n_tiles, 'T'):
                    # check if image axes and n_tiles are compatible
                    widgets_valid(plugin_star_parameters.n_tiles, valid=False)
                    err = 'number of tiles must be 1 for T axis'
                    plugin_star_parameters.n_tiles.tooltip = err
                    _restore()
                
                
                else:
                    # check if image and models are compatible
                    ch_model_den = config_den['n_channel_in']
                    
                    ch_image = (
                        get_data(image).shape[axes_dict(axes_image)['C']]
                        if 'C' in axes_image
                        else 1
                    )
                    all_valid = (
                        set(axes_model_den.replace('C', ''))
                        == set(axes_image.replace('C', '').replace('T', ''))
                        and ch_model_den == ch_image
                    )

                    widgets_valid(
                        plugin.image,
                        plugin.model_den,
                        plugin.model_folder_den.line_edit,
                        valid=all_valid,
                    )
                    if all_valid:
                        help_msg = ''
                    else:
                        help_msg = f'Model with axes {axes_model_den.replace("C", f"C[{ch_model_den}]")} and image with axes {axes_image.replace("C", f"C[{ch_image}]")} not compatible'
            else:
                
                _image_axes(self.valid.image_axes)
                _n_tiles(self.valid.n_tiles)
                _model(self.valid.model_den)

                _restore()

            self.help(help_msg)
            plugin.call_button.enabled = all_valid
            if self.debug:
                print(
                    f'valid ({all_valid}):',
                    ', '.join([f'{k}={v}' for k, v in vars(self.valid).items()]),
                )
 
          
    class Unet_updater:
        def __init__(self, debug=DEBUG):
            from types import SimpleNamespace

            self.debug = debug
            self.valid = SimpleNamespace(
                **{
                    k: False
                    for k in ('image_axes', 'model_unet', 'n_tiles')
                }
            )
            self.args = SimpleNamespace()
            self.viewer = None

        def __call__(self, k, valid, args=None):
            assert k in vars(self.valid)
            setattr(self.valid, k, bool(valid))
            setattr(self.args, k, args)
            self._update()

        def help(self, msg):
            if self.viewer is not None:
                self.viewer.help = msg
            elif len(str(msg)) > 0:
                print(f'HELP: {msg}')
        
        def _update(self):

            # try to get a hold of the viewer (can be None when plugin starts)
            if self.viewer is None:
                # TODO: when is this not safe to do and will hang forever?
                # while plugin.viewer.value is None:
                #     time.sleep(0.01)
                if plugin.viewer.value is not None:
                    self.viewer = plugin.viewer.value
                    if DEBUG:
                        print('GOT viewer')

                   

            def _model(valid):
                widgets_valid(
                    plugin.model_unet, plugin.model_folder_unet.line_edit, valid=valid,
                )
                print('unet',plugin.model_unet, plugin.model_folder_unet.line_edit, valid)
                if valid:
                    config_unet = self.args.model_unet
                    axes_unet = config_unet.get(
                        'axes', 'ZYXC'[-len(config_unet['unet_input_shape']) :]
                    )
                    if 'T' in axes_unet:
                        raise RuntimeError('model with axis "T" not supported')
                    plugin_extra_parameters.unet_model_axes.value = axes_unet.replace('C', f'C[{config_unet["n_channel_in"]}]')    
                    plugin.model_folder_unet.line_edit.tooltip = ''
                    return axes_unet, config_unet
                else:
                    plugin_extra_parameters.unet_model_axes.value = '' 
                    plugin.model_folder_unet.line_edit.tooltip = (
                        'Invalid model directory'
                    )

            def _image_axes(valid):
                axes, image, err = getattr(self.args, 'image_axes', (None, None, None))
                
                widgets_valid(
                    plugin.axes,
                    valid=(
                        valid or (image is None and (axes is None or len(axes) == 0))
                    ),
                )
                

                if valid:
                    plugin.axes.tooltip = '\n'.join(
                        [f'{a} = {s}' for a, s in zip(axes, get_data(image).shape)]
                    )
                    return axes, image
                else:
                    if err is not None:
                        err = str(err)
                        err = err[:-1] if err.endswith('.') else err
                        plugin.axes.tooltip = err
                        # warn(err) # alternative to tooltip (gui doesn't show up in ipython)
                    else:
                        plugin.axes.tooltip = ''

           

            def _n_tiles(valid):
                n_tiles, image, err = getattr(self.args, 'n_tiles', (1, 1, 1))
                widgets_valid(
                    plugin_star_parameters.n_tiles, valid=(valid or image is None)
                )
                if valid:
                    plugin_star_parameters.n_tiles.tooltip = (
                         '\n'.join(
                            [
                                f'{t}: {s}'
                                for t, s in zip(n_tiles, get_data(image).shape)
                            ]
                        )
                    )
                    return n_tiles
                else:
                    msg = str(err) if err is not None else ''
                    plugin_star_parameters.n_tiles.tooltip = msg

            def _no_tiling_for_axis(axes_image, n_tiles, axis):
                if n_tiles is not None and axis in axes_image:
                    return n_tiles[axes_dict(axes_image)[axis]] == 1
                return True

            def _restore():
                widgets_valid(plugin.image, valid=plugin.image.value is not None)

            all_valid = False
            help_msg = ''

            
            if (
                self.valid.image_axes
                and self.valid.n_tiles
                and self.valid.model_unet
            ):
                axes_image, image = _image_axes(True)
                (axes_model_unet, config_unet) = _model(True)
                n_tiles = _n_tiles(True)
                if not _no_tiling_for_axis(axes_image, n_tiles, 'C'):
                    # check if image axes and n_tiles are compatible
                    widgets_valid(plugin_star_parameters.n_tiles, valid=False)
                    err = 'number of tiles must be 1 for C axis'
                    plugin_star_parameters.n_tiles.tooltip = err
                    _restore()
                elif not _no_tiling_for_axis(axes_image, n_tiles, 'T'):
                    # check if image axes and n_tiles are compatible
                    widgets_valid(plugin_star_parameters.n_tiles, valid=False)
                    err = 'number of tiles must be 1 for T axis'
                    plugin_star_parameters.n_tiles.tooltip = err
                    _restore()
                
               
                else:
                    # check if image and models are compatible
                    ch_model_unet = config_unet['n_channel_in']
                    
                    ch_image = (
                        get_data(image).shape[axes_dict(axes_image)['C']]
                        if 'C' in axes_image
                        else 1
                    )
                    all_valid = (
                        set(axes_model_unet.replace('C', ''))
                        == set(axes_image.replace('C', '').replace('T', ''))
                        and ch_model_unet == ch_image
                    )

                    widgets_valid(
                        plugin.image,
                        plugin.model_unet,
                        plugin.model_folder_unet.line_edit,
                        valid=all_valid,
                    )
                    if all_valid:
                        help_msg = ''
                    else:
                        help_msg = f'Model with axes {axes_model_unet.replace("C", f"C[{ch_model_unet}]")} and image with axes {axes_image.replace("C", f"C[{ch_image}]")} not compatible'
            else:
                
                _image_axes(self.valid.image_axes)
                _n_tiles(self.valid.n_tiles)
                _model(self.valid.model_unet)

                _restore()

            self.help(help_msg)
            plugin.call_button.enabled = all_valid
            # widgets_valid(plugin.call_button, valid=all_valid)
            if self.debug:
                print(
                    f'valid ({all_valid}):',
                    ', '.join([f'{k}={v}' for k, v in vars(self.valid).items()]),
                )
                
                


    class Updater:
        def __init__(self, debug=DEBUG):
            from types import SimpleNamespace

            self.debug = debug
            self.valid = SimpleNamespace(
                **{
                    k: False
                    for k in ('image_axes', 'model_star', 'n_tiles', 'norm_axes')
                }
            )
            self.args = SimpleNamespace()
            self.viewer = None

        def __call__(self, k, valid, args=None):
            assert k in vars(self.valid)
            setattr(self.valid, k, bool(valid))
            setattr(self.args, k, args)
            self._update()

        def help(self, msg):
            if self.viewer is not None:
                self.viewer.help = msg
            elif len(str(msg)) > 0:
                print(f'HELP: {msg}')

        def _update(self):
            # try to get a hold of the viewer (can be None when plugin starts)
            if self.viewer is None:
                # TODO: when is this not safe to do and will hang forever?
                # while plugin.viewer.value is None:
                #     time.sleep(0.01)
                if plugin.viewer.value is not None:
                    self.viewer = plugin.viewer.value
                    if DEBUG:
                        print('GOT viewer')

                   

            def _model(valid):
                widgets_valid(
                    plugin.model2d_star,
                    plugin.model3d_star,
                    plugin.model_folder_star.line_edit,
                    valid=valid,
                )
                
                if valid:
                    config_star = self.args.model_star
                    
                    axes_star = config_star.get(
                        'axes', 'ZYXC'[-len(config_star['net_input_shape']) :]
                    )
                    if 'T' in axes_star:
                        raise RuntimeError('model with axis "T" not supported')
                    plugin_star_parameters.star_model_axes.value = axes_star.replace(
                        'C', f'C[{config_star["n_channel_in"]}]'
                    )
                    
                    plugin.model_folder_star.line_edit.tooltip = ''

                    return axes_star, config_star
                else:
                    plugin_star_parameters.star_model_axes.value = ''
                    plugin.model_folder_star.line_edit.tooltip = (
                        'Invalid model directory'
                    )

            def _image_axes(valid):
                axes, image, err = getattr(self.args, 'image_axes', (None, None, None))
                widgets_valid(
                    plugin.axes,
                    valid=(
                        valid or (image is None and (axes is None or len(axes) == 0))
                    ),
                )
                

                if valid:
                    plugin.axes.tooltip = '\n'.join(
                        [f'{a} = {s}' for a, s in zip(axes, get_data(image).shape)]
                    )
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
                norm_axes, err = getattr(self.args, 'norm_axes', (None, None))
                widgets_valid(plugin.norm_axes, valid=valid)
                if valid:
                    plugin.norm_axes.tooltip = f'Axes to jointly normalize (if present in selected input image). Note: channels of RGB images are always normalized together.'
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
                n_tiles, image, err = getattr(self.args, 'n_tiles', (1, 1, 1))
                widgets_valid(
                    plugin_star_parameters.n_tiles, valid=(valid or image is None)
                )
                if valid:
                    plugin_star_parameters.n_tiles.tooltip = (
                         '\n'.join(
                            [
                                f'{t}: {s}'
                                for t, s in zip(n_tiles, get_data(image).shape)
                            ]
                        )
                    )
                    return n_tiles
                else:
                    msg = str(err) if err is not None else ''
                    plugin_star_parameters.n_tiles.tooltip = msg

            def _no_tiling_for_axis(axes_image, n_tiles, axis):
                if n_tiles is not None and axis in axes_image:
                    return n_tiles[axes_dict(axes_image)[axis]] == 1
                return True

            def _restore():
                widgets_valid(plugin.image, valid=plugin.image.value is not None)

            all_valid = False
            help_msg = ''
            
            if (
                self.valid.image_axes
                and self.valid.n_tiles
                and self.valid.model_star
                and self.valid.norm_axes
            ):
                axes_image, image = _image_axes(True)
                (axes_model_star, config_star) = _model(True)
                axes_norm = _norm_axes(True)
                n_tiles = _n_tiles(True)
                if not _no_tiling_for_axis(axes_image, n_tiles, 'C'):
                    # check if image axes and n_tiles are compatible
                    widgets_valid(plugin_star_parameters.n_tiles, valid=False)
                    err = 'number of tiles must be 1 for C axis'
                    plugin_star_parameters.n_tiles.tooltip = err
                    _restore()
                elif not _no_tiling_for_axis(axes_image, n_tiles, 'T'):
                    # check if image axes and n_tiles are compatible
                    widgets_valid(plugin_star_parameters.n_tiles, valid=False)
                    err = 'number of tiles must be 1 for T axis'
                    plugin_star_parameters.n_tiles.tooltip = err
                    _restore()
                elif set(axes_norm).isdisjoint(set(axes_image)):
                    # check if image axes and normalization axes are compatible
                    widgets_valid(plugin.norm_axes, valid=False)
                    err = f'Image axes ({axes_image}) must contain at least one of the normalization axes ({", ".join(axes_norm)})'
                    plugin.norm_axes.tooltip = err
                    _restore()
                
                else:
                    # check if image and models are compatible
                    ch_model_star = config_star['n_channel_in']
                    ch_image = (
                        get_data(image).shape[axes_dict(axes_image)['C']]
                        if 'C' in axes_image
                        else 1
                    )
                    all_valid = (
                        set(axes_model_star.replace('C', ''))
                        == set(axes_image.replace('C', '').replace('T', ''))
                        and ch_model_star == ch_image
                    )

                    widgets_valid(
                        plugin.image,
                        plugin.model2d_star,
                        plugin.model3d_star,
                        plugin.model_folder_star.line_edit,
                        valid=all_valid,
                    )
                    if all_valid:
                        help_msg = ''
                    else:
                        help_msg = f'Model with axes {axes_model_star.replace("C", f"C[{ch_model_star}]")} and image with axes {axes_image.replace("C", f"C[{ch_image}]")} not compatible'
            else:
                
                _image_axes(self.valid.image_axes)
                _norm_axes(self.valid.norm_axes)
                _n_tiles(self.valid.n_tiles)
                _model(self.valid.model_star)
 
                _restore()

            self.help(help_msg)
            plugin.call_button.enabled = all_valid
            # widgets_valid(plugin.call_button, valid=all_valid)
            if self.debug:
                print(
                    f'valid ({all_valid}):',
                    ', '.join([f'{k}={v}' for k, v in vars(self.valid).items()]),
                )
                
    update = Updater()
    update_unet = Unet_updater()
    update_den = Unet_den_updater()
    update_roi = MaskUnet_roi_updater()

    def select_model_star(key_star):
        nonlocal model_selected_star
        if key_star is not None:
            model_selected_star = key_star
            config_star = model_star_configs.get(key_star)
            update('model_star', config_star is not None, config_star)
        if plugin.star_seg_model_type.value == DEFAULTS_MODEL['model_star_none']:
           model_selected_star = None
                

    def select_model_unet(key_unet):
        nonlocal model_selected_unet
        if key_unet is not None:
            model_selected_unet = key_unet
            config_unet = model_unet_configs.get(key_unet)
            update_unet('model_unet', config_unet is not None, config_unet)
        if plugin.unet_seg_model_type.value == DEFAULTS_MODEL['model_unet_none']:
           model_selected_unet = None    
       
    def select_model_den(key_den):
        nonlocal model_selected_den
        if key_den is not None:
            model_selected_den = key_den
            config_den = model_den_configs.get(key_den)
            update_den('model_den', config_den is not None, config_den)
        if plugin.den_model_type.value == DEFAULTS_MODEL['model_den_none']:
           model_selected_den = None

    def select_model_roi(key_roi):
        nonlocal model_selected_roi
        if key_roi is not None:
            model_selected_roi = key_roi
            config_roi = model_roi_configs.get(key_roi)
            update_roi('model_roi', config_roi is not None, config_roi)
        if plugin.roi_model_type.value == DEFAULTS_MODEL['model_roi_none']:
           model_selected_roi = None    
    # -------------------------------------------------------------------------

    # hide percentile selection if normalization turned off
    @change_handler(plugin_star_parameters.norm_image)
    def _norm_image_change(active: bool):
        widgets_inactive(
            plugin_star_parameters.perc_low,
            plugin_star_parameters.perc_high,
            plugin.norm_axes,
            active=active,
        )

    @change_handler(plugin_extra_parameters.dounet)
    def _dounet_change(active: bool):
        plugin_extra_parameters.dounet.value = active
        


    @change_handler(plugin_extra_parameters.seedpool)
    def _seed_pool_change(active: bool):

        plugin_extra_parameters.seedpool.value = active 

    @change_handler(plugin_extra_parameters.slicemerge)
    def _slicemerge_change(active: bool):
        plugin_extra_parameters.slicemerge.value = active
        widgets_inactive(
            plugin_extra_parameters.iouthresh,
            active=active,
        )
        
    @change_handler(plugin_extra_parameters.isRGB)
    def _dorgb_change(active: bool):
        plugin_extra_parameters.isRGB.value = active

    @change_handler(plugin_extra_parameters.prob_map_watershed)
    def _prob_map_watershed_change(active: bool):
        plugin_extra_parameters.prob_map_watershed.value = active
   

    @change_handler(plugin_stop_parameters.stop_button)
    def _stop_computation_change(active: bool):   
            plugin_stop_parameters.stop_button.value = True 
            
            if worker is not None:
                    worker.quit()
                    plugin_stop_parameters.stop_button.native.setStyleSheet('background-color: red')

    @change_handler(plugin_display_parameters.display_prob)
    def _display_prob_map_change(active: bool):
        plugin_display_parameters.display_prob.value = active

    @change_handler(plugin_display_parameters.display_unet)
    def _display_unet_change(active: bool):
        plugin_display_parameters.display_unet.value = active

    @change_handler(plugin_display_parameters.display_stardist)
    def _display_star_change(active: bool):
        plugin_display_parameters.display_stardist.value = active
    
    @change_handler(plugin_display_parameters.display_denoised)
    def _display_den_change(active: bool):
        plugin_display_parameters.display_denoised.value = active

    @change_handler(plugin_display_parameters.display_markers)
    def _display_mark_change(active: bool):
        plugin_display_parameters.display_markers.value = active

    @change_handler(plugin_display_parameters.display_roi)
    def _display_roi_change(active: bool):
        plugin_display_parameters.display_roi.value = active    

    @change_handler(plugin_display_parameters.display_vollseg)
    def _display_vollseg_change(active: bool):
        plugin_display_parameters.display_vollseg.value = active
  
    @change_handler(plugin_display_parameters.display_skeleton)
    def _display_skel_change(active: bool):
        plugin_display_parameters.display_skeleton.value = active

    # ensure that percentile low < percentile high
    @change_handler(plugin_star_parameters.perc_low)
    def _perc_low_change():

        plugin_star_parameters.perc_low.value = min(
            plugin_star_parameters.perc_low.value,
            plugin_star_parameters.perc_high.value - 0.01,
        )

        

    @change_handler(plugin_star_parameters.perc_high)
    def _perc_high_change():


         plugin_star_parameters.perc_high.value = max(
            plugin_star_parameters.perc_low.value + 0.01,
            plugin_star_parameters.perc_high.value,
        )

    @change_handler(plugin.norm_axes, plugin.unet_seg_model_type, plugin.star_seg_model_type, plugin.den_model_type, plugin.roi_model_type)
    def _norm_axes_change(value: str):
        
        value = plugin.norm_axes.value 
        try:
            axes = axes_check_and_normalize(value, disallowed='S')
            if len(axes) >= 1:
                
                if plugin.star_seg_model_type.value != DEFAULTS_MODEL['model_star_none']:
                   update('norm_axes', True, (axes, None))
            else:
                if plugin.star_seg_model_type.value != DEFAULTS_MODEL['model_star_none']:
                    update('norm_axes', False, (axes, 'Cannot be empty'))
        except ValueError as err:
            if plugin.star_seg_model_type.value != DEFAULTS_MODEL['model_star_none']:
                update('norm_axes', False, (value, err))
    # -------------------------------------------------------------------------

    # RadioButtons widget triggers a change event initially (either when 'value' is set in constructor, or via 'persist')
    # TODO: seems to be triggered too when a layer is added or removed (why?)
    @change_handler(plugin.star_seg_model_type, init=False)
    def _seg_model_type_change_star(seg_model_type: Union[str, type]):
        selected = widget_for_modeltype[seg_model_type]
        for w in set(
            (
                plugin.model2d_star,
                plugin.model3d_star,
                plugin.model_star_none,
                plugin.model_folder_star,
            )
        ) - {selected}:
            w.hide()
        selected.show()
        if plugin.star_seg_model_type.value == DEFAULTS_MODEL['model_star_none']:
                plugin_extra_parameters.prob_map_watershed.hide()
                plugin_extra_parameters.seedpool.hide()
                for w in set(plugin_star_parameters):
                    w.hide()
                plugin_star_parameters.n_tiles.show()  
                plugin_star_parameters.norm_image.show()  
                plugin_star_parameters.perc_low.show()
                plugin_star_parameters.perc_high.show()
        else:
                for w in set(plugin_star_parameters):
                    w.show()   
                plugin_extra_parameters.prob_map_watershed.show()
                plugin_extra_parameters.seedpool.show()         
        # trigger _model_change_star()
        selected.changed(selected.value)

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

    @change_handler(plugin.roi_model_type, init=False)
    def _roi_model_type_change(roi_model_type: Union[str, type]):
        selected = widget_for_modeltype[roi_model_type]
        for w in set(
            (plugin.model_roi,  plugin.model_roi_none, plugin.model_folder_roi)
        ) - {selected}:
            w.hide()
        selected.show()

        

        # trigger _model_change_roi
        selected.changed(selected.value)

    # RadioButtons widget triggers a change event initially (either when 'value' is set in constructor, or via 'persist')
    # TODO: seems to be triggered too when a layer is added or removed (why?)
    @change_handler(plugin.den_model_type, init=False)
    def _den_model_type_change(den_model_type: Union[str, type]):
        selected = widget_for_modeltype[den_model_type]
        for w in set(
            (plugin.model_den, plugin.model_den_none, plugin.model_folder_den)
        ) - {selected}:
            w.hide()
        selected.show()

        # trigger _model_change_den
        selected.changed(selected.value)

    
             
             
             
    def return_segment_time(pred):

        res, scale_out, t, x = pred
        if plugin.den_model_type.value != DEFAULTS_MODEL['model_den_none'] and  plugin.star_seg_model_type.value != DEFAULTS_MODEL['model_star_none'] and plugin.roi_model_type.value == DEFAULTS_MODEL['model_roi_none']:
 
            labels, unet_mask, star_labels, probability_map, Markers, Skeleton, denoised_image = zip(*res)
            
            denoised_image = np.asarray(denoised_image)

            denoised_image = np.moveaxis(denoised_image, 0, t)
            
            denoised_image = np.reshape(denoised_image, x.shape)


            
        elif  plugin.den_model_type.value == DEFAULTS_MODEL['model_den_none'] and  plugin.star_seg_model_type.value != DEFAULTS_MODEL['model_star_none'] and plugin.roi_model_type.value == DEFAULTS_MODEL['model_roi_none']:
            
            labels, unet_mask, star_labels, probability_map, Markers, Skeleton = zip(*res)


        elif plugin.den_model_type.value != DEFAULTS_MODEL['model_den_none'] and  plugin.star_seg_model_type.value != DEFAULTS_MODEL['model_star_none'] and plugin.roi_model_type.value != DEFAULTS_MODEL['model_roi_none']:

            labels, unet_mask, star_labels, probability_map, Markers, Skeleton, denoised_image, roi_image = zip(*res)
            
            denoised_image = np.asarray(denoised_image)

            denoised_image = np.moveaxis(denoised_image, 0, t)
            
            denoised_image = np.reshape(denoised_image, x.shape)

            roi_image = np.asarray(roi_image)

            roi_image = np.moveaxis(roi_image, 0, t)
            
            roi_image = np.reshape(roi_image, roi_image.shape)
            
        elif  plugin.den_model_type.value == DEFAULTS_MODEL['model_den_none'] and  plugin.star_seg_model_type.value != DEFAULTS_MODEL['model_star_none'] and plugin.roi_model_type.value != DEFAULTS_MODEL['model_roi_none']:
            
            labels, unet_mask, star_labels, probability_map, Markers, Skeleton, roi_image = zip(*res)
            
            roi_image = np.asarray(roi_image)

            roi_image = np.moveaxis(roi_image, 0, t)
            
            roi_image = np.reshape(roi_image, roi_image.shape)
        
        if plugin.star_seg_model_type.value != DEFAULTS_MODEL['model_star_none']: 
                labels = np.asarray(labels)
                labels = np.moveaxis(labels, 0, t)
                labels = np.reshape(labels, x.shape)
                
                star_labels = np.asarray(star_labels)
                star_labels = np.moveaxis(star_labels, 0, t)
                star_labels = np.reshape(star_labels, x.shape)
                
                unet_mask = np.asarray(unet_mask)
                unet_mask = np.moveaxis(unet_mask, 0, t)
                unet_mask = np.reshape(unet_mask, x.shape)
                
                probability_map = np.asarray(probability_map)
                probability_map = np.moveaxis(probability_map, 0, t)
                probability_map = np.reshape(probability_map, x.shape)
                
                Skeleton = np.asarray(Skeleton)
                Skeleton = np.moveaxis(Skeleton, 0, t)
                Skeleton = np.reshape(Skeleton, x.shape)
                
                Markers = np.asarray(Markers)
                Markers = np.moveaxis(Markers, 0, t)
                Markers = np.reshape(Markers, x.shape)     
 
              
        name_remove = ('VollSeg Binary', 'Base Watershed Image','VollSeg labels', 'StarDist', 'Markers', 'Skeleton','Denoised Image', 'Roi' )
        for layer in list(plugin.viewer.value.layers):
                if  any(name in layer.name for name in name_remove):
                          plugin.viewer.value.layers.remove(layer)                                     
        if plugin.roi_model_type.value != DEFAULTS_MODEL['model_roi_none']:

                    if plugin_display_parameters.display_roi.value:
                             
                              plugin.viewer.value.add_labels(
                            
                                roi_image,
                            
                                    name='Roi', scale= scale_out, opacity=0.5,  visible = plugin_display_parameters.display_roi.value
                            
                        )


        if plugin.star_seg_model_type.value != DEFAULTS_MODEL['model_star_none']:
                    if plugin_display_parameters.display_prob.value:

                        
                        plugin.viewer.value.add_image(
                                    
                                        probability_map,
                                        
                                            name='Base Watershed Image',
                                            scale=scale_out,
                                            visible=plugin_display_parameters.display_prob.value,
                                    
                                )
                    if plugin_display_parameters.display_vollseg.value:
                        plugin.viewer.value.add_labels(
                            
                                labels,
                            
                                    name='VollSeg labels', scale= scale_out, opacity=0.5,  visible = plugin_display_parameters.display_vollseg.value
                            
                        )
                    if plugin_display_parameters.display_stardist.value:
                        plugin.viewer.value.add_labels(
                            
                                star_labels,
                                
                                    name='StarDist',
                                    scale=scale_out,
                                    opacity=0.5,
                                    visible=plugin_display_parameters.display_stardist.value,
                            
                        )
                    if plugin_display_parameters.display_unet.value:    
                        plugin.viewer.value.add_labels(
                            
                                unet_mask,
                            
                                    name='VollSeg Binary',
                                    scale=scale_out,
                                    opacity=0.5,
                                    visible=plugin_display_parameters.display_unet.value,
                            
                        )
                    if plugin_display_parameters.display_markers.value:
                        plugin.viewer.value.add_labels(
                            
                                Markers,
                                
                                    name='Markers',
                                    scale=scale_out,
                                    opacity=0.5,
                                    visible=plugin_display_parameters.display_markers.value,
                            
                        )
                    if plugin_display_parameters.display_skeleton.value:    
                        plugin.viewer.value.add_labels(
                            
                                Skeleton,
                                
                                    name='Skeleton',
                                    scale=scale_out,
                                    opacity=0.5,
                                    visible=plugin_display_parameters.display_skeleton.value,
                                
                        )
                    if plugin_display_parameters.display_denoised.value:     
                        if plugin.den_model_type.value != DEFAULTS_MODEL['model_den_none']:
                            plugin.viewer.value.add_image(
                                
                                    denoised_image,
                                
                                        name='Denoised Image',
                                        scale=scale_out,
                                        visible=plugin_display_parameters.display_denoised.value,
                                    
                                )
        
    
    def return_segment(pred):
              
        
                res, scale_out = pred
                
                if plugin.den_model_type.value != DEFAULTS_MODEL['model_den_none'] and plugin.star_seg_model_type.value != DEFAULTS_MODEL['model_star_none']and plugin.roi_model_type.value == DEFAULTS_MODEL['model_roi_none']:

                    labels, unet_mask, star_labels, probability_map, Markers, Skeleton, denoised_image = res
                    
                elif plugin.den_model_type.value == DEFAULTS_MODEL['model_den_none'] and plugin.star_seg_model_type.value != DEFAULTS_MODEL['model_star_none']and plugin.roi_model_type.value == DEFAULTS_MODEL['model_roi_none']:
                    labels, unet_mask, star_labels, probability_map, Markers, Skeleton = res
                    
                elif plugin.star_seg_model_type.value == DEFAULTS_MODEL['model_star_none']and plugin.roi_model_type.value == DEFAULTS_MODEL['model_roi_none']:
                    
                    unet_mask, denoised_image = res

                if plugin.den_model_type.value != DEFAULTS_MODEL['model_den_none'] and plugin.star_seg_model_type.value != DEFAULTS_MODEL['model_star_none'] and plugin.roi_model_type.value != DEFAULTS_MODEL['model_roi_none']:

                    labels, unet_mask, star_labels, probability_map, Markers, Skeleton, denoised_image, roi_image = res
                    
                elif plugin.den_model_type.value == DEFAULTS_MODEL['model_den_none'] and plugin.star_seg_model_type.value != DEFAULTS_MODEL['model_star_none'] and plugin.roi_model_type.value != DEFAULTS_MODEL['model_roi_none']:
                    labels, unet_mask, star_labels, probability_map, Markers, Skeleton, roi_image = res
                    
                elif plugin.star_seg_model_type.value == DEFAULTS_MODEL['model_star_none'] and plugin.roi_model_type.value != DEFAULTS_MODEL['model_roi_none']:
                    
                    unet_mask, denoised_image, roi_image = res

                elif plugin.star_seg_model_type.value == DEFAULTS_MODEL['model_star_none'] and plugin.unet_seg_model_type.value != DEFAULTS_MODEL['model_unet_none'] and plugin.den_model_type.value == DEFAULTS_MODEL['model_den_none']:
                    
                    unet_mask = res
                    
                
                name_remove = ('VollSeg Binary', 'Base Watershed Image','VollSeg labels', 'StarDist', 'Markers', 'Skeleton','Denoised Image', 'Roi' )
                for layer in list(plugin.viewer.value.layers):
                        if  any(name in layer.name for name in name_remove):
                          plugin.viewer.value.layers.remove(layer) 
                if plugin.roi_model_type.value != DEFAULTS_MODEL['model_roi_none']:

                            if plugin_display_parameters.display_roi.value:
                                    
                                    plugin.viewer.value.add_labels(
                                    
                                        roi_image,
                                    
                                            name='Roi', scale= scale_out, opacity=0.5,  visible = plugin_display_parameters.display_roi.value
                                    
                                )
                if plugin.unet_seg_model_type.value != DEFAULTS_MODEL['model_unet_none']:

                            if plugin_display_parameters.display_unet.value:
                                    plugin.viewer.value.add_labels(
                                        
                                            unet_mask,
                                            
                                                name='VollSeg Binary',
                                                scale=scale_out,
                                                opacity=0.5,
                                                visible=plugin_display_parameters.display_unet.value,
                                            
                                    )     
                if plugin.star_seg_model_type.value != DEFAULTS_MODEL['model_star_none']:

                    if plugin_display_parameters.display_prob.value:
                        plugin.viewer.value.add_image(
                            
                                probability_map,
                                
                                    name='Base Watershed Image',
                                    scale=scale_out,
                                    visible=plugin_display_parameters.display_prob.value,
                                
                        )
                    if plugin_display_parameters.display_vollseg.value:
                            plugin.viewer.value.add_labels(
                                
                                    labels,
                                    
                                        name='VollSeg labels', scale= scale_out, opacity=0.5, visible = plugin_display_parameters.display_vollseg.value 
                                
                            )
                    if plugin_display_parameters.display_stardist.value:
                            plugin.viewer.value.add_labels(
                                
                                    star_labels,
                                    
                                        name='StarDist',
                                        scale=scale_out,
                                        opacity=0.5,
                                        visible=plugin_display_parameters.display_stardist.value,
                                    
                            )
                    if plugin_display_parameters.display_unet.value:
                            plugin.viewer.value.add_labels(
                                
                                    unet_mask,
                                    
                                        name='VollSeg Binary',
                                        scale=scale_out,
                                        opacity=0.5,
                                        visible=plugin_display_parameters.display_unet.value,
                                    
                            )
                    if plugin_display_parameters.display_markers.value:
                            plugin.viewer.value.add_labels(
                                
                                    Markers,
                                    
                                        name='Markers',
                                        scale=scale_out,
                                        opacity=0.5,
                                        visible=plugin_display_parameters.display_markers.value,
                                    
                            )

                    if plugin_display_parameters.display_skeleton.value:      
                            plugin.viewer.value.add_labels(
                                
                                    Skeleton,
                                    
                                        name='Skeleton',
                                        scale=scale_out,
                                        opacity=0.5,
                                        visible=plugin_display_parameters.display_skeleton.value,
                                    
                            )
                    if plugin_display_parameters.display_denoised.value:      
                            if plugin.den_model_type.value != DEFAULTS_MODEL['model_den_none']:
                                    plugin.viewer.value.add_image(
                                        
                                            denoised_image,
                                        
                                                name='Denoised Image',
                                                scale=scale_out,
                                                visible=plugin_display_parameters.display_denoised.value,
                                            
                                        )

                                  
    def return_segment_unet_time(pred):
        
        
                     
                     
              res, scale_out, t, x = pred
              unet_mask, skeleton, denoised_image = zip(*res)
              
              unet_mask = np.asarray(unet_mask)
              unet_mask = unet_mask > 0
              unet_mask = np.moveaxis(unet_mask, 0, t)
              unet_mask = np.reshape(unet_mask, x.shape)
              
              skeleton = np.asarray(skeleton)
              skeleton = skeleton > 0
              skeleton = np.moveaxis(skeleton, 0, t)
              skeleton = np.reshape(skeleton, x.shape)

              denoised_image = np.asarray(denoised_image)
              denoised_image = np.moveaxis(denoised_image, 0, t)
              denoised_image = np.reshape(denoised_image, x.shape)

              name_remove = ('VollSeg Binary', 'Base Watershed Image','VollSeg labels', 'StarDist', 'Markers', 'Skeleton','Denoised Image', 'Roi' )
              for layer in list(plugin.viewer.value.layers):
                        if  any(name in layer.name for name in name_remove):
                          plugin.viewer.value.layers.remove(layer)
              if plugin_display_parameters.display_unet.value:
                    plugin.viewer.value.add_labels(
                        
                            unet_mask, name ='VollSeg Binary',
                                scale= scale_out,
                                opacity=0.5,
                                visible=plugin_display_parameters.display_unet.value)
              if plugin_display_parameters.display_skeleton.value:
                    plugin.viewer.value.add_labels(
                        
                            skeleton, name ='Skeleton',
                                scale= scale_out,
                                opacity=0.5,
                                visible=plugin_display_parameters.display_skeleton.value)

              if plugin.den_model_type.value != DEFAULTS_MODEL['model_den_none']:

                  if plugin_display_parameters.display_denoised.value:
                        plugin.viewer.value.add_image(
                            
                                denoised_image,
                            
                                    name='Denoised Image',
                                    scale=scale_out,
                                    visible=plugin_display_parameters.display_denoised.value,
                                
                            )
                     


    def return_segment_unet(pred):
            
                      
                    res, scale_out = pred
                    unet_mask, skeleton, denoised_image = res
                    name_remove = ('VollSeg Binary', 'Base Watershed Image','VollSeg labels', 'StarDist', 'Markers', 'Skeleton','Denoised Image', 'Roi' )
                    for layer in list(plugin.viewer.value.layers):
                        if  any(name in layer.name for name in name_remove):
                          plugin.viewer.value.layers.remove(layer)
                    
                    
                    if plugin_display_parameters.display_unet.value:             
                            plugin.viewer.value.add_labels(
                                
                                    unet_mask, name ='VollSeg Binary',
                                        scale= scale_out,
                                        opacity=0.5,
                                        visible=plugin_display_parameters.display_unet.value)
                    if plugin_display_parameters.display_skeleton.value:
                            plugin.viewer.value.add_labels(
                                
                                    skeleton, name ='Skeleton',
                                        scale= scale_out,
                                        opacity=0.5,
                                        visible=plugin_display_parameters.display_skeleton.value)            
                    if plugin.den_model_type.value != DEFAULTS_MODEL['model_den_none']:

                        if  plugin_display_parameters.display_denoised.value:
                                plugin.viewer.value.add_image(
                                    
                                        denoised_image,
                                    
                                            name='Denoised Image',
                                            scale=scale_out,
                                            visible=plugin_display_parameters.display_denoised.value,
                                        
                                    )
               
              

    @thread_worker(connect = {"returned": return_segment_time } )         
    def _VollSeg_time( model_star, model_unet, model_roi, x_reorder, axes_reorder, noise_model, scale_out, t, x):
       
      
       pre_res = []
       for  count, _x in enumerate(x_reorder):
            
           yield count
           
           pre_res.append(VollSeg(
                       _x,
                       model_unet,
                       model_star,
                       axes=axes_reorder,
                       noise_model=noise_model,
                       roi_model = model_roi,
                       prob_thresh=plugin_star_parameters.prob_thresh.value,
                       nms_thresh=plugin_star_parameters.nms_thresh.value,
                       min_size_mask=plugin_extra_parameters.min_size_mask.value,
                       seedpool= plugin_extra_parameters.seedpool.value,
                       min_size=plugin_extra_parameters.min_size.value,
                       max_size=plugin_extra_parameters.max_size.value,
                       n_tiles=plugin_star_parameters.n_tiles.value,
                       donormalize=plugin_star_parameters.norm_image.value,
                       lower_perc=plugin_star_parameters.perc_low.value, 
                       upper_perc=plugin_star_parameters.perc_high.value,
                       UseProbability=plugin_extra_parameters.prob_map_watershed.value,
                       dounet=plugin_extra_parameters.dounet.value,
                       RGB = plugin_extra_parameters.isRGB.value,
                       iou_threshold = plugin_extra_parameters.iouthresh.value,slice_merge = plugin_extra_parameters.slicemerge.value
                   ))
                   
       pred = pre_res, scale_out, t, x
       return pred            
              
              
    @thread_worker(connect = {"returned": return_segment_unet_time } )         
    def _Unet_time( model_unet, model_roi, x_reorder, axes_reorder, noise_model, scale_out, t, x):
        pre_res = []
        for  count, _x in enumerate(x_reorder):
             
            yield count
            pre_res.append(VollSeg(_x, unet_model = model_unet, roi_model = model_roi,
                       n_tiles=plugin_star_parameters.n_tiles.value, axes = axes_reorder, noise_model = noise_model,  RGB = plugin_extra_parameters.isRGB.value,
                                min_size_mask=plugin_extra_parameters.min_size_mask.value, seedpool= plugin_extra_parameters.seedpool.value, donormalize=plugin_star_parameters.norm_image.value,
                       lower_perc=plugin_star_parameters.perc_low.value, 
                       upper_perc=plugin_star_parameters.perc_high.value,
                       max_size=plugin_extra_parameters.max_size.value, iou_threshold = plugin_extra_parameters.iouthresh.value,slice_merge = plugin_extra_parameters.slicemerge.value))
        
        pred = pre_res, scale_out, t, x
        return pred           
              
    @thread_worker(connect = {"returned": return_segment_unet } )         
    def _Unet( model_unet, model_roi, x, axes, noise_model, scale_out):
    
        res = VollSeg(x, unet_model = model_unet, roi_model = model_roi, n_tiles=plugin_star_parameters.n_tiles.value, axes = axes, noise_model = noise_model, donormalize=plugin_star_parameters.norm_image.value,
                       lower_perc=plugin_star_parameters.perc_low.value, 
                       upper_perc=plugin_star_parameters.perc_high.value,  RGB = plugin_extra_parameters.isRGB.value,
        min_size_mask=plugin_extra_parameters.min_size_mask.value, seedpool= plugin_extra_parameters.seedpool.value,
                       max_size=plugin_extra_parameters.max_size.value,
                     iou_threshold = plugin_extra_parameters.iouthresh.value,slice_merge = plugin_extra_parameters.slicemerge.value)
                    
        pred = res, scale_out
        return pred           
             
    @thread_worker (connect = {"returned": return_segment } )        
    def _Segment(model_star, model_unet,  model_roi, x, axes, noise_model, scale_out):
    
        res = VollSeg(
            x,
            model_unet,
            model_star,
            axes=axes,
            noise_model=noise_model,
            roi_model =  model_roi,
          
            prob_thresh=plugin_star_parameters.prob_thresh.value,
            nms_thresh=plugin_star_parameters.nms_thresh.value,
            min_size_mask=plugin_extra_parameters.min_size_mask.value,
            min_size=plugin_extra_parameters.min_size.value,
            seedpool= plugin_extra_parameters.seedpool.value,
            max_size=plugin_extra_parameters.max_size.value,
            n_tiles=plugin_star_parameters.n_tiles.value,
            UseProbability=plugin_extra_parameters.prob_map_watershed.value,
            dounet=plugin_extra_parameters.dounet.value,
            donormalize=plugin_star_parameters.norm_image.value,
            lower_perc=plugin_star_parameters.perc_low.value, 
            upper_perc=plugin_star_parameters.perc_high.value,
            RGB = plugin_extra_parameters.isRGB.value,           
            slice_merge = plugin_extra_parameters.slicemerge.value,
            iou_threshold = plugin_extra_parameters.iouthresh.value
            )   
               
        pred = res, scale_out   
        return pred  


    
   
    
    return plugin, plugin_star_parameters, plugin_extra_parameters, plugin_display_parameters, plugin_stop_parameters, get_data


def inrimage_file_reader(path):
    array = inrimage.read_inrimage(path)
    # return it as a list of LayerData tuples,
    # here with no optional metadata
    return [(array,)]


# def klbimage_file_reader(path):
# array = klb.read_klb(path)
# return it as a list of LayerData tuples,
# here with no optional metadata
# return [(array,)]


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


@napari_hook_implementation(specname='napari_get_reader')
def napari_get_reader(path: str):
    # If we recognize the format, we return the actual reader function
    if isinstance(path, str) and path.endswith('.inr') or path.endswith('.inr.gz'):
        return inrimage_file_reader
    # if isinstance(path, str) and path.endswith('.klb'):
    # return klbimage_file_reader
    if isinstance(path, str) and path.endswith('.tif'):
        return tifimage_file_reader
    if isinstance(path, str) and path.endswith('.h5'):
        return h5image_file_reader

    else:
        # otherwise we return None.
        return None


