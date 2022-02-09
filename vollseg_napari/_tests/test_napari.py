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
    DEBUG = False
    def change_handler(*widgets, init=True, debug=DEBUG):
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
    model_star_configs = dict()
    model_unet_configs = dict()
    model_den_configs = dict()
    model_star_threshs = dict()
    model_selected_star = None
    model_selected_unet = None
    model_selected_den = None
    widget_for_modeltype = {
        StarDist2D: plugin.model2d_star,
        StarDist3D: plugin.model3d_star,
        UNET: plugin.model_unet,
        'NODEN': plugin.model_den_none,
        'NOUNET': plugin.model_unet_none,
        'NOSTAR': plugin.model_star_none
    }
    def widgets_valid(*widgets, valid):
            for widget in widgets:
                widget.native.setStyleSheet('' if valid else 'background-color: red')

    def get_data(image):
        image = image.data[0] if image.multiscale else image.data
        # enforce dense numpy array in case we are given a dask array etc
        return np.asarray(image)            
    class Unet_updater:
        def __init__(self, debug=DEBUG):
            from types import SimpleNamespace

            self.debug = debug
            self.valid = SimpleNamespace(
                **{
                    k: False
                    for k in ('image_axes', 'model_unet', 'n_tiles', 'norm_axes')
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

                    @self.viewer.layers.events.removed.connect
                    def _layer_removed(event):
                        layers_remaining = event.source
                        if len(layers_remaining) == 0:
                            plugin.image.tooltip = ''
                            plugin.axes.value = ''
                            plugin_star_parameters.n_tiles.value = 'None'

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
                n_tiles, image, err = getattr(self.args, 'n_tiles', (None, None, None))
                widgets_valid(
                    plugin_star_parameters.n_tiles, valid=(valid or image is None)
                )
                if valid:
                    plugin_star_parameters.n_tiles.tooltip = (
                        'no tiling'
                        if n_tiles is None
                        else '\n'.join(
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
                _norm_axes(self.valid.norm_axes)
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
                
    update_unet = Unet_updater()
    def select_model_unet(key_unet):
        nonlocal model_selected_unet
        model_selected_unet = key_unet
        config_unet = model_unet_configs.get(key_unet)
        update_unet('model_unet', config_unet is not None, config_unet)


    @change_handler(plugin.model2d_star, plugin.model3d_star, plugin.model_star_none, plugin.model_unet, plugin.model_unet_none,plugin.model_den, plugin.model_den_none, init=False) 
    def _model_change_unet(model_name_unet: str):
        
        
        if Signal.sender() is not plugin.model_unet_none:
                model_class_unet = ( UNET if Signal.sender() is plugin.model_unet else UNET if plugin.model_unet.value is not None and Signal.sender() is None else None ) 
                
                if model_class_unet is not None:
                        plugin_extra_parameters.dounet.value = True  
                        if Signal.sender is not None:
                             model_name = model_name_unet
                        elif plugin.model_unet.value is not None:
                            model_name = plugin.model_unet.value
                        key_unet = model_class_unet, model_name
                        if key_unet not in model_unet_configs:
                
                            @thread_worker
                            def _get_model_folder():
                                return get_model_folder(*key_unet)
                
                            def _process_model_folder(path):
                
                                try:
                                    model_unet_configs[key_unet] = load_json(str(path / 'config.json'))
                                    
                                finally:
                
                                        select_model_unet(key_unet)
                                        plugin.progress_bar.hide()
                
                            worker = _get_model_folder()
                            worker.returned.connect(_process_model_folder)
                            worker.start()
                
                            # delay showing progress bar -> won't show up if model already downloaded
                            # TODO: hacky -> better way to do this?
                            time.sleep(0.1)
                            plugin.call_button.enabled = False
                            plugin.progress_bar.label = 'Downloading UNET model'
                            plugin.progress_bar.show()
                
                        else:
                            select_model_unet(key_unet)
        else:
                 plugin.call_button.enabled = True
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
   

