import numpy as np
from tifffile import imwrite
import pytest

from vollseg import UNET, StarDist3D, StarDist2D,CARE, MASKUNET,test_image_carcinoma_3dt, data
from typing import List, Union
from vollseg_napari import _test_dock_widget
import napari
from csbdeep.utils import load_json
from vollseg.pretrained import  get_model_folder
from vollseg import VollSeg
def test_defaults(make_napari_viewer):

    fake_plugin_master = _test_dock_widget.plugin_wrapper_vollseg()
    fake_plugin, fake_plugin_star_parameters, fake_plugin_extra_parameters, fake_plugin_display_parameters,  fake_plugin_stop_parameters, get_data = fake_plugin_master
    fake_viewer = make_napari_viewer()
    #get a slice in time and it is a TZYX shape
    image = data.get_test_data()[0:2,0:10, 0:30, 0:30]
    threed_image = image[0,:]
    twod_image = threed_image[0,:]
    name = 'test_3d'
    fake_viewer.add_image(image, name = name )
    
    #Test the pre-trained models if they are compatiable with the images they are supposed to work on 
    fake_plugin_star_parameters.n_tiles.value = (1,1,1,1)
    fake_plugin_star_parameters.star_model_axes.value = 'ZYX'
    fake_plugin.star_seg_model_type.value = StarDist3D
    fake_plugin.unet_seg_model_type.value ='NOUNET'
    fake_plugin.roi_model_type.value ='NOROI'
    fake_plugin.model3d_star.value = 'Carcinoma_cells'
    key_star = fake_plugin.star_seg_model_type.value, fake_plugin.model3d_star.value
    model_path = data.get_stardist_modelpath()
    model_star = fake_plugin.star_seg_model_type.value(None, name=fake_plugin.model3d_star.value, basedir= str(model_path))
    path = str(model_path) + '/' + 'config.json'
    model_star_configs = dict()
    model_star_configs[key_star] = load_json(path)
    config_star = model_star_configs.get(key_star)
    axes_star = config_star.get(
                            'axes', 'ZYXC'[-len(config_star['net_input_shape']) :])
    
        
    key_den = fake_plugin.den_model_type.value, fake_plugin.model_den.value    
    fake_plugin.den_model_type.value = CARE    
    fake_plugin.model_den.value = 'Denoise_carcinoma'
    fake_plugin_extra_parameters.den_model_axes.value = 'ZYX'
  
    model_path = data.get_denoising_modelpath()
    model_den = fake_plugin.den_model_type.value(None, name=fake_plugin.model_den.value, basedir=str(model_path))
    path = str(model_path) + '/' + 'config.json'
    model_den_configs = dict()
    model_den_configs[key_den] = load_json(path)
    config_den = model_den_configs.get(key_den)
    axes_den = config_den.get(
                            'axes', 'ZYXC'[-len(config_den['unet_input_shape']) :])
   
    key_mask = fake_plugin.roi_model_type.value, fake_plugin.model_roi.value    
    fake_plugin.roi_model_type.value = MASKUNET    
    fake_plugin.model_roi.value = 'Xenopus_Cell_Tissue_Segmentation'
    model_path = data.get_maskunet_modelpath()
    model_roi = fake_plugin.roi_model_type.value(None, name=fake_plugin.model_roi.value, basedir=str(model_path))
    valid_star = update(fake_plugin_star_parameters, model_star, model_den, image )


    fake_plugin_star_parameters.n_tiles.value = (1,1,1)
    valid_unet = update_single(fake_plugin_star_parameters,fake_plugin_extra_parameters, model_den, threed_image) 
    valid_star_unet =  update_duo(fake_plugin_star_parameters,model_star, threed_image ) 
    varid_star_den_roi = update_trio(fake_plugin_star_parameters,model_star, model_den, threed_image)
    fake_plugin_extra_parameters.unet_model_axes.value = 'YX'
    valid_roi = update_quad(fake_plugin_star_parameters,fake_plugin_extra_parameters,model_roi,twod_image )
    assert valid_star == True
    assert valid_unet == True 
    assert valid_star_unet == True
    assert varid_star_den_roi == True
    assert valid_roi == True

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def update(fake_plugin_star_parameters, star_model, noise_model, image ):

    
    res = VollSeg(image, star_model = star_model, noise_model = noise_model,  n_tiles = fake_plugin_star_parameters.n_tiles.value, axes = 'ZYX')
    if len(res) == 7:
      valid = True
    else:
      valid = False

    return valid

def update_single(fake_plugin_star_parameters,fake_plugin_extra_parameters, unet_model, image ):


    res = VollSeg(image, unet_model = unet_model,  star_model = None, noise_model = None, roi_model = None,  n_tiles = fake_plugin_star_parameters.n_tiles.value, axes = 'ZYX')
    print(len(res))
    if len(res) == 3:
      valid = True
    else:
      valid = False

    return valid    

def update_duo(fake_plugin_star_parameters, star_model, image ):


    res = VollSeg(image, unet_model = None, star_model = star_model, noise_model = None,roi_model = None,  n_tiles = fake_plugin_star_parameters.n_tiles.value, axes = 'ZYX')
   
    if len(res) == 6:
      valid = True
    else:
      valid = False

    return valid 

def update_trio(fake_plugin_star_parameters, star_model, noise_model, image ):

    res = VollSeg(image, roi_model = noise_model, star_model = star_model, noise_model = noise_model, n_tiles = fake_plugin_star_parameters.n_tiles.value, axes = 'ZYX')
    
    if len(res) == 8:
      valid = True
    else:
      valid = False

    return valid 

def update_quad(fake_plugin_star_parameters,fake_plugin_extra_parameters, noise_model, image ):

    res = VollSeg(image, roi_model = noise_model, n_tiles = fake_plugin_star_parameters.n_tiles.value, axes = 'YX')
  
    if len(res) == 3:
      valid = True
    else:
      valid = False

    return valid    