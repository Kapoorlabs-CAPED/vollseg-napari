import numpy as np
import vollseg
from vollseg import UNET, StarDist3D, StarDist2D,CARE, MASKUNET
from typing import List, Union
from vollseg_napari import _test_dock_widget
from vollseg import VollSeg
def test_default_star_den_XYZT(make_napari_viewer):
     
    fake_plugin_master = _test_dock_widget.plugin_wrapper_vollseg()
    fake_plugin, fake_plugin_star_parameters, fake_plugin_extra_parameters, fake_plugin_display_parameters,  fake_plugin_stop_parameters, get_data = fake_plugin_master
    fake_viewer = make_napari_viewer()
    #get a slice in time and it is a TZYX shape
    image = vollseg.get_data.get_test_data()
    
    image = np.asarray(vollseg.get_data.get_test_data()[0:2,image.shape[1] - 5:image.shape[1] + 5, image.shape[2] - 5:image.shape[2] + 5, image.shape[3] - 5:image.shape[3] + 5])
    threed_image = image[0,:]
    twod_image = threed_image[0,:]
    name = 'test_3dt'
    fake_viewer.add_image(image, name = name )
    
    #Test the pre-trained models if they are compatiable with the images they are supposed to work on 
    fake_plugin_star_parameters.n_tiles.value = (1,2,2,2)
    fake_plugin_star_parameters.star_model_axes.value = 'ZYX'
    fake_plugin.star_seg_model_type.value = StarDist3D
    fake_plugin.unet_seg_model_type.value ='NOUNET'
    fake_plugin.roi_model_type.value ='NOROI'
    fake_plugin.model3d_star.value = 'Carcinoma_cells'
    key_star = fake_plugin.star_seg_model_type.value, fake_plugin.model3d_star.value
    model_path = vollseg.get_data.get_stardist_modelpath()
    model_star = fake_plugin.star_seg_model_type.value(None, name=fake_plugin.model3d_star.value, basedir= str(model_path))
    key_den = fake_plugin.den_model_type.value, fake_plugin.model_den.value    
    fake_plugin.den_model_type.value = CARE    
    fake_plugin.model_den.value = 'Denoise_carcinoma'
    fake_plugin_extra_parameters.den_model_axes.value = 'ZYX'
    model_path = vollseg.get_data.get_denoising_modelpath()
    model_den = fake_plugin.den_model_type.value(None, name=fake_plugin.model_den.value, basedir=str(model_path))
    # Test stardist model with denoising model on a timelase image
    valid_star = update(fake_plugin_star_parameters, model_star, model_den, image )
    assert valid_star == True
def test_default_star_unet_XYZ(make_napari_viewer):
     
    fake_plugin_master = _test_dock_widget.plugin_wrapper_vollseg()
    fake_plugin, fake_plugin_star_parameters, fake_plugin_extra_parameters, fake_plugin_display_parameters,  fake_plugin_stop_parameters, get_data = fake_plugin_master
    fake_viewer = make_napari_viewer()
    #get a slice in time and it is a TZYX shape
    image = vollseg.get_data.get_test_data()
    image = np.asarray(vollseg.get_data.get_test_data()[0:2,image.shape[1] - 5:image.shape[1] + 5, image.shape[2] - 5:image.shape[2] + 5, image.shape[3] - 5:image.shape[3] + 5])
    key_den = fake_plugin.den_model_type.value, fake_plugin.model_den.value    
    fake_plugin.den_model_type.value = CARE    
    fake_plugin.model_den.value = 'Denoise_carcinoma'
    fake_plugin_extra_parameters.den_model_axes.value = 'ZYX'
    model_path = vollseg.get_data.get_denoising_modelpath()
    model_den = fake_plugin.den_model_type.value(None, name=fake_plugin.model_den.value, basedir=str(model_path))
    
    threed_image = image[0,:]
    
    name = 'test_3d'
    fake_viewer.add_image(threed_image, name = name )   
     
    valid_unet = update_single(fake_plugin_star_parameters, model_den, threed_image) 
    assert valid_unet == True

def test_default_star_XYZ(make_napari_viewer):
     
    fake_plugin_master = _test_dock_widget.plugin_wrapper_vollseg()
    fake_plugin, fake_plugin_star_parameters, fake_plugin_extra_parameters, fake_plugin_display_parameters,  fake_plugin_stop_parameters, get_data = fake_plugin_master
    fake_viewer = make_napari_viewer()
    #get a slice in time and it is a TZYX shape
    image = vollseg.get_data.get_test_data()
    image = np.asarray(vollseg.get_data.get_test_data()[0:2,image.shape[1] - 5:image.shape[1] + 5, image.shape[2] - 5:image.shape[2] + 5, image.shape[3] - 5:image.shape[3] + 5])
    threed_image = image[0,:]
       
    name = 'test_3d'
    fake_viewer.add_image(threed_image, name = name ) 
    key_star = fake_plugin.star_seg_model_type.value, fake_plugin.model3d_star.value
    model_path = vollseg.get_data.get_stardist_modelpath()
    model_star = fake_plugin.star_seg_model_type.value(None, name=fake_plugin.model3d_star.value, basedir= str(model_path))
    fake_plugin_star_parameters.n_tiles.value = (1,2,2)
    valid_star_unet =  update_duo(fake_plugin_star_parameters,model_star, threed_image ) 
    assert valid_star_unet == True
    
def test_default_star_den_XYZ(make_napari_viewer):
 
    fake_plugin_master = _test_dock_widget.plugin_wrapper_vollseg()
    fake_plugin, fake_plugin_star_parameters, fake_plugin_extra_parameters, fake_plugin_display_parameters,  fake_plugin_stop_parameters, get_data = fake_plugin_master
    fake_viewer = make_napari_viewer()
    #get a slice in time and it is a TZYX shape
    image = vollseg.get_data.get_test_data()
    image = np.asarray(vollseg.get_data.get_test_data()[0:2,image.shape[1] - 5:image.shape[1] + 5, image.shape[2] - 5:image.shape[2] + 5, image.shape[3] - 5:image.shape[3] + 5])
    threed_image = image[0,:]
       
    name = 'test_3d'
    fake_viewer.add_image(threed_image, name = name ) 
    key_star = fake_plugin.star_seg_model_type.value, fake_plugin.model3d_star.value
    model_path = vollseg.get_data.get_stardist_modelpath()
    model_star = fake_plugin.star_seg_model_type.value(None, name=fake_plugin.model3d_star.value, basedir= str(model_path))
    fake_plugin_star_parameters.n_tiles.value = (1,2,2) 

    key_den = fake_plugin.den_model_type.value, fake_plugin.model_den.value    
    fake_plugin.den_model_type.value = CARE    
    fake_plugin.model_den.value = 'Denoise_carcinoma'
    fake_plugin_extra_parameters.den_model_axes.value = 'ZYX'
    model_path = vollseg.get_data.get_denoising_modelpath()
    model_den = fake_plugin.den_model_type.value(None, name=fake_plugin.model_den.value, basedir=str(model_path))
    varid_star_den_roi = update_trio(fake_plugin_star_parameters,model_star, model_den, threed_image)
    assert varid_star_den_roi == True

def test_default_star_roi_XY(make_napari_viewer):


    fake_plugin_master = _test_dock_widget.plugin_wrapper_vollseg()
    fake_plugin, fake_plugin_star_parameters, fake_plugin_extra_parameters, fake_plugin_display_parameters,  fake_plugin_stop_parameters, get_data = fake_plugin_master
    fake_viewer = make_napari_viewer()
    #get a slice in time and it is a TZYX shape
    image = vollseg.get_data.get_test_data()
    
    image = np.asarray(vollseg.get_data.get_test_data()[0:2,image.shape[1] - 5:image.shape[1] + 5, image.shape[2] - 5:image.shape[2] + 5, image.shape[3] - 5:image.shape[3] + 5])
    threed_image = image[0,:]
    twod_image = threed_image[0,:]
    name = 'test_2d'
    fake_viewer.add_image(twod_image, name = name )
    
    #Test the pre-trained models if they are compatiable with the images they are supposed to work on 
    fake_plugin_star_parameters.n_tiles.value = (1,2,2,2)
    fake_plugin_star_parameters.star_model_axes.value = 'ZYX'
    fake_plugin.star_seg_model_type.value = StarDist3D
    fake_plugin.unet_seg_model_type.value ='NOUNET'
    fake_plugin.roi_model_type.value ='NOROI'
    fake_plugin.model3d_star.value = 'Carcinoma_cells'
    key_star = fake_plugin.star_seg_model_type.value, fake_plugin.model3d_star.value
    model_path = vollseg.get_data.get_stardist_modelpath()
    model_star = fake_plugin.star_seg_model_type.value(None, name=fake_plugin.model3d_star.value, basedir= str(model_path))

    key_mask = fake_plugin.roi_model_type.value, fake_plugin.model_roi.value    
    fake_plugin.roi_model_type.value = MASKUNET    
    fake_plugin.model_roi.value = 'Xenopus_Cell_Tissue_Segmentation'
    model_path = vollseg.get_data.get_maskunet_modelpath()
    model_roi = fake_plugin.roi_model_type.value(None, name=fake_plugin.model_roi.value, basedir=str(model_path))
    fake_plugin_star_parameters.n_tiles.value = (1,2,2)
    fake_plugin_extra_parameters.unet_model_axes.value = 'YX'
    valid_roi = update_quad(fake_plugin_star_parameters,model_roi,twod_image )
    assert valid_roi == True
def test_default_roi_XY(make_napari_viewer):


    fake_plugin_master = _test_dock_widget.plugin_wrapper_vollseg()
    fake_plugin, fake_plugin_star_parameters, fake_plugin_extra_parameters, fake_plugin_display_parameters,  fake_plugin_stop_parameters, get_data = fake_plugin_master
    fake_viewer = make_napari_viewer()
    #get a slice in time and it is a TZYX shape
    image = vollseg.get_data.get_test_data()
    image = np.asarray(vollseg.get_data.get_test_data()[0:2,image.shape[1] - 5:image.shape[1] + 5, image.shape[2] - 5:image.shape[2] + 5, image.shape[3] - 5:image.shape[3] + 5])
    threed_image = image[0,:]
    twod_image = threed_image[0,:]
    name = 'test_2d'
    fake_viewer.add_image(twod_image, name = name )
    
    #Test the pre-trained models if they are compatiable with the images they are supposed to work on 
    key_mask = fake_plugin.roi_model_type.value, fake_plugin.model_roi.value    
    fake_plugin.roi_model_type.value = MASKUNET    
    fake_plugin.model_roi.value = 'Xenopus_Cell_Tissue_Segmentation'
    model_path = vollseg.get_data.get_maskunet_modelpath()
    model_roi = fake_plugin.roi_model_type.value(None, name=fake_plugin.model_roi.value, basedir=str(model_path))
    fake_plugin_star_parameters.n_tiles.value = (1,2,2)
    fake_plugin_extra_parameters.unet_model_axes.value = 'YX'
    valid_roi = update_2D_roi(fake_plugin_star_parameters,model_roi,twod_image )
    assert valid_roi == True

def test_default_unet_roi_XY(make_napari_viewer):


    fake_plugin_master = _test_dock_widget.plugin_wrapper_vollseg()
    fake_plugin, fake_plugin_star_parameters, fake_plugin_extra_parameters, fake_plugin_display_parameters,  fake_plugin_stop_parameters, get_data = fake_plugin_master
    fake_viewer = make_napari_viewer()
    #get a slice in time and it is a TZYX shape
    image = vollseg.get_data.get_test_data()
    image = np.asarray(vollseg.get_data.get_test_data()[0:2,image.shape[1] - 5:image.shape[1] + 5, image.shape[2] - 5:image.shape[2] + 5, image.shape[3] - 5:image.shape[3] + 5])
    threed_image = image[0,:]
    twod_image = threed_image[0,:]
    name = 'test_2d'
    fake_viewer.add_image(twod_image, name = name )
    
    #Test the pre-trained models if they are compatiable with the images they are supposed to work on 
    key_mask = fake_plugin.roi_model_type.value, fake_plugin.model_roi.value    
    fake_plugin.roi_model_type.value = MASKUNET    
    fake_plugin.model_roi.value = 'Xenopus_Cell_Tissue_Segmentation'
    model_path = vollseg.get_data.get_maskunet_modelpath()
    model_roi = fake_plugin.roi_model_type.value(None, name=fake_plugin.model_roi.value, basedir=str(model_path))
    fake_plugin_star_parameters.n_tiles.value = (1,2,2)
    fake_plugin_extra_parameters.unet_model_axes.value = 'YX'
    unet_model = model_roi
    valid_roi = update_2D_unet_roi(fake_plugin_star_parameters, unet_model, model_roi,twod_image )
    assert valid_roi == True    

def test_default_star_roi_XYZ(make_napari_viewer):


    fake_plugin_master = _test_dock_widget.plugin_wrapper_vollseg()
    fake_plugin, fake_plugin_star_parameters, fake_plugin_extra_parameters, fake_plugin_display_parameters,  fake_plugin_stop_parameters, get_data = fake_plugin_master
    fake_viewer = make_napari_viewer()
    #get a slice in time and it is a TZYX shape
    image = vollseg.get_data.get_test_data()
    
    image = np.asarray(vollseg.get_data.get_test_data()[0:2,image.shape[1] - 5:image.shape[1] + 5, image.shape[2] - 5:image.shape[2] + 5, image.shape[3] - 5:image.shape[3] + 5])
    threed_image = image[0,:]
    name = 'test_3d'
    fake_viewer.add_image(threed_image, name = name )
    
    #Test the pre-trained models if they are compatiable with the images they are supposed to work on 
    fake_plugin_star_parameters.n_tiles.value = (1,2,2,2)
    fake_plugin_star_parameters.star_model_axes.value = 'ZYX'
    fake_plugin.star_seg_model_type.value = StarDist3D
    fake_plugin.unet_seg_model_type.value ='NOUNET'
    fake_plugin.roi_model_type.value ='NOROI'
    fake_plugin.model3d_star.value = 'Carcinoma_cells'
    key_star = fake_plugin.star_seg_model_type.value, fake_plugin.model3d_star.value
    model_path = vollseg.get_data.get_stardist_modelpath()
    model_star = fake_plugin.star_seg_model_type.value(None, name=fake_plugin.model3d_star.value, basedir= str(model_path))

    key_mask = fake_plugin.roi_model_type.value, fake_plugin.model_roi.value    
    fake_plugin.roi_model_type.value = MASKUNET    
    fake_plugin.model_roi.value = 'Xenopus_Cell_Tissue_Segmentation'
    model_path = vollseg.get_data.get_maskunet_modelpath()
    model_roi = fake_plugin.roi_model_type.value(None, name=fake_plugin.model_roi.value, basedir=str(model_path))
    fake_plugin_star_parameters.n_tiles.value = (1,2,2)
    fake_plugin_extra_parameters.unet_model_axes.value = 'YX'
    valid_roi = update_poly(fake_plugin_star_parameters,model_roi,threed_image )
    assert valid_roi == True

def update(fake_plugin_star_parameters, star_model, noise_model, image ):

    
    res = VollSeg(image, star_model = star_model, noise_model = noise_model,  n_tiles = fake_plugin_star_parameters.n_tiles.value, axes = 'ZYX')
    if len(res) == 7:
      valid = True
    else:
      valid = False

    return valid

def update_single(fake_plugin_star_parameters, unet_model, image ):


    res = VollSeg(image, unet_model = unet_model,  star_model = None, noise_model = None, roi_model = None,  n_tiles = fake_plugin_star_parameters.n_tiles.value, axes = 'ZYX')
    
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

def update_quad(fake_plugin_star_parameters, roi_model, image ):

    res = VollSeg(image, roi_model = roi_model, n_tiles = fake_plugin_star_parameters.n_tiles.value, axes = 'YX')
  
    if len(res) == 3:
      valid = True
    else:
      valid = False

    return valid    

def update_poly(fake_plugin_star_parameters, roi_model, image ):

    res = VollSeg(image, roi_model = roi_model, n_tiles = fake_plugin_star_parameters.n_tiles.value, axes = 'ZYX')
  
    if len(res) == 3:
      valid = True
    else:
      valid = False

    return valid     

def update_2D_roi(fake_plugin_star_parameters, roi_model, image ):

    res = VollSeg(image, roi_model = roi_model, n_tiles = fake_plugin_star_parameters.n_tiles.value, axes = 'YX')
  
    if len(res) == 3:
      valid = True
    else:
      valid = False

    return valid    

def update_2D_unet_roi(fake_plugin_star_parameters, unet_model, roi_model, image ):

    res = VollSeg(image, unet_model = unet_model, roi_model = roi_model, n_tiles = fake_plugin_star_parameters.n_tiles.value, axes = 'YX')
  
    if len(res) == 3:
      valid = True
    else:
      valid = False

    return valid    