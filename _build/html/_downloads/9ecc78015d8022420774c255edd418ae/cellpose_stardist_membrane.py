import os
import glob
from tifffile import imread
from vollseg import StarDist3D, UNET, MASKUNET
from vollseg.utils import VollCellSeg
from pathlib import Path 
from pynvml.smi import nvidia_smi
import tensorflow as tf
nvsmi = nvidia_smi.getInstance()


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        memory = nvsmi.DeviceQuery("memory.free")["gpu"][0]["fb_memory_usage"][
            "free"
        ]
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=0.5 * memory)],
        )
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def main():
    dual_channel_image_dir = '/path/to/dual_channeldir'
    save_dir = os.path.join(dual_channel_image_dir, 'CellPose2DResults')
    Path(save_dir).mkdir(exist_ok=True)

    unet_model_nuclei_name = 'unetnucleimodelname'  
    star_model_nuclei_name = 'starnucleimodelname' 
    roi_model_nuclei_name = 'roinucleimodelname' 

    unet_model_membrane_name = 'unetmembranemodelname'    
    star_model_membrane_name = 'starmembranemodelname' 
    cellpose_model_name = 'cellposemodelname' 

    unet_model_dir = 'unetnucleimodeldir' 
    star_model_dir = 'starnucleimodeldir' 
    roi_model_dir = 'roimodeldir'
    cellpose_model_dir = 'cellposemodeldir'

    diameter_cellpose = 34
    stitch_threshold = 0.5
    channel_membrane = 0
    channel_nuclei = 1
    flow_threshold = 0.7
    cellprob_threshold = 0
    gpu = True

    unet_model_nuclei = UNET(config=None, name=unet_model_nuclei_name, basedir=unet_model_dir)
    star_model_nuclei = StarDist3D(config=None, name=star_model_nuclei_name, basedir=star_model_dir)
    roi_model_nuclei = MASKUNET(config=None, name=roi_model_nuclei_name, basedir=roi_model_dir)

    unet_model_membrane = UNET(config=None, name=unet_model_membrane_name, basedir=unet_model_dir)
    star_model_membrane = StarDist3D(config=None, name=star_model_membrane_name, basedir=star_model_dir)


    Raw_path = os.path.join(dual_channel_image_dir, '*.tif')
    filesRaw = glob.glob(Raw_path)
    filesRaw.sort
    min_size = 10
    min_size_mask = 10
    max_size = 1000
    do_3D = True
    n_tiles = (1,1,1)
    dounet = False
    seedpool = False
    slice_merge = True
    UseProbability = True
    donormalize = True
    axes = 'ZYX'
    ExpandLabels = False
    z_thresh = 3
    for fname in filesRaw:
        image = imread(fname)
        Name = os.path.basename(os.path.splitext(fname)[0])
        VollCellSeg(
                    image,
                    diameter_cellpose = diameter_cellpose,
                    stitch_threshold = stitch_threshold,
                    channel_membrane = channel_membrane,
                    channel_nuclei = channel_nuclei,
                    flow_threshold = flow_threshold,
                    cellprob_threshold = cellprob_threshold,
                    unet_model_nuclei=unet_model_nuclei,
                    star_model_nuclei=star_model_nuclei,
                    roi_model_nuclei=roi_model_nuclei,
                    unet_model_membrane=unet_model_membrane,
                    star_model_membrane=star_model_membrane,
                    cellpose_model_path= os.path.join(cellpose_model_dir, cellpose_model_name),
                    gpu = gpu,
                    axes = axes,
                    min_size_mask = min_size_mask,
                    min_size = min_size,
                    max_size = max_size,
                    n_tiles = n_tiles,
                    UseProbability= UseProbability,
                    ExpandLabels = ExpandLabels,
                    donormalize = donormalize,
                    dounet = dounet,
                    seedpool=seedpool,
                    save_dir=save_dir,
                    Name = Name,
                    slice_merge=slice_merge,
                    do_3D=do_3D,
                    z_thresh = z_thresh
                )



if __name__ == '__main__':
    main()
