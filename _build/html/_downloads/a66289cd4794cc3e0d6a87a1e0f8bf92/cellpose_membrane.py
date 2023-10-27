import os
import glob
from tifffile import imread
from vollseg import StarDist3D, UNET, MASKUNET
from vollseg.utils import MembraneSeg

from pathlib import Path 

def main():
    
    image_dir = '/path(to/dual_channel_dir'
   
    save_dir = os.path.join(image_dir, 'MembraneSeg')
    Path(save_dir).mkdir(exist_ok=True)
    
    cellpose_model_name = 'cellpose2D_model_name'

    diameter_cellpose = 34.6
    stitch_threshold = 0.25
    channel_membrane = 1
    cellpose_model_dir = 'cellpose2D_model_dir'
    flow_threshold = 0.7
    cellprob_threshold = 0.0
    gpu = True


    Raw_path = os.path.join(image_dir, '*.tif')
    filesRaw = glob.glob(Raw_path)
    filesRaw.sort
    
    min_size = 1
    min_size_mask = 1
    max_size = 1000000
    n_tiles = (1,1,1)
    dounet = False
    seedpool = False
    slice_merge = False
    UseProbability = True
    donormalize = True
    axes ='ZYX'
    do_3D = False
    ExpandLabels = False
    z_thresh = 2
    for fname in filesRaw:
    
                    image = imread(fname)
                    image_membrane = image[:, channel_membrane, :, :]
                    Name = os.path.basename(os.path.splitext(fname)[0])
                    MembraneSeg( image_membrane, 
                            diameter_cellpose= diameter_cellpose,
                            stitch_threshold = stitch_threshold,
                            flow_threshold = flow_threshold,
                            cellprob_threshold = cellprob_threshold,
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
                            z_thresh = z_thresh) 

    
    
if __name__ == '__main__':
    main()    