import os
import glob
from tifffile import imread
from vollseg import StarDist3D, UNET, CARE
from vollseg.utils import VollSeg
from pathlib import Path 
def main():
    
    image_dir = '/path/to/image/'
    save_dir = os.path.join(image_dir, 'VollSeg')
    Path(save_dir).mkdir(exist_ok=True)
    
    unet_model_name = 'unet_nuclei_model'
    star_model_name = 'nuclei_model'

    unet_model = UNET(config = None, name = unet_model_name, basedir = '/path/to/unet/modeldir/')
    star_model = StarDist3D(config = None, name = star_model_name, basedir = '/path/to/stardist/modeldir/')
    Raw_path = os.path.join(image_dir, '*.tif')
    filesRaw = glob.glob(Raw_path)
    filesRaw.sort
    min_size = 10
    min_size_mask = 10
    max_size = 10000
    n_tiles = (2,2,2)
    dounet = True
    seedpool = True
    slice_merge = False 
    UseProbability = True
    donormalize = True
    axes = 'ZYX'
    ExpandLabels = False
    for fname in filesRaw:
    
                    image = imread(fname)
                    Name = os.path.basename(os.path.splitext(fname)[0])
                    VollSeg( image, 
                            unet_model = unet_model, 
                            star_model = star_model, 
                            seedpool = seedpool, 
                            axes = axes, 
                            min_size = min_size,  
                            min_size_mask = min_size_mask,
                            max_size = max_size,
                            donormalize=donormalize,
                            n_tiles = n_tiles,
                            ExpandLabels = ExpandLabels,
                            slice_merge = slice_merge, 
                            UseProbability = UseProbability, 
                            save_dir = save_dir, 
                            Name = Name,
                            dounet = dounet) 

    
    
if __name__ == '__main__':
    main()    