from vollseg import VollOne, UNET, CARE, StarDist3D
import os
from csbdeep.models import ProjectionCARE
import numpy as np
from tqdm import tqdm
from tifffile import imread
from pathlib import Path


def main():
    membrane_image_dir = 'path/to/membrane/imagedir'
    nuclei_image_dir = 'path/to/nuclei/imagedir'
    save_dir = 'path/to/savedir'

    unet_model_membrane_name = 'path/to/unetmembranemodel'  
    den_model_membrane_name = 'path/to/denmembranemodel'
    roi_model_name = 'path/to/roimodel'
    star_nuclei_model_name = 'path/to/starnucleimodel'

    unet_model_dir = 'path/to/unetmodeldir'
    roi_model_dir = 'path/to/roimodeldir'
    denoise_model_dir = 'path/to/denoisemodeldir'
    star_model_dir = 'path/to/starmodeldir'

    nuclei_star_model = StarDist3D(config = None, name = star_nuclei_model_name, basedir = star_model_dir)
    membrane_unet_model = UNET(config=None, name=unet_model_membrane_name, basedir=unet_model_dir)
    membrane_noise_model =  CARE(config=None, name=den_model_membrane_name, basedir=denoise_model_dir)    
    membrane_roi_model = ProjectionCARE(config=None, name=roi_model_name, basedir=roi_model_dir)
    n_tiles = (1,1,1)
    channel_membrane = 0
    channel_nuclei = 1
    file_type = '*.tif'
    
    # Check if re_run condition is met
    re_run = os.path.exists(os.path.join(save_dir, 'nuclei_labels'))

    # Iterate through files and directories
    possibleChildren = [os.path.join(membrane_image_dir, file) for file in os.listdir(membrane_image_dir)]
    for dir in tqdm(possibleChildren):
        file = os.listdir(dir)
        for fname in file:
            if fname.endswith(file_type.replace('*','')):
                if re_run or fname not in os.listdir(os.path.join(save_dir, 'nuclei_labels')):
                    image_channel_0 = imread(os.path.join(dir, fname))
                    fname = fname.replace('ch_0', 'ch_2')
                    image_channel_2 = imread(os.path.join(nuclei_image_dir, os.path.basename(dir), fname))
                    dual_image = np.asarray([image_channel_0, image_channel_2])

                    Name = os.path.basename(os.path.splitext(fname)[0])
                    VollOne(
                        dual_image,
                        channel_membrane = channel_membrane,
                        channel_nuclei = channel_nuclei,
                        star_model_nuclei= nuclei_star_model,
                        unet_model_membrane = membrane_unet_model,
                        noise_model_membrane= membrane_noise_model,
                        roi_model = membrane_roi_model,
                        n_tiles= n_tiles,
                        save_dir = save_dir,
                        Name = Name,
                        axes = "CZYX",
                    )

if __name__=='__main__':
    main()
