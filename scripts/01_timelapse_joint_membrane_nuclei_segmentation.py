from vollseg import VollOne, UNET, CARE, StarDist3D
import os
from csbdeep.models import ProjectionCARE
import numpy as np
from tqdm import tqdm
from tifffile import imread
from pathlib import Path
def main():
    timelapse_membrane_image_dir = 'timelapse_membrane_directory'
    timelapse_nuclei_image_dir = 'timelapse_nuclei_directory'
    save_dir = Path(timelapse_nuclei_image_dir) /'dual_timelapse_segmentation'
    Path(save_dir).mkdir(exist_ok=True)

    unet_model_membrane_name = 'unet_membrane_model_name'   
    den_model_membrane_name = 'den_membrane_model_name'
    roi_model_name = 'roi_model_name'
    star_nuclei_model_name = 'star_nuclei_model_name'

    unet_model_dir = 'unet_model_dir'
    roi_model_dir = 'roi_model_dir'
    denoise_model_dir = 'den_model_dir'
    star_model_dir = 'star_model_dir'

    nuclei_star_model = StarDist3D(config = None, name = star_nuclei_model_name, basedir = star_model_dir)
    membrane_unet_model = UNET(config=None, name=unet_model_membrane_name, basedir=unet_model_dir)
    membrane_noise_model =  CARE(config=None, name=den_model_membrane_name, basedir=denoise_model_dir)    
    membrane_roi_model = ProjectionCARE(config=None, name=roi_model_name, basedir=roi_model_dir)
    n_tiles = (1,1,1)
    channel_membrane = 0
    channel_nuclei = 1
    file_type = '*.tif'
    file = os.listdir(timelapse_membrane_image_dir)
    re_run = os.path.exists(os.path.join(save_dir, 'membrane_labels'))
    for fname in file:
        if fname.endswith(file_type.replace('*', '')):
            if re_run:
                if fname not in os.listdir(os.path.join(save_dir, 'membrane_labels')):
                    process_image(fname, save_dir, timelapse_membrane_image_dir, timelapse_nuclei_image_dir, channel_membrane, channel_nuclei, nuclei_star_model, membrane_unet_model, membrane_noise_model, membrane_roi_model, n_tiles)
            else:
                process_image(fname, save_dir, timelapse_membrane_image_dir, timelapse_nuclei_image_dir, channel_membrane, channel_nuclei, nuclei_star_model, membrane_unet_model, membrane_noise_model, membrane_roi_model, n_tiles)

def process_image(image_path, save_dir, timelapse_membrane_image_dir, timelapse_nuclei_image_dir, channel_membrane, channel_nuclei, nuclei_star_model, membrane_unet_model, membrane_noise_model, membrane_roi_model, n_tiles):
    image_channel_0 = imread(os.path.join(timelapse_membrane_image_dir, image_path))
    image_path = image_path.replace('channel_0', 'channel_2')
    image_channel_2 = imread(os.path.join(timelapse_nuclei_image_dir, os.path.basename(timelapse_membrane_image_dir), image_path))
    dual_image = np.asarray([image_channel_0, image_channel_2])
    Name = os.path.basename(os.path.splitext(image_path)[0])
    VollOne(
        dual_image,
        channel_membrane=channel_membrane,
        channel_nuclei=channel_nuclei,
        star_model_nuclei=nuclei_star_model,
        unet_model_membrane=membrane_unet_model,
        noise_model_membrane=membrane_noise_model,
        roi_model=membrane_roi_model,
        n_tiles=n_tiles,
        save_dir=save_dir,
        Name=Name,
        axes="CTZYX",
    )
                

if __name__=='__main__':

        main()
