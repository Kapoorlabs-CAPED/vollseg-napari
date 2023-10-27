
from cellshape_helper import conversions
import os


def main( ):

        base_dir = 'base_nuclei_dir'
        cloud_dataset_dir = os.path.join(base_dir, 'cloud_mask_nuclei_dir')
        real_mask_dir = os.path.join(base_dir, 'real_mask_nuclei_patch_dir')
        num_points = 2048
        min_size = (5,16,16)
        conversions.label_tif_to_pc_directory(real_mask_dir, cloud_dataset_dir,num_points, min_size = min_size)


if __name__=='__main__':
        main()         
