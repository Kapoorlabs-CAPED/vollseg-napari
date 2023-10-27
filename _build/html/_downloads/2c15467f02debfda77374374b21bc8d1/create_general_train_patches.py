

from vollseg import SmartPatches

def main( ):


        base_membrane_dir =  'base_membrane_dir' 
        raw_membrane_dir = 'raw_membrane_dir'
        base_nuclei_dir =  'base_nuclei_dir' 
        raw_nuclei_dir = 'raw_nuclei_dir'
        

        membrane_channel_results_directory = 'real_mask_membrane_dir'
        membrane_raw_save_dir = 'raw_membrane_patch_dir'
        membrane_real_mask_patch_dir = 'real_mask_membrane_patch_dir'
        membrane_binary_mask_patch_dir = 'binary_mask_membrane_patch_dir'
        nuclei_channel_results_directory = 'real_mask_nuclei_dir'
        nuclei_raw_save_dir = 'raw_nuclei_patch_dir'
        nuclei_real_mask_patch_dir = 'real_mask_nuclei_patch_dir'
        nuclei_binary_mask_patch_dir = 'binary_mask_nuclei_patch_dir'
        lower_ratio_fore_to_back=0.6
        upper_ratio_fore_to_back=1
        patch_size = (8,256,256)
        erosion_iterations = 2
        create_for_channel = 'nuclei'

        
        SmartPatches(base_membrane_dir,
        raw_membrane_dir,
        base_nuclei_dir,
        raw_nuclei_dir,
        nuclei_channel_results_directory,
        membrane_channel_results_directory,
        nuclei_raw_save_dir,
        membrane_raw_save_dir,
        nuclei_real_mask_patch_dir,
        membrane_real_mask_patch_dir,
        nuclei_binary_mask_patch_dir,
        membrane_binary_mask_patch_dir, patch_size, erosion_iterations,
        create_for_channel = create_for_channel, 
        lower_ratio_fore_to_back=lower_ratio_fore_to_back,
        upper_ratio_fore_to_back=upper_ratio_fore_to_back)

if __name__=='__main__':
        main()        
