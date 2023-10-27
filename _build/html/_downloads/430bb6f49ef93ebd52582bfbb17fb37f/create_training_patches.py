from vollseg import NucleiPatches


def main( ):


        base_nuclei_dir =  'base_nuclei_dir' 
        raw_nuclei_dir = 'raw_nuclei_dir'
        

        nuclei_channel_results_directory = 'nuclei_channel_results_directory'
        nuclei_raw_save_dir = 'nuclei_raw_save_dir'
        nuclei_real_mask_patch_dir = 'nuclei_real_mask_patch_dir'
        nuclei_binary_mask_patch_dir = 'nuclei_binary_mask_patch_dir'
        lower_ratio_fore_to_back=0.1
        upper_ratio_fore_to_back=0.9
        patch_size = (8, 128, 128)
        erosion_iterations = 0

        
        NucleiPatches(base_nuclei_dir,
        raw_nuclei_dir,
        nuclei_channel_results_directory,
        nuclei_raw_save_dir,
        nuclei_real_mask_patch_dir,
        nuclei_binary_mask_patch_dir,
        patch_size, erosion_iterations,
        lower_ratio_fore_to_back=lower_ratio_fore_to_back,
        upper_ratio_fore_to_back=upper_ratio_fore_to_back)

if __name__=='__main__':
        main()        
