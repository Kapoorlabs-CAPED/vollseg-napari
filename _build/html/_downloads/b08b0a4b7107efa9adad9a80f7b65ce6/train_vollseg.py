

from vollseg import SmartSeeds3D

def main( ):

        base_membrane_dir =  'base_membrane_dir' 
        raw_membrane_dir = 'raw_membrane_patch_dir'
        real_mask_membrane_dir = 'real_mask_membrane_dir'
        binary_mask_membrane_dir = 'binary_mask_membrane_patch_dir'
        nuclei_npz_filename =  'nuclei_npz_filename' 
        membrane_npz_filename = 'membrane_npz_filename'

        base_nuclei_dir =  'base_nuclei_dir' 
        raw_nuclei_dir = 'raw_nuclei_patch_dir'
        real_mask_nuclei_dir = 'real_mask_nuclei_patch_dir'
        binary_mask_nuclei_dir = 'binary_mask_nuclei_patch_dir'


        unet_model_dir = unet_model_dir
        star_model_dir = star_model_dir
        unet_nuclei_model_name = unet_nuclei_model_name
        unet_membrane_model_name = unet_membrane_model_name
        star_nuclei_model_name = star_nuclei_model_name
        star_membrane_model_name = star_membrane_model_name



        in_channels = 1
        grid_x = 1
        grid_y = 1
        backbone = "unet"
        batch_size = 16
        learning_rate = 0.001
        patch_size = (8,256,256)
        generate_npz = True
        load_data_sequence = True
        validation_split = 0.01
        n_patches_per_image = 20
        train_loss = "mae"
        train_unet = True
        train_star = True
        erosion_iterations = 0
        use_gpu_opencl = True
        depth = 3
        kern_size = 3
        startfilter = 64
        n_rays = 96
        epochs = 100

        train_nuclei = True 
        train_membrane = True 
        if train_nuclei:
                SmartSeeds3D(base_dir = base_nuclei_dir, 
                            unet_model_name = unet_nuclei_model_name,
                            star_model_name = star_nuclei_model_name,  
                            unet_model_dir = unet_model_dir,
                            star_model_dir = star_model_dir,
                            npz_filename = nuclei_npz_filename, 
                            raw_dir = raw_nuclei_dir,
                            real_mask_dir = real_mask_nuclei_dir,
                            binary_mask_dir = binary_mask_nuclei_dir,
                            n_channel_in = in_channels,
                            backbone = backbone,
                            load_data_sequence = load_data_sequence, 
                            validation_split = validation_split, 
                            n_patches_per_image = n_patches_per_image, 
                            generate_npz = generate_npz,
                            patch_size = patch_size,
                            grid_x = grid_x,
                            grid_y = grid_y,
                            erosion_iterations = erosion_iterations,  
                            train_loss = train_loss,
                            train_star = train_star,
                            train_unet = train_unet,
                            use_gpu = use_gpu_opencl,  
                            batch_size = batch_size, 
                            depth = depth, 
                            kern_size = kern_size, 
                            startfilter = startfilter, 
                            n_rays = n_rays, 
                            epochs = epochs, 
                            learning_rate = learning_rate)
                
        if train_membrane:
               
                SmartSeeds3D(base_dir = base_membrane_dir, 
                            unet_model_name = unet_membrane_model_name,
                            star_model_name = star_membrane_model_name,  
                            unet_model_dir = unet_model_dir,
                            star_model_dir = star_model_dir,
                            npz_filename = membrane_npz_filename, 
                            raw_dir = raw_membrane_dir,
                            real_mask_dir = real_mask_membrane_dir,
                            binary_mask_dir = binary_mask_membrane_dir,
                            n_channel_in = in_channels,
                            backbone = backbone,
                            load_data_sequence = load_data_sequence, 
                            validation_split = validation_split, 
                            n_patches_per_image = n_patches_per_image, 
                            generate_npz = generate_npz,
                            patch_size = patch_size,
                            grid_x = grid_x,
                            grid_y = grid_y,
                            erosion_iterations = erosion_iterations,  
                            train_loss = train_loss,
                            train_star = train_star,
                            train_unet = train_unet,
                            use_gpu = use_gpu_opencl,  
                            batch_size = batch_size, 
                            depth = depth, 
                            kern_size = kern_size, 
                            startfilter = startfilter, 
                            n_rays = n_rays, 
                            epochs = epochs, 
                            learning_rate = learning_rate) 
                       

if __name__ == "__main__":
    main()        