import hydra
from scenario_train_vollseg_cellpose_sam import TrainCellPose
from hydra.core.config_store import ConfigStore
from vollseg import SmartSeeds2D

configstore = ConfigStore.instance()
configstore.store(name = 'TrainCellPose', node = TrainCellPose)

@hydra.main(config_path = 'conf', config_name = 'scenario_train_vollseg_cellpose_sam')
def main( config : TrainCellPose):

    base_roi_dir = config.train_data_paths.base_roi_dir
    raw_roi_dir = config.train_data_paths.raw_roi_dir
    binary_mask_roi_dir = config.train_data_paths.binary_mask_roi_dir
    roi_model_dir = config.model_paths.roi_model_dir
    roi_model_name = config.model_paths.roi_nuclei_model_name
    roi_npz_filename = config.model_paths.roi_npz_filename

    roi_model_dir = config.model_paths.roi_model_dir
    roi_model_name = config.model_paths.roi_nuclei_model_name

    batch_size = config.parameters.batch_size
    learning_rate = config.parameters.learning_rate
    patch_x = config.parameters.patch_x
    patch_y = config.parameters.patch_y
    generate_npz = config.parameters.generate_npz
    n_patches_per_image = config.parameters.n_patches_per_image
    depth = config.parameters.depth
    kern_size = config.parameters.kern_size
    startfilter = config.parameters.startfilter
    train_unet = True 
    train_star = False
    train_seed_unet = False
    epochs = config.parameters.epochs
    SmartSeeds2D(base_dir = base_roi_dir, 
                    npz_filename = roi_npz_filename, 
                    model_name = roi_model_name, 
                    model_dir = roi_model_dir,
                    raw_dir = raw_roi_dir,
                    binary_mask_dir = binary_mask_roi_dir,
                    n_channel_in = 1,
                    n_patches_per_image = n_patches_per_image, 
                    generate_npz = generate_npz,
                    patch_x= patch_x, 
                    patch_y= patch_y, 
                    erosion_iterations = 0,  
                    train_seed_unet = train_seed_unet,
                    train_star = train_star,
                    train_unet = train_unet,
                    batch_size = batch_size, 
                    depth = depth, 
                    kern_size = kern_size, 
                    startfilter = startfilter, 
                    epochs = epochs, 
                    learning_rate = learning_rate)
   
if __name__ == '__main__':
    main()         