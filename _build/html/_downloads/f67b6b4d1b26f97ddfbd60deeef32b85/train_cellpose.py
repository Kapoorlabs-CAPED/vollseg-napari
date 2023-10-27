
from vollseg import CellPose

def main( ):


        base_membrane_dir =  'base_membrane_dir'
        raw_membrane_dir = 'raw_membrane_patch_dir' 
        real_mask_membrane_dir = 'real_mask_membrane_patch_dir'
        test_raw_membrane_patch_dir = 'test_raw_membrane_patch_dir' 
        test_real_mask_membrane_patch_dir = 'test_real_mask_membrane_patch_dir' 

        cellpose_model_dir = 'cellpose2D_model_dir'
        cellpose_model_name = 'cellpose2D_model_name'

        epochs = 100
        learning_rate = 0.001
        gpu = True
        in_channels = 1


        CellPose(base_membrane_dir,
            cellpose_model_name,
            cellpose_model_dir,
            raw_membrane_dir,
            real_mask_membrane_dir,
            test_raw_membrane_patch_dir,
            test_real_mask_membrane_patch_dir,
            n_epochs=epochs,
            learning_rate=learning_rate,
            channels=in_channels,
            gpu=gpu)
        
if __name__=='__main__':
    main()        