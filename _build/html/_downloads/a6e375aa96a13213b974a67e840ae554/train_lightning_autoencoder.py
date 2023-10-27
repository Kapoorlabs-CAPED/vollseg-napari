from kapoorlabs_lightning.optimizers import Adam
from kapoorlabs_lightning.pytorch_losses import ChamferLoss
from kapoorlabs_lightning.lightning_trainer import AutoLightningTrain
from kapoorlabs_lightning import  PointCloudDataset
from cellshape_cloud import CloudAutoEncoder
import os

def main( ):


        base_dir = 'base_membrane_dir'
        cloud_dataset_dir = os.path.join(base_dir, 'cloud_mask_membrane_dir')
        point_cloud_file = os.path.join(cloud_dataset_dir, 'point_cloud_filename')
        cloud_model_dir = 'cloud_membrane_model_dir'
        
        cloud_model_name = 'cloud_membrane_model_name'
        batch_size = 16
        num_features = 64
        encoder_type = 'dgcnn'
        decoder_type = 'foldingnet'
        k_nearest_neighbours = 16
        learning_rate = 0.001
        num_epochs = 100
        output_dir = os.path.join(cloud_model_dir, cloud_model_name)
        ckpt_file = output_dir
        scale_z = 8
        scale_xy = 16
        model = CloudAutoEncoder(
        num_features=num_features,
        k=k_nearest_neighbours,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        )
        
       

        dataset = PointCloudDataset(point_cloud_file, scale_z = scale_z, scale_xy = scale_xy)
        loss = ChamferLoss()
        optimizer = Adam(lr=learning_rate)
         
        
        # Now we have everything (except the logger) to start the training
        lightning_special_train = AutoLightningTrain(dataset, loss, model, optimizer,output_dir,  ckpt_file = ckpt_file, batch_size= batch_size, epochs = num_epochs,
                                                 accelerator = 'gpu', devices = -1)

        lightning_special_train._train_model()

if __name__=='__main__':
      main()        
