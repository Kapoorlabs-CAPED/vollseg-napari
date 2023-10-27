# Model Training

## Create training patches

In order to create training data you can point this script to a directory of raw and integer label images and it will create training patches of chosen size: [create_training_patches](scripts/create_training_patches.py) or [create_general_train_patches](scripts/create_general_train_patches.py)

## Train VollSeg Model
To train a vollseg model, you can use this script that will work with the training patches (raw and integer labels) created in the previous step, it will create a directory of binary images too and then train a U-Net model for semantic segmentation and an instance segmentation model using StarDist [train_vollseg](scripts/train_vollseg.py)

## Train CellPose Model
To train a cellpose model for membrane segmentation, you can use this script that will work with the training patches (raw and integer labels) created in the previous step for the membrane channel,it will train an instance segmentation model using CellPose [train_cellpose](scripts/train_cellpose.py)

## Convert Segmentation to Point Clouds
To train a lightning model for autoencoders we need to convert the integer labels to point clouds, this script creates a directory of mesh and point clouds for the input label images, that can be the same as used for segmentation training networks above [create_point_clouds](scripts/create_point_clouds.py).

## Train Autoencoder for cell shape
To train a lightning model for autoencoders we have created a lightning absed repository that is used in this script to train and autoencoder model for refining the raw input point clouds [train_lightning_autoencoder](scripts/train_lightning_autoencoder.py)

