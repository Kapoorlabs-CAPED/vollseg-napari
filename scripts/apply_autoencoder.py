import os
from pathlib import Path

from cellshape_cloud import CloudAutoEncoder
from kapoorlabs_lightning.lightning_trainer import AutoLightningModel
from kapoorlabs_lightning.optimizers import Adam
from kapoorlabs_lightning.pytorch_losses import ChamferLoss
from napatrackmater import load_json
from napatrackmater.Trackmate import TrackMate
from tifffile import imread


def main():

    xml_path = Path("path/to/xml/file")
    track_csv = Path("path/to/track/csv/file")
    spot_csv = Path("path/to/spot/csv/file")
    edges_csv = Path("path/to/edges/csv/file")
    modelconfig = load_json("path/to/model/config/json/file")
    master_extra_name = "Master_"
    learning_rate = 0.001
    accelerator = "gpu"
    devices = 1
    model_class_cloud_auto_encoder = CloudAutoEncoder
    loss = ChamferLoss()
    optimizer = Adam(lr=learning_rate)
    scale_z = 8
    scale_xy = 16

    cloud_autoencoder = model_class_cloud_auto_encoder(
        num_features=modelconfig["num_features"],
        k=modelconfig["k_nearest_neighbours"],
        encoder_type=modelconfig["encoder_type"],
        decoder_type=modelconfig["decoder_type"],
    )

    autoencoder_model = AutoLightningModel.load_from_checkpoint(
        os.path.join("model_dir", "model_name"),
        network=cloud_autoencoder,
        loss_func=loss,
        optim_func=optimizer,
        scale_z=scale_z,
        scale_xy=scale_xy,
    )

    axes = "ZYX"

    num_points = modelconfig["num_points"]
    seg_image = imread("path/to/segmentation/image")
    mask_image = imread("path/to/mask/image")
    batch_size = 24
    AttributeBoxname = "AttributeIDBox"
    TrackAttributeBoxname = "TrackAttributeIDBox"
    TrackidBox = "All"
    TrackMate(
        xml_path,
        spot_csv,
        track_csv,
        edges_csv,
        AttributeBoxname,
        TrackAttributeBoxname,
        TrackidBox,
        axes,
        seg_image=seg_image,
        mask=mask_image,
        autoencoder_model=autoencoder_model,
        num_points=num_points,
        batch_size=batch_size,
        master_extra_name=master_extra_name,
        accelerator=accelerator,
        devices=devices,
        scale_z=scale_z,
        scale_xy=scale_xy,
    )


if __name__ == "__main__":
    main()
