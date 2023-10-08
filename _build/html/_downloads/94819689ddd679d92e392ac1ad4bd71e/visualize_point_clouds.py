import numpy as np
import pandas as pd
import torch
import trimesh
from cellshape_cloud import CloudAutoEncoder
from kapoorlabs_lightning.lightning_trainer import AutoLightningModel
from kapoorlabs_lightning.optimizers import Adam
from kapoorlabs_lightning.pytorch_losses import ChamferLoss
from napatrackmater import load_json
from pyntcloud import PyntCloud
from skimage.measure import marching_cubes
from tifffile import imread


def main():
    binary_image = imread("binary_image.tif")

    autoencoder_model_path = "autoencoder.ckpt"
    model_path_json = "autoencoder.json"

    vertices, faces, normals, values = marching_cubes(binary_image)
    mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    mesh_obj.sample(2048)
    x_coords = mesh_obj.vertices[:, 0]  # X coordinates
    y_coords = mesh_obj.vertices[:, 1]  # Y coordinates
    z_coords = mesh_obj.vertices[:, 2]  # Z coordinates
    points = np.vstack((x_coords, y_coords, z_coords)).T
    point_cloud = PyntCloud(pd.DataFrame(data=points, columns=["x", "y", "z"]))
    point_cloud.plot(mesh=True)
    loss = ChamferLoss()

    optimizer = Adam(lr=0.001)
    modelconfig = load_json(model_path_json)
    cloud_autoencoder = CloudAutoEncoder(
        num_features=modelconfig["num_features"],
        k=modelconfig["k_nearest_neighbours"],
        encoder_type=modelconfig["encoder_type"],
        decoder_type=modelconfig["decoder_type"],
    )
    autoencoder = AutoLightningModel.load_from_checkpoint(
        autoencoder_model_path,
        network=cloud_autoencoder,
        loss_func=loss,
        optim_func=optimizer,
    )

    point_cloud = torch.tensor(point_cloud.points.values)
    mean = torch.mean(point_cloud, 0)
    scale = torch.tensor([[8, 16, 16]])
    point_cloud = (point_cloud - mean) / scale

    outputs, features = autoencoder(point_cloud.unsqueeze(0).to("cuda"))
    outputs = outputs.detach().cpu().numpy()
    outputs = (
        outputs * scale.detach().cpu().numpy() + mean.detach().cpu().numpy()
    )
    outputs = outputs[0, :]
    points = pd.DataFrame(outputs)
    points = pd.DataFrame(points.values, columns=["x", "y", "z"])
    cloud = PyntCloud(points)
    cloud.plot(mesh=True)


if __name__ == "__main__":

    main()
