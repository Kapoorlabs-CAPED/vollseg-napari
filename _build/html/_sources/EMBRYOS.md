# Confocal and Light Sheet imaged Embryonic cells

## Ascadian Embryo
In this example we consider a dataset imaged using Light sheet fused from four angles to create a single channel 3D image of Phallusia Mammillata Embryo created using live SPIM imaging. The training data can be found [here](https://figshare.com/articles/dataset/Astec-half-Pm1_Cut_at_2-cell_stage_half_Phallusia_mammillata_embryo_live_SPIM_imaging_stages_6-16_/11309570?backTo=/s/765d4361d1b073beedd5). For this imaging modality we trained only a [UNet model](https://zenodo.org/record/6337699) to segment the interior region of the cells and by using ```slice_merge=True``` and 
```expand_labels=True``` in the VollSeg parameter setting we obtained the following segmentation result along with the metrics compared to the ground truth.
<img src="https://github.com/kapoorlab/vollseg-napari/blob/main/vollseg_napari/images/Ascadian_raw.png"  title="Raw Ascadian Embryo" width="200">
<img src="https://github.com/kapoorlab/vollseg-napari/blob/main/vollseg_napari/images/Ascadian_pred.png" title="Prediction Ascadian Embryo" width="200" > | <img src="https://github.com/kapoorlab/vollseg-napari/blob/main/vollseg_napari/images/Metrics_Ascadian.png" title="Metrics Ascadian Embryo" width="200" > 
