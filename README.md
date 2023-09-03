# VollSeg Napari Plugin



[![PyPI version](https://img.shields.io/pypi/v/vollseg-napari.svg)](https://pypi.org/project/vollseg-napari)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/vollseg-napari)](https://napari-hub.org/plugins/vollseg-napari)
[![License](https://img.shields.io/pypi/l/napari-metroid.svg?color=green)](https://github.com/kapoorlab/napari-vollseg/raw/main/LICENSE)
[![codecov](https://codecov.io/gh/kapoorlab/napari-vollseg/branch/main/graph/badge.svg)](https://codecov.io/gh/kapoorlab/napari-vollseg)
[![Twitter Badge](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/entracod)


VollSeg is more than just a single segmentation algorithm; it is a meticulously designed modular segmentation tool tailored to diverse model organisms and imaging methods. While a U-Net might suffice for certain image samples, others might benefit from utilizing StarDist, and some could require a blend of both, potentially coupled with denoising or region of interest models. The pivotal decision left to make is how to select the most appropriate VollSeg configuration for your dataset, a question we comprehensively address in our [documentation website](https://kapoorlabs-caped.github.io/vollseg-napari/).

This project provides the [napari](https://napari.org/) plugin for [VollSeg](https://github.com/kapoorlab/vollseg), a deep learning based 2D and 3D segmentation tool for irregular shaped cells. VollSeg has originally been developed (see [papers](http://conference.scipy.org/proceedings/scipy2021/varun_kapoor.html)) for the segmentation of densely packed membrane labelled cells in challenging images with low signal-to-noise ratios. The plugin allows to apply pretrained and custom trained models from within napari.
For detailed demo of the plugin see these [videos](https://www.youtube.com/watch?v=W_gKrLWKNpQ) and a short video about the [parameter selection](https://www.youtube.com/watch?v=7tQMn_u8_7s&t=1s) 


## Installation & Usage

Install the plugin with `pip install vollseg-napari` or from within napari via `Plugins > Install/Uninstall Package(s)…`. 

You can activate the plugin in napari via `Plugins > VollSeg: VollSeg`. Example images for testing are provided via `File > Open Sample > VollSeg`.

If you use this plugin for your research, please [cite us](http://conference.scipy.org/proceedings/scipy2021/varun_kapoor.html).


## Examples

VollSeg comes with different options to combine CARE based denoising with UNET, StarDist and segmentation in a region of interest (ROI). We present some examples which are represent optimal combination of these different modes for segmenting different cell types. We summarize this in the table below:

| Example Image | Description | Training Data | Trained Model | GT image | Optimal combination | Notebook Code | Model Prediction | Metrics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ![Raw Ascadian Embryo](images/Ascadian_raw.png) | Light sheet fused from four angles 3D single channel | [Training Data ~320 GB](https://figshare.com/articles/dataset/Astec-half-Pm1_Cut_at_2-cell_stage_half_Phallusia_mammillata_embryo_live_SPIM_imaging_stages_6-16_/11309570?backTo=/s/765d4361d1b073beedd5) | [UNET model](https://zenodo.org/record/6337699) | ![GT Ascadian Embryo](images/Ascadian_GT.png) | UNET model, slice_merge = False | [Colab Notebook](https://github.com/kapoorlab/VollSeg/blob/main/examples/Predict/Colab_VollSeg_Ascadian_Embryo.ipynb) | ![Prediction Ascadian Embryo](images/Ascadian_pred.png) | ![Metrics Ascadian Embryo](images/Metrics_Ascadian.png) |
| ![Raw Carcinoma](images/Carcinoma_raw.png) | Confocal microscopy 3D single channel 8 bit | [Training Data](https://zenodo.org/record/5904082#.Yi8-BnrMJD8) | [Denoising Model](https://zenodo.org/record/5910645/) and [StarDist Model](https://zenodo.org/record/6354077/) | ![GT Carcinoma](images/Carcinoma_GT.png) | StarDist model + Denoising Model, dounet = False | [Colab Notebook](https://github.com/kapoorlab/VollSeg/blob/main/examples/Predict/Colab_VollSeg_Mamary_gland.ipynb) | ![Prediction Carcinoma Cells](images/Carcinoma_pred.png) | ![Metrics Carcinoma Cells](images/Metrics_carcinoma.png) |
| ![Raw Xenopus Tissue](images/Xenopus_tissue_raw.png) | LaserScanningConfocalMicroscopy 2D single channel | [Dataset](https://zenodo.org/record/6076614#.YjBaNnrMJD8) | [UNET Model](https://zenodo.org/record/6060378/) | ![GT Xenopus Tissue](images/Xenopus_tissue_GT.png) | UNET model | [Colab Notebook](https://github.com/kapoorlab/VollSeg/blob/main/examples/Predict/Colab_VollSeg_tissue_segmentation.ipynb) | ![Prediction Xenopus Tissue](images/Xenopus_tissue_pred.png) | No Metrics |
| ![Raw Microtubule Kymograph](images/microtubule_kymo_raw.png) | TIRF + MultiKymograph Fiji tool 2D single channel | [Training Dataset](https://zenodo.org/record/6355705/files/Microtubule_edgedetector_training.zip) | [UNET Model](https://zenodo.org/record/6355705/) | ![GT Microtubule Kymograph](images/microtubule_kymo_GT.png) | UNET model | [Colab Notebook](https://github.com/kapoorlab/VollSeg/blob/main/examples/Predict/Colab_Microtubule_kymo_segmentation.ipynb) | ![Prediction Microtubule Kymograph](images/microtubule_kymo_pred.png) | No Metrics |
| ![Raw Lung Xray](images/lung_xray_raw.png) | XRay of Lung 2D single channel | [Training Dataset](https://www.kaggle.com/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset) | [UNET Model](https://zenodo.org/record/6060177/) | ![GT Lung Xray](images/lung_xray_GT.png) | UNET model | [Colab Notebook](https://github.com/kapoorlab/VollSeg/blob/main/examples/Predict/Colab_Microtubule_kymo_segmentation.ipynb) | ![Prediction Lung Xray](images/lung_xray_pred.png) | ![Metrics Lung Xray](images/Metrics_lung_xray.png) |
| ![Raw Nuclei Mask](images/nuclei_mask_raw.png) | LaserScanningConfocalMicroscopy 2D single channel | [Test Dataset](https://zenodo.org/record/6359349/) | Private | ![GT Nuclei Mask](images/nuclei_mask_GT.png) | UNET model | [Colab Notebook](https://github.com/kapoorlab/VollSeg/blob/main/examples/Predict/Colab_Microtubule_kymo_segmentation.ipynb) | ![Prediction Nuclei Mask](images/nuclei_mask_pred.png) | No Metrics |
| ![Raw Nuclei](images/nuclei_raw.png) | LaserScanningConfocalMicroscopy 3D single channel | [Test Dataset](https://zenodo.org/record/6359295/) | Private | ![GT Nuclei](images/nuclei_GT.png) | UNET model + StarDist model + ROI model | [Colab Notebook](https://github.com/kapoorlab/VollSeg/blob/main/examples/Predict/Colab_VollSeg_star_roi.ipynb) | ![Prediction Nuclei](images/nuclei_pred.png) | ![Metrics Nuclei](images/Metrics_nuclei.png) |


## Troubleshooting & Support

- The [image.sc forum](https://forum.image.sc/tag/vollseg) is the best place to start getting help and support. Make sure to use the tag `vollseg`, since we are monitoring all questions with this tag.
- If you have technical questions or found a bug, feel free to [open an issue](https://github.com/kapoorlab/vollseg-napari/issues).

