# VollSeg Napari Plugin

# Developed by KapoorLabs


<img src="images/mtrack.png" alt="Logo1" width="150"/>
<img src="images/kapoorlablogo.png" alt="Logo2" width="150"/>

This product is a testament to our expertise at KapoorLabs, where we specialize in creating cutting-edge solutions. We offer bespoke pipeline development services, transforming your developmental biology questions into publishable figures with our advanced computer vision and AI tools. Leverage our expertise and resources to achieve end-to-end solutions that make your research stand out.

**Note:** The tools and pipelines showcased here represent only a fraction of what we can achieve. For tailored and comprehensive solutions beyond what was done in the referenced publication, engage with us directly. Our team is ready to provide the expertise and custom development you need to take your research to the next level. Visit us at [KapoorLabs](https://www.kapoorlabs.org/).


[![PyPI version](https://img.shields.io/pypi/v/vollseg-napari.svg)](https://pypi.org/project/vollseg-napari)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/vollseg-napari)](https://napari-hub.org/plugins/vollseg-napari)
[![License](https://img.shields.io/pypi/l/napari-metroid.svg?color=green)](https://github.com/kapoorlab/napari-vollseg/raw/main/LICENSE)
[![codecov](https://codecov.io/gh/kapoorlab/napari-vollseg/branch/main/graph/badge.svg)](https://codecov.io/gh/kapoorlab/napari-vollseg)
[![Twitter Badge](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/entracod)

## Segmentation Algorithm

VollSeg is more than just a single segmentation algorithm; it is a meticulously designed modular segmentation tool tailored to diverse model organisms and imaging methods. While a U-Net might suffice for certain image samples, others might benefit from utilizing StarDist, and some could require a blend of both, potentially coupled with denoising or region of interest models. The pivotal decision left to make is how to select the most appropriate VollSeg configuration for your dataset, a question we comprehensively address in our [documentation website](https://kapoorlabs-caped.github.io/vollseg-napari/).

This project provides the [napari](https://napari.org/) plugin for [VollSeg](https://github.com/kapoorlab/vollseg), a deep learning based 2D and 3D segmentation tool for irregular shaped cells. VollSeg has originally been developed (see [papers](http://conference.scipy.org/proceedings/scipy2021/varun_kapoor.html)) for the segmentation of densely packed membrane labelled cells in challenging images with low signal-to-noise ratios. The plugin allows to apply pretrained and custom trained models from within napari.
For detailed demo of the plugin see these [videos](https://www.youtube.com/watch?v=W_gKrLWKNpQ) and a short video about the [parameter selection](https://www.youtube.com/watch?v=7tQMn_u8_7s&t=1s) 


## Installation & Usage

Install the plugin with `pip install vollseg-napari` or from within napari via `Plugins > Install/Uninstall Package(s)…`. 

You can activate the plugin in napari via `Plugins > VollSeg: VollSeg`. Example images for testing are provided via `File > Open Sample > VollSeg`.

If you use this plugin for your research, please [cite us](http://conference.scipy.org/proceedings/scipy2021/varun_kapoor.html).


## Examples

VollSeg comes with different options to combine CARE based denoising with UNET, StarDist and segmentation in a region of interest (ROI). We present some examples which are represent optimal combination of these different modes for segmenting different cell types. We summarize this in the table below:
| Example Image | Description | Training Data | Trained Model |
| --- | --- |--- | --- |
| ![Raw Ascadian Embryo](images/Ascadian_raw.png)| Light sheet fused from four angles 3D single channel| [Training Data ~320 GB](https://figshare.com/articles/dataset/Astec-half-Pm1_Cut_at_2-cell_stage_half_Phallusia_mammillata_embryo_live_SPIM_imaging_stages_6-16_/11309570?backTo=/s/765d4361d1b073beedd5)| [UNET model](https://zenodo.org/record/6337699) |
| ![Raw Carcinoma](images/Carcinoma_raw.png)| Confocal microscopy 3D single channel 8 bit| [Training Data](https://zenodo.org/record/5904082#.Yi8-BnrMJD8)| [Denoising Model](https://zenodo.org/record/5910645/) and [StarDist Model](https://zenodo.org/record/6354077/) |
| ![Raw Xenopus Tissue](images/Xenopus_tissue_raw.png)| LaserScanningConfocalMicroscopy 2D single channel| [Dataset](https://zenodo.org/record/6076614#.YjBaNnrMJD8)| [UNET Model](https://zenodo.org/record/6060378/)  |



## Troubleshooting & Support

- The [image.sc forum](https://forum.image.sc/tag/vollseg) is the best place to start getting help and support. Make sure to use the tag `vollseg`, since we are monitoring all questions with this tag.
- If you have technical questions or found a bug, feel free to [open an issue](https://github.com/kapoorlab/vollseg-napari/issues).

## Authors

- Varun Kapoor <randomaccessiblekapoor@gmail.com>
- Mari Tolonen
- Jakub Sedzinski