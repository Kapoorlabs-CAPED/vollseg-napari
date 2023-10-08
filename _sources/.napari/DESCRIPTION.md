# VollSeg Napari Plugin



[![PyPI version](https://img.shields.io/pypi/v/vollseg-napari.svg)](https://pypi.org/project/vollseg-napari)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/vollseg-napari)](https://napari-hub.org/plugins/vollseg-napari)
[![License](https://img.shields.io/pypi/l/napari-metroid.svg?color=green)](https://github.com/kapoorlab/napari-vollseg/raw/main/LICENSE)
[![codecov](https://codecov.io/gh/kapoorlab/napari-vollseg/branch/main/graph/badge.svg)](https://codecov.io/gh/kapoorlab/napari-vollseg)
[![Twitter Badge](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/entracod)


VollSeg is more than just a single segmentation algorithm; it is a meticulously designed modular segmentation tool tailored to diverse model organisms and imaging methods. While a U-Net might suffice for certain image samples, others might benefit from utilizing StarDist, and some could require a blend of both, potentially coupled with denoising or region of interest models. The pivotal decision left to make is how to select the most appropriate VollSeg configuration for your dataset, a question we comprehensively address in our [documentation website](https://kapoorlabs-caped.github.io/vollseg-napari/).

This project provides the [napari](https://napari.org/) plugin for [VollSeg](https://github.com/kapoorlab/vollseg), a deep learning based 2D and 3D segmentation tool for irregular shaped cells. VollSeg has originally been developed (see [papers](http://conference.scipy.org/proceedings/scipy2021/varun_kapoor.html)) for the segmentation of densely packed membrane labelled cells in challenging images with low signal-to-noise ratios. The plugin allows to apply pretrained and custom trained models from within napari.
For detailed demo of the plugin see these [videos](https://www.youtube.com/watch?v=W_gKrLWKNpQ) and a short video about the [parameter selection](https://www.youtube.com/watch?v=7tQMn_u8_7s&t=1s) 