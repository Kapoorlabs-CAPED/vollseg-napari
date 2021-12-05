# VollSeg Napari Plugin

[![PyPI version](https://img.shields.io/pypi/v/vollseg-napari.svg)](https://pypi.org/project/vollseg-napari)


This project provides the [napari](https://napari.org/) plugin for [VollSeg](https://github.com/kapoorlab/vollseg), a deep learning based 2D and 3D segmentation tool for irregular shaped cells. VollSeg has originally been developed (see [papers](http://conference.scipy.org/proceedings/scipy2021/varun_kapoor.html)) for the segmentation of densely packed membrane labelled cells in challenging images with low signal-to-noise ratios. The plugin allows to apply pretrained and custom trained models from within napari.



## Installation & Usage

Install the plugin with `pip install vollseg-napari` or from within napari via `Plugins > Install/Uninstall Package(s)…`. If you want GPU-accelerated prediction, please read the more detailed [installation instructions](https://github.com/kapoorlab/vollseg-napari#gpu_installation) for VollSeg.

You can activate the plugin in napari via `Plugins > VollSeg: VollSeg`. Example images for testing are provided via `File > Open Sample > VollSeg`.

If you use this plugin for your research, please [cite us](http://conference.scipy.org/proceedings/scipy2021/varun_kapoor.html).

## GPU_Installation

This package is compatible with Python 3.6 - 3.9.

1. Please first [install TensorFlow](https://www.tensorflow.org/install)
(TensorFlow 2) by following the official instructions.
For [GPU support](https://www.tensorflow.org/install/gpu), it is very
important to install the specific versions of CUDA and cuDNN that are
compatible with the respective version of TensorFlow. (If you need help and can use `conda`, take a look at [this](https://github.com/CSBDeep/CSBDeep/tree/master/extras#conda-environment).)

2. *VollSeg* can then be installed with `pip`:

    - If you installed TensorFlow 2 (version *2.x.x*):

          pip install vollseg



## Troubleshooting & Support

- The [image.sc forum](https://forum.image.sc/tag/vollseg) is the best place to start getting help and support. Make sure to use the tag `vollseg`, since we are monitoring all questions with this tag.
- If you have technical questions or found a bug, feel free to [open an issue](https://github.com/kapoorlab/vollseg-napari/issues).

