# Segmentation of mammary gland cells

VollSeg specializes with its seed pooling approach to segment irregular shapes of mammary gland cells

![Mammary Gland Cells](images/Seg_compare-big.png) 

from human or mouse samples. In the orignal algorithm we use a CARE trained denoising model and either use the U-Net model for semantic segmentation or use a denoising model for the semantic segmentation depending on which model has a better prediction. Denoising model is used to denoise the image first and the result is then passed to the segmentation models. For using combination of (U-Net, CARE and StarDist) model with U-Net as the model for semantic segmentation use this [script](scripts/mammary_gland_us.py) if you want to use the denoised image as the base image for creating the semantic segmentation map using Otsu threshold set the parameter ```dounet=False``` in that same script.