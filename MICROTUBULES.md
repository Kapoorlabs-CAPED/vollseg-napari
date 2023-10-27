# VollSeg Extension : vollseg-napari-mtrack

## Segmentation of Microtubule and Actin Kymographs
VollSeg extension [mtrack] is a Napari plugin available from the Napari [hub]. This plugin takes a single kymograph or nultiple kymographs of differnet image sizes, segments the edges of the kymographs using a U-Net model. The segmented result is displayed as labels layer and can be manually corrected. Post segmentation the user can choose to fit linear model or a polynomial model to detect multiple growth and shrink events in all the kymographs using RANSAC based fits. These RANSAC based fits are based on our [publication] where we show the efficiency of this method for quantifying dynamic instability parameters of microtubules and by extension to Actin kymographs. The user will see the detected linear fits as shapes layer in the Napari menu and even they can be manually corrected, with each update to the labels or the shape layer the computed dynamic instability parameters table and plots are updated. Please make sure that your image calibration settings are properly set as the computed results are in physical units than pixel units.  


We provide example microtubule data to test the plugin. To get the example data automatically, install the [mtrack] plugin and then click on File -> Open Sample -> Test Microtubule Kymographs.

Detailed explanation of this extension plugin can be found at the documentation website of the plugin [documentation]






[documentation]: https://kapoorlabs-caped.github.io/vollseg-napari-mtrack
[mtrack]: https://github.com/Kapoorlabs-CAPED/vollseg-napari-mtrack
[hub]: https://www.napari-hub.org/plugins/vollseg-napari-mtrack
[publication]: https://www.nature.com/articles/s41598-018-37767-1