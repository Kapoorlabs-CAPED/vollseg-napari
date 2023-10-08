# Spheroids joint Nuclei and Membrane Segmentation

For segmentation of such cells in 3D we developed a new class inside VollSeg called [VollOne](https://github.com/Kapoorlabs-CAPED/VollSeg/blob/18b33de516e691cbc488e0aedb46b08b2ffd992e/src/vollseg/utils.py#L4612) that assumes a one-to-one correspondence between the nuclei and membrane labelled cells and used the nuclei segmentation channel to segment also the membranes. The steps in this segmentation approach:

## Denoising Membrane
In this workflow we train a model to enhance the edges of the membrane channel using a trained CARE model. Optionally this class also accepts a ROI or ProjectionCARE model to create a 2D region of interest that contains the signal. 

## VollOne Algorithm

The method used to segment the dual channels relies on the VollOne method inside the VollSeg package. Input to this method are the dual channel images, the channel number specifying which channel is membrane and which is nuclei, a denoising model for membrane, a region of interest model for membrane and a StarDist model for nuclei. The denoising model for membrane transforms the input membrane channel to a probability map of the edges of the membrane. The region of interest model for the membrane channel segments the obtained probability map of the previous step into a 2D mask that contains the cells. The StarDist model segments the nuclei channel and outputs the seeds of each segmented region. We then use these seeds on the probability map of the membrane to do a marker controlled watershed in 3D giving instance segmentation labels for the membrane channel.

## Script

The script used for this step are: (for CZYX) [script](scripts/spheroid_nuclei_membrane_segmentation.py) and (for CTZYX) [script](scripts/timelapse_spheroids_joint_segmentation.py). 



