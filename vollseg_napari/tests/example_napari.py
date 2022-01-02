from stardist.data import  test_image_nuclei_2d
import napari
from tifffile import imread

def show_napari_3d():
    x = imread("/home/varunkapoor/Downloads/Testdatasets/Nuclei3D-2.tif")
    viewer =  napari.Viewer()
    viewer.add_image(x)
    viewer.window.add_plugin_dock_widget('VollSeg')
    
    
def show_napari_2d():
    x = test_image_nuclei_2d()
    viewer =  napari.Viewer()
    viewer.add_image(x)
    viewer.window.add_plugin_dock_widget('VollSeg')

def show_napari_3d_test():
    x = imread("/home/varunkapoor/Downloads/Testdatasets/Nuclei3D-2.tif")
    viewer =  napari.Viewer()
    viewer.add_image(x)
    viewer.window.add_plugin_dock_widget('TestVollSeg')
    
    
def show_napari_2d_test():
    x = test_image_nuclei_2d()
    viewer =  napari.Viewer()
    viewer.add_image(x)
    viewer.window.add_plugin_dock_widget('TestVollSeg')

if __name__ == '__main__':
    viewer = show_napari_2d()
  
    napari.run()
