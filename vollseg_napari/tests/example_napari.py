from stardist.data import  test_image_nuclei_2d
import napari




def show_napari_2d():
    x = test_image_nuclei_2d()
    viewer =  napari.Viewer()
    viewer.add_image(x)
    viewer.window.add_plugin_dock_widget('VollSeg')



if __name__ == '__main__':
    viewer = show_napari_2d()
  
    napari.run()
