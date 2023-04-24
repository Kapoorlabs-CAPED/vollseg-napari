import napari
from tifffile import imread

def show_napari_3d():
    viewer =  napari.Viewer()
    viewer.window.add_plugin_dock_widget('vollseg-napari','VollSeg')
    napari.run()
    
 
if __name__ == '__main__':
    viewer = show_napari_3d()
  
    napari.run()
    