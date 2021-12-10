import os
import sys
sys.path.append('../')
import napari
from _dock_widget import *



if __name__ == '__main__':
    viewer = napari.Viewer()
    viewer.window.add_plugin_dock_widget(plugin_wrapper())
    napari.run()

