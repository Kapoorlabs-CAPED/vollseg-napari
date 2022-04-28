from vollseg import test_image_carcinoma_3dt

def test_open(make_napari_viewer):
    viewer = make_napari_viewer()
  
    viewer.add_image(test_image_carcinoma_3dt())
   