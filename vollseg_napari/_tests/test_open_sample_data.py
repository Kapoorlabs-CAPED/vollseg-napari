def test_open(make_napari_viewer):
    viewer = make_napari_viewer()
  
    viewer.open_sample(plugin='vollseg-napari', sample='carcinoma_cells_3dt')
   