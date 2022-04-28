from vollseg import data


def test_open(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(data.get_test_data())

   