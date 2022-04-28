from vollseg import get_data


def test_open(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(get_data.get_test_data())

   