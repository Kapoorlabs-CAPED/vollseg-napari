from vollseg import get_data


def test_open(make_napari_viewer):
    viewer = make_napari_viewer()
    image = get_data.get_test_data()
    viewer.add_image(get_data.get_test_data()[0:2,image.shape[1] - 5:image.shape[1] + 5, image.shape[2] - 5:image.shape[2] + 5, image.shape[3] - 5:image.shape[3] + 5])

   