from vollseg_napari import napari_get_reader
import numpy as np
from tifffile import imwrite
import pytest

@pytest.fixture
def write_results(tmp_path):
    

    def write_func(filename):

       test_file, original_data, reader = test_get_reader_returns_callable(tmp_path, filename)
       
       return test_file, original_data, reader


    return write_func
#tmp_path is a pytest fixture    
def test_get_reader_returns_callable(tmp_path, filename = 'file.tif'):
     
       test_file = str(tmp_path/filename)
       original_data = np.random.rand(20,20,20)
       reader = napari_get_reader(test_file)
       imwrite(test_file, original_data)
       assert callable(reader), f'{reader} is not a valid file to be read by this function' 

       return test_file, original_data, reader


def test_reader_round_trip(write_results):


           test_file, original_data, reader = write_results("file.tif")

           layer_data_list = reader(test_file)

           assert isinstance(layer_data_list, list) and len(layer_data_list) > 0

           layer_data_tuple = layer_data_list[0]

           layer_data = layer_data_tuple[0]

           np.testing.assert_allclose(layer_data, original_data)
   

