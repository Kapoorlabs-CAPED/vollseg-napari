from vollseg import  test_image_ascadian_3d, test_image_carcinoma_3dt, test_image_arabidopsis_3d

def _test_image_cell_3d():
   
    return [(test_image_ascadian_3d(), {'name': 'ascadian_embryo_3d'})]

def _test_image_cell_3dt():
    
    return [(test_image_carcinoma_3dt(), {'name': 'carcinoma_cells_3dt'})]


def _test_image_arabidopsis_3d():
   
    return [(test_image_arabidopsis_3d(), {'name': 'arabidopsis_3d'})]    