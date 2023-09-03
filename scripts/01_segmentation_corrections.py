import os
from vollseg.utils import SegCorrect

def main( ):

    imagedir = 'raw_membrane_seg_corrections_dir'
    segmentationdir = 'membrane_seg_corrections_dir'

    segcorrect = SegCorrect(imagedir, segmentationdir)

    segcorrect.showNapari()

if __name__ == '__main__':

    main() 
