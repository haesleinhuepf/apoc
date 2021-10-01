import apoc
from skimage.io import imread, imsave
from pathlib import Path
import numpy as np

def test_overwriting_cl_file():
    root = Path(apoc.__file__).parent
    img_path = str(root / '..' / 'demo' / 'blobs.tif')
    image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'annotations.tif')
    gt_image = imread(img_path)

    feature_specs = "original gaussian_blur=1 sobel_of_gaussian_blur=1"

    classifier = apoc.ObjectSegmenter(positive_class_identifier=2)
    classifier.train(feature_specs, gt_image, image)

    classifier = apoc.ObjectSegmenter(positive_class_identifier=3)
    classifier.train(feature_specs, gt_image, image)
    assert classifier.positive_class_identifier == 3
