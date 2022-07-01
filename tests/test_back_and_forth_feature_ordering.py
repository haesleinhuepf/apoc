import apoc
from skimage.io import imread, imsave
from pathlib import Path
import numpy as np


def test_back_and_forth_feature_ordering():
    root = Path(apoc.__file__).parent
    img_path = str(root / '..' / 'demo' / 'blobs.tif')
    image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'annotations.tif')
    gt_image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'reference_labels.tif')
    ref_image = imread(img_path)

    feature_specs = "original gaussian_blur=1 sobel_of_gaussian_blur=1"

    classifier = apoc.ObjectSegmenter(positive_class_identifier=2, num_ensembles=100)
    classifier.train(feature_specs, gt_image, image)

    result1 = classifier.predict(image=image)

    feature_specs = "sobel_of_gaussian_blur=1 gaussian_blur=1 original"

    classifier = apoc.ObjectSegmenter(positive_class_identifier=2, num_ensembles=100)
    classifier.train(feature_specs, gt_image, image)

    result2 = classifier.predict(image=image)

    binary1 = result1 > 0
    binary2 = result2 > 0

    imsave("binary1.tif", binary1)
    imsave("binary2.tif", binary2)

    intersection = binary1 * binary2
    union = (binary1 + binary2) > 0

    jaccard_index = intersection.sum() / union.sum()

    print(jaccard_index)

    assert jaccard_index > 0.999


