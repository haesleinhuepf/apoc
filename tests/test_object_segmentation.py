import apoc
from skimage.io import imread, imsave
from pathlib import Path
import numpy as np

def test_object_segmentation():
    root = Path(apoc.__file__).parent
    img_path = str(root / '..' / 'demo' / 'blobs.tif')
    image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'annotations.tif')
    gt_image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'reference_labels.tif')
    ref_image = imread(img_path)

    feature_specs = "original gaussian_blur=1 sobel_of_gaussian_blur=1"

    classifier = apoc.ObjectSegmenter(positive_class_identifier=2)
    classifier.train(feature_specs, gt_image, image)

    result = classifier.predict(features=feature_specs, image=image)

    assert result.dtype == np.uint32

    assert np.allclose(result, ref_image)
