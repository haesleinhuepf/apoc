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

    apoc.erase_classifier("test.cl")
    classifier = apoc.ObjectSegmenter(opencl_filename="test.cl", positive_class_identifier=2, num_ensembles=10)

    assert len(classifier.feature_importances().keys()) == 0

    print(classifier)
    assert 'ObjectSegmenter' in str(classifier)
    assert 'Positive class identifier: 2' in str(classifier)

    classifier.train(feature_specs, gt_image, image)

    assert len(classifier.feature_importances().keys()) == 3

    print(classifier)
    assert 'ObjectSegmenter' in str(classifier)
    assert 'Positive class identifier: 2' in str(classifier)

    result = classifier.predict(image=image)

    assert result.dtype == np.uint32

    assert np.allclose(result, ref_image)

    classifier = apoc.ObjectSegmenter(opencl_filename="test.cl")

    print(classifier)
    assert 'ObjectSegmenter' in str(classifier)
    assert 'Positive class identifier: 2' in str(classifier)

    result = classifier.predict(image=image)

    print(classifier)
    assert 'ObjectSegmenter' in str(classifier)

