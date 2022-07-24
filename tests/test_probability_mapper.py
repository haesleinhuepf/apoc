import apoc
from skimage.io import imread, imsave
from pathlib import Path
import numpy as np

def test_probability_mapper():
    root = Path(apoc.__file__).parent
    img_path = str(root / '..' / 'demo' / 'blobs.tif')
    image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'annotations.tif')
    gt_image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'reference_probability.tif')
    ref_image = imread(img_path)

    feature_specs = "original gaussian_blur=1 sobel_of_gaussian_blur=1"

    classifier = apoc.ProbabilityMapper(output_probability_of_class=2, num_ensembles=10)

    print(classifier)
    assert "ProbabilityMapper" in str(classifier)
    assert "Output probability of class: 2" in str(classifier)

    classifier.train(feature_specs, gt_image, image)

    print(classifier)
    assert "ProbabilityMapper" in str(classifier)
    assert "Output probability of class: 2" in str(classifier)

    result = classifier.predict(image=image)

    assert result.dtype == np.float32

    assert np.allclose(result, ref_image)

    print(classifier)
    assert "ProbabilityMapper" in str(classifier)

    classifier = apoc.ProbabilityMapper(output_probability_of_class=2, num_ensembles=10)

    print(classifier)
    assert "ProbabilityMapper" in str(classifier)
    assert "Output probability of class: 2" in str(classifier)

    result = classifier.predict(image=image)

    assert result.dtype == np.float32

    assert np.allclose(result, ref_image)

    print(classifier)
    assert "ProbabilityMapper" in str(classifier)
