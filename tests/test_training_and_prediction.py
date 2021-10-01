import apoc
from skimage.io import imread, imsave
from pathlib import Path
import numpy as np

def test_training_and_prediction():
    root = Path(apoc.__file__).parent
    img_path = str(root / '..' / 'demo' / 'blobs.tif')
    image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'annotations.tif')
    gt_image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'reference.tif')
    ref_image = imread(img_path)

    feature_specs = "original gaussian_blur=1 sobel_of_gaussian_blur=1"

    classifier = apoc.PixelClassifier()
    classifier.train(feature_specs, gt_image, image)

    result = classifier.predict(feature_specs, image)

    assert np.allclose(result, ref_image)
