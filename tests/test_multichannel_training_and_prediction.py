from skimage.data import cells3d
from skimage.io import imshow, imsave, imread
import napari
import numpy as np
import apoc
from pathlib import Path

def test_multichannel_training_and_prediction():
    image = cells3d()
    image_ch1 = image[30, 0]
    image_ch2 = image[30, 1]

    root = Path(apoc.__file__).parent

    filename = str(root / '..' / 'demo' / 'cells_annotation.tif')
    annotation = imread(filename)

    img_path = str(root / '..' / 'demo' / 'cells_result.tif')
    ref_image = imread(img_path)

    from apoc import PixelClassifier

    # define features: original image, a blurred version and an edge image
    features = "original gaussian_blur=2 sobel_of_gaussian_blur=2"

    # this is where the model will be saved
    cl_filename = 'test_pixel_classifier_multichannel.cl'

    clf = PixelClassifier(opencl_filename=cl_filename)
    clf.train(features=features, ground_truth=annotation, image=[image_ch1, image_ch2])

    result = clf.predict(image=[image_ch1, image_ch2])

    assert np.allclose(result, ref_image)

    clf = PixelClassifier(opencl_filename=cl_filename)

    result = clf.predict(image=[image_ch1, image_ch2])

    assert np.allclose(result, ref_image)

