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

    filename = "test_pixel_classifier.cl"
    apoc.erase_classifier(filename)
    classifier = apoc.PixelClassifier(opencl_filename=filename, num_ensembles=10)

    assert len(classifier.feature_importances().keys()) == 0

    classifier.train(feature_specs, gt_image, image)

    assert len(classifier.feature_importances().keys()) == 3

    print(classifier)
    assert "Ground truth dimensions: 2" in str(classifier)

    result = classifier.predict(image=image)

    assert result.dtype == np.uint32

    assert np.allclose(result, ref_image)

    feature_importances = classifier.feature_importances()
    print(feature_importances)

    assert feature_importances["original"] > 0.3
    assert feature_importances["gaussian_blur=1"] > 0.3
    assert feature_importances["sobel_of_gaussian_blur=1"] < 0.1


def test_multichannel_training_and_prediction():
    root = Path(apoc.__file__).parent
    img_path = str(root / '..' / 'demo' / 'blobs.tif')
    image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'annotations.tif')
    gt_image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'reference_multichannel.tif')
    ref_image = imread(img_path)

    feature_specs = "original gaussian_blur=1 sobel_of_gaussian_blur=1"

    filename = "test_pixel_classifier.cl"
    apoc.erase_classifier(filename)
    classifier = apoc.PixelClassifier(opencl_filename=filename, num_ensembles=10)

    assert len(classifier.feature_importances().keys()) == 0

    classifier.train(feature_specs, gt_image, [image, image])

    assert len(classifier.feature_importances().keys()) == 6

    result = classifier.predict(image=[image, image])

    assert result.dtype == np.uint32

    assert np.allclose(result, ref_image)


def test_continue_training_and_prediction():
    root = Path(apoc.__file__).parent
    img_path = str(root / '..' / 'demo' / 'blobs.tif')
    image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'annotations.tif')
    gt_image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'reference_continue_training.tif')
    ref_image = imread(img_path)

    feature_specs = "original gaussian_blur=1 sobel_of_gaussian_blur=1"

    classifier = apoc.PixelClassifier(num_ensembles=10)
    classifier.train(feature_specs, gt_image, image, continue_training=True)
    classifier.train(feature_specs, gt_image, image, continue_training=True)

    result = classifier.predict(image=image)

    assert result.dtype == np.uint32

    assert np.allclose(result, ref_image)

def test_compare_cpu_gpu_prediction():
    root = Path(apoc.__file__).parent
    img_path = str(root / '..' / 'demo' / 'blobs.tif')
    image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'annotations.tif')
    gt_image = imread(img_path)

    feature_specs = "original gaussian_blur=1 sobel_of_gaussian_blur=1"

    classifier = apoc.PixelClassifier()
    classifier.train(feature_specs, gt_image, image)

    result_cpu = classifier._predict_cpu(image=image)
    result_gpu = classifier.predict(image=image)

    import pyclesperanto_prototype as cle

    same = cle.equal(result_gpu, result_cpu)
    num_same = cle.sum_of_all_pixels(same)

    different = cle.not_equal(result_gpu, result_cpu)
    num_different = cle.sum_of_all_pixels(different)

    print("same", num_same)
    print("different", num_different)

    differently_classified_ratio = num_different / (num_same + num_different)
    print("differently classified ratio", differently_classified_ratio)

    assert differently_classified_ratio < 0.05