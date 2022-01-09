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

    result = classifier.predict(features=feature_specs, image=image)

    assert np.allclose(result, ref_image)

def test_multichannel_training_and_prediction():
    root = Path(apoc.__file__).parent
    img_path = str(root / '..' / 'demo' / 'blobs.tif')
    image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'annotations.tif')
    gt_image = imread(img_path)

    img_path = str(root / '..' / 'demo' / 'reference_multichannel.tif')
    ref_image = imread(img_path)

    feature_specs = "original gaussian_blur=1 sobel_of_gaussian_blur=1"

    classifier = apoc.PixelClassifier()
    classifier.train(feature_specs, gt_image, [image, image])

    result = classifier.predict(features=feature_specs, image=[image, image])

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

    classifier = apoc.PixelClassifier()
    classifier.train(feature_specs, gt_image, image, continue_training=True)
    classifier.train(feature_specs, gt_image, image, continue_training=True)

    result = classifier.predict(features=feature_specs, image=image)

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

    result_cpu = classifier._predict_cpu(features=feature_specs, image=image)
    result_gpu = classifier.predict(features=feature_specs, image=image)

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