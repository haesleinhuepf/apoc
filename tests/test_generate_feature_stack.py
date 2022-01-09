from apoc import generate_feature_stack
import apoc
from skimage.io import imread
from pathlib import Path
import pyclesperanto_prototype as cle

def test_generate_feature_stack():
    root = Path(apoc.__file__).parent
    img_path = str(root / '..' / 'demo' / 'blobs.tif')

    image = imread(img_path)

    feature_specs = "original gaussian_blur=5 sobel_of_gaussian_blur=5 laplace_box_of_gaussian_blur=3 difference_of_gaussian=4"

    feature_stack = generate_feature_stack(image, feature_specs)

    assert len(feature_stack) == len(feature_specs.split(" "))

    print("a")
    assert cle.mean_squared_error(feature_stack[0], image) == 0
    print("b")
    assert cle.mean_squared_error(feature_stack[1], cle.gaussian_blur(image, sigma_x=5, sigma_y=5, sigma_z=5)) == 0
    print("c")
    assert cle.mean_squared_error(feature_stack[2], cle.sobel(cle.gaussian_blur(image, sigma_x=5, sigma_y=5, sigma_z=5))) == 0
    print("d")
    assert cle.mean_squared_error(feature_stack[3], cle.laplace_box(cle.gaussian_blur(image, sigma_x=3, sigma_y=3, sigma_z=3))) == 0
    print("e")
    assert cle.mean_squared_error(feature_stack[4], cle.difference_of_gaussian(image, sigma1_x=4 * 0.9, sigma1_y=4 * 0.9, sigma1_z=4 * 0.9, sigma2_x=4 * 1.1, sigma2_y=4 * 1.1, sigma2_z=4 * 1.1)) == 0


def test_generate_feature_stack_default():
    root = Path(apoc.__file__).parent
    img_path = str(root / '..' / 'demo' / 'blobs.tif')

    image = imread(img_path)

    feature_specs = None

    feature_stack = generate_feature_stack(image, feature_specs)

    print("a")
    assert cle.mean_squared_error(feature_stack[0], image) == 0
    print("b")
    assert cle.mean_squared_error(feature_stack[1], cle.gaussian_blur(image, sigma_x=2, sigma_y=2, sigma_z=2)) == 0
    print("c")
    assert cle.mean_squared_error(feature_stack[2], cle.sobel(cle.gaussian_blur(image, sigma_x=2, sigma_y=2, sigma_z=2))) == 0




