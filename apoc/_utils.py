from typing import Union

import pyclesperanto_prototype as cle
import inspect
import pyopencl
from ._feature_sets import PredefinedFeatureSet
import os

def generate_feature_stack(image, features_specification : Union[str, PredefinedFeatureSet] = None):
    """
    Creates a feature stack from a given image.

    Parameters
    ----------
    image : ndarray
        2D or 3D image to generate a feature stack from
    features_specification : str or PredefinedFeatureSet
        a space-separated list of features, e.g.
        original gaussian=4 sobel_of_gaussian=4 or a
        PredefinedFeatureSet

    Returns
    -------
    a list of OCLarray images
    """

    image = cle.push(image)

    # default features
    if features_specification is None:
        blurred = cle.gaussian_blur(image, sigma_x=2, sigma_y=2, sigma_z=2)
        edges = cle.sobel(blurred)
        stack = [
            image,
            blurred,
            edges
        ]

        return stack
    if isinstance(features_specification, PredefinedFeatureSet):
        features_specification = features_specification.value

    while "  " in features_specification:
        features_specification = features_specification.replace("  ", " ")
    while "\t" in features_specification:
        features_specification = features_specification.replace("\t", " ")

    features_specs = features_specification.split(" ")
    generated_features = {}

    result_features = []

    for spec in features_specs:
        if spec.lower() == 'original':
            generated_features['original'] = image
            result_features.append(image)
        elif "=" in spec:
            temp = spec.split("=")
            operation = temp[0]
            numeric_parameter = float(temp[1])

            if not hasattr(cle, operation) and "_of_" in operation:
                temp = operation.split("_of_")
                outer_operation = temp[0]
                inner_operation = temp[1]

                if (inner_operation+"="+str(numeric_parameter)) not in generated_features.keys():
                    new_image = cle.create_like(image)
                    _apply_operation(inner_operation, image, new_image, numeric_parameter)
                    generated_features[inner_operation+"="+str(numeric_parameter)] = new_image

                if (operation+"="+str(numeric_parameter)) not in generated_features.keys():
                    new_image2 = cle.create_like(image)
                    _apply_operation(outer_operation, generated_features[inner_operation+"="+str(numeric_parameter)], new_image2, numeric_parameter)
                    generated_features[operation+"="+str(numeric_parameter)] = new_image2
            else:
                if (operation+"="+str(numeric_parameter)) not in generated_features:
                    new_image = cle.create_like(image)
                    _apply_operation(operation, image, new_image, numeric_parameter)
                    generated_features[operation+"="+str(numeric_parameter)] = new_image

            result_features.append(generated_features[operation+"="+str(numeric_parameter)])

    return result_features

def _apply_operation(operation, input_image, output_image, numeric_parameter):
    """Apply a given image-filter to an image and save the result into another new_image.

    Parameters
    ----------
    operation: callable
    input_image: ndimage
    output_image: ndimage
    numeric_parameter: float or int
        The filters typically have numeric parameters, such as radius or sigma.
    """
    func = getattr(cle, operation)
    sig = inspect.signature(func)
    if len(sig.parameters.keys()) == 2:
        func(input_image, output_image)
    elif len(sig.parameters.keys()) == 3:
        func(input_image, output_image, numeric_parameter)
    elif len(sig.parameters.keys()) == 4:
        func(input_image, output_image, numeric_parameter, numeric_parameter)
    elif len(sig.parameters.keys()) == 5:
        func(input_image, output_image, numeric_parameter, numeric_parameter, numeric_parameter)
    elif len(sig.parameters.keys()) == 8:
        # e.g. difference_of_gaussian
        func(input_image, output_image, numeric_parameter * 0.9, numeric_parameter * 0.9, numeric_parameter * 0.9, numeric_parameter * 1.1, numeric_parameter * 1.1, numeric_parameter * 1.1)
    else:
        func(input_image, output_image, numeric_parameter, numeric_parameter, numeric_parameter)

def _read_something_from_opencl_file(opencl_filename, some_key:str, default_value=None):
    """APOC's OpenCL files have a header in ini-format. Using this method, we can read entries from that header.

    Parameters
    ----------
    opencl_filename: str, filename
    some_key: str
        We'll search for a line that starts with that string and return the rest of the line where we found it.
    default_value
        If we can't find any line starting with some_key, we return this default value

    Returns
    -------
    str
    """
    if not os.path.exists(opencl_filename):
        return default_value

    with open(opencl_filename) as f:
        line = ""
        count = 0
        while line != "*/" and line is not None and count < 25:
            count = count + 1
            line = f.readline()
            if line.startswith(some_key):
                return line.replace(some_key, "").replace("\n","")

    return default_value


def erase_classifier(filename):
    """Deletes a file in case it exists.

    Parameters
    ----------
    filename: str
    """
    if os.path.exists(filename):
        os.remove(filename)


def list_available_object_classification_features():
    """List available features for object classifiaction

    Returns
    -------
    list(str)
    """
    import numpy as np
    from pyclesperanto_prototype import statistics_of_labelled_pixels

    dummy_image = np.asarray([[0,1]])
    stats = statistics_of_labelled_pixels(dummy_image, dummy_image)

    return list(stats.keys()) + \
        ["touching_neighbor_count",
        "average_distance_of_touching_neighbors",
        "average_distance_of_n_nearest_neighbors=?",
        "average_distance_of_n_nearest_neighbors=?",
        ]


def train_classifier_from_image_folders(classifier, features, **kwargs):
    """Train any classifier from pairs or tuples of image folders.

    When passing multiple folders for the training, it is necessary that
    the images in these folders have identical filenames. For example,
    to train a ObjectSegmenter, you need to pass two folders where the
    images are organized like this:

    * folder/
      * images/
        * image1.tif
        * image2.tif
        * image3.tif
      * masks/
        * image1.tif
        * image2.tif
        * image3.tif

    Parameters
    ----------
    classifier: APOC classifier
        e.g. PixelClassifier, ObjectClassifier, ObjectSegmenter, ProbabilityMapper
    features: str
        list of features, space separated
    kwargs
        depends on the classifier. See the classifier's train() function for details
    """
    import os
    from skimage.io import imread

    any_folder = kwargs[list(kwargs.keys())[0]]
    file_list = os.listdir(any_folder)
    continue_training = False
    for filename in file_list:

        # assemble parameters for training
        kwargs_to_pass = {
            'features': features,
            'continue_training': continue_training
        }
        for key, value in kwargs.items():
            kwargs_to_pass[key] = imread(kwargs[key] + filename)

        # run training
        classifier.train(**kwargs_to_pass)
        continue_training = True
